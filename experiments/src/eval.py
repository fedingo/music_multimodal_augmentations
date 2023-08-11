from .utils import batchify, tqdm, MultiEpochsDataLoader, LoadAudioLive
import torch
from typing import List, Callable, Dict
from tasks import AbstractTask
import wandb
import pandas as pd


def evaluate(
    model,
    dataset: List[dict],
    task: AbstractTask,
    load_audio: bool = False,
    batch_size: int = 16,
    save_predictions = False,
):
            model.train(False)
            outputs = []
            pbar = batchify(
                    dataset, 
                    batch_size=batch_size, 
                    epochs = 1,
                    load_audio = load_audio,
                )

            with torch.no_grad():
                losses = []
                modal_distributions = []
                for batch in pbar:   
                    inputs = model.preprocess_inputs(batch, device="cuda")
                    targets = task.format_targets(batch).cuda()
                    output = model(labels=targets, **inputs)
                    losses.append(output.loss)
                    if "modal_distribution" in output:
                        modal_distributions.append(output.modal_distribution)

                    outputs.append(output.logits)
            
                # Append on Batch Axis
                outputs = torch.cat(outputs, axis=0)
                if modal_distributions:
                    modal_distributions = torch.cat(modal_distributions, axis=0)
                    modal_distributions = torch.mean(modal_distributions, axis=0)
                    modal_distributions = modal_distributions.cpu()*100

            if save_predictions:
                assert len(dataset) == outputs.size(0),\
                    f"Mismatch of length between dataset ({len(dataset)}) and predictions ({outputs.size(0)})"
                
                for obj, prediction in zip(dataset, outputs):
                    obj['prediction'] = prediction.tolist()

            model.train(True)

            if modal_distributions:
                return {
                    "Loss": torch.mean(torch.Tensor(losses)),
                    "Modal_Distribution": modal_distributions,
                    **task.eval(dataset, outputs)
                }
            else:
                return {
                    "Loss": torch.mean(torch.Tensor(losses)),
                    **task.eval(dataset, outputs)
                }


def extract_table_predictions(dataset: List[dict]) -> None:

    columns = ["filename", "prediction"]

    def extract(obj):
        # Fails if one of the columns is not Defined
        return {k:obj[k] for k in columns}

    subset_dataset = [extract(obj) for obj in dataset]
    df = pd.DataFrame(subset_dataset)

    return wandb.Table(dataframe=df)


class Evaluator:

    def __init__(self, 
                dataset: List[dict],
                task: AbstractTask,
                load_audio: bool = False,
                batch_size: int = 16,
            ) -> None:

        self.task = task
        # Save dataset as base object even if we need the audio
        self.dataset = dataset

        if load_audio:
            dataset = LoadAudioLive(dataset)

        self.loader = MultiEpochsDataLoader(
            dataset=dataset,  
            batch_size=batch_size, 
            collate_fn=lambda x: x, # Skip Collation
            num_workers=batch_size,
            num_epochs=1,
            prefetch_factor=2,
        )
    
    def __call__(self,
                model,
                save_predictions = False,
                ) -> dict:
        
        model.train(False)

        with torch.no_grad():
            outputs = []
            for batch in self.loader:   
                # Model Inference
                inputs = model.preprocess_inputs(batch, device="cuda")
                targets = self.task.format_targets(batch).to("cuda")
                output = model(labels=targets, **inputs)
                # Collecting Outputs
                outputs.append(output)
                
        
            # Append on Batch Axis
            logits = torch.cat([out.logits for out in outputs], axis=0)
            losses = torch.mean(torch.Tensor([out.loss for out in outputs]))

            if "modal_outputs" in outputs[0]:
                modal_logits = [torch.cat([out.modal_outputs[i].logits for out in outputs], axis=0) for i in range(2)]
            # Collect Modal Distributions if present
            modal_distributions = None
            if "modal_distribution" in outputs[0]:
                modal_distributions = [out.modal_distribution for out in outputs]
                modal_distributions = torch.cat(modal_distributions, axis=0)
                modal_distributions = torch.mean(modal_distributions, axis=0)
                modal_distributions = modal_distributions.cpu()*100

        if save_predictions:
            assert len(self.dataset) == logits.size(0),\
                f"Mismatch of length between dataset ({len(self.dataset)}) and predictions ({logits.size(0)})"
            
            for obj, prediction in zip(self.dataset, logits):
                obj['prediction'] = prediction.tolist()

        model.train(True)

        if modal_distributions:
            return {
                "Loss": torch.mean(torch.Tensor(losses)),
                "Modal_Distribution": modal_distributions,
                **self.task.eval(self.dataset, logits),
                **{ 
                    f"Mod{idx}_{metric}": score
                    for idx, modal_logit in enumerate(modal_logits)
                    for metric, score in self.task.eval(self.dataset, modal_logit).items()
                }
            }
        else:
            return {
                "Loss": torch.mean(torch.Tensor(losses)),
                **self.task.eval(self.dataset, logits),
            }

import torch
import wandb
import collections
import numpy as np
from typing import Optional, Type, List
from time import time, gmtime, strftime
from .utils import multiline_tqdm, fold_generator, batchify
from tasks import AbstractTask

from .checkpointer import Checkpointer
from .eval import extract_table_predictions, Evaluator


def get_optimizer_config(model, lr_conf) -> list:
    if type(lr_conf) == list:
        res = []
        for component, sub_lr_conf in lr_conf:
            res.extend(get_optimizer_config(getattr(model, component), sub_lr_conf))
        return res
    elif type(lr_conf) == float:
        return [
            {
                'params': model.parameters(),
                'lr': lr_conf
            }
        ]


def train(
    model_class: Type,
    task_class: Type,
    model_kwargs: dict = {},
    dataset_path: Optional[str] = None,
    batch_size: int = 128,
    epochs: int = 10,
    load_audio: bool = False,
    text_representation: list = [],
    learning_rate: float = 1e-4,
    lr_decay_interval: int = 100,
    n_log_steps: int = 20, # Fixed amount of logging steps (To fix graph length)
    tags: List[str] = [],
    maximize_metric: str = "",
    n_runs: int = 3,
    ):

    title_lines = [
        f"Task: {task_class.__name__}",
        f"Data: {dataset_path}",
        f"Model: {model_class.__name__}",
    ]
    width = np.max([len(line) for line in title_lines])

    print("") # Add a bit of spacing before the final result
    print("#", "-"*width, "#")
    for line in title_lines:

        print("|", line.ljust(width), "|")
    print("#", "-"*width, "#")

    start_time = time()

    #! LOADING dataset_class and DATASET
    task: AbstractTask
    if dataset_path:
        task = task_class(dataset_path)
    else:
        task = task_class()

    if len(text_representation) > 0 :
        print("### LOADING TEXTUAL REPRESENTATIONS INTO MEMORY ###")
        task.preload_text(text_representation)

    # task.preload_audio()

    scores_list = []
    for fold_idx, (train, validation, test) in enumerate(fold_generator(task, n_runs=n_runs)):

        run = wandb.init(
            project="SongAugmentation",
            reinit=True,
            config={
                "model": model_class.__name__,
                "task": task_class.__name__,
                "data": dataset_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "text_representation": text_representation,
                "model_kwargs": model_kwargs,
                "learning_rate": learning_rate,
                "lr_decay_interval": lr_decay_interval,
                "fold": fold_idx,
            },
            tags=tags
        )

        print(f"\n### {fold_idx} FOLD ###")
        config = task.get_model_config()
        model = model_class(**config, **model_kwargs).cuda()
        model.train(True)

        #! learning_rate contains List[Tuple(str, float)] to configure the 
        #! lrs across the model
        optimizer_config = get_optimizer_config(model, learning_rate)


        total_steps = int(np.ceil(len(train)/batch_size)*epochs)
        log_interval = total_steps//n_log_steps

        optimizer = torch.optim.AdamW(optimizer_config)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=list(range(lr_decay_interval, total_steps, lr_decay_interval)), 
            gamma=0.9
        )                                    

        last_losses = collections.deque([], maxlen=len(train)//batch_size+1)
        formatted_scores = []

        metric = maximize_metric or "Loss"

        #? Define Evaluation objects
        evaluate_val = Evaluator(validation, task, load_audio, batch_size)

        with Checkpointer(model, minimize=(metric == "Loss")) as checkpointer:
        
            pbar = multiline_tqdm(
                batchify(
                    train, 
                    batch_size=batch_size, 
                    epochs = epochs,
                    load_audio = load_audio,
                ),
                total= total_steps,
                leave=True
                )
        
            for step, batch in enumerate(pbar):
                # Building Batch

                targets = task.format_targets(batch).cuda()
                inputs = model.preprocess_inputs(batch, device="cuda")
                    
                # Training Step
                optimizer.zero_grad()
                output = model(**inputs, labels = targets)
                output.loss.backward()
                optimizer.step()
                scheduler.step()

                last_losses.append(output.loss.detach())
                mean_loss = torch.mean(torch.Tensor(last_losses))

                if (step+1) % log_interval == 0:
                    val_scores = evaluate_val(model)
                    scores = {'Val_'+key: score for key, score in val_scores.items()}

                    checkpointer.submit_score(scores['Val_'+metric])
                    wandb.log({
                        "Loss": mean_loss,
                        **scores
                    })

                    formatted_scores = [f"{k}: {s:.2f}%" for k,s in scores.items()]

                description = ", ".join([f"Loss: {mean_loss:.4f}"] + formatted_scores)
                pbar.set_description(description)

        
        evaluate_train = Evaluator(train, task, load_audio, batch_size)
        evaluate_test = Evaluator(test, task, load_audio, batch_size)
        print("** Computing Final Metrics **")

        # ! FINAL EVALUATION ON TRAIN, VALIDATION AND TEST
        trian_scores = evaluate_train(model, save_predictions=True)
        train_scores = {'Train_'+key: score for key, score in trian_scores.items()}

        val_scores = evaluate_val(model, save_predictions=True)
        val_scores = {'Val_'+key: score for key, score in val_scores.items()}

        test_scores = evaluate_test(model, save_predictions=True)
        test_scores = {'Test_'+key: score for key, score in test_scores.items()}

        scores = {
            **train_scores,
            **val_scores,
            **test_scores,
        }

        # ? Logging on WANDB, Terminal and CrossValidation
        wandb.log(scores)
        for metric, score in scores.items():
            print(f"{metric}: {score:.2f}%")
        scores_list.append(scores)

        # ? Logging on WANDB the predictions produces by the final model
        table = extract_table_predictions([*train, *validation, *test])

        predictions = wandb.Artifact(dataset_path, type="predictions")
        predictions.add(table, "predictions")
        run.log_artifact(predictions)

        run.finish()


    final_scores = {
        metric: [score[metric] for score in scores_list] for metric in scores_list[0].keys()
    }

    time_elapsed = time() - start_time
    
    display_results(
        final_scores=final_scores,
        time_str=strftime('%H:%M:%S',gmtime(time_elapsed))
    )

    return final_scores 


def display_results(final_scores, time_str):

    result_lines = [
        "RESULTS"
    ] + [
        f"Mean {metric}: {np.mean(scores):.2f} ± {np.std(scores):.2f}%" 
            for metric, scores in final_scores.items()
    ] + [
        f"Time Elapsed: {time_str}"
    ]
    width = np.max([len(line) for line in result_lines])

    print("") # Add a bit of spacing before the final result
    print("#", "-"*width, "#")
    for line in result_lines:
        print("|", line.ljust(width), "|")
    print("#", "-"*width, "#")

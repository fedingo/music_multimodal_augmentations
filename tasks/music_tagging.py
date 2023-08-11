import json
import torch
import os
from typing import Iterable, List, Dict, Any
from . import AbstractTask
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC
import torch.nn.functional as F


class MusicTagging(AbstractTask):

    def __init__(self, path):
        super().__init__(path)
        self.labels = sorted(list(set([tag for obj in self.data for tag in obj['tags']])))
        self.mAP = MultilabelAveragePrecision(num_labels=len(self.labels), average="macro")
        self.ROC = MultilabelAUROC(num_labels=len(self.labels), average="macro")

    def get_model_config(self) -> Dict[str, Any]:
        return {
            'task': 'multi_label_classification',
            'num_targets': len(self.labels),
            # 'pos_weights': [1]*len(self.labels)
        }

    def eval(self, batch, model_outputs) -> List[float]:

        targets = self.format_targets(batch).to(model_outputs.device)

        # ** Sanity Check **
        assert targets.shape == model_outputs.shape, f"Mismatch shape between Targets "\
            f"({targets.shape}) and Predictions ({model_outputs.shape})"

        model_outputs = F.sigmoid(model_outputs)
        
        return {
            "mAP": self.mAP(model_outputs, targets).cpu()*100,
            "ROC": self.ROC(model_outputs, targets).cpu()*100,
        }

    def format_targets(self, batch) -> torch.Tensor:

        batch_size = len(batch)
        target_ids = [[self.labels.index(tag) for tag in obj['tags']] for obj in batch]

        targets = torch.zeros(batch_size, len(self.labels))

        for idx in range(batch_size):
            targets[idx].scatter_(dim=0, index=torch.Tensor(target_ids[idx]).to(torch.int64), value=1.)

        return targets
import os
from typing import Iterable, List, Dict, Any
from . import AbstractTask
import json
import torch
import numpy as np


class GenreClassification(AbstractTask):

    def __init__(self, path):
        super().__init__(path)
        self.labels = sorted(list(set([x['genre'] for x in self.data])))

    def get_model_config(self) -> Dict[str, Any]:
        return {
            'task': 'single_label_classification',
            'num_targets': len(self.labels)
        }

    def eval(self, batch, model_outputs) -> List[float]:

        predictions = torch.argmax(model_outputs, axis=-1)

        # ** Sanity Check **
        assert len(batch) == len(predictions), f"Mismatch length between labels "\
            f"({len(batch)}) and predictions ({len(predictions)})"

        accuracies = []
        for obj, pred in zip(batch, predictions):
            if self.labels[int(pred)] == obj['genre']:
                score = 100.0
            else:
                score = 0.0
            accuracies.append(score)
        
        return {
            "Accuracy": np.mean(accuracies)
        }

    def format_targets(self, batch) -> torch.Tensor:
        targets = [self.labels.index(obj['genre']) for obj in batch]
        return torch.Tensor(targets).long()

    

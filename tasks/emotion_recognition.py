import json
import torch
import os
from typing import Iterable, List, Dict, Any
from . import AbstractTask
from torchmetrics import R2Score, MeanSquaredError


class EmotionRecognition(AbstractTask):

    def __init__(self, path):
        super().__init__(path)
        self.score = R2Score(num_outputs=2).cuda()

    def get_model_config(self) -> Dict[str, Any]:
        return {
            'task': 'regression',
            'num_targets': 2
        }

    def eval(self, batch, model_outputs) -> List[float]:

        targets = self.format_targets(batch).to(model_outputs.device)

        # ** Sanity Check **
        assert targets.shape == model_outputs.shape, f"Mismatch shape between Targets "\
            f"({targets.shape}) and Predictions ({model_outputs.shape})"
        
        return {
            "R2Score": self.score(model_outputs, targets).cpu()*100
        }

    def format_targets(self, batch) -> torch.Tensor:
        targets = [(obj['valence'], obj['arousal']) for obj in batch]
        return torch.Tensor(targets)
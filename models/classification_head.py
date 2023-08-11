from torch import nn, Tensor
from typing import Optional
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import functional as F


class ClassificationHead(nn.Module):

    def __init__(
        self, 
        representation_size: int, 
        *, 
        task: str, 
        num_targets: int,
        pos_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        assert task in [
            "regression",
            "single_label_classification",
            "multi_label_classification",
        ], f"The configured task {task} is not one of the supported tasks for this classification head"

        if pos_weights is not None and type(pos_weights) is not Tensor:
            pos_weights = Tensor(pos_weights)

        if task == "regression":
            self.loss_fct = nn.MSELoss()
        elif task == "single_label_classification":
            self.loss_fct = nn.CrossEntropyLoss()
        elif task == "multi_label_classification":
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        self.task = task

        self.layer_norm = nn.LayerNorm(representation_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(representation_size, num_targets)

        self.num_labels = num_targets

    def forward(self, x, labels=None) -> SequenceClassifierOutput:
        # x SHAPE: batch_size X representation_size

        
        inner_x = self.layer_norm(x)
        inner_x = F.relu(inner_x)
        inner_x = self.dropout(inner_x)
        logits = self.classifier(inner_x)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=x
        )
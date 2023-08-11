import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import functional as F
from pytorch_metric_learning.losses import NTXentLoss
from dataclasses import dataclass
from .multimodal_for_classification import Projection
from typing import List

@dataclass
class PretrainingOutput:
    loss: torch.Tensor


class PretrainingHead(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.pretrain_loss = NTXentLoss()

    def forward(self, audio_embeddings, text_embeddings, labels) -> SequenceClassifierOutput:
        # audio SHAPE: batch_size X representation_size
        # text SHAPE: batch_size X representation_size
        # labels SHAPE: batch_size

        assert audio_embeddings.shape == text_embeddings.shape, "Text and Audio Embeddings do not match dimensions"

        embeddings = torch.cat([audio_embeddings, text_embeddings], dim=0)
        batched_labels = torch.cat([labels, labels], dim=0)

        return PretrainingOutput(
            loss = self.pretrain_loss(embeddings, batched_labels)
        )
    

class PretrainingModel(torch.nn.Module):

    def __init__(self, audio_encoder, text_encoder) -> None:
        super().__init__()

        self.audio_encoder = Projection(text_encoder.OUTPUT_SHAPE, audio_encoder())
        self.text_encoder = text_encoder()

        self.loss = PretrainingHead()

    def preprocess_inputs(self, batch: List[dict], *, device: str = "cpu"):

        audio_inputs = self.audio_encoder.preprocess_inputs(batch, device=device)
        inputs = self.text_encoder.preprocess_inputs(batch, device=device)

        inputs.update(audio_inputs)
        return inputs
    
    def forward(self, audio_batch, labels, **kwargs):
        audio_encoder_output = self.audio_encoder(audio_batch)
        text_encoder_output  = self.text_encoder(**kwargs)

        return self.loss(audio_embeddings = audio_encoder_output.pooler_output,
                         text_embeddings = text_encoder_output.pooler_output,
                         labels = labels)


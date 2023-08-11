from torch import nn
import torch

from models.encoders.text_encoders import T5Encoder
from models.encoders.audio_encoders import VGGishProjectionEncoder 

from typing import List, Optional, Type
from models.multimodal_head import ConcatHead
from models.base_model_output import AbstractModelOutput



class ModelOutput(AbstractModelOutput):
    PROPERTIES = [
        "pooler_output",
        "modal_distribution",
    ]


class MultiSourceFusion(nn.Module):

    def __init__(
        self, 
        modal_fusion_head: Optional[Type] = None,
        n_sources: int = 1,
        **kwargs
    ) -> None:
        super().__init__()

        self.text_encoder = T5Encoder()
        self.audio_encoder = VGGishProjectionEncoder(1024)

        assert self.text_encoder.OUTPUT_SHAPE == self.audio_encoder.OUTPUT_SHAPE,\
            "Mismatching output dimension for Modal Encoders"

        self.OUTPUT_SHAPE = self.text_encoder.OUTPUT_SHAPE

        if modal_fusion_head is None:
            # Defaults to ConcatHead
            modal_fusion_head = ConcatHead

        self.multimodal_head = modal_fusion_head(self.OUTPUT_SHAPE, n_inputs=n_sources+1)

    @staticmethod
    def preprocess_text(batch: List[dict], device: str = "cpu"):
        num_source = len(batch[0]['text_representations'])

        encoded_sources = []
        for i in range(num_source):
            text_samples = [obj['text_representations'][i] for obj in batch]
            inputs = T5Encoder.tokenizer(
                text_samples,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ).to(device)
            encoded_sources.append(dict(inputs))

        return encoded_sources

    @staticmethod
    def preprocess_inputs(batch: List[dict], *, device: str = "cpu"):

        audio_inputs = VGGishProjectionEncoder.preprocess_inputs(batch, device=device)
        inputs = MultiSourceFusion.preprocess_text(batch, device=device)

        return {
            **audio_inputs,
            "encoded_sources": inputs
        }

    def forward(self, audio_batch, encoded_sources):
        # Assumes inputs have already been preprocess
        # (AKA tokenized, resampled and loaded into vMem)

        audio_encoder_output = self.audio_encoder(audio_batch)

        sources_outputs = []
        for inputs in encoded_sources:
            sources_outputs.append(self.text_encoder(**inputs))
        
        mm_head_out = self.multimodal_head(audio_encoder_output, *sources_outputs)
        
        return ModelOutput(
            **mm_head_out,
        )

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


class MultiModalFusion(nn.Module):

    def __init__(
        self, 
        modal_fusion_head: Optional[Type] = None, 
        **kwargs
    ) -> None:
        super().__init__()

        self.text_encoder = T5Encoder()
        self.audio_encoder = VGGishProjectionEncoder(1024)

        assert self.text_encoder.OUTPUT_SHAPE == self.audio_encoder.OUTPUT_SHAPE,\
            "Mismatching output dimension for Modal Encoders"

        modal_emb_size = self.text_encoder.OUTPUT_SHAPE

        if modal_fusion_head is None:
            # Defaults to ConcatHead
            modal_fusion_head = ConcatHead

        self.multimodal_head = modal_fusion_head(modal_emb_size, n_inputs=2)

    @staticmethod
    def preprocess_inputs(batch: List[dict], *, device: str = "cpu"):

        audio_inputs = VGGishProjectionEncoder.preprocess_inputs(batch, device=device)
        inputs = T5Encoder.preprocess_inputs(batch, device=device)

        inputs.update(audio_inputs)
        return inputs


    def forward(self, audio_batch, **kwargs):
        # Assumes inputs have already been preprocess
        # (AKA tokenized, resampled and loaded into vMem)

        audio_encoder_output = self.audio_encoder(audio_batch)
        text_encoder_output  = self.text_encoder(**kwargs)
        
        mm_head_out = self.multimodal_head(audio_encoder_output, text_encoder_output)
        
        return ModelOutput(
            **mm_head_out,
        )

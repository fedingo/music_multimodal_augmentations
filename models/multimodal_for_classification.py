from torch import nn
import torch

from .encoders.text_encoders import T5Encoder
from .encoders.audio_encoders import VGGishProjectionEncoder, VGGishEncoder

import torch.nn.functional as F
from typing import List, Optional, Type
from .classification_head import ClassificationHead
from .multimodal_head import ResidualHead
from .base_model_output import AbstractModelOutput
from torch.distributions import Categorical

from transformers.modeling_outputs import BaseModelOutputWithPooling
from .multimodal import MultiSourceFusion, SourceEmbedding



class ModelOutput(AbstractModelOutput):
    PROPERTIES = [
        "loss",
        "logits",
        "hidden_states",
        "modal_distribution",
        "modal_outputs",
    ]


class MultiModalForClassification(nn.Module):

    def __init__(
        self, 
        modal_regularization: Optional[float] = None,
        modal_specific_training: bool = True,
        modal_fusion_head: Optional[Type] = None,
        audio_encoder_class: Optional[Type] = None,
        **kwargs
    ) -> None:
        super().__init__()

        if not audio_encoder_class:
            audio_encoder_class = VGGishEncoder

        self.text_encoder = T5Encoder()

        # self.audio_encoder = VGGishProjectionEncoder(1024)
        self.audio_encoder = Projection(1024, audio_encoder_class())
        self.modal_regularization = modal_regularization

        self.modal_specific_training = modal_specific_training

        assert self.text_encoder.OUTPUT_SHAPE == self.audio_encoder.OUTPUT_SHAPE,\
            "Mismatching output dimension for Modal Encoders"

        modal_emb_size = self.text_encoder.OUTPUT_SHAPE

        if modal_fusion_head is None:
            # Defaults to ResidualHead
            modal_fusion_head = ResidualHead

        self.multimodal_head = modal_fusion_head(modal_emb_size, n_inputs=2)

        self.classifier = ClassificationHead(
            modal_emb_size,
            **kwargs
        )

        self.modal_classifiers = nn.ModuleList([
            ClassificationHead(
                modal_emb_size,
                **kwargs
            ) for _ in range(2)
        ])

    def preprocess_inputs(self, batch: List[dict], *, device: str = "cpu"):

        audio_inputs = self.audio_encoder.preprocess_inputs(batch, device=device)
        inputs = self.text_encoder.preprocess_inputs(batch, device=device)

        inputs.update(audio_inputs)
        return inputs
    
    def step(self, step):
        self.modal_specific -= step
        self.modal_specific = self.modal_specific if self.modal_specific >= 0 else 0

        assert self.modal_specific <= 1.0


    def forward(self, audio_batch, labels=None, **kwargs):
        # Assumes inputs have already been preprocess
        # (AKA tokenized, resampled and loaded into vMem)

        audio_encoder_output = self.audio_encoder(audio_batch)
        text_encoder_output  = self.text_encoder(**kwargs)
        
        mm_head_out = self.multimodal_head(audio_encoder_output, text_encoder_output)
        shared_emb = mm_head_out.pooler_output

        classifier_output = self.classifier(shared_emb, labels = labels)

        
        #! MODAL SPECIFIC TRAINING
        
        encoders_output = [audio_encoder_output, text_encoder_output]
        encoders_pooled_output = [output.pooler_output for output in encoders_output]
        if self.modal_specific_training:
            encoders_pooled_output = [output.detach() for output in encoders_pooled_output]

        modal_outputs = [classifier(pooled_output, labels = labels) 
                        for classifier, pooled_output in zip(self.modal_classifiers, 
                                                            encoders_pooled_output
                                                            )
                        ]
        modal_specific_loss = sum([m_o.loss for m_o in modal_outputs])*0.5
        classifier_output.loss = modal_specific_loss + classifier_output.loss

        #! INFO REGULARIZATION
        if self.modal_regularization is not None:
            modal_coeff = mm_head_out.modal_distribution

            distribution = torch.stack([modal_coeff, 1-modal_coeff], dim=-1)
            modal_entropy = Categorical(distribution).entropy()
            classifier_output.loss -= self.modal_regularization * torch.mean(modal_entropy)

        return ModelOutput(
            **classifier_output,
            modal_distribution = mm_head_out.modal_distribution,
            modal_outputs = modal_outputs
        )


class MultiModalForClassificationV2(nn.Module):

    def __init__(
        self, 
        modal_regularization: Optional[float] = None,
        modal_fusion_head: Optional[Type] = None, 
        # multi_modal_encoder: Optional[Type] = None,
        **kwargs
    ) -> None:
        super().__init__()

        if modal_fusion_head is None:
            # Defaults to ConcatHead
            modal_fusion_head = ResidualHead

        self.multi_modal_encoder = MultiSourceFusion(modal_fusion_head, n_sources=3)
        self.modal_regularization = modal_regularization


        self.classifier = ClassificationHead(
            self.multi_modal_encoder.OUTPUT_SHAPE,
            **kwargs
        )

    @staticmethod
    def preprocess_inputs(*args, **kwargs):
        return MultiSourceFusion.preprocess_inputs(*args, **kwargs)

    def forward(self, labels=None, **inputs):
        # Assumes inputs have already been preprocess
        # (AKA tokenized, resampled and loaded into vMem)

        mm_out = self.multi_modal_encoder(**inputs)
        shared_emb = mm_out.pooler_output

        classifier_output = self.classifier(shared_emb, labels = labels)

        if self.modal_regularization is not None:
            modal_coeff = mm_out.modal_distribution

            distribution = torch.stack([modal_coeff, 1-modal_coeff], dim=-1)
            modal_entropy = Categorical(distribution).entropy()
            classifier_output.loss -= self.modal_regularization * torch.mean(modal_entropy)

        return ModelOutput(
            **classifier_output,
            # modal_distribution = mm_out.modal_distribution
        )
    


class Projection(nn.Module):

    def __init__(self, out_shape, encoder) -> None:
        super().__init__()

        self.encoder = encoder
        self.norm = torch.nn.LayerNorm(self.encoder.OUTPUT_SHAPE)
        self.projection = torch.nn.Linear(self.encoder.OUTPUT_SHAPE, out_shape) 

        self.OUTPUT_SHAPE = out_shape
        self.activation = torch.relu

    def preprocess_inputs(self, *args, **kwargs):
        return self.encoder.preprocess_inputs(*args, **kwargs)

    def forward(self, audio_batch):

        last_hidden_state = self.encoder(audio_batch).last_hidden_state

        # Normalization
        last_hidden_state = self.norm(last_hidden_state)
        # Activation
        last_hidden_state = self.activation(last_hidden_state)
        # Projection
        last_hidden_state = self.projection(last_hidden_state)

        #Average Pooling for each batch
        pooler_output = torch.mean(last_hidden_state, axis=1)

        return BaseModelOutputWithPooling(
            pooler_output=pooler_output,
            last_hidden_state=last_hidden_state,
        )
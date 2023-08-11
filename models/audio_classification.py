from typing import List
from torch import nn
from .classification_head import ClassificationHead
from .multimodal_head import SequenceAttention
from .encoders.audio_encoders import *


class VGGishForClassification(VGGishEncoder):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.classifier = ClassificationHead(self.OUTPUT_SHAPE, **kwargs)

    def forward(self, audio_batch, labels=None):
        
        encoded_audio = super().forward(audio_batch)
        sequence_vector = encoded_audio.pooler_output

        return self.classifier(sequence_vector, labels=labels)

        
class VGGishWithSequenceAttentionForClassification(VGGishEncoder):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.sequence_head = SequenceAttention(self.OUTPUT_SHAPE)
        self.classifier = ClassificationHead(self.OUTPUT_SHAPE, **kwargs)

    def forward(self, audio_batch, labels=None):
        
        encoded_audio = super().forward(audio_batch)
        sequence_output = self.sequence_head(encoded_audio.last_hidden_state)

        return self.classifier(sequence_output.pooler_output, labels=labels)
    

class AudioClassification(nn.Module):

    def __init__(self, audio_encoder_class, **kwargs) -> None:
        super().__init__()

        self.audio_encoder = audio_encoder_class()
        self.classifier = ClassificationHead(self.audio_encoder.OUTPUT_SHAPE, **kwargs)

    def preprocess_inputs(self, *args, **kwargs):
        return self.audio_encoder.preprocess_inputs(*args, **kwargs)

    def forward(self, audio_batch, labels=None):
        
        encoded_audio = self.audio_encoder(audio_batch)

        return self.classifier(encoded_audio.pooler_output, labels=labels)
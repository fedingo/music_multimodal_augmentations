import torch
from torch import nn
from models.vggish.torch_vggish import TorchVGGish, SAMPLE_RATE
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio
import torchaudio.functional as F
from typing import List
from transformers import (
    AutoModelForAudioClassification,
)
from .utils import TorchWhisperExtractor


class VGGishEncoder(TorchVGGish):

    OUTPUT_SHAPE = 128

    @staticmethod
    def preprocess_inputs(batch: List[dict], *, device: str = "cpu"):

        audio_batch = []
        for obj in batch:
            audio_batch.append(obj['tensor'].to(device))

        audio_batch = torch.nested.nested_tensor(audio_batch).to_padded_tensor(0)

        return {
            "audio_batch": audio_batch
        }
    
    @staticmethod
    def obj_to_audio(obj):
        audio, sample_rate = torchaudio.load(obj['filename'])
        if sample_rate != SAMPLE_RATE:
            audio = F.resample(audio, sample_rate, SAMPLE_RATE, lowpass_filter_width=6)
        # If stereo convert to Mono
        if audio.size(0) > 1:
            audio = torch.mean(audio, axis=0)

        if len(audio) > 1e6:
            audio = audio[:960000]

        return audio.squeeze()
    

class VGGishProjectionEncoder(VGGishEncoder):


    def __init__(self, out_shape) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(self.OUTPUT_SHAPE)
        self.projection = torch.nn.Linear(self.OUTPUT_SHAPE, out_shape) 

        # Update new output shape
        self.OUTPUT_SHAPE = out_shape
        
        self.activation = torch.relu

    def forward(self, audio_batch):

        last_hidden_state = super().forward(audio_batch).last_hidden_state

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


class Wav2Vec2Encoder(nn.Module):

    OUTPUT_SHAPE = 768

    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    def preprocess_inputs(self, batch: List[dict], *, device: str = "cpu"):
        audio_inputs = self.feat_extractor(
            [obj['tensor'].numpy() for obj in batch],
            sampling_rate=self.feat_extractor.sampling_rate,  
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length= 200_000,
        ).to(device)

        return {
            "audio_batch": audio_inputs
        }
    
    def forward(self, audio_batch):

        model_output = self.encoder(**audio_batch)
        pooler_output = torch.mean(model_output.last_hidden_state, dim=1) # Average across time

        return BaseModelOutputWithPooling(
            pooler_output=pooler_output,
            last_hidden_state=model_output.last_hidden_state,
        )
    

class WhisperEncoder(nn.Module):

    """
    tiny   -> 384
    base   -> 512
    small  -> 768
    medium -> 1024
    large  -> 1280
    """

    OUTPUT_SHAPE = 768

    def __init__(self) -> None:
        super().__init__()
        
        model_checkpoint = "openai/whisper-small"
        # self.feat_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        self.feat_extractor = TorchWhisperExtractor("cuda")
        self.encoder = AutoModelForAudioClassification.from_pretrained(model_checkpoint).encoder
    
    def preprocess_inputs(self, batch: List[dict], *, device: str = "cpu"):
        audio_inputs = self.feat_extractor(
            [obj['tensor'].to(device) for obj in batch],
        )

        return {
            "audio_batch": audio_inputs
        }
    
    def forward(self, audio_batch):

        model_output = self.encoder(**audio_batch)
        pooler_output = torch.mean(model_output.last_hidden_state, dim=1) # Average across time

        return BaseModelOutputWithPooling(
            pooler_output=pooler_output,
            last_hidden_state=model_output.last_hidden_state,
        )
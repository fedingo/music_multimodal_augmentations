from .vggish import VGGish
from torchaudio.transforms import MelSpectrogram
from transformers.modeling_outputs import BaseModelOutputWithPooling
import math
import torch
from torch import nn
from . import vggish_params
from typing import List 
from .vggish_params import SAMPLE_RATE

class TorchVGGish(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        model_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
                        'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
                    'releases/download/v0.1/vggish_pca_params-970ea276.pth'
        }

        self.mel_spec = MelSpectrogram(
            sample_rate=vggish_params.SAMPLE_RATE,
            f_max=vggish_params.MEL_MAX_HZ,
            f_min=vggish_params.MEL_MIN_HZ,
            n_mels=vggish_params.NUM_MEL_BINS,
            win_length=400,
            hop_length=160,
            center=False,
            n_fft=512,
            power=1,
        )

        self.vggish = VGGish(urls=model_urls, pretrained = True, preprocess = False, postprocess = True)

    def forward(self, audio_batch):
        
        log_mel = torch.log(self.mel_spec(audio_batch)+0.01).T

        features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
        window_length = int(round(
            vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        hop_length = int(round(
            vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
        
        
        num_frames = 1 + int(math.floor((log_mel.size(0) - window_length) / hop_length))
        shape = (num_frames, window_length) + log_mel.shape[1:]
        strides = (log_mel.stride(0) * hop_length,) + log_mel.stride()
        framed_log_mel = torch.as_strided(log_mel, size=shape, stride=strides)

        # Reorder as (Time, batch_size, W, H)
        framed_log_mel = framed_log_mel.permute(0,3,1,2).contiguous()

        # Add channels dimension
        vggish_input = framed_log_mel.unsqueeze(2)

        # Permute to Batch x Seq_len x Emb
        out_vectors = self.vggish(vggish_input).permute(1,0,2)

        #Average Pooling for each batch
        sequence_vector = torch.mean(out_vectors, axis=1)

        return BaseModelOutputWithPooling(
            pooler_output=sequence_vector, 
            last_hidden_state=out_vectors
        )

    def freeze(self, freeze=False):
        for param in self.vggish.parameters():
            param.requires_grad = not freeze
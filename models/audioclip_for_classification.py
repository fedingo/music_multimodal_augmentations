import torch
from .audioclip.esresnet import ESResNeXtFBSP
from typing import Optional, Union, List
from .classification_head import ClassificationHead


class AudioCLIPForClassification(torch.nn.Module):
    
    def __init__(
        self,
        embed_dim: int = 527,
        **kwargs,
    ) -> None:
        super().__init__()

        n_fft: int = 2048
        hop_length: Optional[int] = 561
        win_length: Optional[int] = 1654
        window: Optional[str] = 'blackmanharris'
        normalized: bool = True
        onesided: bool = True
        spec_height: int = -1
        spec_width: int = -1
        apply_attention: bool = True

        self.audio = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=False
        )

        # state_dict = torch.load("/home/user/projects/git/AudioCLIP/assets/ESRNXFBSP.pt")
        # self.audio.load_state_dict(state_dict=state_dict, strict=True)

        self.classifier = ClassificationHead(embed_dim, **kwargs)
    
    @staticmethod
    def preprocess_inputs(batch: List[dict], *, device: str = "cpu"):

        audio_batch = []
        for obj in batch:
            audio_batch.append(obj['tensor'].to(device))

        audio_batch = torch.nested.nested_tensor(audio_batch).to_padded_tensor(0)

        return {
            "audio_batch": audio_batch
        }

    def forward(self, audio_batch, labels=None):
        
        hidden_state = self.audio(audio_batch)
        return self.classifier(hidden_state, labels=labels)
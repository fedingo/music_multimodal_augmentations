import torch
import numpy as np
from transformers import WhisperFeatureExtractor
from typing import *
from torch import TensorType


class TorchWhisperExtractor(WhisperFeatureExtractor):

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.mel_filters = torch.tensor(self.mel_filters).to(device)

    def _np_extract_fbank_features(self, waveform: TensorType) -> TensorType:
        """
        Compute the log-Mel spectrogram of the provided audio, using torch.
        """

        device = waveform.device
        window = torch.hann_window(self.n_fft, periodic=True, device=device)

        waveform = waveform.unsqueeze(0)
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=window,
                        center=True, normalized=False, onesided=True, pad_mode='reflect', return_complex=True)
        magnitudes = stft.abs().pow(2).squeeze()[:,:-1]

        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        log_spec = torch.clamp(log_spec, min=log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec
    
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
    ) -> Dict:
        """
        Main method to featurize and prepare for the model one or several sequence(s).
        """

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], torch.Tensor))
        )

        batch_size = len(raw_speech)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # Pad
        padded_inputs = torch.nested.nested_tensor(raw_speech).to_padded_tensor(0)

        if padded_inputs.size(-1) < self.n_samples:
            padded_inputs = torch.nested.nested_tensor(raw_speech).to_padded_tensor(0, output_size=(batch_size, self.n_samples))
        else:
            # Truncate
            padded_inputs = padded_inputs[:,:self.n_samples]


        input_features = [self._np_extract_fbank_features(waveform) for waveform in padded_inputs]

        return {
            "input_features": torch.stack(input_features)
        }

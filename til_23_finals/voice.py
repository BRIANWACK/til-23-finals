"""Code to extract clean voice from any noisy audio file."""

import torch
import torch.nn as nn
from demucs.apply import apply_model
from demucs.pretrained import get_model
from noisereduce.torchgate import TorchGate as TG
from torchaudio.functional import resample

__all__ = ["load_demucs_model", "VoiceExtractor"]

DEMUCS_MODEL = "htdemucs_ft"
DEMUCS_MODEL_REPO = None

# Competition audio files are 22050Hz.
DEFAULT_SR = 22050
BEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_demucs_model(name=DEMUCS_MODEL, repo=DEMUCS_MODEL_REPO):
    """Load demucs model."""
    return get_model(name=name, repo=repo)


class VoiceExtractor(nn.Module):
    """Class to extract voice from audio file."""

    # TODO: Expose below for configuration.
    spectral_gate_std_thres = 1.5
    spectral_noise_remove = 0.98
    spectral_n_fft = 1024

    def __init__(self, sr: int = DEFAULT_SR, model=None):
        super(VoiceExtractor, self).__init__()
        # See: https://github.com/timsainb/noisereduce.
        self.denoise1 = TG(
            sr=sr,
            nonstationary=False,
            n_std_thresh_stationary=self.spectral_gate_std_thres,
            prop_decrease=self.spectral_noise_remove,
            n_fft=self.spectral_n_fft,
            # Below smooth out the noise mask to prevent artifacting.
            freq_mask_smooth_hz=300,
            time_mask_smooth_ms=100,
        )
        self.demucs = load_demucs_model() if model is None else model

    def _denoise1(self, wav: torch.Tensor, sr: int):
        wav = resample(wav, orig_freq=sr, new_freq=self.denoise1.sr)
        wav = self.denoise1(wav[None])[0]
        return wav, self.denoise1.sr

    def _demucs(self, wav: torch.Tensor, sr: int):
        wav = resample(wav, orig_freq=sr, new_freq=self.demucs.samplerate)
        wav = wav[None].expand(self.demucs.audio_channels, -1)
        # Copied from `demucs.separate.main` (v4.0.0) to ensure correctness.
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            self.demucs,
            wav[None],  # BCT
            shifts=1,
            split=True,
            progress=False,
            overlap=0.25,
            num_workers=0,
        )[0]
        sources = sources * ref.std() + ref.mean()
        wav = sources[self.demucs.sources.index("vocals")]
        wav = wav.mean(0)
        return wav, self.demucs.samplerate

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor, sr: int):
        """Extract voice from audio file."""
        assert len(wav.shape) == 1, "Input must be T."

        wav, sr = self._demucs(wav, sr)
        wav, sr = self._denoise1(wav, sr)
        return wav, sr

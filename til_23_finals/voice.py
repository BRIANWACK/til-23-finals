"""Code to extract clean voice from any noisy audio file."""

from typing import Optional

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

    # TODO: Expose below kwargs for tuning in the config file.
    def __init__(
        self,
        sr: int = DEFAULT_SR,
        spectral_gate_std_thres: float = 1.0,
        spectral_noise_remove: float = 1.0,
        spectral_n_fft: int = 512,
        spectral_freq_mask_hz: Optional[int] = None,
        spectral_time_mask_ms: Optional[int] = None,
        skip_demucs: bool = False,
        skip_denoise1: bool = False,
        use_ori: bool = False,
        model=None,
    ):
        super(VoiceExtractor, self).__init__()
        # See: https://github.com/timsainb/noisereduce.
        self.denoise1 = TG(
            sr=sr,
            nonstationary=False,
            n_std_thresh_stationary=spectral_gate_std_thres,
            prop_decrease=spectral_noise_remove,
            n_fft=spectral_n_fft,
            # Below smoothes out the noise mask to prevent artifacting.
            # NOTE: Given a good noise sample, there is no artifacting even with
            # the masks disabled.
            freq_mask_smooth_hz=spectral_freq_mask_hz,
            time_mask_smooth_ms=spectral_time_mask_ms,
        )
        self.demucs = load_demucs_model() if model is None else model
        self.skip_demucs = skip_demucs
        self.skip_denoise1 = skip_denoise1
        self.use_ori = use_ori

    def _denoise1(
        self, wav: torch.Tensor, sr: int, noise: Optional[torch.Tensor] = None
    ):
        wav = resample(wav, orig_freq=sr, new_freq=self.denoise1.sr)
        noise = noise[None] if noise is not None else None
        wav = self.denoise1(wav[None], noise)[0]
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
            shifts=2,
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

        ori_wav, ori_sr = wav, sr
        noise = None
        if not self.skip_demucs:
            wav, sr = self._demucs(wav, sr)
            # Use extracted voice sample to find noise sample.
            noisy = (
                ori_wav
                if self.use_ori
                else resample(ori_wav, orig_freq=ori_sr, new_freq=sr)
            )
            voice = (
                resample(wav, orig_freq=sr, new_freq=ori_sr) if self.use_ori else wav
            )
            noise = noisy - voice
        if not self.skip_denoise1:
            if self.use_ori:
                wav, sr = self._denoise1(ori_wav, ori_sr, noise)
            else:
                wav, sr = self._denoise1(wav, sr, noise)
        return wav, sr

"""Various implementations for `AbstractSpeakerIDService`."""

import logging
from typing import List

import numpy as np
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel as NeMoModel
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from til_23_asr import VoiceExtractor
from torchaudio.functional import resample

from til_23_finals.types import SpeakerID
from til_23_finals.utils import cos_sim, thres_strategy_naive

from .abstract import AbstractSpeakerIDService
from .speaker_hack import CUSTOM_EXTRACTOR_CFG, CUSTOM_OTHER_CFG

__all__ = ["NeMoSpeakerIDService"]

log = logging.getLogger("SpeakID")

BEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DEFAULT_EXTRACTOR_CFG = dict(
    skip_vol_norm=False,
    skip_df=False,
    skip_spectral=True,
    spectral_first=False,
    use_ori=False,
    noise_removal_limit_db=5,
)

DEFAULT_OTHER_CFG = dict(
    use_shifts=True,
    shift=0.0025,
    use_agc=False,
    agc_window=0.4,
    speed=1.0,
)

USE_EXTRACTOR = True


def shift_and_pad(wav: torch.Tensor, shift: float) -> torch.Tensor:
    """Shift and pad the input waveform."""
    T = len(wav)
    n_shift = int(T * shift)
    idx = torch.arange(T, device=wav.device)
    idx = idx.roll(n_shift)
    wav = wav[idx]
    if n_shift > 0:
        wav[:n_shift] = 0
    else:
        wav[n_shift:] = 0
    return wav


def agc(wav: torch.Tensor, sr: int, window: float = 1.0) -> torch.Tensor:
    """Automatic Gain Control."""
    size = int(sr * window)
    ori_device = wav.device
    wav = wav.numpy(force=True)
    gains = 1 / maximum_filter1d(abs(wav), size=size)
    # gains = minimum_filter1d(gains, size=size)
    # gains = gaussian_filter1d(gains, sigma=size / 2)
    wav = (wav * gains).clip(-1, 1)
    return torch.tensor(wav, device=ori_device)


class NeMoSpeakerIDService(AbstractSpeakerIDService):
    """Speaker ID service using NeMo."""

    model_sr = 16000
    fallback_thres = 0.2

    def __init__(self, model_dir, denoise_model_dir, device=BEST_DEVICE):
        """Initialize NeMoSpeakerIDService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        device : str
            the torch device to use for computation.
        """
        self.device = device
        # nvidia/speakerverification_en_titanet_large
        self.model: NeMoModel = NeMoModel.restore_from(restore_path=model_dir)
        self.model.eval()
        self.extractor = VoiceExtractor(denoise_model_dir)

        self.identities: List[SpeakerID] = []
        self.deactivate()

    def activate(self):
        """Any preparation before actual use."""
        super(NeMoSpeakerIDService, self).activate()
        self.model.to(self.device)
        self.extractor.to(self.device)

    def deactivate(self):
        """Any cleanup after actual use."""
        super(NeMoSpeakerIDService, self).deactivate()
        self.model.to("cpu")
        self.extractor.to("cpu")

    @torch.inference_mode()
    def embed_speaker(self, audio_waveform, sampling_rate, team_id=""):
        """Embed speaker."""
        if not self.activated:
            log.critical("NeMoSpeakerIDService not activated!")

        cfg = {**DEFAULT_EXTRACTOR_CFG, **CUSTOM_EXTRACTOR_CFG.get(team_id, {})}
        ecfg = {**DEFAULT_OTHER_CFG, **CUSTOM_OTHER_CFG.get(team_id, {})}
        raw, raw_sr = torch.tensor(audio_waveform, device=self.device), sampling_rate

        if USE_EXTRACTOR:
            clean, clean_sr = self.extractor.forward(raw, raw_sr, **cfg)

        raw = resample(raw, orig_freq=raw_sr, new_freq=self.model_sr)
        raw = raw[None]
        wav_len = torch.tensor([raw.shape[1]], device=self.device)
        _, raw_embed = self.model.forward(input_signal=raw, input_signal_length=wav_len)
        raw_embed = raw_embed.mean(0).numpy(force=True)

        if USE_EXTRACTOR:
            clean = resample(
                clean, orig_freq=clean_sr // ecfg["speed"], new_freq=self.model_sr
            )
            if ecfg["use_agc"]:
                clean = agc(clean, clean_sr, ecfg["agc_window"])
            if ecfg["use_shifts"]:
                v = ecfg["shift"]
                clean_a = shift_and_pad(clean, v)
                clean_b = shift_and_pad(clean, -v)
                clean = torch.stack([clean, clean_a, clean_b])
                clean_len = torch.tensor([clean.shape[1]] * 3, device=self.device)
            else:
                clean = clean[None]
                clean_len = torch.tensor([clean.shape[1]], device=self.device)
            _, clean_embed = self.model.forward(
                input_signal=clean, input_signal_length=clean_len
            )
            clean_embed = clean_embed.mean(0).numpy(force=True)
        else:
            clean_embed = np.ones_like(raw_embed)

        return raw_embed, clean_embed

    def clear_speakers(self):
        """Clear all enrolled speakers."""
        self.identities.clear()

    @torch.inference_mode()
    def enroll_speaker(self, audio_waveform, sampling_rate, team_id, member_id):
        """Enroll a speaker.

        Parameters
        ----------
        audio_waveform : np.ndarray
            Input waveform.
        sampling_rate : int
            The sampling rate of the audio file.
        team_id : str
            The team ID of the speaker.
        member_id : str
            The member ID of the speaker.
        """
        if not self.activated:
            log.critical("NeMoSpeakerIDService not activated!")

        raw_embed, clean_embed = self.embed_speaker(
            audio_waveform, sampling_rate, team_id
        )

        identity = SpeakerID(team_id, member_id, raw_embed, clean_embed)
        log.info(f"Enrolled: {team_id}_{member_id}")
        self.identities.append(identity)

    @torch.inference_mode()
    def identify_speaker(
        self, audio_waveform, sampling_rate, team_id="", return_all=False
    ):
        """Identify the speaker in the audio file.

        Parameters
        ----------
        audio_waveform : np.ndarray
            input waveform.
        sampling_rate : int
            the sampling rate of the audio file.
        team_id : str
            Optional filter to identify within a specific team, defaults to "".

        Returns
        -------
        scores : Dict[Tuple[str, str], float]
            Map of the team ID & member ID to the score.
        """
        if not self.activated:
            log.critical("NeMoSpeakerIDService not activated!")

        # TODO: Save audio files for debugging.
        raw_embed, clean_embed = self.embed_speaker(
            audio_waveform, sampling_rate, team_id
        )

        compare = [i for i in self.identities if i.team_id.upper() == team_id.upper()]
        raws = [i.raw_embed for i in compare]
        cleans = [i.clean_embed for i in compare]

        raw_sims = [cos_sim(raw_embed, r) for r in raws]
        clean_sims = [cos_sim(clean_embed, c) for c in cleans]

        scores_raw = {}
        scores_clean = {}
        for identity, raw_score, clean_score in zip(compare, raw_sims, clean_sims):
            # TODO: Weighted average? How to detect failure case, by low sigma?
            log.info(
                f"Identity: {identity.team_id}_{identity.member_id}, Raw: {raw_score:.3f}, Clean: {clean_score:.3f}"
            )
            scores_raw[(identity.team_id, identity.member_id)] = raw_score
            scores_clean[(identity.team_id, identity.member_id)] = clean_score

        # NOTE: My efforts in voice extraction were wasted. NeMo's TitaNet is already
        # noise resilient and it is clear that overall, the voice extraction harms
        # the accuracy.
        if return_all:
            return scores_raw, scores_clean
        return scores_raw

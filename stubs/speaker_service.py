"""Various implementations from `AbstractSpeakerIDService`."""

from pathlib import Path

import librosa
import numpy as np
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel as NeMoModel
from til_23_cv import cos_sim, thres_strategy_naive

from .abstract_ai_services import AbstractSpeakerIDService

BEST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NeMoSpeakerIDService(AbstractSpeakerIDService):
    """Speaker ID service using NeMo."""

    model_sr = 16000
    id_thres = 0.2

    def __init__(self, model_dir: str, speaker_dir: str = "", device=BEST_DEVICE):
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
        self.model: NeMoModel = NeMoModel.from_pretrained(model_dir)
        self.model.to(device).eval()

        # TODO: Cache embeddings.
        self.speaker_ids = []
        self.speaker_embeds = []
        for wav_path in Path(speaker_dir).glob("*.wav"):
            speaker_id = wav_path.stem
            embed = self.model.get_embedding(str(wav_path))
            self.speaker_ids.append(speaker_id)
            # Remove batch dimension.
            self.speaker_embeds.append(embed[0].numpy(force=True))

    # TODO: Modify abstract interface to accept filepath instead of waveform for convenience.
    # TODO: Ability to exclude speaker ids if sure our team's speaker is already identified.
    def identify_speaker(self, audio_waveform: np.ndarray, sampling_rate: int) -> str:
        """Identify the speaker in the audio file.

        Parameters
        ----------
        audio_waveform : np.ndarray
            input waveform.
        sampling_rate : int
            the sampling rate of the audio file.

        Returns
        -------
        result : str
            string representing the speaker's ID corresponding to the list of speaker IDs in the training data set.
        """
        # `audio_waveform` is monochannel and has shape (n_samples,) already.
        if sampling_rate != self.model_sr:
            audio_waveform = librosa.resample(
                audio_waveform, orig_sr=sampling_rate, target_sr=self.model_sr
            )
        audio_len = len(audio_waveform)

        with torch.inference_mode():
            _, embed = self.model.forward(
                input_signal=torch.tensor([audio_waveform], device=self.device),
                input_signal_length=torch.tensor([audio_len], device=self.device),
            )

        embed = embed[0].numpy(force=True)
        speaker_sims = [cos_sim(embed, e) for e in self.speaker_embeds]
        idx = thres_strategy_naive(speaker_sims, self.id_thres)
        assert idx != -1
        return self.speaker_ids[idx]

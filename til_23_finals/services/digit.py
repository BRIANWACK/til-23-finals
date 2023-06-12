"""Various implementations for `AbstractDigitDetectionService`."""

import re

import librosa
import torch
import whisper
from til_23_asr import VoiceExtractor

from .abstract import AbstractDigitDetectionService

__all__ = ["WhisperDigitDetectionService"]

WHISPER_SAMPLE_RATE = 16000
BEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class WhisperDigitDetectionService(AbstractDigitDetectionService):
    """Digit Detection service using OpenAI's Whisper."""

    def __init__(self, model_dir: str, device=BEST_DEVICE):
        """Initialize WhisperDigitDetectionService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        device : str
            the torch device to use for computation.
        """
        self.extractor = VoiceExtractor()
        # large-v2
        self.model = whisper.load_model(model_dir, device="cpu")
        self.prompt = """0 1 2 3 4 5 6 7 8 9."""
        self.options = whisper.DecodingOptions(
            fp16=False,
            language="en",
            prompt=self.prompt,
        )
        self.device = device
        self.extractor.to("cpu")

    @torch.inference_mode()
    def transcribe_audio_to_digits(self, audio_waveform):
        """Transcribe audio waveform to a tuple of ints.

        Parameters
        ----------
        audio_waveform : numpy.ndarray
            Numpy 1d array of floats that represent the audio file.
            It is assumed that the sampling rate of the audio is 22050Hz.

        Returns
        -------
        results : Tuple[int]
            The ordered tuple of digits found in the input audio file.
        """
        self.extractor.to(self.device)
        wav = torch.tensor(audio_waveform, device=self.device)
        wav, sr = self.extractor(wav, 22050)
        self.extractor.to("cpu")

        wav = librosa.resample(wav.numpy(force=True), sr, WHISPER_SAMPLE_RATE)
        wav = whisper.pad_or_trim(wav)
        mel = whisper.log_mel_spectrogram(wav)
        mel = mel.to(self.device)

        self.model.to(self.device)
        text = whisper.decode(self.model, mel, self.options)[0].text
        self.model.to("cpu")

        return tuple(int(s) for s in re.findall(r"\d+", text))

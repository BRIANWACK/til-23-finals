"""Various implementations for `AbstractDigitDetectionService`."""

import logging
import re

import numpy as np
import torch
import whisper
from faster_whisper import WhisperModel
from num2words import num2words
from til_23_asr import VoiceExtractor, normalize_distribution
from torchaudio.functional import resample

from .abstract import AbstractDigitDetectionService

__all__ = ["WhisperDigitDetectionService", "FasterWhisperDigitDetectionService"]

WHISPER_SAMPLE_RATE = 16000
BEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# fmt: off
NUMBERS = dict(
    ZERO=0, ONE=1, TWO=2, THREE=3, FOUR=4, FIVE=5, SIX=6, SEVEN=7, EIGHT=8, NINE=9,
    ZEROTH=0, FOURTH=4, SIXTH=6, SEVENTH=7, EIGHTH=8, NINTH=9, # Common mistakes.
)
# fmt: on

log = logging.getLogger("Digit")


def extract_digits(text: str):
    """Extract digits from text sorted by closeness to middle."""
    # Given number must be between 0 and 9, ignore if number is not alone.
    text = re.sub(r"(?<!\d)\d(?!\d)", lambda m: f" {num2words(int(m.group()))} ", text)
    text = text.upper()
    text = re.sub(r"[^A-Z ]", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()

    words = text.split(" ")
    middle = int(len(words) * 0.7)
    digits = []
    for i, word in enumerate(words):
        if word in NUMBERS:
            digits.append((NUMBERS[word], abs(i - middle)))
    digits = sorted(digits, key=lambda x: x[1])
    return tuple(d[0] for d in digits)


class WhisperDigitDetectionService(AbstractDigitDetectionService):
    """Digit Detection service using OpenAI's Whisper."""

    def __init__(self, model_dir, denoise_model_dir, device=BEST_DEVICE):
        """Initialize WhisperDigitDetectionService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        device : str
            the torch device to use for computation.
        """
        self.extractor = VoiceExtractor(denoise_model_dir)
        # large-v2
        self.model = whisper.load_model(model_dir, device="cpu")
        # self.prompt = """0 1 2 3 4 5 6 7 8 9."""
        self.options = whisper.DecodingOptions(
            fp16=True,
            language="en",
            # Don't use prompt as it might hurt model accuracy.
            # prompt=self.prompt,
            without_timestamps=True,
        )
        self.device = device

        self.deactivate()

    def activate(self):
        """Any preparation before actual use."""
        super(WhisperDigitDetectionService, self).activate()
        self.model.to(self.device)
        self.extractor.to(self.device)

    def deactivate(self):
        """Any cleanup after actual use."""
        super(WhisperDigitDetectionService, self).deactivate()
        self.model.to("cpu")
        self.extractor.to("cpu")

    @torch.inference_mode()
    def transcribe_audio_to_digits(self, audio_waveform, sampling_rate):
        """Transcribe audio waveform to a tuple of ints.

        Note, digits are sorted by closeness to center of text for now.

        Parameters
        ----------
        audio_waveform : numpy.ndarray
            Numpy 1d array of floats that represent the audio file.
        sampling_rate : int
            Sampling rate of the audio.

        Returns
        -------
        results : Tuple[int, ...]
            Tuple of digits sorted by confidence.
        """
        if not self.activated:
            log.critical("WhisperDigitDetectionService not activated!")

        # TODO: Save audio files for debugging.
        wav = torch.tensor(audio_waveform, device=self.device)
        # NOTE: The denoiser hurts digit recognition performance.
        # wav, sr = self.extractor.forward(wav, sampling_rate)
        wav, sr = wav, sampling_rate

        wav = resample(wav, orig_freq=sr, new_freq=WHISPER_SAMPLE_RATE)
        # See: https://github.com/huggingface/transformers/pull/21263
        wav = normalize_distribution(wav)
        wav = whisper.pad_or_trim(wav)
        mel = whisper.log_mel_spectrogram(wav)
        mel = mel.to(self.device)

        result = whisper.decode(self.model, mel, self.options)
        # TODO: Use result.no_speech_prob to determine if VoiceExtractor failed,
        # in which case, fallback to source audio or less aggressive extraction.
        # Also fallback if no digits or more than one digit detected?
        # TODO: Use digit with highest confidence/attention.
        text = result.text
        digits = extract_digits(text)
        log.info(f"Extracted text: {text}")
        log.info(f"Extracted digits: {digits}")
        return digits


class FasterWhisperDigitDetectionService(AbstractDigitDetectionService):
    """Digit Detection service using FasterWhisper."""

    def __init__(self, model_dir, denoise_model_dir, device=BEST_DEVICE):
        """Initialize WhisperDigitDetectionService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        device : str
            the torch device to use for computation.
        """
        self.extractor = VoiceExtractor(denoise_model_dir)
        # large-v2
        self.model = WhisperModel(model_dir, device="cuda", compute_type="int8_float16")
        self.options = dict(
            language="en",
            compression_ratio_threshold=10.0,
            log_prob_threshold=-10.0,
            no_speech_threshold=1.0,
            beam_size=5,
            patience=1,
            without_timestamps=True,
        )
        self.device = device
        # Crash if model is broken.
        self.model.transcribe(np.ones((16000,)), **self.options)

        self.deactivate()

    def activate(self):
        """Any preparation before actual use."""
        super(FasterWhisperDigitDetectionService, self).activate()
        self.extractor.to(self.device)

    def deactivate(self):
        """Any cleanup after actual use."""
        super(FasterWhisperDigitDetectionService, self).deactivate()
        self.extractor.to("cpu")

    @torch.inference_mode()
    def transcribe_audio_to_digits(self, audio_waveform, sampling_rate):
        """Transcribe audio waveform to a tuple of ints.

        Note, digits are sorted by closeness to center of text for now.

        Parameters
        ----------
        audio_waveform : numpy.ndarray
            Numpy 1d array of floats that represent the audio file.
        sampling_rate : int
            Sampling rate of the audio.

        Returns
        -------
        results : Tuple[int, ...]
            Tuple of digits sorted by confidence.
        """
        wav = torch.tensor(audio_waveform, device=self.device)
        # NOTE: The denoiser hurts digit recognition performance.
        # wav, sr = self.extractor.forward(wav, sampling_rate)
        wav, sr = wav, sampling_rate

        wav = resample(wav, orig_freq=sr, new_freq=WHISPER_SAMPLE_RATE)
        # See: https://github.com/huggingface/transformers/pull/21263
        wav = normalize_distribution(wav)
        wav = wav.numpy(force=True)

        segments, _ = self.model.transcribe(wav, **self.options)
        segments = list(segments)
        if len(segments) > 1:
            log.critical(f"Segments:\n{segments}")
        text = segments[0].text
        # TODO: Use result.no_speech_prob to determine if VoiceExtractor failed,
        # in which case, fallback to source audio or less aggressive extraction.
        # Also fallback if no digits or more than one digit detected?
        # TODO: Use digit with highest confidence/attention.
        digits = extract_digits(text)
        log.info(f"Extracted text: {text}")
        log.info(f"Extracted digits: {digits}")
        return digits

"""Various implementations for `AbstractDigitDetectionService`."""

import re

import torch
import whisper
from num2words import num2words
from til_23_asr import VoiceExtractor, normalize_distribution
from torchaudio.functional import resample

from .abstract import AbstractDigitDetectionService

__all__ = ["WhisperDigitDetectionService"]

WHISPER_SAMPLE_RATE = 16000
BEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NUMBERS = dict(
    ZERO=0, ONE=1, TWO=2, THREE=3, FOUR=4, FIVE=5, SIX=6, SEVEN=7, EIGHT=8, NINE=9
)


def extract_digits(text: str):
    """Extract digits from text sorted by closeness to middle."""
    # Given number must be between 0 and 9, ignore if number is not alone.
    text = re.sub(r"(?<!\d)\d(?!\d)", lambda m: f" {num2words(int(m.group()))} ", text)
    text = text.upper()
    text = re.sub(r"[^A-Z ]", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()

    words = text.split(" ")
    middle = len(words) // 2
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
            # TODO: Test & tune below:
            beam_size=5,
            patience=2,
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
        assert self.activated

        wav = torch.tensor(audio_waveform, device=self.device)
        # TODO: Save audio files for debugging.
        wav, sr = self.extractor.forward(
            wav,
            sampling_rate,
            # TODO: Skip demucs if voice extractor fails.
            # skip_demucs=True,
        )

        wav = resample(wav, orig_freq=sr, new_freq=WHISPER_SAMPLE_RATE)
        # See: https://github.com/huggingface/transformers/pull/21263
        wav = normalize_distribution(wav)
        wav = whisper.pad_or_trim(wav)
        mel = whisper.log_mel_spectrogram(wav)
        mel = mel.to(self.device)

        result = whisper.decode(self.model, mel, self.options)[0]
        # TODO: Use result.no_speech_prob to determine if VoiceExtractor failed,
        # in which case, fallback to source audio or less aggressive extraction.
        # Also fallback if no digits or more than one digit detected?
        # TODO: Use digit with highest confidence/attention.
        return extract_digits(result.text)

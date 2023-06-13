"""Mock implementations of AI services for testing."""

from typing import Tuple, Union

import numpy as np
from tilsdk.cv.types import BoundingBox

from .abstract import (
    AbstractDigitDetectionService,
    AbstractObjectReIDService,
    AbstractSpeakerIDService,
)

__all__ = ["MockDigitDetectionService", "MockSpeakerIDService", "MockObjectReIDService"]


class MockDigitDetectionService(AbstractDigitDetectionService):
    """Implementation of the Digit Detection Service based on Automatic Speech Recognition."""

    def __init__(self, model_dir: str):
        """Initialize MockDigitDetectionService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        pass

    def transcribe_audio_to_digits(self, audio_waveform: np.ndarray) -> Tuple[int, ...]:
        """Transcribe audio waveform to a tuple of ints.

        Parameters
        ----------
        audio_waveform : numpy.ndarray
            Numpy 1d array of floats that represent the audio file.
            It is assumed that the sampling rate of the audio is 16K.

        Returns
        -------
        results : Tuple[int]
            The ordered tuple of digits found in the input audio file.
        """
        return (1, 2)


class MockSpeakerIDService(AbstractSpeakerIDService):
    """Implementation of the Speaker ID service."""

    def __init__(self, model_dir: str):
        """Initialize MockSpeakerIDService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        pass

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
        if audio_waveform[0] == 1:
            return "TeamName1_Member1"
        else:
            return "TeamName2_Member3"


class MockObjectReIDService(AbstractObjectReIDService):
    """Implementation of the Object Re-ID service."""

    def __init__(self, yolo_model_path: str, reid_model_path: str):
        """Initialize MockObjectReIDService.

        Parameters
        ----------
        yolo_model_dir : str
            Path of yolo model file to load.
        reid_model_path : str
            Path of reid model file to load.
        """
        pass

    def targets_from_image(self, scene_img, target_embed) -> Union[BoundingBox, None]:
        """Process image with re-id pipeline and return the bounding box of the target_embed.

        Returns None if the model doesn't believe that the target is within scene.

        Parameters
        ----------
        scene_img : np.ndarray
            Input image representing the scene to search through.
        target_embed : np.ndarray
            Target embedding.

        Returns
        -------
        results : BoundingBox or None
            BoundingBox of target within scene.
            Assume the values are NOT normalized, i.e. the bbox values are based on the raw
            pixel coordinates of the `scene_img`.
        """
        bbox = BoundingBox(100, 100, 300, 50)
        return bbox

    def embed_images(self, ims):
        """Embed images into mock embeddings of 512D."""
        B = len(ims)
        return np.ones((B, 512))

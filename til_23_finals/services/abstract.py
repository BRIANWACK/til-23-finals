"""Abstract classes for AI services."""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from tilsdk.cv.types import BoundingBox

__all__ = [
    "AbstractSpeakerIDService",
    "AbstractDigitDetectionService",
    "AbstractObjectReIDService",
]


class ActivatableService:
    """Interface for services that need to be activated and deactivated."""

    def activate(self):
        """Any preparation before actual use."""
        pass

    def deactivate(self):
        """Any cleanup after actual use."""
        pass

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate()


class AbstractDigitDetectionService(ABC, ActivatableService):
    """Interface for Digit Detection.

    This interface should be inherited from, and the following methods should be implemented.
    """

    @abstractmethod
    def __init__(self, model_dir: str):
        """Initialize AbstractDigitDetectionService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe_audio_to_digits(self, audio_waveform: np.ndarray) -> Tuple[int, ...]:
        """Transcribe audio waveform to a tuple of ints.

        Parameters
        ----------
        audio_waveform : numpy.ndarray
            Numpy 1d array of floats that represent the audio file.
            It is assumed that the sampling rate of the audio is 16K.

        Returns
        -------
        results : Tuple[int, ...]
            The ordered tuple of digits found in the input audio file.
        """
        raise NotImplementedError


class AbstractObjectReIDService(ABC, ActivatableService):
    """Interface for Object ReID.

    This interface should be inherited from, and the following methods should be implemented.
    """

    @abstractmethod
    def __init__(self, yolo_model_path: str, reid_model_path: str):
        """Initialize AbstractObjectReIDService.

        Parameters
        ----------
        yolo_model_dir : str
            Path of yolo model file to load.
        reid_model_path : str
            Path of reid model file to load.
        """
        raise NotImplementedError

    @abstractmethod
    def targets_from_image(
        self, scene_img: np.ndarray, target_img: np.ndarray
    ) -> Union[BoundingBox, None]:
        """Process image with re-id pipeline and return the bounding box of the target_img.

        Returns None if the model doesn't believe that the target is within scene.

        Parameters
        ----------
        scene_img : np.ndarray
            Input image representing the scene to search through.
        target_img : np.ndarray
            Target image representing the object to re-identify.

        Returns
        -------
        results : BoundingBox or None
            BoundingBox of target within scene.
            Assume the values are NOT normalized, i.e. the bbox values are based on the raw
            pixel coordinates of the `scene_img`.
        """
        raise NotImplementedError


class AbstractSpeakerIDService(ABC, ActivatableService):
    """Abstract class for the Speaker ID service."""

    def __init__(self, model_dir: str):
        """Initialize AbstractSpeakerIDService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

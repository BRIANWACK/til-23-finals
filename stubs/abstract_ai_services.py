"""Abstract classes for AI services."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from tilsdk.cv.types import BoundingBox, DetectedObject


class AbstractDigitDetectionService(ABC):
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
        pass

    @abstractmethod
    def transcribe_audio_to_digits(self, audio_waveform: np.ndarray) -> Tuple[int]:
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
        pass


class AbstractObjectReIDService(ABC):
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
        device : str
            the torch device to use for computation.
        """
        pass

    @abstractmethod
    def targets_from_image(self, scene_img, target_img) -> BoundingBox:
        """Process image with re-id pipeline and return the detected objects and their classes.

        Parameters
        ----------
        img : Any
            Input image.

        Returns
        -------
        results : List[DetectedObject]
            List of DetectedObjects.
        """
        # dummy data
        bbox = BoundingBox(100, 100, 300, 50)
        obj = DetectedObject("1", "1", bbox)
        # DetectedObject: ['id', 'cls', 'bbox']
        # e.g. cls=0 means match class 0, cls=1 means match class 1, cls=2 means match class2.
        pass


class AbstractSpeakerIDService(ABC):
    """Abstract class for the Speaker ID service."""

    def __init__(self, model_dir: str):
        """Initialize AbstractSpeakerIDService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        pass

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
        pass

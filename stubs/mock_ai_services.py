from typing import List, Tuple

import numpy as np
from abstract_ai_services import (
    AbstractDigitDetectionService,
    AbstractObjectReIDService,
    AbstractSpeakerIDService,
)
from tilsdk.cv.types import *


class MockDigitDetectionService(AbstractDigitDetectionService):
    """Implementation of the Digit Detection Service based on Automatic Speech Recognition."""

    def __init__(self, model_dir: str):
        """
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        pass

    def transcribe_audio_to_digits(self, audio_waveform: np.array) -> Tuple[int]:
        """Transcribe audio waveform to a tuple of ints.

        Parameters
        ----------
        audio_waveform : numpy.array
            Numpy array of floats that represent the audio file. It is assumed that the sampling rate of the audio is 16K.
        Returns
        -------
        results  :
            The ordered tuple of digits found in the input audio file.
        """
        return (1, 2)  # mock value


class MockSpeakerIDService(AbstractSpeakerIDService):
    """Implementation of the Speaker ID service."""

    def __init__(self, model_dir: str):
        """
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        pass

    def identify_speaker(self, audio_waveform: np.array, sampling_rate: int) -> str:
        """
        Parameters
        ----------
        audio_waveform : np.array
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
    """
    Implementation of the Object Re-ID service.
    """

    def __init__(self, yolo_model_path: str, reid_model_path: str, device=None):
        pass

    def targets_from_image(self, scene_img, target_img) -> BoundingBox:
        """Process image with re-id pipeline and return the bounding box of the target_img.
        Returns None if the model doesn't believe that the target is within scene.

        Parameters
        ----------
        scene_img : ndarray
            Input image representing the scene to search through.

        target_img: ndarray
            Target image representing the object to re-identify.

        Returns
        -------
        results  : BoundingBox or None
            BoundingBox of target within scene.
            Assume the values are NOT normalized, i.e. the bbox values are based on the raw
            pixel coordinates of the `scene_img`.
        """
        # dummy data
        bbox = BoundingBox(100, 100, 300, 50)
        return bbox  # return mock value for now. please change this.

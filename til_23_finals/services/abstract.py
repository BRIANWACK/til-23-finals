"""Abstract classes for AI services."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from til_23_finals.types import ReIDClass, ReIDObject

__all__ = [
    "AbstractSpeakerIDService",
    "AbstractDigitDetectionService",
    "AbstractObjectReIDService",
]


class ActivatableService:
    """Interface for services that need to be activated and deactivated."""

    def activate(self):
        """Any preparation before actual use."""
        self.activated = True

    def deactivate(self):
        """Any cleanup after actual use."""
        self.activated = False

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
    def __init__(self, model_dir: str, denoise_model_dir: str):
        """Initialize AbstractDigitDetectionService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe_audio_to_digits(
        self, audio_waveform: np.ndarray, sampling_rate: int
    ) -> Tuple[int, ...]:
        """Transcribe audio waveform to a tuple of ints.

        Parameters
        ----------
        audio_waveform : numpy.ndarray
            Numpy 1d array of floats that represent the audio file.
        sampling_rate : int
            Sampling rate of the audio.

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
    def targets_from_image(self, scene_img: np.ndarray) -> List[ReIDObject]:
        """Process image with re-id pipeline to return a list of `ReIDObject`.

        Each `ReIDObject` contains the absolute xywh coordinates of the bbox, the
        embedding of the object, the similarity score to the class (default 0.0),
        and the class (default `ReIDClass.CIVILIAN`).

        Parameters
        ----------
        scene_img : np.ndarray
            Input image representing the scene to search through.

        Returns
        -------
        results : List[ReIDObject]
            BoundingBox of targets within the scene. The coordinates are absolute.
        """
        raise NotImplementedError

    @abstractmethod
    def identity_target(
        self,
        targets: List[ReIDObject],
        suspect_embed: np.ndarray,
        hostage_embed: np.ndarray,
    ) -> Tuple[List[ReIDObject], ReIDClass, int]:
        """Identify if the suspect or hostage and present and which one.

        Note, as per the competition rules, it is assumed either the suspect or
        hostage is present in the scene, but not both. The returned list of targets
        will have suspect and hostage set nonetheless for visualization purposes.

        Parameters
        ----------
        targets : List[ReIDObject]
            List of targets to search through.
        suspect_embed : np.ndarray
            Embedding of the suspect.
        hostage_embed : np.ndarray
            Embedding of the hostage.

        Returns
        -------
        results : Tuple[List[ReIDObject], ReIDClass, int]
            List of targets with suspect and hostage set, the class of the target,
            and the index of the target in the list.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, ims: np.ndarray) -> np.ndarray:
        """Embed images into vectors."""
        raise NotImplementedError


class AbstractSpeakerIDService(ABC, ActivatableService):
    """Abstract class for the Speaker ID service."""

    def __init__(self, model_dir: str, denoise_model_dir: str):
        """Initialize AbstractSpeakerIDService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        raise NotImplementedError

    @abstractmethod
    def enroll_speaker(
        self,
        audio_waveform: np.ndarray,
        sampling_rate: int,
        team_id: str,
        member_id: str,
    ):
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
        raise NotImplementedError

    @abstractmethod
    def identify_speaker(
        self, audio_waveform: np.ndarray, sampling_rate: int, team_id: str = ""
    ) -> Dict[Tuple[str, str], float]:
        """Identify the speaker in the audio file.

        Parameters
        ----------
        audio_waveform : np.ndarray
            Input waveform.
        sampling_rate : int
            The sampling rate of the audio file.
        team_id : str
            Optional filter to identify within a specific team, defaults to "".

        Returns
        -------
        scores : Dict[Tuple[str, str], float]
            Map of the team ID & member ID to the score.
        """
        raise NotImplementedError

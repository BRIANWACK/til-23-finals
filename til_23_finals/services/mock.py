"""Mock implementations of AI services for testing."""

from typing import Dict, Tuple

import numpy as np

from til_23_finals.types import ReIDClass, ReIDObject

from .abstract import (
    AbstractDigitDetectionService,
    AbstractObjectReIDService,
    AbstractSpeakerIDService,
)

__all__ = ["MockDigitDetectionService", "MockSpeakerIDService", "MockObjectReIDService"]


class MockDigitDetectionService(AbstractDigitDetectionService):
    """Implementation of the Digit Detection Service based on Automatic Speech Recognition."""

    def __init__(self, model_dir: str, denoise_model_dir: str):
        """Initialize MockDigitDetectionService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        pass

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
        return (1, 2)


class MockSpeakerIDService(AbstractSpeakerIDService):
    """Implementation of the Speaker ID service."""

    def __init__(self, model_dir: str, denoise_model_dir: str):
        """Initialize MockSpeakerIDService.

        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        """
        pass

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
        pass

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
        if audio_waveform[0] == 1:
            return {("BRIANWACK", "MemberA"): 0.9}
        else:
            return {("PALMTREE", "MemberB"): 0.9}


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

    def targets_from_image(self, scene_img):
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
        det1 = ReIDObject(80, 240, 240, 240, np.ones((512,)), 0.0, ReIDClass.CIVILIAN)
        det2 = ReIDObject(1200, 240, 240, 240, np.ones((512,)), 0.0, ReIDClass.CIVILIAN)
        return [det1, det2]

    def identity_target(self, targets, suspect_embed, hostage_embed):
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
        return targets, ReIDClass.CIVILIAN, -1

    def embed_images(self, ims):
        """Embed images into mock embeddings of 512D."""
        B = len(ims)
        return np.ones((B, 512))

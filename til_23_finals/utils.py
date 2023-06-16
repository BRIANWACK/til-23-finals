"""Utilities."""

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import librosa
import numpy as np

from til_23_finals.types import ReIDClass, ReIDObject

__all__ = [
    "load_audio_from_dir",
    "enable_camera",
    "cos_sim",
    "thres_strategy_A",
    "thres_strategy_naive",
    "thres_strategy_softmax",
    "viz_reid",
]

data_log = logging.getLogger("Data")


def load_audio_from_dir(save_path: str) -> Dict[str, Tuple[np.ndarray, int]]:
    """Load audio files from a directory.

    Parameters
    ----------
    save_path : str
        Path to directory containing audio files.

    Returns
    -------
    audio_dict : Dict[str, Tuple[np.ndarray, int]]
        Key is the filename without extension ("audio1") and value is a tuple of the audio and sampling rate.
    """
    audio_dict = {}
    for audio_path in Path(save_path).resolve().glob("*.wav"):
        data_log.debug(f"Load audio: {audio_path}")
        wav, sr = librosa.load(audio_path, sr=None)
        audio_dict[audio_path.stem] = (wav, int(sr))
    return audio_dict


@contextmanager
def enable_camera(robot, photo_dir: Optional[Path] = None):
    """Context manager to open the video stream to take photos."""
    robot.camera.start_video_stream(display=False, resolution="720p")

    if photo_dir is not None:
        photo_dir = photo_dir.resolve()
        photo_dir.mkdir(parents=True, exist_ok=True)

    def _take_photo():
        img = robot.camera.read_cv2_image(strategy="newest")
        data_log.debug(f"Captured photo with shape: {img.shape}.")
        if photo_dir is None:
            return img

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        photo_path = photo_dir / f"cam_{timestamp}.jpg"
        if not cv2.imwrite(str(photo_path), img):
            data_log.warning(f"Could not save photo: {photo_path}")
        else:
            data_log.debug(f"Photo saved: {photo_path}")
        return img

    try:
        yield _take_photo
    finally:
        robot.camera.stop_video_stream()


def cos_sim(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def thres_strategy_A(scores: list, accept_thres=0.3, vote_thres=0.0, sd_thres=4.4):
    """Strategy based on standard deviation.

    If any in `scores` is greater than `accept_thres`, return the index of the
    max score. If any in `scores` is greater than `vote_thres`, return the index
    of the max score if it is greater than `sd` standard deviations away from
    the mean of `scores` (excluding the max score). Otherwise, return -1.

    This strategy is not found in any literature and is the author's own.

    Parameters
    ----------
    scores : List[float]
        List of scores.
    accept_thres : float, optional
        Threshold for accepting a prediction, by default 0.7
    vote_thres : float, optional
        Threshold for voting, by default 0.1
    sd_thres : float, optional
        Number of standard deviations away from the mean, by default 5.0

    Returns
    -------
    int
        The index of the max score if it meets the criteria, otherwise -1.
    """
    if np.max(scores) > accept_thres:
        return np.argmax(scores)
    elif np.max(scores) > vote_thres:
        scores = np.array(scores).clip(0.0)  # type: ignore
        mean = np.mean(scores[scores < np.max(scores)])
        std = np.std(scores[scores < np.max(scores)])
        if np.max(scores) - mean > sd_thres * std:
            return np.argmax(scores)
    return -1


def thres_strategy_naive(scores: list, thres=0.3):
    """Naive thresholding strategy."""
    if len(scores) < 1:
        return -1
    if np.max(scores) > thres:
        return np.argmax(scores)
    return -1


def thres_strategy_softmax(scores: list, temp=0.8, ratio=1.4):
    """Threshold using softmax."""
    x = np.array(scores) / temp  # type: ignore
    ex = np.exp(x - np.max(x))
    ex /= ex.sum() + 1e-12
    # TODO: Figure out proper solution to sensitivity.
    if np.max(ex) > ratio / (len(ex) + 1):
        return np.argmax(ex)
    return -1


def viz_reid(img: np.ndarray, objects: List[ReIDObject]):
    """Visualize the reid results."""
    img = img.copy()
    for obj in objects:
        if obj.cls == ReIDClass.CIVILIAN:
            col = (255, 0, 0)
        elif obj.cls == ReIDClass.SUSPECT:
            col = (0, 0, 255)
        elif obj.cls == ReIDClass.HOSTAGE:
            col = (0, 255, 0)
        else:
            data_log.critical(f"Invalid reid class: {obj.cls}")
            continue
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_pt = (obj.x, max(0, obj.y - 10))
        img = cv2.rectangle(img, (obj.x, obj.y, obj.w, obj.h), col, 3)
        img = cv2.putText(
            img, f"{obj.cls.value} {obj.sim:.3f}", text_pt, font, 0.5, col, 2
        )

    return img

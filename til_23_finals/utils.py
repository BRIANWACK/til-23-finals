"""Utilities."""

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import librosa
import numpy as np
from scipy.ndimage import distance_transform_cdt
from tilsdk.localization import GridLocation, RealPose, SignedDistanceGrid
from tilsdk.localization.types import GridLocation

from til_23_finals.types import Heading, LocOrPose, ReIDClass, ReIDObject

__all__ = [
    "load_audio_from_dir",
    "enable_camera",
    "cos_sim",
    "thres_strategy_naive",
    "viz_reid",
    "viz_pose",
    "get_ang_delta",
    "ang_to_heading",
    "nearest_cardinal",
    "ang_to_waypoint",
    "ManhattanSDGrid",
]

viz_log = logging.getLogger("Viz")
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


def thres_strategy_naive(scores: list, thres=0.3):
    """Naive thresholding strategy."""
    if len(scores) < 1:
        return -1
    if np.max(scores) > thres:
        return np.argmax(scores)
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
            viz_log.critical(f"Invalid reid class: {obj.cls}")
            continue
        font = cv2.FONT_HERSHEY_SIMPLEX
        lbl = f"{obj.cls.value} {obj.sim:.3f}"
        text_pt = (obj.x, max(0, obj.y - 10))
        try:
            img = cv2.rectangle(img, (obj.x, obj.y, obj.w, obj.h), col, 3)
            img = cv2.putText(img, lbl, text_pt, font, 0.5, col, 2)
        except Exception as e:
            viz_log.error(f"Error in `viz_reid`.", exc_info=e)
    return img


def viz_pose(grid: np.ndarray, grid_pt: GridLocation, heading: float, color=0):
    """Visualize the pose on map grid."""
    grid = grid.copy()
    radius = 10
    tip_len = 0.5
    thickness = 2
    x, y = grid_pt.x, grid_pt.y
    try:
        cos = np.cos(np.deg2rad(heading))
        sin = np.sin(np.deg2rad(heading))
        base = (round(x - radius * cos), round(y - radius * sin))
        tip = (round(x + radius * cos), round(y + radius * sin))
        cv2.arrowedLine(grid, base, tip, color, thickness, tipLength=tip_len)
    except Exception as e:
        viz_log.error(f"Error in `viz_pose`.", exc_info=e)
    return grid


def get_ang_delta(cur: float, tgt: float):
    """Get angular difference in degrees of two angles in degrees.

    Returns a value in the range [-180, 180].
    """
    delta = tgt - cur
    if delta < -180:
        delta += 360
    if delta > 180:
        delta -= 360
    return delta


def ang_to_heading(ang: float):
    """Convert relative angle to absolute headings in degrees."""
    while ang < 0:
        ang += 360
    return ang % 360


def nearest_cardinal(heading: float) -> Heading:
    """Get nearest cardinal heading."""
    deltas = {h: abs(get_ang_delta(heading, h)) for h in Heading}
    return min(deltas, key=deltas.get)  # type: ignore


def ang_to_waypoint(pose: RealPose, waypoint: LocOrPose):
    """Get angular difference in degrees of current pose to current waypoint."""
    ang_to_wp = np.degrees(np.arctan2(waypoint.y - pose.y, waypoint.x - pose.x))
    delta = get_ang_delta(ang_to_wp, pose.z)
    return delta


class ManhattanSDGrid(SignedDistanceGrid):
    """Override neighbours to ignore diagonals for faster computation & straight paths."""

    def neighbours(self, id: GridLocation) -> List[Tuple[GridLocation, float, float]]:
        """Ignore diagonals."""
        x, y = id.x, id.y
        neighbours = [
            # (GridLocation(x - 1, y - 1), _SQRT2),  # NW
            (GridLocation(x, y - 1), 1.0),  # N
            # (GridLocation(x + 1, y - 1), _SQRT2),  # NE
            (GridLocation(x - 1, y), 1.0),  # W
            (GridLocation(x + 1, y), 1.0),  # E
            # (GridLocation(x - 1, y + 1), _SQRT2),  # SW
            (GridLocation(x, y + 1), 1.0),  # S
            # (GridLocation(x + 1, y + 1), _SQRT2),  # SE
        ]
        results = []
        for loc, dist in neighbours:
            if self.in_bounds(loc) and self.passable(loc):
                results.append((loc, dist, self.grid[loc.y, loc.x]))
        return results

    @classmethod
    def from_old_class(cls, old: SignedDistanceGrid):
        """Create new instance from old instance."""
        # Convert signed distance grid from euclidean to chessboard.
        grid = old.grid
        # Add walls around the grid.
        b = 5
        grid[:b, :] = grid[-b:, :] = grid[:, :b] = grid[:, -b:] = -1
        grid = grid[:, :] < 0
        grid = distance_transform_cdt(1 - grid) - distance_transform_cdt(grid)
        return cls(grid=grid, scale=old.scale)

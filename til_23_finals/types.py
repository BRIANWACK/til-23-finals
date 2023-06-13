"""Utility types."""

from enum import Enum
from typing import NamedTuple

import numpy as np

__all__ = ["ReIDObject", "ReIDClass"]


class ReIDClass(Enum):
    """Object classes."""

    SUSPECT = "suspect"
    HOSTAGE = "hostage"
    CIVILIAN = "none"


class ReIDObject(NamedTuple):
    """ReID detected plushie."""

    x: int
    y: int
    w: int
    h: int
    emb: np.ndarray
    sim: float
    cls: ReIDClass

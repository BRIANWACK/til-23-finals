"""Utility types."""

from enum import Enum, IntEnum
from typing import NamedTuple, Union

import numpy as np
from tilsdk.localization.types import RealLocation, RealPose

__all__ = ["LocOrPose", "ReIDObject", "ReIDClass", "Heading", "SpeakerID"]

LocOrPose = Union[RealPose, RealLocation]


class ReIDClass(Enum):
    """Object classes."""

    SUSPECT = "suspect"
    HOSTAGE = "hostage"
    CIVILIAN = "none"


class Heading(IntEnum):
    """0 when +x -> +x, 90 when +x -> +y, 180 when +x -> -x, and 360 when +x -> -y."""

    POS_X = 0
    POS_Y = 90
    NEG_X = 180
    NEG_Y = 270


class ReIDObject(NamedTuple):
    """ReID detected plushie."""

    x: int
    y: int
    w: int
    h: int
    emb: np.ndarray
    sim: float
    cls: ReIDClass


class SpeakerID(NamedTuple):
    """Speaker Identity."""

    team_id: str
    member_id: str
    raw_embed: np.ndarray
    clean_embed: np.ndarray

"""Code to emulate missing functionality in the simulator."""

import logging
import time

__all__ = ["bind_robot"]

log = logging.getLogger("Emulate")


class Action:
    """Mock Action."""

    def __init__(self, pause):
        self.pause = pause
        self.start = time.time()

    @property
    def _time_left(self):
        return self.pause + self.start - time.time()

    @property
    def is_completed(self):
        """Whether action is completed."""
        return self._time_left <= 0

    def wait_for_completed(self):
        """Block till action is completed."""
        if self.is_completed:
            return
        time.sleep(self._time_left)


class Gimbal:
    """Mock Gimbal."""

    def __init__(self, robot):
        self.robot = robot

    def move(self, pitch=0, yaw=0, pitch_speed=30, yaw_speed=30):
        """Mock gimbal move."""
        t = max(abs(pitch) / pitch_speed, abs(yaw) / yaw_speed)
        log.info(f"[gimbal.move] pitch: {pitch}, yaw: {yaw}, t: {t}")
        return Action(t)

    def recenter(self):
        """Mock gimbal recenter."""
        log.info("[gimbal.recenter] recentered")
        return Action(0)


def move(self, x=0, y=0, z=0, xy_speed=0.5, z_speed=30):
    """Mock move."""
    assert z == 0 or (x == 0 and y == 0), "Cannot move in xy and z at the same time."
    if z == 0:
        d = (x**2 + y**2) ** 0.5
        t = d / xy_speed
        log.info(f"[chassis.move] x: {x}, y: {y}, t: {t}")
        self.drive_speed(x / t, y / t, 0)
    else:
        t = abs(z) / z_speed
        log.info(f"[chassis.move] z: {z}, t: {t}")
        self.drive_speed(0, 0, z / t)
    return Action(t)


def bind_robot(robot):
    """Bind missing functionality to simulator robot."""
    bound_method = move.__get__(robot.chassis, robot.chassis.__class__)
    setattr(robot.chassis, "move", bound_method)
    setattr(robot, "gimbal", Gimbal(robot))

"""Code to emulate missing functionality in the simulator."""

import logging
import time

__all__ = ["bind_robot"]

log = logging.getLogger("Emulate")

# Multiply speed of everything by this factor to speed up testing.
SPEED = 1


class Action:
    """Mock Action."""

    def __init__(self, pause, callback=None):
        self.pause = pause
        self.start = time.time()
        self.callback = callback
        self.has_succeeded = False

    @property
    def _time_left(self):
        left = self.pause + self.start - time.time()
        if left <= 0:
            self._trigger()
        return left

    @property
    def is_completed(self):
        """Whether action is completed."""
        return self._time_left <= 0

    def _trigger(self):
        self.has_succeeded = True
        if self.callback is not None:
            self.callback()
            self.callback = None

    def wait_for_completed(self):
        """Block till action is completed."""
        if self.is_completed:
            return
        time.sleep(self._time_left)
        self._trigger()


class Gimbal:
    """Mock Gimbal."""

    def __init__(self, robot):
        self.robot = robot

    def move(self, pitch=0, yaw=0, pitch_speed=30, yaw_speed=30):
        """Mock gimbal move."""
        pitch_speed *= SPEED
        yaw_speed *= SPEED
        t = max(abs(pitch) / pitch_speed, abs(yaw) / yaw_speed)
        log.info(f"[gimbal.move] pitch: {pitch}, yaw: {yaw}, t: {t}")
        return Action(t)

    def recenter(self):
        """Mock gimbal recenter."""
        log.info("[gimbal.recenter] recentered")
        return Action(0)


def move(self, x=0, y=0, z=0, xy_speed=0.5, z_speed=30):
    """Mock move."""
    eps = 0.001
    xy_speed *= SPEED
    z_speed *= SPEED
    assert z == 0 or (x == 0 and y == 0), "Cannot move in xy and z at the same time."
    if z == 0:
        d = (x**2 + y**2) ** 0.5
        t = d / xy_speed
        log.info(f"[chassis.move] x: {x}, y: {y}, t: {t}")
        if t < eps:
            return Action(0, lambda: self.drive_speed())
        self.drive_speed(x / t, y / t, 0)
    else:
        t = abs(z) / z_speed
        log.info(f"[chassis.move] z: {z}, t: {t}")
        if t < eps:
            return Action(0, lambda: self.drive_speed())
        self.drive_speed(0, 0, z / t)
    return Action(t, lambda: self.drive_speed())


def bind_robot(robot):
    """Bind missing functionality to simulator robot."""
    bound_method = move.__get__(robot.chassis, robot.chassis.__class__)
    setattr(robot.chassis, "move", bound_method)
    setattr(robot, "gimbal", Gimbal(robot))

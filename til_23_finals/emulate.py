"""Code to emulate missing functionality in the simulator."""

import time

__all__ = ["bind_robot"]


class Action:
    """Mock Action."""

    def __init__(self, pause):
        self.start = time.time()
        self.pause = pause

    def wait_for_completed(self):
        """Block till action is completed."""
        time.sleep(self.pause)

    @property
    def is_completed(self):
        """Whether action is completed."""
        return time.time() - self.start > self.pause


class Gimbal:
    """Mock Gimbal."""

    def __init__(self, robot):
        self.robot = robot

    def move(self, pitch=0, yaw=0):
        """Mock gimbal move."""
        self.robot.chassis.drive_speed(x=0, y=0, z=yaw / 1)
        return Action(1)

    def recenter(self):
        """Mock gimbal recenter."""
        return Action(1)


def move(self, x=0, y=0, z=0, xy_speed=0.5, z_speed=30):
    """Mock move."""
    print(f"move x: {x}, y: {y}, z: {z}")
    self.drive_speed(x / 1, y / 1, z / 1)
    return Action(1)


def bind_robot(robot):
    """Bind missing functionality to simulator robot."""
    bound_method = move.__get__(robot.chassis, robot.chassis.__class__)
    setattr(robot.chassis, "move", bound_method)
    setattr(robot, "gimbal", Gimbal(robot))

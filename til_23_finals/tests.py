"""Test robot and code functionality."""

import cv2
import imutils
import matplotlib
import matplotlib.pyplot as plt
from tilsdk import RealLocation

from .navigation import Navigator
from .utils import viz_pose

matplotlib.use("TkAgg")

__all__ = [
    "WASD_loop",
    "gimbal_stationary_test",
    "gimbal_moving_test",
    "heading_test",
    "basic_navigation_test",
    "TOF_test",
]


def WASD_loop(nav: Navigator, trans_vel_mag=0.5, ang_vel_mag=30):
    """Run manual control loop using WASD keys."""
    forward_vel = 0
    rightward_vel = 0
    ang_vel = 0

    # gimbalPhase = 0
    # self.robot.gimbal.recenter()

    while True:
        pose = nav.get_filtered_pose()
        print(pose)
        if pose is None:
            continue

        real_location = RealLocation(x=pose[0], y=pose[1])
        grid_location = nav.map.real_to_grid(real_location)

        mapMat = nav.map.grid.copy()
        mapMat = viz_pose(mapMat, grid_location, pose[2])

        key = chr(cv2.waitKey(1))
        if key in "wW":
            forward_vel = trans_vel_mag
            rightward_vel = 0
            ang_vel = 0
        elif key in "aA":
            forward_vel = 0
            rightward_vel = -trans_vel_mag
            ang_vel = 0
        elif key in "sS":
            forward_vel = -trans_vel_mag
            rightward_vel = 0
            ang_vel = 0
        elif key in "dD":
            forward_vel = 0
            rightward_vel = trans_vel_mag
            ang_vel = 0
        elif key in "qQ":
            forward_vel = 0
            rightward_vel = 0
            ang_vel = -ang_vel_mag
        elif key in "eE":
            forward_vel = 0
            rightward_vel = 0
            ang_vel = ang_vel_mag
        elif key == chr(27):
            cv2.destroyAllWindows()
            break
        else:
            forward_vel = 0
            rightward_vel = 0
            ang_vel = 0

        nav.robot.chassis.drive_speed(x=forward_vel, y=rightward_vel, z=ang_vel)

        mapMat = imutils.resize(mapMat, width=600)
        cv2.imshow("Map", mapMat)

        plt.ion()
        plt.scatter(grid_location.x, grid_location.y)
        plt.draw()
        plt.pause(0.01)

        nav.robot._modules["DistanceSensor"].sub_distance(
            freq=1, callback=lambda x: print("TOF", x)
        )

        # amplitude = 30
        # pitch = int(round(amplitude*np.sin(gimbalPhase/180)))
        # gimbalPhase += 1
        # self.robot.gimbal.moveto(pitch=pitch)


def gimbal_stationary_test(nav: Navigator, pitchMag=30, yawMag=30):
    """Gimbal stationary test."""
    nav.robot.gimbal.recenter().wait_for_completed()
    nav.robot.gimbal.move(pitch=-pitchMag).wait_for_completed()
    nav.robot.gimbal.move(pitch=pitchMag).wait_for_completed()
    nav.robot.gimbal.move(yaw=-yawMag).wait_for_completed()
    nav.robot.gimbal.move(yaw=yawMag).wait_for_completed()


def gimbal_moving_test(nav: Navigator):
    """Gimbal moving test."""
    robotMoveAction = nav.robot.chassis.move(y=nav.BOARDSCALE)
    gimbalMoveAction = nav.robot.gimbal.move(pitch=-60)
    gimbalMoveAction.wait_for_completed()
    robotMoveAction.wait_for_completed()


def heading_test(nav: Navigator):
    """Test if turning to various headings is correct."""
    import random

    cur_pose = nav.measure_pose(heading_only=True)
    print(f"INITIAL HEADING: {cur_pose.z}")
    tgt = random.randint(0, 359)
    print(f"TARGET HEADING: {tgt}")
    nav.set_heading(cur_pose.z, tgt, spd=30.0).wait_for_completed()
    cur_pose = nav.measure_pose(heading_only=True)
    print(f"FINAL HEADING: {cur_pose.z}")


def basic_navigation_test(nav: Navigator):
    """Move by 1 board length up, down, left, right, and turn 90 deg clockwise and anti-clockwise."""
    nav.robot.chassis.move(y=nav.BOARDSCALE).wait_for_completed()
    nav.robot.chassis.move(y=-nav.BOARDSCALE).wait_for_completed()
    nav.robot.chassis.move(x=-nav.BOARDSCALE).wait_for_completed()
    nav.robot.chassis.move(x=nav.BOARDSCALE).wait_for_completed()
    nav.robot.chassis.move(z=90).wait_for_completed()
    nav.robot.chassis.move(z=-90).wait_for_completed()


def TOF_test(nav: Navigator):
    """Test TOF."""
    print(f"Distance Sensor Version No.: {nav.robot.sensor.get_version()}")

    def cb_distance(val):
        print("[left,right,front,back]", val)

    tof = nav.robot.sensor
    tof.sub_distance(freq=1, callback=cb_distance)

    while True:
        pass

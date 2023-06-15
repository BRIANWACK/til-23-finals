"""Navigation code."""

import logging
import time
from typing import List

import cv2
import imutils
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tilsdk.localization import (
    RealLocation,
    RealPose,
    SignedDistanceGrid,
    euclidean_distance,
)
from tilsdk.utilities import PIDController

# Exceptions for path planning.
from .planner import InvalidStartException, NoPathFoundException

matplotlib.use("TkAgg")

# This can be tuned.
DEFAULT_PID = dict(Kp=(0.5, 0.20), Ki=(0.2, 0.1), Kd=(0.0, 0.0))

main_log = logging.getLogger("Main")
nav_log = logging.getLogger("Nav")
ctrl_log = logging.getLogger("Ctrl")

def ang_difference(ang1, ang2):
    """Get angular difference in degrees of two angles in degrees.

    Returns a value in the range [-180, 180].
    """
    ang_diff = -(ang1 - ang2)  # body frame

    # ensure ang_diff is in [-180, 180]
    if ang_diff < -180:
        ang_diff += 360

    if ang_diff > 180:
        ang_diff -= 360
    return ang_diff

def ang_diff_to_wp(pose: RealPose, curr_wp):
    """Get angular difference in degrees of current pose to current waypoint."""
    ang_to_wp = np.degrees(np.arctan2(curr_wp[1] - pose[1], curr_wp[0] - pose[0]))
    ang_diff = ang_difference(ang_to_wp, pose[2])
    return ang_diff

class Navigator:
    """Navigator class."""

    def __init__(self, map_, robot, loc_service, planner, pose_filter, cfg):
        self.map: SignedDistanceGrid = map_
        self.loc_service = loc_service
        self.planner = planner
        self.pose_filter = pose_filter
        self.controller = PIDController(**DEFAULT_PID)

        IS_SIM = not cfg["use_real_localization"]
        if IS_SIM:
            from tilsdk.mock_robomaster.robot import Robot
        else:
            from robomaster.robot import Robot

        self.robot: Robot = robot

        self.VISUALIZE_FLAG = cfg["VISUALIZE_FLAG"]
        self.REACHED_THRESHOLD_M = cfg["REACHED_THRESHOLD_M"]
        self.ANGLE_THRESHOLD_DEG = cfg["ANGLE_THRESHOLD_DEG"]

        self.BOARDSCALE = 0.5 # length of board in m
    
    def plan_path(self, start: list, goal) -> List[RealLocation]:
        """Plan path."""
        try:
            current_coord = RealLocation(x=start[0], y=start[1])
            path = self.planner.plan(current_coord, goal)
            main_log.info("Path planned.")
            return path

        except InvalidStartException as e:
            ## Ensure only valid start positions are passed to the planner.
            nav_log.warning(f"{e}")
            # TODO: find and use another valid start point.
            return None

    def get_filtered_pose(self):
        """Get filtered pose."""
        pose = self.loc_service.get_pose()
        # no new pose data, continue to next iteration.
        if not pose:
            return None
        return self.pose_filter.update(pose)

    def drawPose(self, mapMat, grid_location, heading):
        """Draw pose on map."""
        cosTheta = np.cos(np.deg2rad(heading))
        sinTheta = np.sin(np.deg2rad(heading))
        arrowRadius = 10
        arrowStart = tuple(
            map(
                lambda x: int(round(x)),
                (
                    grid_location.x - arrowRadius * cosTheta,
                    grid_location.y - arrowRadius * sinTheta,
                ),
            )
        )
        arrowEnd = tuple(
            map(
                lambda x: int(round(x)),
                (
                    grid_location.x + arrowRadius * cosTheta,
                    grid_location.y + arrowRadius * sinTheta,
                ),
            )
        )

        cv2.arrowedLine(mapMat, arrowStart, arrowEnd, 0, 2, tipLength=0.5)

    def turnRobot(self, target_rotation):
        nav_log.info("Turning robot to face target angle...")
        while abs(rel_ang) > 20:
            pose = self.get_filtered_pose()
            rel_ang = ang_difference(
                pose[2], target_rotation
            )  # current heading vs target heading

            if rel_ang < -20:
                # rotate counter-clockwise
                nav_log.info(f"Trying to turn clockwise... ang left: {rel_ang}")
                self.robot.chassis.drive_speed(x=0, z=10)
            elif rel_ang > 20:
                # rotate clockwise
                nav_log.info(
                    f"Trying to turn counter-clockwise... ang left: {rel_ang}"
                )
                self.robot.chassis.drive_speed(x=0, z=-10)
            time.sleep(1)
        nav_log.info("Robot should now be facing close to target angle.")

    def given_navigation_loop(self, last_valid_pose, curr_loi, target_rotation):
        """Run navigation loop."""
        path = self.plan_path(last_valid_pose, curr_loi)
        if path is None: # Due to invalid pose
            return

        while True:
            pose = self.get_filtered_pose()
            if pose is None:
                continue

            real_location = RealLocation(x=pose[0], y=pose[1])
            grid_location = self.map.real_to_grid(real_location)

            if self.VISUALIZE_FLAG:
                mapMat = self.map.grid.copy()
                self.drawPose(mapMat, grid_location, pose[2])
                cv2.waitKey(1)

            if self.map.in_bounds(grid_location) and self.map.passable(grid_location):
                last_valid_pose = pose
            else:
                main_log.warning(
                    f"Invalid pose received from localization server. Skipping."
                )
                continue

            dist_to_goal = euclidean_distance(last_valid_pose, curr_loi)
            if round(dist_to_goal, 2) <= self.REACHED_THRESHOLD_M:  # Reached checkpoint.
                nav_log.info(
                    f"Reached checkpoint {last_valid_pose[0]:.2f},{last_valid_pose[1]:.2f}"
                )
                path = []  # flush path.
                self.controller.reset()

                # ROTATE ROBOT TO TARGET ORIENTATION.
                rel_ang = ang_difference(
                    last_valid_pose[2], target_rotation
                )  # current heading vs target heading
                self.turnRobot(rel_ang)

                curr_wp = None
                prev_loi = curr_loi
                curr_loi = None
                try_start_tasks = True  # try to start ai tasks in the next main iter.
                break

            elif path:
                curr_wp = path[0]  # nearest waypoint is at the start of list.

                if self.VISUALIZE_FLAG:
                    curr_wp_grid = self.map.real_to_grid(curr_wp)
                    cv2.circle(mapMat, (curr_wp_grid.x, curr_wp_grid.y), 5, 0, -1)

                nav_log.info(f"Num wps left: {len(path)}")

                dist_to_wp = euclidean_distance(real_location, curr_wp)
                if round(dist_to_wp, 2) < self.REACHED_THRESHOLD_M:
                    path = path[1:]  # remove the nearest waypoint.
                    self.controller.reset()
                    continue  # start navigating to next waypoint.

                pose_str = f"x:{last_valid_pose[0]:.2f}, y:{last_valid_pose[1]:.2f}, rot:{last_valid_pose[2]:.2f}"
                curr_wp_str = f"{curr_wp[0]:.2f}, {curr_wp[1]:.2f}"
                curr_loi_str = f"{curr_loi[0]:.2f}, {curr_loi[1]:.2f}"
                dist_to_wp = euclidean_distance(real_location, curr_wp)
                nav_log.info(
                    f"Goal: {curr_loi_str}, {target_rotation} \t Pose: {pose_str}"
                )
                nav_log.info(f"WP: {curr_wp_str} \t dist_to_wp: {dist_to_wp:.2f}\n")

                ang_diff = ang_diff_to_wp(last_valid_pose, curr_wp)

                # Move robot until next waypoint is reached.
                # Determine velocity commands given distance to waypoint and heading to waypoint.
                vel_cmd = self.controller.update((dist_to_wp, ang_diff))
                vel_cmd[0] *= np.cos(
                    np.radians(ang_diff)
                )  # reduce forward velocity based on angular difference.

                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > self.ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0

                forward_vel, ang_vel = vel_cmd[0], vel_cmd[1]
                ctrl_log.info(
                    f"input for final forward speed and rotation speed: {forward_vel:.2f}, {ang_vel:.2f}"
                )

                self.robot.chassis.drive_speed(x=forward_vel, z=ang_vel)
                time.sleep(1)
            else:
                nav_log.info("Did not reach checkpoint and no waypoints left.")
                raise Exception("Did not reach checkpoint and no waypoints left.")

            mapMat = imutils.resize(mapMat, width=600)
            cv2.imshow("Map", mapMat)

            plt.ion()
            plt.scatter(grid_location.x, grid_location.y)
            plt.draw()
            plt.pause(0.01)

        return curr_wp, prev_loi, curr_loi, try_start_tasks

    def basic_navigation_loop(self, last_valid_pose: RealPose, curr_loi, target_rotation):
        # TODO: Test movement accuracy, no simulator equivalent
        # TODO: Measure length of board, assumed to be 0.5 m now
        # TODO: Add visualization code

        path = self.plan_path(last_valid_pose, curr_loi)
        if path is None: # Due to invalid pose
            return
        
        curr_estimated_pose = [last_valid_pose.x, last_valid_pose.y]
        while path:
            wp = path[0]
            path = path[1:]
            deltaX = self.BOARDSCALE*(wp.x - curr_estimated_pose.x)
            deltaY = self.BOARDSCALE*(wp.y - curr_estimated_pose.y)
            self.robot.chassis.move(x=deltaX, y=deltaY).wait_for_completed()
            curr_estimated_pose = wp

        # Check direction
        rel_ang = ang_difference(
                    last_valid_pose.z, target_rotation
                )  # current heading vs target heading
        self.robot.chassis.move(z=rel_ang).wait_for_completed()
    
    def WASD_loop(self, trans_vel_mag=0.5, ang_vel_mag=30):
        """Run manual control loop using WASD keys."""
        forward_vel = 0
        rightward_vel = 0
        ang_vel = 0

        # gimbalPhase = 0
        # self.robot.gimbal.recenter()

        while True:
            pose = self.get_filtered_pose()
            print(pose)
            if pose is None:
                continue

            real_location = RealLocation(x=pose[0], y=pose[1])
            grid_location = self.map.real_to_grid(real_location)

            mapMat = self.map.grid.copy()
            self.drawPose(mapMat, grid_location, pose[2])

            key = cv2.waitKey(1)
            if key == ord("w") or key == ord("W"):
                forward_vel = trans_vel_mag
                rightward_vel = 0
                ang_vel = 0
            elif key == ord("a") or key == ord("A"):
                forward_vel = 0
                rightward_vel = -trans_vel_mag
                ang_vel = 0
            elif key == ord("s") or key == ord("S"):
                forward_vel = -trans_vel_mag
                rightward_vel = 0
                ang_vel = 0
            elif key == ord("d") or key == ord("D"):
                forward_vel = 0
                rightward_vel = trans_vel_mag
                ang_vel = 0
            elif key == ord("q") or key == ord("Q"):
                forward_vel = 0
                rightward_vel = 0
                ang_vel = -ang_vel_mag
            elif key == ord("e") or key == ord("E"):
                forward_vel = 0
                rightward_vel = 0
                ang_vel = ang_vel_mag
            elif key == 27:
                cv2.destroyAllWindows()
                break
            else:
                forward_vel = 0
                rightward_vel = 0
                ang_vel = 0

            self.robot.chassis.drive_speed(x=forward_vel, y=rightward_vel, z=ang_vel)

            mapMat = imutils.resize(mapMat, width=600)
            cv2.imshow("Map", mapMat)

            plt.ion()
            plt.scatter(grid_location.x, grid_location.y)
            plt.draw()
            plt.pause(0.01)

            self.robot._modules["DistanceSensor"].sub_distance(
                freq=1, callback=lambda x: print("TOF", x)
            )

            # amplitude = 30
            # pitch = int(round(amplitude*np.sin(gimbalPhase/180)))
            # gimbalPhase += 1
            # self.robot.gimbal.moveto(pitch=pitch)

    def gimbal_stationary_test(self, pitchMag, yawMag):
        self.robot.gimbal.recenter().wait_for_completed()
        self.robot.gimbal.move(pitch=-pitchMag).wait_for_completed()
        self.robot.gimbal.move(pitch=pitchMag).wait_for_completed()
        self.robot.gimbal.move(yaw=-yawMag).wait_for_completed()
        self.robot.gimbal.move(yaw=yawMag).wait_for_completed()

    def gimbal_moving_test(self):
        robotMoveAction = self.robot.chassis.move(y=self.BOARDSCALE)
        gimbalMoveAction = self.robot.gimbal.move(pitch=-60)
        gimbalMoveAction.wait_for_completed()
        robotMoveAction.wait_for_completed()

    def gimbal_moving_test2(self):
        pass
        
    def TOF_test(self):
        print(f"Distance Sensor Version No.: {self.robot.sensor.get_version()}")

        def cb_distance(val):
            print("[left,right,front,back]", val)

        tof = self.robot.sensor
        tof.sub_distance(freq=1, callback=cb_distance)
        
        while True:
            pass

    def basic_navigation_test(self):
        """Move by 1 board length up, down, left, right, and turn 90 deg clockwise and anti-clockwise."""
        self.robot.chassis.move(y=self.BOARDSCALE).wait_for_completed()
        self.robot.chassis.move(y=-self.BOARDSCALE).wait_for_completed()
        self.robot.chassis.move(x=-self.BOARDSCALE).wait_for_completed()
        self.robot.chassis.move(x=self.BOARDSCALE).wait_for_completed()
        self.robot.chassis.move(z=90).wait_for_completed()
        self.robot.chassis.move(z=-90).wait_for_completed()


# if __name__ == "__main__":

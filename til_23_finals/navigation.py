"""Localization API based navigation code."""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Union

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
from tilsdk.localization.service import LocalizationService
from tilsdk.utilities import PIDController

# Exceptions for path planning.
from .planner import InvalidStartException, NoPathFoundException, Planner
from .types import LocOrPose
from .utils import ang_to_heading, ang_to_waypoint, get_ang_delta, viz_pose

matplotlib.use("TkAgg")

nav_log = logging.getLogger("Nav")


class Navigator(ABC):
    """Base Navigator class."""

    def __init__(
        self,
        arena_map: SignedDistanceGrid,
        robot,
        loc_service: LocalizationService,
        planner: Planner,
        pose_filter,
        cfg,
    ):
        IS_SIM = not cfg["use_real_localization"]
        if IS_SIM:
            from tilsdk.mock_robomaster.robot import Robot
        else:
            from robomaster.robot import Robot

        self.map = arena_map
        self.robot: Robot = robot
        self.loc_service = loc_service
        self.planner = planner
        self.pose_filter = pose_filter

        self.VISUALIZE_FLAG = cfg["VISUALIZE_FLAG"]
        self.REACHED_THRESHOLD_M = cfg["REACHED_THRESHOLD_M"]
        self.ANGLE_THRESHOLD_DEG = cfg["ANGLE_THRESHOLD_DEG"]
        self.FLIP_X = cfg["FLIP_X"]
        self.FLIP_Y = cfg["FLIP_Y"]
        self.FLIP_Z = cfg["FLIP_Z"]
        self.SWAP_XY = cfg["SWAP_XY"]

    @abstractmethod
    def plan_path(
        self, start: LocOrPose, goal: LocOrPose
    ) -> Union[List[RealLocation], None]:
        """Plan path."""
        raise NotImplementedError

    def get_raw_pose(self, correct_heading=True):
        """Get raw pose. Server has hard rate limit of 5 per second."""
        pose = self.loc_service.get_pose()
        # NOTE: loc_service returns (None, None).
        if not isinstance(pose, RealPose):
            return None
        # Update filter for `get_filtered_pose()`.
        self.pose_filter.update(pose)
        z = ang_to_heading(pose.z) if correct_heading else pose.z
        return RealPose(pose.x, pose.y, z)

    def get_filtered_pose(self):
        """Get filtered pose."""
        pose = self.get_raw_pose()
        if pose is None:
            return None
        return self.pose_filter.get_value()

    def is_pose_valid(self, loc: LocOrPose):
        """Check whether the pose is in bounds."""
        real_loc = RealLocation(loc.x, loc.y)
        grid_loc = self.map.real_to_grid(real_loc)
        if not self.map.in_bounds(grid_loc):
            nav_log.warning(f"Pose is out of bounds: {loc}, {grid_loc}")
            return False
        if not self.map.passable(grid_loc):
            nav_log.warning(f"Pose is inside wall: {loc}, {grid_loc}")
            return False
        return True

    # TODO: LocNavigator should update to use this.
    @abstractmethod
    def navigation_loop(self, tgt_pose: LocOrPose, ini_pose: Optional[RealPose] = None):
        """Navigation loop."""
        raise NotImplementedError


class LocNavigator(Navigator):
    """Localization API Navigator class."""

    # This can be tuned.
    DEFAULT_PID = dict(Kp=(0.5, 0.20), Ki=(0.2, 0.1), Kd=(0.0, 0.0))

    def __init__(
        self,
        arena_map: SignedDistanceGrid,
        robot,
        loc_service: LocalizationService,
        planner: Planner,
        pose_filter,
        cfg,
    ):
        super(LocNavigator, self).__init__(
            arena_map, robot, loc_service, planner, pose_filter, cfg
        )
        self.controller = PIDController(**self.DEFAULT_PID)

    def plan_path(self, start, goal):
        """Plan path."""
        try:
            path = self.planner.plan(start, goal)
            nav_log.info("Path planned.")
            return path
        except InvalidStartException as e:
            nav_log.error(f"Invalid start position: {start}", exc_info=e)
            return None
        except NoPathFoundException as e:
            nav_log.error("No path found.", exc_info=e)
            return None

    def wait_for_valid_pose(self, ignore_invalid=False):
        """Block until the pose received is valid."""
        while True:
            pose = self.get_filtered_pose()
            if pose is None:
                nav_log.warning("No pose received from localization server.")
                continue
            if not self.is_pose_valid(pose):
                nav_log.warning("Invalid pose received from localization server.")
                if ignore_invalid:
                    return pose
                continue
            return pose

    def turnRobot(self, target_rotation):
        """Turn robot."""
        nav_log.info("Turning robot to face target angle...")
        rel_ang = 180
        while abs(rel_ang) > 20:
            pose = self.wait_for_valid_pose()
            rel_ang = get_ang_delta(
                pose[2], target_rotation
            )  # current heading vs target heading

            if rel_ang < -20:
                # rotate counter-clockwise
                nav_log.info(f"Trying to turn clockwise... ang left: {rel_ang}")
                self.robot.chassis.drive_speed(x=0, z=10)
            elif rel_ang > 20:
                # rotate clockwise
                nav_log.info(f"Trying to turn counter-clockwise... ang left: {rel_ang}")
                self.robot.chassis.drive_speed(x=0, z=-10)
            time.sleep(1)
        nav_log.info("Robot should now be facing close to target angle.")

    def given_navigation_loop(self, last_valid_pose, curr_loi, target_rotation):
        """Run navigation loop."""
        path = self.plan_path(last_valid_pose, curr_loi)
        if path is None:  # Due to invalid pose
            return

        while True:
            pose = self.wait_for_valid_pose()
            if pose is None:
                continue

            real_location = RealLocation(x=pose[0], y=pose[1])
            grid_location = self.map.real_to_grid(real_location)

            if self.VISUALIZE_FLAG:
                mapMat = self.map.grid.copy()
                mapMat = viz_pose(mapMat, grid_location, pose[2])
                cv2.waitKey(1)

            if self.map.in_bounds(grid_location) and self.map.passable(grid_location):
                last_valid_pose = pose
            else:
                nav_log.warning(
                    f"Invalid pose received from localization server. Skipping."
                )
                continue

            dist_to_goal = euclidean_distance(last_valid_pose, curr_loi)
            if (
                round(dist_to_goal, 2) <= self.REACHED_THRESHOLD_M
            ):  # Reached checkpoint.
                nav_log.info(
                    f"Reached checkpoint {last_valid_pose[0]:.2f},{last_valid_pose[1]:.2f}"
                )
                path = []  # flush path.
                self.controller.reset()

                # ROTATE ROBOT TO TARGET ORIENTATION.
                rel_ang = get_ang_delta(
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

                ang_diff = ang_to_waypoint(last_valid_pose, curr_wp)

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
                nav_log.info(
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

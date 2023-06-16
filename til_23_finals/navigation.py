"""Navigation code."""

import logging
import time
from typing import List, Optional

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
from .types import Heading
from .utils import (
    ang_to_heading,
    ang_to_waypoint,
    get_ang_delta,
    nearest_cardinal,
    viz_pose,
)

matplotlib.use("TkAgg")

# This can be tuned.
DEFAULT_PID = dict(Kp=(0.5, 0.20), Ki=(0.2, 0.1), Kd=(0.0, 0.0))

main_log = logging.getLogger("Main")
nav_log = logging.getLogger("Nav")
ctrl_log = logging.getLogger("Ctrl")


class Navigator:
    """Navigator class."""

    def __init__(
        self,
        arena_map: SignedDistanceGrid,
        robot,
        loc_service: LocalizationService,
        planner: Planner,
        pose_filter,
        cfg,
    ):
        self.map = arena_map
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
        self.FLIP_X = cfg["FLIP_X"]
        self.FLIP_Y = cfg["FLIP_Y"]
        self.FLIP_Z = cfg["FLIP_Z"]
        self.SWAP_XY = cfg["SWAP_XY"]
        self.BOARDSCALE = cfg["BOARDSCALE"]
        self.EXTRASCALE = cfg["EXTRASCALE"]

    def plan_path(self, start, goal):
        """Plan path."""
        try:
            current_coord = RealLocation(start.x, start.y)
            path = self.planner.plan(current_coord, goal)
            main_log.info("Path planned.")
            return path
        except InvalidStartException as e:
            nav_log.error(f"Invalid start position: {start}", exc_info=e)
            return None
        except NoPathFoundException as e:
            nav_log.error("No path found.", exc_info=e)
            return None

    def get_filtered_pose(self):
        """Get filtered pose."""
        nav_log.critical(
            "`get_filtered_pose` is deprecated! Use `measure_pose` instead."
        )
        pose = self.loc_service.get_pose()
        # no new pose data, continue to next iteration.
        if not pose:
            return None
        return self.pose_filter.update(pose)

    def get_raw_pose(self, correct_heading=True):
        """Get raw pose."""
        pose = self.loc_service.get_pose()
        # NOTE: loc_service returns (None, None).
        if not isinstance(pose, RealPose):
            return None
        z = ang_to_heading(pose.z) if correct_heading else pose.z
        return RealPose(pose.x, pose.y, z)

    def turnRobot(self, target_rotation):
        nav_log.info("Turning robot to face target angle...")
        rel_ang = 180
        while abs(rel_ang) > 20:
            pose = self.get_filtered_pose()
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
            pose = self.get_filtered_pose()
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
                main_log.warning(
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

    def _gim_pose(self, rate_limit, **kwargs) -> List[RealPose]:
        """Read pose while moving gimbal (kwargs passed to `gimbal.move`)."""
        reads = []
        action = self.robot.gimbal.move(**kwargs)
        while not action.is_completed:
            pose = self.get_raw_pose(correct_heading=False)
            if pose is not None:
                reads.append(pose)
            time.sleep(rate_limit)
        action.wait_for_completed()
        return reads

    def set_heading(self, cur: float, tgt: float, spd=30.0):
        """Set the heading of the robot."""
        ang = get_ang_delta(cur, tgt)
        ang = -ang if self.FLIP_Z else ang
        return self.robot.chassis.move(z=ang, z_speed=spd)

    def measure_pose(
        self,
        yaw=20,
        pitch=20,
        yaw_spd=20,
        pitch_spd=20,
        heading_only=False,
        rate_limit=0.25,
        min_reliable=4,
    ):
        """Get accurate measurement of pose.

        Default magnitude of 20 & speed of 20 implies 1s per action.
        With rate limit of 0.25s, implies 3-4 measurements per action.
        For 4 actions, implies 12-16 measurements of location, 6-8 measurements of heading.
        """
        nav_log.debug("Measuring pose...")

        pitches = []
        yaws = []

        # TODO: Would internal compass of `chassis.sub_position` be more accurate
        # for heading/z-angle than 8 samples of localization API?
        self.robot.gimbal.recenter().wait_for_completed()
        pitches += self._gim_pose(rate_limit, pitch=-pitch, pitch_speed=pitch_spd)
        pitches += self._gim_pose(rate_limit, pitch=pitch, pitch_speed=pitch_spd)

        if len(pitches) < min_reliable:
            return None

        if not heading_only:
            self.robot.gimbal.recenter().wait_for_completed()
            yaws += self._gim_pose(rate_limit, yaw=-yaw, yaw_speed=yaw_spd)
            yaws += self._gim_pose(rate_limit, yaw=yaw, yaw_speed=yaw_spd)

        self.robot.gimbal.recenter().wait_for_completed()

        # `np.mean` crashes if list is empty.
        combined = pitches + yaws
        avg_x = np.mean([p.x for p in combined])
        avg_y = np.mean([p.y for p in combined])
        avg_z = np.mean([p.z for p in pitches])  # Yaw bad for z-heading.
        avg_z = ang_to_heading(avg_z)
        real_pose = RealPose(avg_x, avg_y, avg_z)
        nav_log.debug(f"Measured: {real_pose}")
        return real_pose

    def is_pose_valid(self, pose: RealPose):
        """Check whether the pose is in bounds."""
        real_loc = RealLocation(x=pose.x, y=pose.y)
        grid_loc = self.map.real_to_grid(real_loc)
        if not self.map.in_bounds(grid_loc):
            nav_log.warning(f"Pose is out of bounds: {real_loc}, {grid_loc}")
            return False
        if not self.map.passable(grid_loc):
            nav_log.warning(f"Pose is inside wall: {real_loc}, {grid_loc}")
            return False
        return True

    def transform_axes(self, x: float, y: float, heading: Heading):
        """Transform movement values to account for mismatch with map axes and heading."""
        if self.SWAP_XY:
            x, y = y, x
        if self.FLIP_X:
            x, y = -x, y
        if self.FLIP_Y:
            x, y = x, -y
        if heading == Heading.POS_X:
            x, y = x, y
        elif heading == Heading.POS_Y:
            x, y = y, -x
        elif heading == Heading.NEG_X:
            x, y = -x, -y
        elif heading == Heading.NEG_Y:
            x, y = -y, x
        return x, y

    def wait_for_valid_pose(self, ignore_invalid=False):
        """Block until the pose received is valid."""
        while True:
            pose = self.measure_pose()
            if pose is None:
                nav_log.warning("Insufficient poses received from localization server.")
                continue
            if not self.is_pose_valid(pose):
                nav_log.warning("Invalid pose received from localization server.")
                if ignore_invalid:
                    return pose
                continue
            return pose

    def basic_navigation_loop(
        self, tgt_pose: RealPose, ini_pose: Optional[RealPose] = None
    ):
        """Navigate to target location, disregarding target heading.

        Returns whether the measured initial pose is close to the target pose and
        the measured initial pose. If the initial pose is close to the target pose,
        no movement is performed.
        """
        # TODO: Test movement accuracy, no simulator equivalent
        # TODO: Measure length of board, assumed to be 0.5 m now
        ini_pose = self.wait_for_valid_pose() if ini_pose is None else ini_pose
        nav_log.info(f"Start: {ini_pose}")
        nav_log.info(f"Target: {tgt_pose}")

        if (
            (ini_pose.x - tgt_pose.x) ** 2 + (ini_pose.y - tgt_pose.y) ** 2
        ) ** 0.5 < self.REACHED_THRESHOLD_M:
            nav_log.info("Already at target location.")
            return True, ini_pose

        # path = self.plan_path(ini_pose, tgt_pose)
        path: list = []
        # Due to invalid pose?
        if path is None:
            nav_log.warning("Unable to plan path.")
            return False, ini_pose

        skips = 1
        if self.VISUALIZE_FLAG:
            mapMat = self.map.grid.copy()

            cur_grid_loc = self.map.real_to_grid(ini_pose)
            tgt_grid_loc = self.map.real_to_grid(tgt_pose)
            viz_pose(mapMat, cur_grid_loc, ini_pose.z)
            viz_pose(mapMat, tgt_grid_loc, tgt_pose.z)

            for wp in path[::skips]:
                grid_wp = self.map.real_to_grid(wp)
                cv2.circle(mapMat, (grid_wp.x, grid_wp.y), 3, 0, 1)

            cv2.imshow("Map", imutils.resize(mapMat, width=600))
            cv2.waitKey(1)

        # while path:
        #     wp = path[0]
        #     path = path[skips:]
        #
        #     if self.VISUALIZE_FLAG:
        #         grid_wp = self.map.real_to_grid(wp)
        #
        #         cv2.circle(mapMat, (grid_wp.x, grid_wp.y), 5, 0, -1)
        #         cv2.imshow("Map", imutils.resize(mapMat, width=600))
        #         cv2.waitKey(1)
        #
        #     deltaX = self.BOARDSCALE / 1 * (wp.x - ini_pose.x)
        #     deltaY = self.BOARDSCALE / 1 * (wp.y - ini_pose.y)
        #     self.robot.chassis.move(x=deltaY, y=deltaX).wait_for_completed()
        #     # NOTE: This isn't our real pose, so no point drawing.
        #     ini_pose = wp

        xy_spd = 0.7
        z_spd = 30
        align = nearest_cardinal(ini_pose.z)
        delta_x = self.BOARDSCALE / self.EXTRASCALE * (tgt_pose.x - ini_pose.x)
        delta_y = self.BOARDSCALE / self.EXTRASCALE * (tgt_pose.y - ini_pose.y)
        mov_x, mov_y = self.transform_axes(delta_x, delta_y, align)

        # TODO: What if we hit a wall while rotating?
        self.set_heading(ini_pose.z, align, spd=z_spd).wait_for_completed()
        self.robot.chassis.move(x=mov_x, xy_speed=xy_spd).wait_for_completed()
        self.robot.chassis.move(y=mov_y, xy_speed=xy_spd).wait_for_completed()

        nav_log.info(f"Navigation done! (Current pose unknown till next measurement)")
        return False, ini_pose

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
            mapMat = viz_pose(mapMat, grid_location, pose[2])

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

    def heading_test(self):
        """Test if turning to various headings is correct."""
        import random

        cur_pose = self.measure_pose(heading_only=True)
        print(f"INITIAL HEADING: {cur_pose.z}")
        tgt = random.randint(0, 359)
        print(f"TARGET HEADING: {tgt}")
        self.set_heading(cur_pose.z, tgt, spd=30.0).wait_for_completed()
        cur_pose = self.measure_pose(heading_only=True)
        print(f"FINAL HEADING: {cur_pose.z}")

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

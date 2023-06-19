"""Grid based navigation code."""

import logging
import time
from typing import List

import cv2
import imutils
import numpy as np
from tilsdk.localization import RealPose, SignedDistanceGrid
from tilsdk.localization.service import LocalizationService, euclidean_distance

from .navigation import Navigator

# Exceptions for path planning.
from .planner2 import GridPlanner, InvalidStartException, NoPathFoundException
from .types import LocOrPose
from .utils import ang_to_heading, get_ang_delta, nearest_cardinal, viz_pose

nav_log = logging.getLogger("Nav")


class GridNavigator(Navigator):
    """Grid Navigator class."""

    def __init__(
        self,
        arena_map: SignedDistanceGrid,
        robot,
        loc_service: LocalizationService,
        planner: GridPlanner,
        pose_filter,
        cfg,
    ):
        super(GridNavigator, self).__init__(
            arena_map, robot, loc_service, planner, pose_filter, cfg  # type: ignore
        )
        self.SCALE = cfg["BOARDSCALE"] / cfg["EXTRASCALE"]
        self.CARDINAL_MOVE = cfg["CARDINAL_MOVE"]
        self.XY_SPEED = cfg["XY_SPEED"]
        self.XY_MICRO_SPEED = cfg["XY_MICRO_SPEED"]
        self.Z_SPEED = cfg["Z_SPEED"]
        self.POSE_YAW_SPEED = cfg["POSE_YAW_SPEED"]
        self.POSE_PITCH_SPEED = cfg["POSE_PITCH_SPEED"]

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

    def measure_pose(
        self,
        yaw=30,
        pitch=20,
        yaw_spd=None,
        pitch_spd=None,
        heading_only=True,
        rate_limit=0.22,
        min_reliable=4,
    ):
        """Get accurate measurement of pose.

        Default magnitude of 20 & speed of 20 implies 1 second per action.
        Server rate limit is 5 per second, use rate limit of 0.22s to be safe.
        With rate limit of 0.22s, implies 4 measurements per action.
        For 4 actions, implies 16 measurements of location, 8 measurements of heading.
        """
        yaw_spd = self.POSE_YAW_SPEED if yaw_spd is None else yaw_spd
        pitch_spd = self.POSE_PITCH_SPEED if pitch_spd is None else pitch_spd

        nav_log.debug("Measuring pose...")

        pitches = []
        yaws = []

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

    # TODO: Maybe have an extra slow and accurate mode.
    def wait_for_valid_pose(self, ignore_invalid=False, quick=False):
        """Block until the pose received is valid."""
        while True:
            pose = self.measure_pose(heading_only=quick)
            if pose is None:
                nav_log.warning("Insufficient poses received from localization server.")
                continue
            if not self.is_pose_valid(pose):
                nav_log.warning("Invalid pose received from localization server.")
                quick = False  # Try and get more accurate measurement.
                if ignore_invalid:
                    return pose
                continue
            return pose

    def transform_axes(self, x: float, y: float, heading: float):
        """Transform movement values to account for mismatch with map axes and heading."""
        a = np.deg2rad(heading)
        R = np.array(((np.cos(a), np.sin(a)), (-np.sin(a), np.cos(a))))
        x, y = R @ (x, y)
        if self.SWAP_XY:
            x, y = y, x
        if self.FLIP_X:
            x, y = -x, y
        if self.FLIP_Y:
            x, y = x, -y
        return x, y

    def set_heading(self, cur: float, tgt: float, spd: float = 45.0, tries: int = -1):
        """Set the heading of the robot."""
        ang = get_ang_delta(cur, tgt)
        ang = -ang if self.FLIP_Z else ang
        act = self.robot.chassis.move(z=ang, z_speed=spd)
        act.wait_for_completed()
        if not act.has_succeeded and tries > 0:
            cur = self.wait_for_valid_pose(quick=True).z
            return self.set_heading(cur, tgt, spd, tries=tries - 1)
        return act.has_succeeded

    def move_location(
        self,
        cur: LocOrPose,
        tgt: LocOrPose,
        alignment: float,
        spd: float = 0.5,
        tries: int = -1,
    ):
        """Move location taking into account alignment."""
        delta_x = self.SCALE * (tgt.x - cur.x)
        delta_y = self.SCALE * (tgt.y - cur.y)
        mov_x, mov_y = self.transform_axes(delta_x, delta_y, alignment)
        act = self.robot.chassis.move(x=mov_x, y=mov_y, xy_speed=spd)
        act.wait_for_completed()
        if not act.has_succeeded and tries > 0:
            cur = self.wait_for_valid_pose(quick=False)
            return self.move_location(cur, tgt, cur.z, spd, tries=tries - 1)
        return act.has_succeeded

    def navigation_loop(self, tgt_pose, ini_pose=None):
        """Navigate to target location, disregarding target heading.

        Returns whether the measured initial pose is close to the target pose and
        the measured initial pose. If the initial pose is close to the target pose,
        no movement is performed.
        """
        skips = 1
        xy_spd = self.XY_SPEED
        z_spd = self.Z_SPEED
        # NOTE: Turn cardinal_move on if diagonal movement proves too inaccurate!
        cardinal_move = self.CARDINAL_MOVE

        if ini_pose is None:
            ini_pose = self.wait_for_valid_pose(quick=False)
        nav_log.info(f"Start: {ini_pose}")
        nav_log.info(f"Target: {tgt_pose}")

        dist = euclidean_distance(ini_pose, tgt_pose)
        if dist < self.REACHED_THRESHOLD_M:
            nav_log.info("Already at target location.")
            return True, ini_pose
        elif dist < 3 * self.REACHED_THRESHOLD_M:
            nav_log.info("Near target location. Using slow speed.")
            xy_spd = self.XY_MICRO_SPEED

        path = self.plan_path(ini_pose, tgt_pose)
        if path is None:
            nav_log.warning("Unable to plan path.")
            return False, ini_pose

        if self.VISUALIZE_FLAG:
            mapMat = self.map.grid[:, :, None].repeat(3, axis=2)

            # Visualize start and end poses.
            cur_grid_loc = self.map.real_to_grid(ini_pose)
            tgt_grid_loc = self.map.real_to_grid(tgt_pose)
            mapMat = viz_pose(mapMat, cur_grid_loc, ini_pose.z, (0, 255, 0))
            mapMat = viz_pose(mapMat, tgt_grid_loc, tgt_pose.z, (0, 0, 255))

            # Visualize path taken.
            for wp in path[::skips]:
                grid_wp = self.map.real_to_grid(wp)
                cv2.circle(mapMat, (grid_wp.x, grid_wp.y), 4, 0, 2)

            cv2.imshow("Map", imutils.resize(mapMat, width=600))
            cv2.waitKey(1)

        if cardinal_move:
            align = nearest_cardinal(ini_pose.z)
            nav_log.info(f"Nearest cardinal: {align}")
            self.set_heading(ini_pose.z, align, spd=z_spd, tries=3)
        else:
            align = ini_pose.z

        cur_pose = ini_pose
        while len(path) > 0:
            wp = path[0]
            path = path[skips:]

            if self.VISUALIZE_FLAG:
                grid_wp = self.map.real_to_grid(wp)

                # Mark current waypoint.
                cv2.circle(mapMat, (grid_wp.x, grid_wp.y), 3, (255, 0, 0), -1)
                cv2.imshow("Map", imutils.resize(mapMat, width=600))
                cv2.waitKey(1)

            self.move_location(cur_pose, wp, align, spd=xy_spd, tries=3)
            # TODO: Measure pose again every N waypoints to correct for drift.
            # cur_pose = self.wait_for_valid_pose(quick=True)
            cur_pose = RealPose(wp.x, wp.y, align)

        nav_log.info(f"Navigation done! (Current pose unknown till next measurement)")
        return False, ini_pose

import logging
import time

import cv2
import imutils
from planner import (  # Exceptions for path planning.
    InvalidStartException,
    NoPathFoundException,
    Planner,
)

# Import necessary and useful things from til2023 SDK
from tilsdk import *  # import the SDK
from tilsdk.utilities import (  # import optional useful things
    PIDController,
    SimpleMovingAverage,
)

# === Initialize movement controller ===
controller = PIDController(
    Kp=(0.5, 0.20), Ki=(0.2, 0.1), Kd=(0.0, 0.0)
)  # this can be tuned.

# === Initialize pose filter to smooth out noisy pose data ===
pose_filter = SimpleMovingAverage(n=3)  # Smoothens out noisy localization data.

def get_pose(loc_service, pose_filter):
    pose = loc_service.get_pose()

    if not pose:
        # no new pose data, continue to next iteration.
        return None

    return pose_filter.update(pose)

def plan_path(planner, start: list, goal):
    current_coord = RealLocation(x=start[0], y=start[1])
    path = planner.plan(current_coord, goal)
    return path

def ang_difference(ang1, ang2):
    """
    Get angular difference in degrees of two angles in degrees,

    Returns a value in the range [-180, 180].
    """
    ang_diff = -(ang1 - ang2)  # body frame

    # ensure ang_diff is in [-180, 180]
    if ang_diff < -180:
        ang_diff += 360

    if ang_diff > 180:
        ang_diff -= 360
    return ang_diff

def ang_diff_to_wp(pose, curr_wp):
    ang_to_wp = np.degrees(np.arctan2(curr_wp[1] - pose[1], curr_wp[0] - pose[0]))
    ang_diff = ang_difference(ang_to_wp, pose[2])
    return ang_diff

class Navigator():
    def __init__(self, map_, robot, loc_service, planner, pose_filter, cfg):
        self.map: SignedDistanceGrid = map_
        self.loc_service = loc_service
        self.planner = planner
        self.pose_filter = pose_filter

        if cfg["use_real_localization"] == True:
            import robomaster
            from robomaster.robot import Robot  # Use this for real robot.
        else:
            from tilsdk.mock_robomaster.robot import (
                Robot,  # Use this for simulated robot.
            )

        self.robot: Robot = robot
        
        self.VISUALIZE_FLAG = cfg["VISUALIZE_FLAG"]
        self.REACHED_THRESHOLD_M = cfg["REACHED_THRESHOLD_M"]
        self.ANGLE_THRESHOLD_DEG = cfg["ANGLE_THRESHOLD_DEG"]

    def navigation_loop(self, last_valid_pose, curr_loi, target_rotation):
        try:
            path = plan_path(
                self.planner, last_valid_pose, curr_loi
            )  ## Ensure only valid start positions are passed to the planner.

        except InvalidStartException as e:
            logging.getLogger("Navigation").warn(f"{e}")
            # TODO: find and use another valid start point.
            return
        
        logging.getLogger("Main").info("Path planned.")

        while True:
            pose = get_pose(self.loc_service, pose_filter)
            if pose is None:
                continue

            real_location = RealLocation(x=pose[0], y=pose[1])
            grid_location = self.map.real_to_grid(real_location)

            # TODO: Add visualization code here.
            if self.VISUALIZE_FLAG:
                plt.ion()
                plt.scatter(real_location.x, real_location.y)
                plt.draw()
                plt.pause(0.01)

                mapMat = imutils.resize(self.map.grid, width=600)
                cv2.circle(mapMat, (grid_location.x * 2, grid_location.y * 2), 20, 0, -1)
                cv2.imshow("Map", mapMat)
                cv2.waitKey(1)

            if self.map.in_bounds(grid_location) and self.map.passable(grid_location):
                last_valid_pose = pose
            else:
                logging.getLogger("Main").warning(
                    f"Invalid pose received from localization server. Skipping."
                )
                continue

            dist_to_goal = euclidean_distance(last_valid_pose, curr_loi)
            if round(dist_to_goal, 2) <= self.REACHED_THRESHOLD_M:  # Reached checkpoint.
                logging.getLogger("Navigation").info(
                    f"Reached checkpoint {last_valid_pose[0]:.2f},{last_valid_pose[1]:.2f}"
                )
                path = []  # flush path.
                controller.reset()

                # ROTATE ROBOT TO TARGET ORIENTATION.
                rel_ang = ang_difference(
                    last_valid_pose[2], target_rotation
                )  # current heading vs target heading
                logging.getLogger("Navigation").info(
                    "Turning robot to face target angle..."
                )
                while abs(rel_ang) > 20:
                    pose = get_pose(self.loc_service, pose_filter)
                    rel_ang = ang_difference(
                        pose[2], target_rotation
                    )  # current heading vs target heading

                    if rel_ang < -20:
                        # rotate counter-clockwise
                        logging.getLogger("Navigation").info(
                            f"Trying to turn clockwise... ang left: {rel_ang}"
                        )
                        self.robot.chassis.drive_speed(x=0, z=10)
                    elif rel_ang > 20:
                        # rotate clockwise
                        logging.getLogger("Navigation").info(
                            f"Trying to turn counter-clockwise... ang left: {rel_ang}"
                        )
                        self.robot.chassis.drive_speed(x=0, z=-10)
                    time.sleep(1)
                logging.getLogger("Navigation").info(
                    "Robot should now be facing close to target angle."
                )

                curr_wp = None
                prev_loi = curr_loi
                curr_loi = None
                try_start_tasks = True  # try to start ai tasks in the next main iter.
                break

            elif path:
                curr_wp = path[0]  # nearest waypoint is at the start of list.

                logging.getLogger("Navigation").info(f"Num wps left: {len(path)}")

                dist_to_wp = euclidean_distance(real_location, curr_wp)
                if round(dist_to_wp, 2) < self.REACHED_THRESHOLD_M:
                    path = path[1:]  # remove the nearest waypoint.
                    controller.reset()
                    curr_wp = path[0]
                    continue  # start navigating to next waypoint.

                pose_str = f"x:{last_valid_pose[0]:.2f}, y:{last_valid_pose[1]:.2f}, rot:{last_valid_pose[2]:.2f}"
                curr_wp_str = f"{curr_wp[0]:.2f}, {curr_wp[1]:.2f}"
                curr_loi_str = f"{curr_loi[0]:.2f}, {curr_loi[1]:.2f}"
                dist_to_wp = euclidean_distance(real_location, curr_wp)
                logging.getLogger("Navigation").info(
                    f"Goal: {curr_loi_str}, {target_rotation} \t Pose: {pose_str}"
                )
                logging.getLogger("Navigation").info(
                    f"WP: {curr_wp_str} \t dist_to_wp: {dist_to_wp:.2f}\n"
                )

                ang_diff = ang_diff_to_wp(last_valid_pose, curr_wp)

                # Move robot until next waypoint is reached.
                # Determine velocity commands given distance to waypoint and heading to waypoint.
                vel_cmd = controller.update((dist_to_wp, ang_diff))
                vel_cmd[0] *= np.cos(
                    np.radians(ang_diff)
                )  # reduce forward velocity based on angular difference.

                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > self.ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0

                forward_vel, ang_vel = vel_cmd[0], vel_cmd[1]
                logging.getLogger("Control").info(
                    f"input for final forward speed and rotation speed: {forward_vel:.2f}, {ang_vel:.2f}"
                )

                self.robot.chassis.drive_speed(x=forward_vel, z=ang_vel)
                time.sleep(1)
            else:
                logging.getLogger("Navigation").info(
                    "Did not reach checkpoint and no waypoints left."
                )
                raise Exception("Did not reach checkpoint and no waypoints left.")
            
        return curr_wp, prev_loi, curr_loi, try_start_tasks

    def WASD_loop(self, delay=5, forward_vel=0.1, ang_vel=5):
        key = cv2.waitKey(delay)
        if key == ord('w') or key == ord('W'):
            self.robot.chassis.drive_speed(x=forward_vel)
        elif key == ord('a') or key == ord('A'):
            self.robot.chassis.drive_speed(y=-forward_vel)
        elif key == ord('s') or key == ord('S'):
            self.robot.chassis.drive_speed(x=-forward_vel)
        elif key == ord('d') or key == ord('D'):
            self.robot.chassis.drive_speed(y=forward_vel)
        elif key == ord('q') or key == ord('Q'):
            self.robot.chassis.drive_speed(z=-ang_vel)
        elif key == ord('e') or key == ord('E'):
            self.robot.chassis.drive_speed(z=ang_vel)
        else:
            self.robot.chassis.drive_speed()

# if __name__ == "__main__":
    
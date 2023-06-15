"""SAMPLE AUTONOMY CODE FOR TIL2023.

This is not the most optimal solution. However, it is a fully-running codebase
with major integrations done for you already so that you can concentrate more on
improving your algorithms and models and less on integration work. Participants
are free to modify any thing in the "stubs" folder. You do not need to modify the
"src" folder. Have fun!
"""

# Setup some cache directories before everything else.
import os

os.environ["HF_HOME"] = "models/"
os.environ["TORCH_HOME"] = "models/"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import logging
import time
from typing import List

import yaml
from tilsdk import LocalizationService, ReportingService
from tilsdk.localization import GridLocation, RealLocation, RealPose, SignedDistanceGrid
from tilsdk.utilities.filters import SimpleMovingAverage

from .ai import prepare_ai_loop
from .navigation import Navigator
from .planner import Planner

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)

main_log = logging.getLogger("Main")


def main():
    """Run main loop."""
    # === Initialize admin services ===
    # If passthrough sim, use sim's ip address.
    loc_service = LocalizationService(
        host=LOCALIZATION_SERVER_IP, port=LOCALIZATION_SERVER_PORT
    )
    rep_service = ReportingService(host=SCORE_SERVER_IP, port=SCORE_SERVER_PORT)

    # === Initialize robot ===
    robot = Robot()
    robot.initialize(conn_type="ap")
    robot.set_robot_mode(mode="chassis_lead")
    if not IS_SIM:
        robot.gimbal.recenter().wait_for_completed()

    if IS_SIM:

        class Action:
            def __init__(self):
                pass

            def wait_for_completed(self):
                """Block till action is completed."""
                time.sleep(1)

        def move(self, x=0, y=0, z=0, xy_speed=0.5, z_speed=30):
            self.drive_speed(x, y, z)
            return Action()

        bound_method = move.__get__(robot.chassis, robot.chassis.__class__)
        setattr(robot.chassis, "move", bound_method)

    # === Initialize planner ===
    map_: SignedDistanceGrid = loc_service.get_map()
    # Dilate obstacles virtually so that planner avoids bringing robot too close
    # to real obstacles.
    map_ = map_.dilated(ROBOT_RADIUS_M)
    planner = Planner(map_, sdf_weight=0.5)

    # === Loop State/Flags ===
    # Current location of interest.
    curr_loi: RealLocation = None
    # List of way points to get from starting location to location of interest.
    path: List[RealLocation] = []
    # New RealLocation received from Reporting Server.
    new_loi = None
    prev_loi = new_loi
    # A bearing between [-180, 180].
    target_rotation = None
    # Whether to try and start AI tasks.
    try_start_tasks = False
    last_pose = None

    # === Initialize pose filter to smooth out noisy pose data ===
    pose_filter = SimpleMovingAverage(n=3)

    navigator = Navigator(map_, robot, loc_service, planner, pose_filter, cfg)

    # TODO: Run initialization of AI services concurrently.
    ai_loop = prepare_ai_loop(cfg, rep_service, navigator)

    # Tell reporting server we are ready.
    res = rep_service.start_run()
    if res.status == 200:
        initial_target_pose = eval(res.data)
        new_loi = RealLocation(x=initial_target_pose[0], y=initial_target_pose[1])
        target_rotation = initial_target_pose[2]
    else:
        main_log.error("Bad response from challenge server.")
        return

    # NOTE: Use nav.getStartPose() instead.
    # for _ in range(10):
    #     main_log.info(f"Warming up pose filter to reduce initial noise.")
    #     pose = loc_service.get_pose()  # TODO: remove `clues`.
    #     time.sleep(0.25)

    #     pose = pose_filter.update(pose)

    main_log.info(f">>>>> Autobot rolling out! <<<<<")

    # Main loop
    while True:
        # navigator.gimbal_stationary_test(30, 90)
        # pose = navigator.getStartPose()
        # if pose is None:
        #     continue
        # real_location = RealLocation(x=pose[0], y=pose[1])
        # grid_location = map_.real_to_grid(real_location)
        # if map_.in_bounds(grid_location) and map_.passable(grid_location):
        #     last_valid_pose = pose
        # else:
        #     mapGridUpperBounds = GridLocation(map_.width, map_.height)
        #     # mapGridLowerBounds = GridLocation(0, 0)

        #     print(grid_location, (map_.width, map_.height))
        #     print(real_location, map_.grid_to_real(mapGridUpperBounds))
        #     # print(f"{map_.grid_to_real(mapGridLowerBounds)} to {map_.grid_to_real(mapGridUpperBounds)}")
        #     main_log.warning(f"Invalid pose received from localization server.")
        #     continue

        # TEMP
        # try_start_tasks = True
        if try_start_tasks:
            ## Check whether robot's pose is a checkpoint or not.
            info = rep_service.check_pose(last_pose)
            if type(info) == str:
                if info == "End Goal Reached":
                    rep_service.end_run()
                    print("=== YOU REACHED THE END ===")
                    break
                elif info == "Task Checkpoint Reached":
                    pass
                elif info == "Not An Expected Checkpoint":
                    main_log.info(
                        f"Not yet at task checkpoint. status: {res.status}, data: {res.data}, curr pose: {pose}"
                    )
                    # If we reached this execution branch, it means the autonomy code thinks the
                    # robot has reached close enough to the checkpoint, but the Reporting server
                    # is expecting the robot to be even closer to the checkpoint.
                    # TODO:  Robot should try to get closer to the checkpoint.
                    try_start_tasks = False
                    # new_loi = prev_loi  # Try to navigate to prev loi again. May need to get bot to do
                    # a pre-programmed sequence (e.g. shifting around slowly until its
                    # position is close enough to the checkpoint).
                else:
                    raise Exception("Unexpected string value.")
            elif (
                type(info) == RealPose
            ):  # robot reached detour checkpoint and received new coordinates to go to.
                main_log.info(
                    f"Not goal, not task checkpt. Received a new target pose: {info}."
                )

                new_loi = RealLocation(x=info[0], y=info[1])
                target_rotation = info[2]
                main_log.info(f"Setting {new_loi} as new LOI.")
                try_start_tasks = False
            else:
                raise Exception(f"Unexpected return type: {type(info)}.")

            # TEMP
            # try_start_tasks = True
            # AI Loop.
            if try_start_tasks:
                main_log.info("===== Starting AI tasks =====")
                target_pose = ai_loop(robot, last_pose)
                new_loi = RealLocation(x=target_pose[0], y=target_pose[1])
                target_rotation = target_pose[2]
                main_log.info("===== AI tasks complete =====")
                try_start_tasks = False

        # Navigation loop.
        # navigator.given_navigation_loop(last_valid_pose, new_loi, target_rotation)
        at_pos, last_pose = navigator.basic_navigation_loop(None, new_loi, target_rotation)
        # NOTE: last_pose is only the current pose if at_pos is true
        if at_pos:
            delta_z = last_pose.z - target_rotation
            robot.gimbal.move(yaw=delta_z).wait_for_completed()
            try_start_tasks = True

        ##################
        #   Test Cases   #
        ##################

        # Test basic movement (drive_speed) + visualisation
        # navigator.WASD_loop()

        # Gimbal Tests
        ## Test if gimbal responds to command
        # navigator.gimbal_stationary_test()
        ## Test if can command gimbal while moving
        # navigator.gimbal_moving_test()

        # TOF
        # navigator.TOF_test()

        # Test accuracy of DJI Robomaster SDK's move
        # navigator.basic_navigation_test()

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.5)  # set stop for safety
    main_log.info("===== Mission Terminated =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomy implementation")
    parser.add_argument(
        "--config",
        type=str,
        help="path to configuration YAML file.",
        default="config/autonomy_cfg.yml",
    )
    # As a best practice, use a configuration file to configure ur software.

    args = parser.parse_args()

    cfg_path = args.config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

        IS_SIM = not cfg["use_real_localization"]

        if IS_SIM:
            from tilsdk.mock_robomaster.robot import Robot
        else:
            from robomaster.robot import Robot

        VISUALIZE = cfg["VISUALIZE_FLAG"]

        SCORE_SERVER_IP = cfg["SCORE_SERVER_IP"]
        SCORE_SERVER_PORT = cfg["SCORE_SERVER_PORT"]
        LOCALIZATION_SERVER_IP = cfg["LOCALIZATION_SERVER_IP"]
        LOCALIZATION_SERVER_PORT = cfg["LOCALIZATION_SERVER_PORT"]
        ROBOT_IP = cfg["ROBOT_IP"]

        REACHED_THRESHOLD_M = cfg["REACHED_THRESHOLD_M"]
        ANGLE_THRESHOLD_DEG = cfg["ANGLE_THRESHOLD_DEG"]
        ROBOT_RADIUS_M = cfg["ROBOT_RADIUS_M"]

    main()

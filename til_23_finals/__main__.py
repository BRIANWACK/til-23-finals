"""sAmPlE autoNOMY coDE foR TIL2023."""

import os

# Setup some cache directories before everything else.
os.environ["HF_HOME"] = "models/"
os.environ["TORCH_HOME"] = "models/"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import logging
import time

import yaml
from tilsdk import LocalizationService, ReportingService
from tilsdk.localization import RealPose
from tilsdk.utilities.filters import SimpleMovingAverage

from .ai import prepare_ai_loop
from .emulate import bind_robot
from .navigation import Navigator
from .planner import Planner

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)

main_log = logging.getLogger("Main")
ckpt_log = logging.getLogger("Ckpt")


def start_run(rep: ReportingService):
    """Start run and return first target pose."""
    res = rep.start_run()
    if res.status == 200:
        data = eval(res.data)
        assert isinstance(data, RealPose) or isinstance(
            data, tuple
        ), f"Bad response from service: {data}"
        return RealPose(*data)
    else:
        main_log.error(f"Bad response from service.")
        main_log.error(f"Response code: {res.status}")
        raise NotImplementedError


def check_checkpoint(rep: ReportingService, pose: RealPose):
    """Check checkpoint response."""
    ckpt_log.info("===== Check Checkpoint =====")
    data = rep.check_pose(pose)
    if isinstance(data, str):
        if data == "End Goal Reached":
            ckpt_log.info("Endpoint reached!")
            rep.end_run()
            return None
        elif data == "Task Checkpoint Reached":
            ckpt_log.info("Task checkpoint reached!")
            return True
        elif data == "Not An Expected Checkpoint":
            # TODO: Micro-adjustment proccedure?
            ckpt_log.info("Not within checkpoint!")
        elif data == "You Still Have Checkpoints":
            # NOTE: IDK what this means but the reporting service might return this?
            ckpt_log.info("????????")
        else:
            ckpt_log.warning(f"Unexpected response: {data}")
    elif isinstance(data, RealPose) or isinstance(data, tuple):
        ckpt_log.info("Arrived at detour checkpoint!")
        return RealPose(*data)
    else:
        ckpt_log.warning(f"Unexpected response: {data}")
    return False


def main():
    """Run main loop."""
    # ===== Initialize Robot =====
    robot = Robot()
    if IS_SIM:
        bind_robot(robot)

    robot.initialize(conn_type="ap")
    robot.set_robot_mode(mode="free")

    # ===== Initialize API Connections =====
    loc_service = LocalizationService(
        host=LOCALIZATION_SERVER_IP, port=LOCALIZATION_SERVER_PORT
    )
    rep_service = ReportingService(host=SCORE_SERVER_IP, port=SCORE_SERVER_PORT)

    # ===== Initialize Planning & Navigation =====
    arena_map = loc_service.get_map()
    # Dilate obstacles virtually to avoid collision.
    arena_map = arena_map.dilated(ROBOT_RADIUS_M)
    planner = Planner(arena_map, sdf_weight=0.5)
    pose_filter = SimpleMovingAverage(n=3)
    navigator = Navigator(arena_map, robot, loc_service, planner, pose_filter, cfg)
    # TODO: Run initialization of AI services concurrently.
    ai_loop = prepare_ai_loop(cfg, rep_service)

    # === Loop State/Flags ===
    # Whether to try and start AI tasks.
    start_ai = False
    # Whether we should check if at checkpoint.
    check_ckpt = False
    # Current pose (only used after AI task since stationary).
    cur_pose = None
    # Target location & rotation.
    tgt_pose = start_run(rep_service)
    main_log.info(f"Initial target: {tgt_pose}")

    main_log.info(f">>>>> Autobot rolling out! <<<<<")
    while True:
        # If measured pose is close to target pose, no movement is performed and
        # both True and the measured initial pose is returned.
        check_ckpt, last_pose = navigator.basic_navigation_loop(tgt_pose, cur_pose)
        cur_pose = None  # Now unknown.

        if check_ckpt:
            check_ckpt = False
            # TODO: What if we hit wall while rotating?
            navigator.set_heading(last_pose.z, tgt_pose.z).wait_for_completed()
            # TODO: Is this necessary if the robot is accurate? Do we just sleep
            # till localization server catches up? What if the robot is wrong?
            cur_pose = navigator.measure_pose(heading_only=True)
            cur_pose = RealPose(last_pose.x, last_pose.y, cur_pose.z)
            status = check_checkpoint(rep_service, cur_pose)
            if isinstance(status, RealPose):
                tgt_pose = status
                main_log.info(f"New target: {tgt_pose}")
            elif status is None:
                break
            else:
                start_ai = status

        if start_ai:
            start_ai = False
            tgt_pose = ai_loop(robot, last_pose)
            main_log.info(f"New target: {tgt_pose}")

    # ===== Test Cases =====
    if False:
        from .tests import (
            TOF_test,
            WASD_loop,
            basic_navigation_test,
            gimbal_moving_test,
            gimbal_stationary_test,
            heading_test,
        )

        # Test heading.
        heading_test(navigator)
        # Test basic movement (drive_speed) + visualisation
        WASD_loop(navigator)
        # Test if gimbal responds to command
        gimbal_stationary_test(navigator)
        # Test if can command gimbal while moving
        gimbal_moving_test(navigator)
        # TOF
        TOF_test(navigator)
        # Test accuracy of DJI Robomaster SDK's move
        basic_navigation_test(navigator)

    robot.chassis.drive_speed()  # Brake.
    main_log.info("===== Mission Terminated =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomy implementation")
    parser.add_argument(
        "--config",
        type=str,
        help="path to configuration YAML file.",
        default="config/autonomy_cfg.yml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    IS_SIM = not cfg["use_real_localization"]
    if IS_SIM:
        from tilsdk.mock_robomaster.robot import Robot

        # Wait for simulator to start.
        time.sleep(1)
    else:
        from robomaster.robot import Robot

    SCORE_SERVER_IP = cfg["SCORE_SERVER_IP"]
    SCORE_SERVER_PORT = cfg["SCORE_SERVER_PORT"]
    LOCALIZATION_SERVER_IP = cfg["LOCALIZATION_SERVER_IP"]
    LOCALIZATION_SERVER_PORT = cfg["LOCALIZATION_SERVER_PORT"]
    ROBOT_RADIUS_M = cfg["ROBOT_RADIUS_M"]

    main()

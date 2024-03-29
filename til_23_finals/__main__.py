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
from .navigation2 import GridNavigator
from .planner2 import GridPlanner
from .utils import get_ang_delta

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
            # NOTE: How would this even happen?
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
    robot.set_robot_mode(mode="chassis_lead")

    # ===== Initialize API Connections =====
    loc_service = LocalizationService(
        host=LOCALIZATION_SERVER_IP, port=LOCALIZATION_SERVER_PORT
    )
    rep_service = ReportingService(host=SCORE_SERVER_IP, port=SCORE_SERVER_PORT)

    # ===== Initialize Planning & Navigation =====
    arena_map = loc_service.get_map()
    # Dilate obstacles virtually to avoid collision.
    arena_map = arena_map.dilated(ROBOT_RADIUS_M)
    planner = GridPlanner(arena_map)
    pose_filter = SimpleMovingAverage(n=5)
    navigator = GridNavigator(arena_map, robot, loc_service, planner, pose_filter, cfg)
    # TODO: Run initialization of AI services concurrently.
    ai_loop = prepare_ai_loop(cfg, rep_service)

    # ===== Test Cases =====
    if False:
        from .tests import advanced_navigation_test, heading_test

        # NOTE: Use these to tune speed and tile size.
        while True:
            heading_test(navigator, spd=60.0)
            advanced_navigation_test(navigator, spd=1.0)

    # === Loop State/Flags ===
    # Whether to try and start AI tasks.
    start_ai = False
    # Whether we should check if at checkpoint.
    check_ckpt = False
    # Used for calibration.
    cal_pose_tgt = None
    # NOTE: The run only starts when the chassis moves; Hence, we can gimbal & measure here.
    # Current pose.
    cur_pose = navigator.wait_for_valid_pose(quick=False)
    # Target location & rotation.
    tgt_pose = start_run(rep_service)
    main_log.info(f"Initial target: {tgt_pose}")

    main_log.info(f">>>>> Autobot rolling out! <<<<<")
    while True:
        # If measured pose is close to target pose, no movement is performed and
        # both True and the measured initial pose is returned. If current pose is
        # not None, then the navigation loop will skip the initial measurement.
        check_ckpt, last_pose = navigator.navigation_loop(tgt_pose, cur_pose)

        if cal_pose_tgt is not None:
            # Calibrate using "previous previous" position and target.
            cal_pose, cal_tgt = cal_pose_tgt
            navigator.calibrate_scale(
                cal_pose, cal_tgt, last_pose, CALIBRATE_SCALE, CALIBRATE_SCALE_AVG
            )

        cal_pose_tgt = last_pose, tgt_pose
        cur_pose = None  # Now unknown.

        if check_ckpt:
            check_ckpt = False
            # Reuse last pose when resuming from detour.
            cur_pose = last_pose
            # NOTE: Scoring server doesn't care about heading.
            status = check_checkpoint(rep_service, cur_pose)
            if isinstance(status, RealPose):
                tgt_pose = status
                main_log.info(f"New target: {tgt_pose}")
            elif status is None:
                break  # Endpoint.
            else:
                start_ai = status

        if start_ai:
            start_ai = False
            cur_z, tgt_z = last_pose.z, tgt_pose.z
            for _ in range(2):
                navigator.set_heading(cur_z, tgt_z, Z_SPD, tries=3)
                cur_pose = navigator.wait_for_valid_pose(quick=True)
                # cur_pose = RealPose(last_pose.x, last_pose.y, cur_pose.z)
                cur_z = cur_pose.z
                if abs(get_ang_delta(cur_z, tgt_z)) < ANGLE_THRESHOLD:
                    break
            tgt_pose = ai_loop(robot, cur_pose)
            main_log.info(f"New target: {tgt_pose}")

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
    ANGLE_THRESHOLD = cfg["ANGLE_THRESHOLD_DEG"]
    Z_SPD = cfg["AI_Z_SPEED"]
    CALIBRATE_SCALE = cfg["CALIBRATESCALE"]
    CALIBRATE_SCALE_AVG = cfg["CALIBRATESCALEAVG"]

    main()

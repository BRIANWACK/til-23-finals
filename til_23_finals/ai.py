"""Handle AI phase of robot."""

import logging
from pathlib import Path
from time import sleep

import cv2
from tilsdk.localization import RealLocation
from tilsdk.mock_robomaster.robot import Robot
from tilsdk.reporting.service import ReportingService

from til_23_finals.navigation import Navigator
from til_23_finals.services.abstract import (
    AbstractDigitDetectionService,
    AbstractSpeakerIDService,
)
from til_23_finals.utils import enable_camera, load_audio_from_dir, viz_reid

reid_log = logging.getLogger("ReID")
sid_log = logging.getLogger("SpeakID")
main_log = logging.getLogger("Main")


def identify_speakers(service: AbstractSpeakerIDService, audio_dir: str):
    """Identify speakers from audio files in audio_dir."""
    audio_dict = load_audio_from_dir(audio_dir)

    speakerid_result = {}
    with service:
        for fname, v in audio_dict.items():
            audio_waveform, rate = v
            speakerid_result[fname] = service.identify_speaker(audio_waveform, rate)

    for fname, speaker_id in speakerid_result.items():
        sid_log.info(f"{fname} speaker is {speaker_id}.")
    return speakerid_result


def detect_digits(service: AbstractDigitDetectionService, audio_dir: str):
    """Detect digits from audio files in audio_dir."""
    digits_result = {}
    audio_dict = load_audio_from_dir(audio_dir)
    sorted_fnames = sorted(audio_dict.keys())

    with service:
        for fname in sorted_fnames:
            audio_waveform, rate = audio_dict[fname]
            digits = service.transcribe_audio_to_digits(audio_waveform)
            digits_result[fname] = (
                digits[0] if digits else None
            )  # as a heuristic, take the first digit detected.
    return digits_result


def prepare_ai_loop(cfg, rep: ReportingService, nav: Navigator):
    """Return function to run AI phase of main loop."""
    NLP_MODEL_DIR = cfg["NLP_MODEL_DIR"]
    CV_MODEL_DIR = cfg["CV_MODEL_DIR"]
    REID_MODEL_DIR = cfg["REID_MODEL_DIR"]
    SPEAKER_ID_MODEL_DIR = cfg["SPEAKER_ID_MODEL_DIR"]

    PHOTO_DIR = Path(cfg["PHOTO_DIR"])
    ZIP_SAVE_DIR = Path(cfg["ZIP_SAVE_DIR"])
    MY_TEAM_NAME = cfg["MY_TEAM_NAME"]

    VISUALIZE = cfg["VISUALIZE_FLAG"]
    IS_SIM = cfg["use_real_localization"]

    if IS_SIM:
        from til_23_finals.services.digit import WhisperDigitDetectionService
        from til_23_finals.services.reid import BasicObjectReIDService
        from til_23_finals.services.speaker import NeMoSpeakerIDService

        REID_SERVICE: type = BasicObjectReIDService
        SPEAKER_SERVICE: type = NeMoSpeakerIDService
        DIGIT_SERVICE: type = WhisperDigitDetectionService

    else:
        from til_23_finals.services.mock import (
            MockDigitDetectionService,
            MockObjectReIDService,
            MockSpeakerIDService,
        )

        REID_SERVICE = MockObjectReIDService
        SPEAKER_SERVICE = MockSpeakerIDService
        DIGIT_SERVICE = MockDigitDetectionService

    main_log.info("===== Loading AI services =====")
    main_log.warning("This will take a while unless we implement concurrent loading!")
    reid_service = REID_SERVICE(CV_MODEL_DIR, REID_MODEL_DIR)
    speaker_service = SPEAKER_SERVICE(SPEAKER_ID_MODEL_DIR)
    digit_service = DIGIT_SERVICE(NLP_MODEL_DIR)

    with reid_service:
        # TODO: Zoom onto each target to scan.
        sus_embed, hostage_embed = reid_service.embed_images(
            [cv2.imread(cfg["SUSPECT_IMG"]), cv2.imread(cfg["HOSTAGE_IMG"])]
        )

    def loop(robot: Robot):
        """Run AI phase of main loop."""
        main_log.info("===== Starting AI tasks =====")

        robot.chassis.drive_speed()
        # TODO: Test if necessary.
        # sleep(1)
        # TODO: Will get_filtered_pose in the future use the calibration routine?
        pose = nav.get_filtered_pose()

        main_log.info("===== Object ReID =====")

        with enable_camera(robot, PHOTO_DIR) as take_photo:
            img = take_photo()

        with reid_service:
            # TODO: Use bboxes to adjust camera.
            # TODO: Use multiple `scene_img` for multiple crops & embeds. Embeds can then
            # be averaged for robustness.
            # TODO: Temporal image denoise & upscale.
            bboxes = reid_service.targets_from_image(img)

        dets, lbl, _ = reid_service.identity_target(bboxes, sus_embed, hostage_embed)
        viz = viz_reid(img, dets)
        save_path = rep.report_situation(viz, pose, lbl.value, ZIP_SAVE_DIR)

        if VISUALIZE:
            cv2.imshow("Object View", viz)
            cv2.waitKey(1)

        main_log.info(f"Saved next task files: {save_path}")
        main_log.info("===== Speaker ID =====")

        speaker_results = identify_speakers(speaker_service, save_path)
        speakerid_submission = None
        for audio_fname, speakerid in speaker_results.items():
            if audio_fname.endswith(".wav"):
                curr_audio_fname = audio_fname[:-4]
                team_and_member = speakerid.split("_")
                team = team_and_member[0]
                if not team.lower() == MY_TEAM_NAME.lower():  # find opponent's clip.
                    speakerid_submission = curr_audio_fname + "_" + speakerid
                    break

        if speakerid_submission is not None:
            main_log.info(f"Submitting {speakerid_submission} to report_audio API.")
            save_path = rep.report_audio(pose, speakerid_submission, ZIP_SAVE_DIR)
        else:
            raise Exception("no valid speakerid was found.")

        main_log.info(f"Saved next task files: {save_path}")
        main_log.info("===== Digit Detection =====")

        digits_result = detect_digits(digit_service, save_path)
        password = tuple([val for _, val in digits_result.items()])

        # submit answer to scoring server and get scoring server's response.
        main_log.info(f"Submitting password {password} to report_digit API.")
        target_pose = rep.report_digit(pose, password)
        main_log.info(f"new target pose received from server: {target_pose}")

        new_loi = RealLocation(x=target_pose[0], y=target_pose[1])
        target_rotation = target_pose[2]

        main_log.info(
            "===== Ending AI tasks. Continuing Navigation to new target pose ======"
        )
        return new_loi, target_rotation

    return loop

"""Handle AI phase of robot."""

import logging
from pathlib import Path
from time import sleep

import cv2
import imutils
import numpy as np
from tilsdk.localization import RealLocation
from tilsdk.mock_robomaster.robot import Robot
from tilsdk.reporting.service import ReportingService

from til_23_finals.navigation import Navigator
from til_23_finals.services.abstract import (
    AbstractDigitDetectionService,
    AbstractObjectReIDService,
    AbstractSpeakerIDService,
)
from til_23_finals.utils import enable_camera, load_audio_from_dir

reid_log = logging.getLogger("ReID")
sid_log = logging.getLogger("SpeakID")
main_log = logging.getLogger("Main")


def reid(
    service: AbstractObjectReIDService,
    hostage_embed: np.ndarray,
    suspect_embed: np.ndarray,
    img: np.ndarray,
):
    """Use reid service to identify whether hostage or suspect is in the image."""
    with service:
        sus_bbox = service.targets_from_image(img, suspect_embed)
        hostage_bbox = service.targets_from_image(img, hostage_embed)
    reid_log.info(f"Suspect: {sus_bbox}, Hostage: {hostage_bbox}")
    return sus_bbox, hostage_bbox


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

        # TODO: Return all bboxes for camera adjustment.
        # TODO: Return confidence & similarity scores since only one of suspect or hostage can be present.
        sus_bbox, hostage_bbox = reid(reid_service, hostage_embed, sus_embed, img)

        if VISUALIZE:
            img_mat = imutils.resize(img, width=1280)
            sus_col = (255, 0, 0)
            hostage_col = (0, 255, 0)
            non_col = (0, 0, 255)

        if sus_bbox:
            save_path = rep.report_situation(img, pose, "suspect", ZIP_SAVE_DIR)
            if VISUALIZE:
                cv2.rectangle(img_mat, sus_bbox, sus_col, 5)
        if hostage_bbox:
            save_path = rep.report_situation(img, pose, "hostage", ZIP_SAVE_DIR)
            if VISUALIZE:
                cv2.rectangle(img_mat, hostage_bbox, hostage_col, 5)
        if not sus_bbox and not hostage_bbox:
            save_path = rep.report_situation(img, pose, "none", ZIP_SAVE_DIR)

        if VISUALIZE:
            cv2.imshow("Object View", img_mat)
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

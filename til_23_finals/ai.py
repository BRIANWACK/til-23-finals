"""Handle AI phase of robot."""

import logging
from pathlib import Path
from time import sleep

import cv2
from tilsdk.mock_robomaster.robot import Robot
from tilsdk.reporting.service import ReportingService

from til_23_finals.navigation import Navigator
from til_23_finals.services.abstract import AbstractSpeakerIDService
from til_23_finals.utils import enable_camera, load_audio_from_dir, viz_reid

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


def prepare_ai_loop(cfg, rep: ReportingService, nav: Navigator):
    """Return function to run AI phase of main loop."""
    NLP_MODEL_DIR = cfg["NLP_MODEL_DIR"]
    CV_MODEL_DIR = cfg["CV_MODEL_DIR"]
    REID_MODEL_DIR = cfg["REID_MODEL_DIR"]
    SPEAKER_ID_MODEL_DIR = cfg["SPEAKER_ID_MODEL_DIR"]
    DENOISE_MODEL_DIR = cfg["DENOISE_MODEL_DIR"]

    PHOTO_DIR = Path(cfg["PHOTO_DIR"])
    ZIP_SAVE_DIR = Path(cfg["ZIP_SAVE_DIR"])
    MY_TEAM_NAME = cfg["MY_TEAM_NAME"]
    OPPONENT_TEAM_NAME = cfg["OPPONENT_TEAM_NAME"]

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
    speaker_service = SPEAKER_SERVICE(SPEAKER_ID_MODEL_DIR, DENOISE_MODEL_DIR)
    digit_service = DIGIT_SERVICE(NLP_MODEL_DIR, DENOISE_MODEL_DIR)

    with reid_service:
        # TODO: Zoom onto each target to scan.
        sus_embed, hostage_embed = reid_service.embed_images(
            [cv2.imread(cfg["SUSPECT_IMG"]), cv2.imread(cfg["HOSTAGE_IMG"])]
        )

    def loop(robot: Robot):
        """Run AI phase of main loop."""
        robot.chassis.drive_speed()
        # TODO: Test if necessary.
        # sleep(1)
        # NOTE: This pose only used by judges to verify robot is near checkpoint.
        # As such, it doesn't have to be correct/constantly measured.
        # TODO: Get last pose from navigation instead to avoid unnecessary relocalization
        # routine. Unless, we can run said routine concurrently.
        pose = nav.get_filtered_pose()

        main_log.info("===== Object ReID =====")

        with enable_camera(robot, PHOTO_DIR) as take_photo:
            img = take_photo()

        with reid_service:
            # TODO: Use bboxes to adjust camera.
            # TODO: Use multiple `scene_img` for multiple crops & embeds. Embeds can then
            # be averaged for robustness.
            # TODO: Temporal image denoise & upscale (can only find 1 library for this and its unusable).
            bboxes = reid_service.targets_from_image(img)

        dets, lbl, _ = reid_service.identity_target(bboxes, sus_embed, hostage_embed)
        viz = viz_reid(img, dets)
        save_path = rep.report_situation(viz, pose, lbl.value, ZIP_SAVE_DIR)

        if VISUALIZE:
            cv2.imshow("Object View", viz)
            cv2.waitKey(1)

        main_log.info(f"Saved next task files: {save_path}")
        main_log.info("===== Speaker ID =====")

        # TODO: Filter out audio samples for speaker identity depending on opponent team.
        speaker_results = identify_speakers(speaker_service, save_path)
        submission_id = None
        for audio_name, speaker_id in speaker_results.items():
            # NOTE: Should be "{team_name}_{member}_{split}".
            team, member = speaker_id.split("_")[:2]
            if team.lower() != MY_TEAM_NAME.lower():  # Find opponent's clip.
                submission_id = f"{audio_name}_{team}_{member}"
                break

        if submission_id is None:
            main_log.error("Could not find opponent's clip!")
            submission_id = "unknown"

        main_log.info(f'Submitting "{submission_id}" to report_audio API.')
        save_path = rep.report_audio(pose, submission_id, ZIP_SAVE_DIR)

        main_log.info(f"Saved next task files: {save_path}")
        main_log.info("===== Digit Detection =====")

        password = []
        with digit_service:
            digit_audio = load_audio_from_dir(save_path)
            # Number of files won't exceed 9, so no need to worry about number sorting.
            for name in sorted(digit_audio.keys()):
                wav, sr = digit_audio[name]
                # Digits already sorted by confidence by service.
                digits = digit_service.transcribe_audio_to_digits(wav, sr)
                password.append(digits[0] if len(digits) > 0 else 8)  # Lucky guess.

        # submit answer to scoring server and get scoring server's response.
        main_log.info(f"Submitting password {password} to report_digit API.")
        target_pose = rep.report_digit(pose, tuple(password))
        main_log.info(f"Received next target pose: {target_pose}")

        return target_pose

    return loop

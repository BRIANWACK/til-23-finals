"""Handle AI phase of robot."""

import logging
import time
from pathlib import Path

import cv2
from tilsdk.localization.types import RealPose
from tilsdk.mock_robomaster.robot import Robot
from tilsdk.reporting.service import ReportingService

from til_23_finals.utils import enable_camera, load_audio_from_dir, save_image, viz_reid

main_log = logging.getLogger("AI")


def prepare_ai_loop(cfg, rep: ReportingService):
    """Return function to run AI phase of main loop."""
    NLP_MODEL_DIR = cfg["NLP_MODEL_DIR"]
    CV_MODEL_DIR = cfg["CV_MODEL_DIR"]
    REID_MODEL_DIR = cfg["REID_MODEL_DIR"]
    SPEAKER_ID_MODEL_DIR = cfg["SPEAKER_ID_MODEL_DIR"]
    DENOISE_MODEL_DIR = cfg["DENOISE_MODEL_DIR"]

    PHOTO_DIR = Path(cfg["PHOTO_DIR"])
    ZIP_SAVE_DIR = Path(cfg["ZIP_SAVE_DIR"])
    SPEAKER_DIR = cfg["SPEAKER_DIR"]
    MY_TEAM_NAME = cfg["MY_TEAM_NAME"]
    OPPONENT_TEAM_NAME = cfg["OPPONENT_TEAM_NAME"]

    REID_THRES = cfg["REID_THRESHOLD"]

    VISUALIZE = cfg["VISUALIZE_FLAG"]

    if cfg["use_real_models"]:
        from til_23_finals.services.digit import FasterWhisperDigitDetectionService
        from til_23_finals.services.reid import BasicObjectReIDService
        from til_23_finals.services.speaker import NeMoSpeakerIDService

        REID_SERVICE: type = BasicObjectReIDService
        SPEAKER_SERVICE: type = NeMoSpeakerIDService
        DIGIT_SERVICE: type = FasterWhisperDigitDetectionService

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
    reid_service = REID_SERVICE(CV_MODEL_DIR, REID_MODEL_DIR, reid_thres=REID_THRES)
    speaker_service = SPEAKER_SERVICE(SPEAKER_ID_MODEL_DIR, DENOISE_MODEL_DIR)
    digit_service = DIGIT_SERVICE(NLP_MODEL_DIR, DENOISE_MODEL_DIR)

    with reid_service:
        sus_embed = reid_service.embed_image(cv2.imread(cfg["SUSPECT_IMG"]))
        hostage_embed = reid_service.embed_image(cv2.imread(cfg["HOSTAGE_IMG"]))

    @speaker_service
    def _register_speaker_id():
        # Scope this to allow garbage collection.
        speaker_service.clear_speakers()
        speaker_audio = load_audio_from_dir(SPEAKER_DIR)
        for name, (wav, sr) in speaker_audio.items():
            if all(
                n.upper() not in name.upper()
                for n in [MY_TEAM_NAME, OPPONENT_TEAM_NAME]
            ):
                continue

            team, member = name.split("_")[:2]
            speaker_service.enroll_speaker(wav, sr, team_id=team, member_id=member)

    _register_speaker_id()

    @reid_service
    def _reid(robot: Robot, pose, save_path):
        with enable_camera(robot, PHOTO_DIR) as take_photo:
            time.sleep(1)
            img = take_photo()

        # TODO: Robust camera logic:
        # - Use bboxes to adjust camera.
        # - Zoom onto each target to scan.
        # - Temporal image denoise & upscale (can only find 1 library for this and its unusable).
        # - Use multiple `scene_img` for multiple crops & embeds. Embeds can then
        #   be averaged for robustness.
        # - Use gimbal to move to prevent invalidating stationary position assumption.
        bboxes = reid_service.targets_from_image(img)

        dets, lbl, _ = reid_service.identity_target(bboxes, sus_embed, hostage_embed)
        viz = viz_reid(img, dets)

        if VISUALIZE:
            save_image(viz, "reid")
            # cv2.imshow("Object View", viz)
            # cv2.waitKey(1)

        return rep.report_situation(viz, pose, lbl.value, ZIP_SAVE_DIR)

    @speaker_service
    def _speaker(robot: Robot, pose, save_path):
        speaker_audio = load_audio_from_dir(save_path)
        us_scores = {}
        them_scores = {}
        for name, (wav, sr) in speaker_audio.items():
            main_log.info(f"Processing: {name}")
            us = speaker_service.identify_speaker(wav, sr, team_id=MY_TEAM_NAME)
            them = speaker_service.identify_speaker(wav, sr, team_id=OPPONENT_TEAM_NAME)
            us_scores[name] = max(us.values())
            them_scores[name] = them

        # Remove our clip.
        them_scores.pop(max(us_scores, key=us_scores.get))  # type: ignore
        name, them = next(iter(them_scores.items()))
        team_id, member_id = max(them, key=them.get)
        submission_id = f"{name}_{team_id}_{member_id}"
        main_log.info(f'Submitting "{submission_id}" to report_audio API.')
        return rep.report_audio(pose, submission_id, ZIP_SAVE_DIR)

    @digit_service
    def _digit(robot: Robot, pose, save_path):
        password = []
        digit_audio = load_audio_from_dir(save_path)
        # Number of files won't exceed 9, so no need to worry about number sorting.
        for name in sorted(digit_audio.keys()):
            main_log.info(f"Processing: {name}")
            wav, sr = digit_audio[name]
            # Digits already sorted by confidence by service.
            digits = digit_service.transcribe_audio_to_digits(wav, sr)
            password.append(digits[0] if len(digits) > 0 else 8)  # Lucky guess.

        # submit answer to scoring server and get scoring server's response.
        main_log.info(f"Submitting password {password} to report_digit API.")
        return rep.report_digit(pose, tuple(password))

    def loop(robot: Robot, pose):
        """Run AI phase of main loop.

        Note, the robot is assumed to be stationary and in the correct pose!
        """
        main_log.info("===== Starting AI Tasks =====")
        main_log.info("===== Object ReID =====")
        save_path = _reid(robot, pose, None)
        main_log.info(f"Saved next task files: {save_path}")
        main_log.info("===== Speaker ID =====")
        save_path = _speaker(robot, pose, save_path)
        main_log.info(f"Saved next task files: {save_path}")
        main_log.info("===== Digit Detection =====")
        target_pose = _digit(robot, pose, save_path)
        main_log.info(f"Received next target: {target_pose}")
        main_log.info("===== AI Tasks Complete =====")
        return RealPose(*target_pose)

    return loop

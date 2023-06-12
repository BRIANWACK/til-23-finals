"""Various implementations for `AbstractObjectReIDService`."""

import locale
import logging
import os

import torch
from til_23_cv import ReIDEncoder, cos_sim, thres_strategy_naive
from tilsdk.cv.types import BoundingBox
from ultralytics import YOLO

from .abstract import AbstractObjectReIDService

__all__ = ["BasicObjectReIDService"]

log = logging.getLogger("ReID")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

BEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Workaround for annoying ultralytics bug.
locale.getpreferredencoding = lambda _: "UTF-8"  # type: ignore


class BasicObjectReIDService(AbstractObjectReIDService):
    """Basic implementation of `AbstractObjectReIDService`."""

    # TODO: Expose below hardcoded configs in `autonomy_cfg.yml` config file.
    det_conf_thres = 0.6
    det_iou_thres = 0.7
    reid_thres = 0.25
    reid_pad = 0.075

    def __init__(self, yolo_model_path, reid_model_path, device=BEST_DEVICE):
        """Initialize BasicObjectReIDService.

        Parameters
        ----------
        yolo_model_path : str
            Path of yolo model file to load.
        reid_model_path : str
            Path of reid model file to load.
        device : str
            Device to run the model on.
        """
        self.yolo_model_path = os.path.abspath(yolo_model_path)
        self.reid_model_path = os.path.abspath(reid_model_path)
        self.device = device

        log.info(
            f"yolo_model_path: {self.yolo_model_path}, reid_model_path: {self.reid_model_path}"
        )

        self.yolo = YOLO(self.yolo_model_path)
        # NOTE: Torchscript model has to be initialized on same device type it was exported from!
        self.reid = ReIDEncoder(self.reid_model_path, device=self.device)
        self.yolo.fuse()

        # Move to CPU to save GPU memory.
        self.yolo.to("cpu")
        self.reid.to("cpu")

    # TODO: Use multiple `scene_img` for multiple crops & embeds. Embeds can then
    # be averaged for robustness.
    # TODO: Temporal image denoise & upscale.
    # TODO: Return all bboxes for use in camera adjustment. (perhaps focus on each?)
    def targets_from_image(self, scene_img, target_img):
        """Process image with re-id pipeline and return the bounding box of the target_img.

        Returns None if the model doesn't believe that the target is within scene.

        Parameters
        ----------
        scene_img : np.ndarray
            Input image representing the scene to search through.
        target_img : np.ndarray
            Target image representing the object to re-identify.

        Returns
        -------
        results : BoundingBox or None
            BoundingBox of target within scene.
            Assume the values are NOT normalized, i.e. the bbox values are based on the raw
            pixel coordinates of the `scene_img`.
        """
        # BGR to RGB
        scene_img = scene_img[:, :, ::-1]
        target_img = target_img[:, :, ::-1]

        h, w = scene_img.shape[:2]

        res = self.yolo.predict(
            scene_img,
            conf=self.det_conf_thres,
            iou=self.det_iou_thres,
            half=False,
            device=self.device,
            imgsz=1280,
            verbose=False,
        )[0]
        self.yolo.to("cpu")

        boxes = res.boxes.xyxy.round().int().tolist()
        crops = []
        for x1, y1, x2, y2 in boxes:
            px = int(self.reid_pad * (x2 - x1))
            py = int(self.reid_pad * (y2 - y1))
            x1 = max(0, x1 - px)
            y1 = max(0, y1 - py)
            x2 = min(w, x2 + px)
            y2 = min(h, y2 + py)
            crops.append(scene_img[y1:y2, x1:x2])

        if len(crops) == 0:
            return None

        self.reid.to(self.device)
        embeds = self.reid([target_img, *crops])
        self.reid.to("cpu")

        box_sims = [cos_sim(embeds[0], e) for e in embeds[1:]]
        idx = thres_strategy_naive(box_sims, self.reid_thres)

        if idx == -1:
            return None

        x1, y1, x2, y2 = boxes[idx]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

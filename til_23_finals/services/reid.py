"""Various implementations for `AbstractObjectReIDService`."""

import locale
import logging
import os

import torch
from til_23_cv import ReIDEncoder
from ultralytics import YOLO

from til_23_finals.types import ReIDClass, ReIDObject
from til_23_finals.utils import cos_sim, thres_strategy_naive

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

        self.deactivate()  # Save VRAM.

    def activate(self):
        """Move models to device."""
        super(BasicObjectReIDService, self).activate()
        self.yolo.to(self.device)
        self.reid.to(self.device)

    def deactivate(self):
        """Move models to CPU."""
        super(BasicObjectReIDService, self).deactivate()
        self.yolo.to("cpu")
        self.reid.to("cpu")

    def targets_from_image(self, scene_img):
        """Process image with re-id pipeline to return a list of `ReIDObject`.

        Each `ReIDObject` contains the absolute xywh coordinates of the bbox, the
        embedding of the object, the similarity score to the class (default 0.0),
        and the class (default `ReIDClass.CIVILIAN`).

        Parameters
        ----------
        scene_img : np.ndarray
            Input image representing the scene to search through.

        Returns
        -------
        results : List[ReIDObject]
            BoundingBox of targets within the scene. The coordinates are absolute.
        """
        assert self.activated

        # BGR to RGB
        scene_img = scene_img[:, :, ::-1]

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

        embeds = self.reid(crops)
        dets = []
        for (x1, y1, x2, y2), embed in zip(boxes, embeds):
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            det = ReIDObject(x1, y1, x2 - x1, y2 - y1, embed, 0.0, ReIDClass.CIVILIAN)
            dets.append(det)

        log.info(f"Found targets: {len(dets)}")
        return dets

    def identity_target(self, targets, suspect_embed, hostage_embed):
        """Identify if the suspect or hostage and present and which one.

        Note, as per the competition rules, it is assumed either the suspect or
        hostage is present in the scene, but not both. The returned list of targets
        will have suspect and hostage set nonetheless for visualization purposes.
        The scores set on the targets is the max of the similarity to the suspect
        or hostage for debugging purposes.

        Parameters
        ----------
        targets : List[ReIDObject]
            List of targets to search through.
        suspect_embed : np.ndarray
            Embedding of the suspect.
        hostage_embed : np.ndarray
            Embedding of the hostage.

        Returns
        -------
        results : Tuple[List[ReIDObject], ReIDClass, int]
            List of targets with suspect and hostage set, the class of the target,
            and the index of the target in the list.
        """
        sus_sims = [cos_sim(suspect_embed, t.emb) for t in targets]
        hos_sims = [cos_sim(hostage_embed, t.emb) for t in targets]
        sus_idx = thres_strategy_naive(sus_sims, self.reid_thres)
        hos_idx = thres_strategy_naive(hos_sims, self.reid_thres)

        if sus_idx == -1 and hos_idx == -1:
            lbl = ReIDClass.CIVILIAN
            idx = -1
        elif sus_idx != -1 and hos_idx == -1:
            lbl = ReIDClass.SUSPECT
            idx = sus_idx
        elif sus_idx == -1 and hos_idx != -1:
            lbl = ReIDClass.HOSTAGE
            idx = hos_idx
        else:  # Both are present.
            log.warning(
                'Both suspect and hostage are present in the scene! Default to "none".'
            )
            lbl = ReIDClass.CIVILIAN
            idx = -1
            # TODO: Should we just assume its a false positive instead and report no target?
            # if sus_sims[sus_idx] > hos_sims[hos_idx]:
            #     lbl = ReIDClass.SUSPECT
            #     idx = sus_idx
            # else:
            #     lbl = ReIDClass.HOSTAGE
            #     idx = hos_idx

        max_sims = [max(s, h) for s, h in zip(sus_sims, hos_sims)]
        results = [
            ReIDObject(t.x, t.y, t.w, t.h, t.emb, s, t.cls)
            for t, s in zip(targets, max_sims)
        ]
        if sus_idx != -1:
            t = results[sus_idx]
            results[sus_idx] = ReIDObject(
                t.x, t.y, t.w, t.h, t.emb, t.sim, ReIDClass.SUSPECT
            )
        if hos_idx != -1:
            t = results[hos_idx]
            results[hos_idx] = ReIDObject(
                t.x, t.y, t.w, t.h, t.emb, t.sim, ReIDClass.HOSTAGE
            )

        log.info(f"Identified: {lbl.value}")
        return results, lbl, idx

    def embed_images(self, ims):
        """Embed images using ReID model."""
        assert self.activated

        # BGR to RGB
        ims = ims[..., ::-1]
        return self.reid(ims)

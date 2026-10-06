# SPDX-License-Identifier: MIT
"""YOLO tooth localization followed by SAM mask prediction."""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO


def normalize_dental_xray(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(gray, (1.0, 99.0))
    if high <= low:
        normalized = np.zeros_like(gray, dtype=np.uint8)
    else:
        normalized = np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(
            np.uint8
        )
    return np.repeat(normalized[..., None], 3, axis=2)


def mask_to_cvat_detector_mask(mask: np.ndarray) -> list[int]:
    """Return cropped raw mask pixels and bounds for CVAT's detector API."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    xtl, ytl = int(xs.min()), int(ys.min())
    xbr, ybr = int(xs.max()), int(ys.max())
    cropped = mask[ytl : ybr + 1, xtl : xbr + 1]
    return [*cropped.astype(np.uint8).ravel().tolist(), xtl, ytl, xbr, ybr]


class ModelHandler:
    def __init__(self) -> None:
        yolo_path = os.environ.get("YOLO_MODEL_PATH", "/opt/nuclio/best.pt")
        sam_path = os.environ.get(
            "SAM_MODEL_PATH", "/opt/nuclio/sam_vit_b_01ec64.pth"
        )
        for path in (yolo_path, sam_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model checkpoint not found: {path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for YOLO + SAM segmentation")

        self.device = os.environ.get("MODEL_DEVICE", "cuda")
        self.imgsz = int(os.environ.get("IMG_SIZE", "960"))
        self.detector = YOLO(yolo_path)
        self.detector.to(self.device)
        sam = sam_model_registry["vit_b"](checkpoint=sam_path)
        sam.to(device=self.device)
        sam.eval()
        self.segmenter = SamPredictor(sam)
        self._lock = threading.Lock()

    def infer(self, image: Image.Image, threshold: float) -> list[dict]:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        rgb = np.asarray(image.convert("RGB"))
        with self._lock, torch.inference_mode():
            result = self.detector.predict(
                rgb,
                imgsz=self.imgsz,
                conf=threshold,
                device=self.device,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            self.segmenter.set_image(normalize_dental_xray(image))
            shapes = []
            for box, score in zip(boxes, scores):
                masks, mask_scores, _ = self.segmenter.predict(
                    box=box.astype(np.float32),
                    multimask_output=True,
                )
                mask = masks[int(np.argmax(mask_scores))]
                detector_mask = mask_to_cvat_detector_mask(mask)
                if not detector_mask:
                    continue
                shapes.append({
                    "confidence": str(float(score)),
                    "label": "tooth",
                    "mask": detector_mask,
                    "type": "mask",
                })
            return shapes

"""Aspect-preserving resize + padding (letterbox) with box forward/inverse mapping.

Boxes are xyxy in pixels. Image is H x W [x C] uint8.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxRecord:
    orig_w: int
    orig_h: int
    scale: float
    new_w: int
    new_h: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    size: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_record(orig_w: int, orig_h: int, size: int = 640) -> LetterboxRecord:
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("image size must be positive")
    scale = min(size / orig_w, size / orig_h)
    new_w = min(size, max(1, round(orig_w * scale)))
    new_h = min(size, max(1, round(orig_h * scale)))
    dw, dh = size - new_w, size - new_h
    left, top = dw // 2, dh // 2
    return LetterboxRecord(
        orig_w, orig_h, scale, new_w, new_h, left, top, dw - left, dh - top, size
    )


def letterbox_image(
    img: np.ndarray,
    size: int = 640,
    pad_value: int = 114,
    interp_down: str = "area",
    interp_up: str = "linear",
) -> tuple[np.ndarray, LetterboxRecord]:
    h, w = img.shape[:2]
    rec = compute_record(w, h, size)
    interp = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}
    flag = interp[interp_down] if rec.scale < 1 else interp[interp_up]
    resized = (
        img
        if (rec.new_w, rec.new_h) == (w, h)
        else cv2.resize(img, (rec.new_w, rec.new_h), interpolation=flag)
    )
    out = cv2.copyMakeBorder(
        resized,
        rec.pad_top,
        rec.pad_bottom,
        rec.pad_left,
        rec.pad_right,
        cv2.BORDER_CONSTANT,
        value=(pad_value,) * (img.shape[2] if img.ndim == 3 else 1),
    )
    return out, rec


def boxes_forward(boxes: np.ndarray, rec: LetterboxRecord) -> np.ndarray:
    """Original-image xyxy -> letterboxed-image xyxy.

    Uses the actual per-axis ratios (new/orig) so a box spanning the full
    image maps exactly onto the resized content.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    sx, sy = rec.new_w / rec.orig_w, rec.new_h / rec.orig_h
    b[:, [0, 2]] = b[:, [0, 2]] * sx + rec.pad_left
    b[:, [1, 3]] = b[:, [1, 3]] * sy + rec.pad_top
    return b


def boxes_inverse(boxes: np.ndarray, rec: LetterboxRecord) -> np.ndarray:
    """Letterboxed-image xyxy -> original-image xyxy (clipped to the image)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    sx, sy = rec.new_w / rec.orig_w, rec.new_h / rec.orig_h
    b[:, [0, 2]] = (b[:, [0, 2]] - rec.pad_left) / sx
    b[:, [1, 3]] = (b[:, [1, 3]] - rec.pad_top) / sy
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, rec.orig_w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, rec.orig_h)
    return b

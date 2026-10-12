"""Letterbox image/mask to a square canvas and map predictions back to the original image."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.letterbox import LetterboxRecord, compute_record, letterbox_image


def letterbox_pair(
    image: np.ndarray,
    mask: np.ndarray | None,
    size: int = 512,
    image_pad: int = 114,
    mask_pad: int = 0,
    interp_down: str = "area",
    interp_up: str = "linear",
) -> tuple[np.ndarray, np.ndarray | None, LetterboxRecord]:
    """Aspect-preserving resize + pad. Mask uses nearest so labels stay in {0,1}."""
    img_out, rec = letterbox_image(image, size, image_pad, interp_down, interp_up)
    if mask is None:
        return img_out, None, rec
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("mask and image must have the same size")
    resized = cv2.resize(mask, (rec.new_w, rec.new_h), interpolation=cv2.INTER_NEAREST)
    m = cv2.copyMakeBorder(
        resized, rec.pad_top, rec.pad_bottom, rec.pad_left, rec.pad_right,
        cv2.BORDER_CONSTANT, value=mask_pad,
    )
    return img_out, m, rec


def restore_to_original(prob: np.ndarray, rec: LetterboxRecord) -> np.ndarray:
    """size x size float map -> orig_h x orig_w float map (padding removed, bilinear)."""
    crop = prob[rec.pad_top : rec.pad_top + rec.new_h, rec.pad_left : rec.pad_left + rec.new_w]
    return cv2.resize(crop.astype(np.float32), (rec.orig_w, rec.orig_h), interpolation=cv2.INTER_LINEAR)


def points_forward(pts: np.ndarray, rec: LetterboxRecord) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2).copy()
    p[:, 0] = p[:, 0] * rec.new_w / rec.orig_w + rec.pad_left
    p[:, 1] = p[:, 1] * rec.new_h / rec.orig_h + rec.pad_top
    return p


def points_inverse(pts: np.ndarray, rec: LetterboxRecord) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2).copy()
    p[:, 0] = (p[:, 0] - rec.pad_left) * rec.orig_w / rec.new_w
    p[:, 1] = (p[:, 1] - rec.pad_top) * rec.orig_h / rec.new_h
    return p


__all__ = ["LetterboxRecord", "compute_record", "letterbox_pair", "restore_to_original",
           "points_forward", "points_inverse"]

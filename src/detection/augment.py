"""Conservative training-only augmentation with geometry-cancel protection.

Order: brightness/contrast -> rotation -> translation (geometry is one affine).
If any transformed box (axis-aligned envelope of its 4 corners) leaves the image,
the WHOLE geometric transform is cancelled; photometric change is kept.
No scaling, flips, mosaic, mixup, cropping or elastic deformation.
Boxes are xyxy pixels on the (already letterboxed) image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PreprocessConfig


@dataclass
class AugResult:
    image: np.ndarray
    boxes: np.ndarray
    photometric_applied: bool
    geometry: str  # "applied" | "cancelled" | "not_selected"
    params: dict
    cancel_reason: str = ""


def transform_boxes(boxes: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine m to xyxy boxes; return envelope boxes."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.zeros_like(b)
    for i, (x1, y1, x2, y2) in enumerate(b):
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]).T
        t = m @ pts
        out[i] = [t[0].min(), t[1].min(), t[0].max(), t[1].max()]
    return out


def boxes_inside(boxes: np.ndarray, w: int, h: int, eps: float = 1e-6) -> bool:
    b = np.asarray(boxes).reshape(-1, 4)
    if len(b) == 0:
        return True
    return bool((b[:, 0] >= -eps).all() and (b[:, 1] >= -eps).all()
                and (b[:, 2] <= w + eps).all() and (b[:, 3] <= h + eps).all())


def augment_once(
    img: np.ndarray,
    boxes: np.ndarray,
    cfg: PreprocessConfig,
    rng: np.random.Generator,
    pad_value: int | None = None,
) -> AugResult:
    a = cfg.augment
    h, w = img.shape[:2]
    pad = cfg.letterbox.pad_value if pad_value is None else pad_value
    out, ob = img, np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    params: dict = {}

    photometric = rng.random() < a.prob
    if photometric:
        alpha = 1 + rng.uniform(-a.contrast, a.contrast)
        beta = rng.uniform(-a.brightness, a.brightness) * 255
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).round().astype(np.uint8)
        params.update(contrast_alpha=float(alpha), brightness_beta=float(beta))

    geometry, reason = "not_selected", ""
    if rng.random() < a.prob:
        ang = rng.uniform(-a.rotate_deg, a.rotate_deg)
        tx = rng.uniform(-a.translate_frac, a.translate_frac) * w
        ty = rng.uniform(-a.translate_frac, a.translate_frac) * h
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(ang), 1.0)
        m[:, 2] += (tx, ty)
        params.update(rotate_deg=float(ang), tx=float(tx), ty=float(ty))
        nb = transform_boxes(ob, m)
        if boxes_inside(nb, w, h):
            out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(pad,) * 3 if out.ndim == 3 else pad)
            ob, geometry = nb, "applied"
        else:
            geometry, reason = "cancelled", "box_out_of_bounds"
    return AugResult(out, ob, photometric, geometry, params, reason)

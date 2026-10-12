"""Training-only conservative augmentation applied jointly to image and mask.

Order: brightness/contrast (image only) -> rotation + translation (one affine, image and
mask together). No scaling, flips, mosaic, mixup, cropping or elastic deformation.

Truncation guard: each tooth instance mask is warped onto a canvas enlarged by
`margin_frac` per side; the fraction of its pixels that land outside the real canvas is
its truncation loss. If any tooth loses more than `tolerance`, the WHOLE geometric
transform is cancelled and the photometric change is kept. Teeth already cut by the
image border are fine as long as the transform does not cut them further. No annotation
is ever removed to accept a transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Augment


@dataclass
class AugResult:
    image: np.ndarray
    mask: np.ndarray
    photometric_applied: bool
    geometry: str  # "applied" | "cancelled" | "not_selected"
    params: dict = field(default_factory=dict)
    cancel_reason: str = ""


def affine_matrix(size: int, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return m


def truncation_losses(
    instances: list[np.ndarray], m: np.ndarray, size: int, margin_frac: float
) -> list[float]:
    """Per-instance fraction of pixels pushed outside the canvas by affine `m`."""
    pad = int(round(size * margin_frac))
    big = size + 2 * pad
    t_in = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float64)
    t_out = np.array([[1, 0, -pad], [0, 1, -pad], [0, 0, 1]], dtype=np.float64)
    m3 = np.vstack([m, [0, 0, 1]])
    me = (t_in @ m3 @ t_out)[:2]
    losses = []
    for inst in instances:
        src = np.zeros((big, big), dtype=np.uint8)
        src[pad : pad + size, pad : pad + size] = inst
        dst = cv2.warpAffine(src, me, (big, big), flags=cv2.INTER_NEAREST, borderValue=0)
        total = int(dst.sum())
        inside = int(dst[pad : pad + size, pad : pad + size].sum())
        losses.append(0.0 if total == 0 else 1.0 - inside / total)
    return losses


def augment_once(
    image: np.ndarray,
    mask: np.ndarray,
    instances: list[np.ndarray],
    cfg: Augment,
    rng: np.random.Generator,
    pad_value: int = 114,
    force: dict | None = None,
) -> AugResult:
    """`force` = {"photometric": (b, c) | None, "geometry": (angle, tx, ty) | None} for tests."""
    size = image.shape[0]
    params: dict = {}
    out_img, out_mask = image, mask

    if force is not None:
        photo = force.get("photometric")
        geo = force.get("geometry")
    else:
        photo = (
            (rng.uniform(-cfg.brightness, cfg.brightness), rng.uniform(-cfg.contrast, cfg.contrast))
            if rng.random() < cfg.prob else None
        )
        geo = None
        if rng.random() < cfg.prob:
            lim = cfg.translate_frac * size
            geo = (rng.uniform(-cfg.rotate_deg, cfg.rotate_deg),
                   rng.uniform(-lim, lim), rng.uniform(-lim, lim))

    if photo is not None:
        b, c = photo
        f = out_img.astype(np.float32) * (1.0 + c) + b * 255.0
        out_img = np.clip(f, 0, 255).round().astype(np.uint8)
        params["brightness"], params["contrast"] = float(b), float(c)

    geometry, reason = "not_selected", ""
    if geo is not None:
        angle, tx, ty = geo
        params.update(angle=float(angle), tx=float(tx), ty=float(ty))
        m = affine_matrix(size, angle, tx, ty)
        losses = truncation_losses(instances, m, size, cfg.truncation.margin_frac)
        worst = max(losses, default=0.0)
        if worst > cfg.truncation.tolerance:
            geometry = "cancelled"
            reason = f"new_truncation: max tooth loss {worst:.4f} > {cfg.truncation.tolerance}"
        else:
            geometry = "applied"
            out_img = cv2.warpAffine(out_img, m, (size, size), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)
            out_mask = cv2.warpAffine(out_mask, m, (size, size), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return AugResult(out_img, out_mask, photo is not None, geometry, params, reason)

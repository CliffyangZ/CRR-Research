"""Single-tooth ROI geometry: expanded crop (out-of-image padded), letterbox, target-only mask, inverse map."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.letterbox import LetterboxRecord

from .transform import letterbox_pair, points_inverse


def expanded_rect(box_xyxy, frac: float) -> list[int]:
    """Box grown by `frac` of its width/height on EACH side; rounded to integer pixels (may leave the image)."""
    x0, y0, x1, y1 = box_xyxy
    w, h = x1 - x0, y1 - y0
    return [int(round(x0 - frac * w)), int(round(y0 - frac * h)), int(round(x1 + frac * w)), int(round(y1 + frac * h))]


def crop_with_pad(img: np.ndarray, rect: list[int], pad_value: int) -> tuple[np.ndarray, dict]:
    """Crop `rect`; parts outside the image are filled with pad_value. -> (crop, outside_image_px per side)."""
    x0, y0, x1, y1 = rect
    h, w = img.shape[:2]
    ch, cw = y1 - y0, x1 - x0
    out = np.full((ch, cw) + img.shape[2:], pad_value, dtype=img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    outside = {"left": max(0, -x0), "top": max(0, -y0), "right": max(0, x1 - w), "bottom": max(0, y1 - h)}
    return out, outside


def polygons_mask_in_crop(anns: list[dict], rect: list[int]) -> np.ndarray:
    """Union mask (uint8 {0,1}) of the annotations' polygons in crop coordinates of `rect`."""
    x0, y0, x1, y1 = rect
    m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for a in anns:
        for poly in a["segmentation"]:
            pts = np.round(np.asarray(poly, dtype=np.float64).reshape(-1, 2) - [x0, y0]).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


def touches_source_border(ann: dict, width: int, height: int, tol: float = 1.0) -> bool:
    for poly in ann["segmentation"]:
        p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if (p[:, 0] <= tol).any() or (p[:, 1] <= tol).any() or (p[:, 0] >= width - tol).any() or (p[:, 1] >= height - tol).any():
            return True
    return False


def _touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def build_roi(gray: np.ndarray, box_xyxy, target: dict, others: list[dict], cfg) -> dict:
    """gray: full-image uint8 (already CLAHE'd). Returns {'quarantine': reason} or the ROI sample.

    cfg: RoiConfig. Target mask = target polygon only; neighbours and Metal are background.
    """
    r = cfg.roi
    h, w = gray.shape
    rect = expanded_rect(box_xyxy, r.expand_frac_per_side)
    crop, outside = crop_with_pad(gray, rect, r.image_pad)
    tmask = polygons_mask_in_crop([target], rect)
    area = tmask.size
    frac = float(tmask.sum()) / area
    if tmask.sum() == 0 or frac < cfg.pairing.min_mask_frac:
        return {"quarantine": "tiny_target", "mask_frac": frac}
    nmask = polygons_mask_in_crop(others, rect) if others else np.zeros_like(tmask)
    img, msk, rec = letterbox_pair(crop, tmask, r.size, r.image_pad, r.mask_pad, r.interp_down, r.interp_up)
    return {
        "image": img, "mask": msk, "letterbox": rec.to_dict(), "expanded_xyxy": rect, "outside_image_px": outside,
        "mask_area_px": int(msk.sum()), "mask_frac_in_expanded": frac,
        "target_truncated_by_roi": _touches_edge(tmask),
        "truncated_at_source": touches_source_border(target, w, h),
        "neighbor_in_roi_frac": float(((nmask > 0) & (tmask == 0)).sum()) / area,
    }


def roi_points_to_original(pts, letterbox: dict, expanded_xyxy: list[int]) -> np.ndarray:
    """ROI-canvas points -> original-image points (inverse of crop + letterbox)."""
    p = points_inverse(np.asarray(pts, dtype=np.float64), LetterboxRecord(**letterbox))
    return p + np.array([expanded_xyxy[0], expanded_xyxy[1]], dtype=np.float64)


def original_points_to_roi(pts, letterbox: dict, expanded_xyxy: list[int]) -> np.ndarray:
    from .transform import points_forward

    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2) - np.array([expanded_xyxy[0], expanded_xyxy[1]], dtype=np.float64)
    return points_forward(p, LetterboxRecord(**letterbox))

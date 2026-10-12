"""Overlays and error-case classification in original-image coordinates."""

from __future__ import annotations

import cv2
import numpy as np


def _components(mask: np.ndarray) -> list[int]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [int(s[cv2.CC_STAT_AREA]) for s in stats[1:n]]


def hd95(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """95th-percentile symmetric boundary distance in pixels; None if either mask is empty."""
    if not pred.any() or not gt.any():
        return None

    def contour(m):
        m = m.astype(np.uint8)
        return m - cv2.erode(m, np.ones((3, 3), np.uint8))

    cp, cg = contour(pred), contour(gt)
    dt_g = cv2.distanceTransform((1 - cg).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((1 - cp).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.concatenate([dt_g[cp > 0], dt_p[cg > 0]])
    return float(np.percentile(d, 95)) if d.size else 0.0


def classify_errors(pred: np.ndarray, gt: np.ndarray, miss_min_frac: float = 0.02,
                    false_min_frac: float = 0.001) -> dict:
    """miss: a FN component >= miss_min_frac of GT foreground; false: a FP component >=
    false_min_frac of image area; boundary: HD95 of the two masks (pixels)."""
    p, g = pred.astype(bool), gt.astype(bool)
    fn_areas = _components(~p & g)
    fp_areas = _components(p & ~g)
    gt_area = int(g.sum())
    miss = max(fn_areas, default=0)
    false = max(fp_areas, default=0)
    return {
        "max_fn_component": miss,
        "max_fp_component": false,
        "miss": bool(gt_area and miss >= miss_min_frac * gt_area),
        "false": bool(false >= false_min_frac * g.size),
        "hd95": hd95(p, g),
        "miss_score": miss / gt_area if gt_area else 0.0,
        "false_score": false / g.size,
    }


def overlay(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """BGR: green = TP, red = FN (missed), blue = FP (false), on the original image."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    p, g = pred.astype(bool), gt.astype(bool)
    layer = base.copy()
    layer[p & g] = (0, 200, 0)
    layer[~p & g] = (0, 0, 230)
    layer[p & ~g] = (230, 80, 0)
    return cv2.addWeighted(layer, 0.45, base, 0.55, 0)

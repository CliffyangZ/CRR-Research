"""Boundary-error metrics in a ROI's original-resolution frame (pixels).

Both masks are restricted to the `valid` region first, so padding never counts. The mask boundary is the set of
foreground pixels with a background 4-neighbour inside the frame (the frame edge itself is not a boundary).
Empty prediction or ground truth gives None plus a reason, never 0.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=1)


def _prep(pred, gt, valid):
    v = np.ones_like(gt, dtype=bool) if valid is None else valid.astype(bool)
    p, g = pred.astype(bool) & v, gt.astype(bool) & v
    if not p.any() or not g.any():
        reason = "pred_empty" if not p.any() and g.any() else "gt_empty" if not g.any() and p.any() else "both_empty"
        return None, None, reason
    return _surface(p), _surface(g), None


def _dists(a_surf: np.ndarray, b_surf: np.ndarray) -> np.ndarray:
    """Distance of every a-surface pixel to the nearest b-surface pixel."""
    return ndimage.distance_transform_edt(~b_surf)[a_surf]


def hd95(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> tuple[float | None, str | None]:
    ps, gs, why = _prep(pred, gt, valid)
    if why:
        return None, why
    return float(max(np.percentile(_dists(ps, gs), 95), np.percentile(_dists(gs, ps), 95))), None


def boundary_f1(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None, tol: float = 2.0):
    ps, gs, why = _prep(pred, gt, valid)
    if why:
        return None, why
    prec = float((_dists(ps, gs) <= tol).mean())
    rec = float((_dists(gs, ps) <= tol).mean())
    return (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)), None

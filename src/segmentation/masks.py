"""Rasterise COCO tooth polygons into a binary union mask. Other categories (Metal) are ignored."""

from __future__ import annotations

import cv2
import numpy as np


def polygon_issue(poly: list[float], width: int, height: int, tol: float = 1.0) -> str | None:
    """Return a reason string if the polygon is unusable, else None."""
    if len(poly) < 6 or len(poly) % 2:
        return "degenerate: fewer than 3 points"
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(pts).all():
        return "degenerate: non-finite"
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area <= 0:
        return "degenerate: zero area"
    if x.min() < -tol or y.min() < -tol or x.max() > width + tol or y.max() > height + tol:
        return "out_of_bounds"
    return None


def rasterize_union(
    annotations: list[dict],
    category_ids: set[int],
    width: int,
    height: int,
) -> tuple[np.ndarray, list[dict]]:
    """Union of polygons whose category_id is in `category_ids` -> uint8 {0,1} mask.

    Returns (mask, issues). Polygons with issues are skipped (out_of_bounds ones within
    tolerance are clipped by the canvas). Annotations of other categories never paint.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    issues: list[dict] = []
    for ann in annotations:
        if ann.get("category_id") not in category_ids:
            continue
        segs = ann.get("segmentation")
        if not isinstance(segs, list) or not segs:
            issues.append({"ann_id": ann.get("id"), "reason": "no_polygon"})
            continue
        for poly in segs:
            reason = polygon_issue(poly, width, height)
            if reason:
                issues.append({"ann_id": ann.get("id"), "reason": reason})
                if reason != "out_of_bounds":
                    continue
            pts = np.round(np.asarray(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask, issues


def instance_masks(
    annotations: list[dict], category_ids: set[int], width: int, height: int
) -> list[np.ndarray]:
    """One {0,1} mask per annotation (used for per-tooth truncation checks)."""
    out = []
    for ann in annotations:
        if ann.get("category_id") not in category_ids:
            continue
        m, _ = rasterize_union([ann], category_ids, width, height)
        if m.any():
            out.append(m)
    return out

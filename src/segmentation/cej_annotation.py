"""CEJ (cemento-enamel junction) point annotation on single-tooth ROIs.

An annotator marks two points per tooth, one on each side of the neck. The two points define the
CEJ line. Points are stored in the ROI canvas and in the source image's pixel coordinates.
Nothing here asserts expert status or a clinical meaning: the record only says who clicked what.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .roi import roi_points_to_original

SCHEMA_VERSION = 1


def order_left_right(points: list[list[float]]) -> list[list[float]]:
    """Left point = smaller x in the image (viewer's left). Ties keep click order."""
    return sorted(([float(x), float(y)] for x, y in points), key=lambda p: p[0])


def cej_line(points: list[list[float]]) -> dict:
    (x0, y0), (x1, y1) = order_left_right(points)
    dx, dy = x1 - x0, y1 - y0
    return {"midpoint": [(x0 + x1) / 2, (y0 + y1) / 2], "length_px": math.hypot(dx, dy),
            "angle_deg_vs_x_axis": math.degrees(math.atan2(dy, dx))}


def build_record(points_roi: list[list[float]], row: dict, annotator: str, size: int = 256) -> dict:
    """Annotated record for one ROI row of roi_index.json. Raises ValueError on invalid clicks."""
    if len(points_roi) != 2:
        raise ValueError("exactly two points are required")
    pts = order_left_right(points_roi)
    if any(not (0 <= x < size and 0 <= y < size) for x, y in pts):
        raise ValueError("points must lie inside the ROI canvas")
    if math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) < 2:
        raise ValueError("the two points coincide")
    orig = roi_points_to_original(pts, row["letterbox"], row["expanded_xyxy"]).tolist()
    return {
        "status": "annotated", "annotator": annotator, "annotated_utc": _now(),
        "points_roi": pts, "points_original": orig,
        "cej_line_roi": cej_line(pts), "cej_line_original": cej_line(orig),
        "image_id": row["image_id"], "subset": row["subset"],
    }


def skip_record(reason: str, annotator: str, row: dict) -> dict:
    return {"status": "skipped", "reason": reason, "annotator": annotator, "annotated_utc": _now(),
            "image_id": row["image_id"], "subset": row["subset"]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CejStore:
    """JSON file with one record per ROI; saved atomically after every change."""

    def __init__(self, path: str | Path, data_version: str, annotator: str, role: str | None = None):
        self.path = Path(path)
        self.meta = {"schema_version": SCHEMA_VERSION, "data_version": data_version,
                     "coordinate_note": "x right, y down, pixel units; points_roi on the 256x256 ROI canvas, points_original on the source image",
                     "point_order": "left point (smaller x) first",
                     "annotator": annotator, "annotator_role": role,
                     "annotation_status": "human-annotated CEJ points; expert confirmation not asserted"}
        self.teeth: dict[str, dict] = {}
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if old.get("data_version") != data_version:
                raise ValueError(f"{self.path} belongs to data_version {old.get('data_version')}, not {data_version}")
            self.teeth = old["teeth"]
            self.meta["created_utc"] = old.get("created_utc")
        self.meta.setdefault("created_utc", _now())
        self.meta["created_utc"] = self.meta["created_utc"] or _now()

    def done(self) -> set[str]:
        return set(self.teeth)

    def set(self, roi_id: str, record: dict) -> None:
        self.teeth[roi_id] = record
        self.save()

    def remove(self, roi_id: str) -> None:
        self.teeth.pop(roi_id, None)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {**self.meta, "updated_utc": _now(),
               "counts": {"annotated": sum(t["status"] == "annotated" for t in self.teeth.values()),
                          "skipped": sum(t["status"] == "skipped" for t in self.teeth.values())},
               "teeth": dict(sorted(self.teeth.items()))}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        os.replace(tmp, self.path)


class CejSession:
    """Click state for the tooth currently shown (display coordinates -> ROI canvas coordinates)."""

    def __init__(self, scale: float, size: int = 256):
        self.scale, self.size = scale, size
        self.points: list[list[float]] = []

    def click(self, x_disp: float, y_disp: float) -> bool:
        if len(self.points) >= 2:
            return False
        x, y = x_disp / self.scale, y_disp / self.scale
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        self.points.append([x, y])
        return True

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def reset(self) -> None:
        self.points.clear()

    def load(self, points: list[list[float]]) -> None:
        self.points = [list(p) for p in points]

    @property
    def ready(self) -> bool:
        return len(self.points) == 2


def render(image: np.ndarray, mask: np.ndarray | None, session: CejSession, header: str, show_mask: bool = True) -> np.ndarray:
    """BGR frame: scaled ROI at the top (clicks map 1:1 to it) and an instruction band below."""
    s = session.scale
    h, w = int(round(image.shape[0] * s)), int(round(image.shape[1] * s))
    base = cv2.cvtColor(cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    if show_mask and mask is not None:
        m = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(base, cnt, -1, (0, 200, 255), 1)
    pts = [(int(round(x * s)), int(round(y * s))) for x, y in order_left_right(session.points)] if session.points else []
    if len(pts) == 2:
        cv2.line(base, pts[0], pts[1], (0, 255, 0), 1, cv2.LINE_AA)
    for k, p in enumerate(pts):
        cv2.drawMarker(base, p, (0, 0, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        cv2.circle(base, p, 3, (0, 255, 0), -1)
    band = np.full((92, w, 3), 30, dtype=np.uint8)
    lines = [header, "click: left & right CEJ | Enter/Space: save | u: undo | r: reset",
             "x: skip (CEJ unclear) | b: back | m: mask on/off | q: save & quit"]
    for i, t in enumerate(lines):
        cv2.putText(band, t, (6, 20 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([base, band])

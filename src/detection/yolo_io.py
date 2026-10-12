"""YOLO txt <-> xyxy pixel conversion and box validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_yolo_labels(path: str | Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_ids[N], xyxy_pixels[N,4]). Missing file -> raises FileNotFoundError
    (an unlabeled image must never be treated as 'no teeth')."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append([float(v) for v in line.split()[:5]])
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return a[:, 0].astype(int), xyxy


def write_yolo_labels(
    path: str | Path, cls: np.ndarray, xyxy: np.ndarray, w: int, h: int
) -> None:
    lines = []
    for c, (x1, y1, x2, y2) in zip(cls, xyxy):
        lines.append(
            f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        )
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def validate_boxes(xyxy: np.ndarray, w: int, h: int, tol: float = 0.05) -> list[str]:
    """Return a list of problems (empty if valid)."""
    problems = []
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(xyxy).reshape(-1, 4)):
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            problems.append(f"box {i}: zero/negative area")
        if x1 < -tol or y1 < -tol or x2 > w + tol or y2 > h + tol:
            problems.append(f"box {i}: out of image bounds")
    return problems


def clip_boxes(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip to the image (used only after validate_boxes accepted sub-pixel rounding overshoot)."""
    b = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b

"""Selective-search region proposals (cached .npy per image) and proposal recall."""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SS_SETTINGS = {"scales": (100, 200, 300, 500), "sigma": 0.8, "min_size": 30, "min_box": 20, "max_boxes": 2000, "ratio_max": 8.0}


def selective_search_boxes(img_rgb: np.ndarray, s: dict = SS_SETTINGS) -> np.ndarray:
    """Union of selective-search boxes over several scales -> unique xyxy float32 [N,4]."""
    import selectivesearch

    h, w = img_rgb.shape[:2]
    out: set[tuple] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sc in s["scales"]:
            _, regions = selectivesearch.selective_search(img_rgb, scale=sc, sigma=s["sigma"], min_size=s["min_size"])
            for r in regions:
                x, y, bw, bh = r["rect"]
                if bw < s["min_box"] or bh < s["min_box"] or max(bw / bh, bh / bw) > s["ratio_max"]:
                    continue
                out.add((x, y, min(x + bw, w), min(y + bh, h)))
    b = np.array(sorted(out), dtype=np.float32).reshape(-1, 4)
    if len(b) > s["max_boxes"]:
        b = b[np.random.default_rng(0).permutation(len(b))[: s["max_boxes"]]]
    return b


def _one(args):
    img_path, out_path = args
    out_path = Path(out_path)
    if out_path.exists():
        return str(out_path), -1
    img = cv2.imread(str(img_path))[..., ::-1].copy()
    b = selective_search_boxes(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, b)
    return str(out_path), len(b)


def proposal_path(data_dir: Path, split: str, name: str) -> Path:
    return Path(data_dir) / "proposals" / split / f"{name}.npy"


def build_proposals(data_dir: str | Path, splits=("train", "val", "test"), workers: int = 4) -> None:
    data_dir = Path(data_dir)
    jobs = [(p, proposal_path(data_dir, s, p.stem)) for s in splits for p in sorted((data_dir / "images" / s).glob("*.png"))]
    with ProcessPoolExecutor(workers) as ex:
        for _ in ex.map(_one, jobs, chunksize=2):
            pass


def proposal_recall(proposals: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    """Fraction of GT boxes covered by some proposal at IoU >= thr."""
    from ..evaluate import iou_matrix

    if len(gt) == 0:
        return float("nan")
    return float((iou_matrix(gt, proposals).max(1) >= thr).mean()) if len(proposals) else 0.0

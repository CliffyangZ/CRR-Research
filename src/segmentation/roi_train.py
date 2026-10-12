"""U-Net training/evaluation on the ROI dataset.

Metrics are computed per ROI in the ROI's own original-resolution frame (the expanded crop of the
source image): letterbox padding is removed, and crop pixels that lie outside the source image
(padded with grey) are excluded. The ground truth is the target polygon rasterised at source
resolution, not the 256x256 label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detection.letterbox import LetterboxRecord

from .dataset import collate
from .metrics import aggregate, evaluate_pair
from .roi import polygons_mask_in_crop
from .roi_dataset import RoiSegDataset
from .transform import restore_to_original


class GroundTruth:
    """Source-resolution target masks for ROI rows (reads the segmentation COCO files lazily)."""

    def __init__(self, seg_root: str | Path):
        self.root = Path(seg_root)
        self._anns: dict[str, dict] = {}

    def _subset(self, subset: str) -> tuple[dict, dict]:
        if subset not in self._anns:
            c = json.loads((self.root / subset / "annotations" / "instances_default.json").read_text())
            self._anns[subset] = ({a["id"]: a for a in c["annotations"]}, {i["id"]: i for i in c["images"]})
        return self._anns[subset]

    def crop(self, row: dict, subset_dir: str) -> tuple[np.ndarray, np.ndarray]:
        """-> (gt {0,1} crop, valid bool crop). `subset_dir` is the segmentation directory name."""
        anns, imgs = self._subset(subset_dir)
        ann = anns[row["polygon_ann_id"]]
        im = imgs[ann["image_id"]]
        x0, y0, x1, y1 = row["expanded_xyxy"]
        gt = polygons_mask_in_crop([ann], row["expanded_xyxy"])
        valid = np.zeros_like(gt, dtype=bool)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, im["width"]), min(y1, im["height"])
        valid[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = True
        return gt, valid


def source_subset(derived: Path, row: dict) -> str:
    """Directory of the segmentation source the ROI's image came from (recorded at build time)."""
    return row.get("source_subset") or row["subset"]


@torch.no_grad()
def predict_rois(model, ds: RoiSegDataset, device, seg_root: str | Path, batch_size: int = 16, amp: bool = True):
    """Yield (row, prob_crop float32, gt_crop uint8, valid bool, image_crop uint8) per ROI."""
    model.eval()
    gt = GroundTruth(seg_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    by_id = {r["roi_id"]: r for r in ds.rows}
    import cv2

    for b in dl:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(b["image"].to(device))
        probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for rid, rec, p, x in zip(b["id"], b["rec"], probs, b["image"]):
            row = by_id[rid]
            prob = restore_to_original(p, rec)
            g, valid = gt.crop(row, source_subset(ds.dir, row))
            img = restore_to_original((x[0].numpy() * 255.0), rec).clip(0, 255).astype(np.uint8)
            yield row, prob, g, valid, img


def evaluate_roi_split(model, ds: RoiSegDataset, device, threshold: float, amp: bool = True) -> dict:
    rows = []
    for row, prob, gt, valid, _ in predict_rois(model, ds, device, ds.cfg.data.seg_root, amp=amp):
        pred = prob >= threshold
        r = evaluate_pair(pred[valid].astype(np.uint8), gt[valid])
        r["image_id"] = row["roi_id"]
        rows.append(r)
    out = aggregate(rows)
    out["per_image"] = rows
    return out


def build_roi_datasets(derived: Path, cfg, seed: int, smoke: bool):
    tr = RoiSegDataset(derived, "train", cfg, seed=seed)
    va = RoiSegDataset(derived, "val", cfg)
    if smoke:
        tr.rows, va.rows = tr.rows[:16], va.rows[:8]
    return tr, va

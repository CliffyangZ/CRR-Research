"""Per-ROI evaluation of one trained model for the architecture comparison (no model construction here)."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from .boundary import boundary_f1, hd95
from .metrics import aggregate, evaluate_pair
from .report import classify_errors
from .roi_dataset import RoiSegDataset
from .roi_train import predict_rois

NEIGHBOR_FRAC_MIN = 0.05        # a ROI "contains a neighbour tooth" when >= 5% of its frame is non-target tooth
BF1_TOL_PX = 2.0
CSV_COLUMNS = ["roi_id", "source_image_id", "subset", "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall",
               "hd95", "hd95_undefined", "boundary_f1", "boundary_f1_undefined", "truncated", "has_neighbor",
               "outside_image", "gt_px", "fn_frac_of_gt", "max_fn_component_frac_of_gt"]


def evaluate_rois(model, ds: RoiSegDataset, device, threshold: float, amp: bool, miss_min_frac: float,
                  false_min_frac: float) -> list[dict]:
    rows = []
    for row, prob, gt, valid, _ in predict_rois(model, ds, device, ds.cfg.data.seg_root, amp=amp):
        pred = (prob >= threshold) & valid
        r = evaluate_pair(pred[valid].astype(np.uint8), gt[valid])
        h, h_why = hd95(pred, gt, valid)
        b, b_why = boundary_f1(pred, gt, valid, BF1_TOL_PX)
        err = classify_errors(pred.astype(np.uint8), gt & valid, miss_min_frac, false_min_frac)
        gt_px = int((gt.astype(bool) & valid).sum())
        r.update(
            roi_id=row["roi_id"], source_image_id=str(row["image_id"]), subset=row["subset"],
            hd95=h, hd95_undefined=h_why, boundary_f1=b, boundary_f1_undefined=b_why,
            truncated=bool(row["target_truncated_by_roi"] or row["truncated_at_source"]),
            has_neighbor=bool(row["neighbor_in_roi_frac"] >= NEIGHBOR_FRAC_MIN),
            outside_image=any(row["outside_image_px"].values()), gt_px=gt_px,
            fn_frac_of_gt=(r["counts"]["fn"] / gt_px) if gt_px else None,
            max_fn_component_frac_of_gt=(err["max_fn_component"] / gt_px) if gt_px else None,
        )
        rows.append(r)
    return rows


def summarise(rows: list[dict]) -> dict:
    agg = aggregate(rows)
    out = {k: agg[k] for k in ("aggregate", "aggregate_undefined", "per_image_mean", "per_image_n_valid", "n_images",
                               "n_valid_pixels", "totals")}
    out["n_source_images"] = len({r["source_image_id"] for r in rows})
    for k in ("hd95", "boundary_f1"):
        xs = [r[k] for r in rows if r[k] is not None]
        out[f"{k}_mean"] = float(np.mean(xs)) if xs else None
        out[f"{k}_n_valid"] = len(xs)
    return out


def write_per_roi_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            c, m = r["counts"], r["metrics"]
            vals = {**r, **c, **{k: m[k] for k in ("dice", "iou", "precision", "recall")}}
            w.writerow(["" if vals.get(k) is None else vals[k] for k in CSV_COLUMNS])


@torch.no_grad()
def time_inference(model, ds: RoiSegDataset, device, amp: bool, warmup: int = 5) -> dict:
    """Model-only forward latency per ROI (batch 1, same autocast as training). Pre/post-processing excluded."""
    model.eval()
    xs = [ds[i]["image"][None] for i in range(len(ds))]
    use_amp = amp and device.type == "cuda"

    def once(x):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            model(x.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1000

    for x in xs[:warmup]:
        once(x)
    ms = [once(x) for x in xs]
    return {"ms_per_roi_mean": float(np.mean(ms)), "ms_per_roi_median": float(np.median(ms)), "n": len(ms),
            "conditions": f"batch 1, amp={use_amp}, device={device}, warm-up {warmup}, model forward only"}

"""Pixel metrics for tooth/background segmentation in original-image coordinates.

Dice=2TP/(2TP+FP+FN), IoU=TP/(TP+FP+FN), precision=TP/(TP+FP), recall=TP/(TP+FN).
Zero denominators yield None plus a reason, never a silent 0 or 1.
"""

from __future__ import annotations

import numpy as np

NAMES = ("dice", "iou", "precision", "recall")


def counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    return {
        "tp": int((p & g).sum()), "fp": int((p & ~g).sum()),
        "fn": int((~p & g).sum()), "tn": int((~p & ~g).sum()),
    }


def from_counts(c: dict) -> tuple[dict, dict]:
    """-> (values, undefined_reasons); undefined metrics are None."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    den = {"dice": 2 * tp + fp + fn, "iou": tp + fp + fn, "precision": tp + fp, "recall": tp + fn}
    num = {"dice": 2 * tp, "iou": tp, "precision": tp, "recall": tp}
    why = {
        "dice": "gt_empty_and_pred_empty", "iou": "gt_empty_and_pred_empty",
        "precision": "pred_empty", "recall": "gt_empty",
    }
    vals, undefined = {}, {}
    for k in NAMES:
        if den[k] == 0:
            vals[k], undefined[k] = None, why[k]
        else:
            vals[k] = num[k] / den[k]
    return vals, undefined


def aggregate(per_image: list[dict]) -> dict:
    """per_image rows hold 'counts'. Returns pixel-accumulated aggregate + per-image mean."""
    tot = {k: sum(r["counts"][k] for r in per_image) for k in ("tp", "fp", "fn", "tn")}
    vals, undefined = from_counts(tot)
    mean, n_valid = {}, {}
    for k in NAMES:
        xs = [r["metrics"][k] for r in per_image if r["metrics"][k] is not None]
        n_valid[k] = len(xs)
        mean[k] = float(np.mean(xs)) if xs else None
    all_bg = [r for r in per_image if r["counts"]["tp"] + r["counts"]["fn"] == 0]
    return {
        "aggregate": vals,
        "aggregate_undefined": undefined,
        "per_image_mean": mean,
        "per_image_n_valid": n_valid,
        "n_images": len(per_image),
        "n_valid_pixels": int(sum(sum(r["counts"].values()) for r in per_image)),
        "n_all_background_gt": len(all_bg),
        "fp_pixels_on_all_background": int(sum(r["counts"]["fp"] for r in all_bg)),
        "totals": tot,
    }


def evaluate_pair(pred: np.ndarray, gt: np.ndarray) -> dict:
    c = counts(pred, gt)
    m, u = from_counts(c)
    return {"counts": c, "metrics": m, "undefined": u}

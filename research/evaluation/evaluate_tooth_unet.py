"""Evaluate a saved ROI U-Net checkpoint on a fixed ROI split (per-ROI original-resolution frame).

The checkpoint was chosen by the run on validation Dice; nothing is selected or tuned here.
Example:
  python -m research.evaluation.evaluate_tooth_unet --run reports/segmentation/<run_id> --weights best
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from research.models.unet import build_unet
from src.detection.provenance import base_manifest, sha256_file, write_manifest
from src.segmentation.metrics import aggregate, evaluate_pair
from src.segmentation.report import classify_errors, overlay
from src.segmentation.roi_config import RoiTrainConfig
from src.segmentation.roi_dataset import RoiSegDataset
from src.segmentation.roi_train import predict_rois


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", choices=["best", "last"], default="best")
    ap.add_argument("--split", choices=["test", "val"], default="test", help="test is the fixed report split; val only for dry runs")
    ap.add_argument("--derived", default="data/derived")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    run = Path(args.run)
    m = json.loads((run / "run_manifest.json").read_text())
    if m["status"] != "succeeded":
        print(f"run status is {m['status']}; refusing to evaluate", file=sys.stderr)
        return 2
    out = Path(args.output) if args.output else run / ("eval" if args.split == "test" else f"eval_{args.split}")
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    cfg = RoiTrainConfig(**m["config"])
    ck = m["checkpoints"][args.weights]
    if sha256_file(ck["path"]) != ck["sha256"]:
        print("checkpoint sha256 differs from the run manifest", file=sys.stderr)
        return 3
    derived = Path(args.derived) / m["data_version"]
    if sha256_file(derived / "splits.json") != m["split_sha256"]:
        print("split file changed since training", file=sys.stderr)
        return 3
    (out / "overlays").mkdir(parents=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_unet(cfg.model).to(dev)
    model.load_state_dict(torch.load(ck["path"], map_location=dev)["model"])
    ds = RoiSegDataset(derived, args.split, cfg)
    thr, ec = cfg.train.threshold, cfg.eval.error_cases

    rows, cases = [], {}
    for row, prob, gt, valid, img in predict_rois(model, ds, dev, cfg.data.seg_root, amp=cfg.train.amp):
        pred = (prob >= thr) & valid
        r = evaluate_pair(pred[valid].astype(np.uint8), gt[valid])
        r["image_id"] = row["roi_id"]
        r["errors"] = classify_errors(pred.astype(np.uint8), gt, ec.miss_min_frac, ec.false_min_frac)
        r["flags"] = {k: row[k] for k in ("target_truncated_by_roi", "truncated_at_source")} | {
            "neighbor_in_roi_frac": row["neighbor_in_roi_frac"], "outside_image": any(row["outside_image_px"].values())}
        rows.append(r)
        cases[row["roi_id"]] = (img, gt, pred.astype(np.uint8))

    agg = aggregate(rows)
    pick = {}
    for kind, key in (("miss", "miss_score"), ("false", "false_score")):
        hits = sorted((r for r in rows if r["errors"][kind]), key=lambda r: -r["errors"][key])
        pick[kind], pick[f"{kind}_count"] = (hits[0]["image_id"] if hits else None), len(hits)
    bd = sorted((r for r in rows if r["errors"]["hd95"] is not None), key=lambda r: -r["errors"]["hd95"])
    pick["boundary"] = bd[0]["image_id"] if bd else None
    for kind in ("miss", "false", "boundary"):
        if pick[kind]:
            gray, gt, pred = cases[pick[kind]]
            ov = overlay(gray, gt, pred)
            s = 512 / max(ov.shape[:2])
            cv2.imwrite(str(out / "overlays" / f"{kind}_{pick[kind]}.png"),
                        cv2.resize(ov, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else ov)

    def sub(flag):
        sel = [r for r in rows if r["flags"][flag]]
        return {"n": len(sel), **{k: v for k, v in aggregate(sel).items() if k in ("aggregate", "per_image_mean")}} if sel else {"n": 0}

    metrics = {
        "run_id": m["run_id"], "weights": args.weights, "weights_sha256": ck["sha256"], "weights_epoch": ck["epoch"],
        "subset": args.split, "unit": "ROI (one detection box paired with one polygon)", "threshold": thr,
        "data_version": m["data_version"], "split_sha256": m["split_sha256"], "annotation_review": m["annotation_review"],
        "n_rois": agg["n_images"], "n_valid_pixels": agg["n_valid_pixels"],
        "frame": "per-ROI source resolution; letterbox padding and out-of-image crop padding excluded",
        "aggregation": "pixel-accumulated over all ROIs (primary); per-ROI mean excludes undefined ROIs",
        "definitions": {"dice": "2TP/(2TP+FP+FN)", "iou": "TP/(TP+FP+FN)", "precision": "TP/(TP+FP)", "recall": "TP/(TP+FN)"},
        **{k: agg[k] for k in ("aggregate", "aggregate_undefined", "per_image_mean", "per_image_n_valid",
                               "n_all_background_gt", "fp_pixels_on_all_background", "totals")},
        "by_flag": {f: sub(f) for f in ("target_truncated_by_roi", "truncated_at_source", "outside_image")},
        "error_cases": {**pick, "definitions": {
            "miss": f"FN connected component >= {ec.miss_min_frac:.1%} of GT tooth area",
            "false": f"FP connected component >= {ec.false_min_frac:.1%} of ROI area",
            "boundary": "95th percentile symmetric boundary distance (pixels); worst ROI shown"}},
        "not_measured": ["seed variance", "robustness", "clinical diagnostic ability", "patient-level leakage",
                         "performance on teeth missing from the segmentation labels (41 quarantined boxes)"],
        "exploratory": True,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(out / "per_roi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall", "max_fn_component",
                    "max_fp_component", "hd95", "miss", "false", "target_truncated_by_roi", "outside_image"])
        for r in rows:
            c, mt, e = r["counts"], r["metrics"], r["errors"]
            w.writerow([r["image_id"], c["tp"], c["fp"], c["fn"], c["tn"],
                        *["" if mt[k] is None else f"{mt[k]:.6f}" for k in ("dice", "iou", "precision", "recall")],
                        e["max_fn_component"], e["max_fp_component"], "" if e["hd95"] is None else f"{e['hd95']:.2f}",
                        int(e["miss"]), int(e["false"]), int(r["flags"]["target_truncated_by_roi"]), int(r["flags"]["outside_image"])])
    write_manifest(out / "eval_manifest.json", base_manifest(m["run_id"] + "-eval", None, status="succeeded",
                                                             evaluated_run=m["run_id"], subset=args.split))
    print(json.dumps({k: metrics[k] for k in ("subset", "n_rois", "aggregate", "per_image_mean", "by_flag", "error_cases")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-time evaluation of the adopted comparison runs on a fixed split (test by default).

Each run's `best.pt` (chosen on validation Dice during training) is evaluated; nothing is tuned here. After all nine
runs are evaluated on test, a lock is written; a second evaluation is refused unless --record-violation is given.
  python -m research.evaluation.evaluate_seg_models --protocol configs/segmentation/compare_protocol_v1.yaml --runs reports/segmentation/<run_id> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from research.models.registry import build_model
from src.detection.provenance import base_manifest, sha256_file, write_manifest
from src.segmentation import run_lock
from src.segmentation.compare_eval import (BF1_TOL_PX, NEIGHBOR_FRAC_MIN, evaluate_rois, summarise, time_inference,
                                           write_per_roi_csv)
from src.segmentation.protocol import ProtocolError, load_protocol
from src.segmentation.roi_config import RoiTrainConfig
from src.segmentation.roi_dataset import RoiSegDataset


def evaluate_run(m: dict, split: str, derived_root: str, device: torch.device, out: Path) -> dict:
    cfg = RoiTrainConfig(**m["config"])
    ck = m["checkpoints"]["best"]
    if sha256_file(ck["path"]) != ck["sha256"]:
        raise ProtocolError(f"{m['run_id']}: checkpoint sha256 differs from the run manifest")
    derived = Path(derived_root) / m["data_version"]
    if sha256_file(derived / "splits.json") != m["split_sha256"]:
        raise ProtocolError(f"{m['run_id']}: split file changed since training")
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(ck["path"], map_location=device)["model"])
    ds = RoiSegDataset(derived, split, cfg)
    ec = cfg.eval.error_cases
    rows = evaluate_rois(model, ds, device, cfg.train.threshold, cfg.train.amp, ec.miss_min_frac, ec.false_min_frac)
    timing = time_inference(model, ds, device, cfg.train.amp)
    out.mkdir(parents=True, exist_ok=False)
    write_per_roi_csv(out / "per_roi.csv", rows)
    metrics = {
        "run_id": m["run_id"], "candidate": m["candidate"], "seed": m["seed"], "subset": split,
        "weights": "best", "weights_sha256": ck["sha256"], "weights_epoch": ck["epoch"],
        "threshold": cfg.train.threshold, "data_version": m["data_version"], "split_sha256": m["split_sha256"],
        "annotation_review": m["annotation_review"], "n_parameters": m["n_parameters"],
        "frame": "per-ROI source resolution; letterbox padding and out-of-image crop padding excluded",
        "aggregation": "pixel-accumulated over all ROIs (primary); per-ROI mean excludes undefined ROIs",
        "boundary": {"hd95": "95th pct of symmetric boundary distance, px", "boundary_f1_tol_px": BF1_TOL_PX},
        "flags": {"has_neighbor": f"neighbor_in_roi_frac >= {NEIGHBOR_FRAC_MIN}"},
        "inference": timing, **summarise(rows),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="the nine adopted run directories")
    ap.add_argument("--split", choices=["test", "val"], default="test", help="val = dry run (no lock, not a result)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--record-violation", default=None, metavar="REASON",
                    help="allow a repeat test evaluation; recorded in the lock and invalidates confirmatory claims")
    a = ap.parse_args(argv)
    try:
        p = load_protocol(a.protocol)
        dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
        try:
            manifests = run_lock.check_ready(p, a.runs)
        except run_lock.TestAlreadyUsed as e:
            if not (a.record_violation and a.split == "test"):
                print(f"REFUSED: {e}. Pass --record-violation REASON to repeat (invalidates confirmatory claims).",
                      file=sys.stderr)
                return 3
            run_lock.record_violation(p, a.record_violation)
            manifests = sorted((json.loads((Path(d) / "run_manifest.json").read_text()) for d in a.runs),
                               key=lambda m: (m["candidate"], m["seed"]))
        sub = "eval" if a.split == "test" else "eval_val_dryrun"
        done = []
        for m in manifests:
            out = Path(m["output_dir"]) / sub
            if a.record_violation and out.exists():
                out = Path(m["output_dir"]) / f"{sub}_repeat_{len(run_lock.load_lock(p)['violations'])}"
            print(f"evaluating {m['run_id']} ({m['candidate']}, seed {m['seed']}) on {a.split}", flush=True)
            res = evaluate_run(m, a.split, p.derived_root, dev, out)
            res["dry_run"] = a.split != "test"
            (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
            write_manifest(out / "eval_manifest.json", base_manifest(
                m["run_id"] + "-eval", sys.argv if argv is None else ["evaluate_seg_models", *argv], status="succeeded",
                evaluated_run=m["run_id"], subset=a.split, protocol_hash=m["protocol_hash"],
                weights_sha256=res["weights_sha256"], dry_run=a.split != "test"))
            done.append({"run_id": m["run_id"], "weights_sha256": res["weights_sha256"]})
            print(f"  dice {res['aggregate']['dice']:.4f}  hd95_mean {res['hd95_mean']}", flush=True)
        if a.split == "test" and not a.record_violation:
            run_lock.acquire(p, done)
            print(f"test lock written: {run_lock.lock_path(p)}")
    except (ProtocolError, FileExistsError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

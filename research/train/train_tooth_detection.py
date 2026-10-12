"""Train one detector (YOLO v8/v11/v26 or Fast R-CNN) on a frozen tooth-detection data version.

  python -m research.train.train_tooth_detection --data-version dv-... --model v8 --device 0 [--smoke]
   [--smoke]

Model selection: YOLO = Ultralytics best.pt (max validation fitness 0.9*AP50-95 + 0.1*AP50);
Fast R-CNN = max validation AP@[.5:.95] (src.detection.evaluate). VALIDATION split only;
the test split is never touched here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.detection.models import candidate
from src.detection.provenance import base_manifest, sha256_file, write_manifest
from src.detection.train_args import assert_no_builtin_aug, build_train_args


def training_run_id(a) -> str:
    """Readable experiment directory with an explicit Taipei timestamp."""
    def component(value):
        return re.sub(r"[^\w.-]+", "-", str(value)).strip(".-") or "experiment"

    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%m%d-%H%M%S")
    parts = [a.model, component(a.data_version)]
    if a.experiment:
        parts.append(component(a.experiment))
    if a.smoke:
        parts.append("smoke")
    parts.append(stamp)
    return "_".join(parts)


def train_fastrcnn(a, data_dir: Path, run_dir: Path, manifest: dict, manifest_path: Path) -> int:
    """Fast R-CNN branch: selective-search proposals (cached) -> train -> best.pt by validation AP."""
    import numpy as np
    import torch

    from src.detection.fast_rcnn.data import ToothSplit
    from src.detection.fast_rcnn.proposals import SS_SETTINGS, build_proposals, proposal_recall
    from src.detection.fast_rcnn.trainer import DEFAULTS, train_fast_rcnn

    dev = f"cuda:{a.device}" if a.device.isdigit() else a.device
    hp = {**DEFAULTS, "epochs": 2 if a.smoke else a.epochs, "patience": a.patience, "imgsz": a.imgsz, "device": dev,
          "seed": a.seed, "selective_search": SS_SETTINGS, "builtin_augmentation": "none (offline conservative copies only; flips disabled)"}
    manifest.update(hyperparams=hp, status="training",
                    selection_rule="max val AP@[.5:.95] (src.detection.evaluate, NMS 0.3); test split unused")
    write_manifest(manifest_path, manifest)
    build_proposals(data_dir, workers=a.workers)
    recall = {}
    for sp in ("train", "val", "test"):
        ds = ToothSplit(data_dir, sp)
        r = [proposal_recall(ds[i][3].numpy(), ds[i][2].numpy()) for i in range(len(ds))]
        recall[sp] = float(np.nanmean(r))
    manifest["proposal_recall_iou50"] = recall  # upper bound on test recall at IoU 0.5 (GT-boxes-as-RoIs excluded)
    write_manifest(manifest_path, manifest)
    res = train_fast_rcnn(data_dir, run_dir / "train", epochs=a.epochs, patience=a.patience, seed=a.seed, device=dev,
                          smoke=a.smoke, log=lambda m: print(m, flush=True))
    best = Path(res["weights"])
    manifest.update(status="trained", selected_weights={"path": str(best), "sha256": sha256_file(best), "epoch": res["best_epoch"]},
                    train_dir=str(run_dir / "train"), val_metrics_last={"best_val_AP": res["best_val_AP"]},
                    effective_args=res["hp"], epochs_run=res["epochs_run"],
                    nondeterminism_note="cudnn.deterministic=True, benchmark=False; RoIPool/atomic-add kernels may still be non-deterministic.")
    write_manifest(manifest_path, manifest)
    print(json.dumps({"run_id": manifest["run_id"], "best": str(best)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-version", required=True)
    ap.add_argument("--data-root", default="data/derived")
    ap.add_argument("--model", required=True, choices=["v8", "v11", "v26", "fastrcnn"])
    ap.add_argument("--weights-dir", default="models/pretrained")
    ap.add_argument("--output", default="reports/detection")
    ap.add_argument("--experiment", help="Optional experiment label included in the run directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on the (small) smoke data version")
    a = ap.parse_args(argv)

    run_id = training_run_id(a)
    run_dir = Path(a.output) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    data_dir = Path(a.data_root) / a.data_version
    cand = candidate(a.model, a.weights_dir)
    manifest = base_manifest(run_id, model=a.model, candidate=cand, data_version=a.data_version, seed=a.seed, smoke=a.smoke,
                             experiment=a.experiment, run_name_timezone="Asia/Taipei")
    print(f"Experiment: {run_id}\nRun directory: {run_dir}", flush=True)
    manifest_path = run_dir / "run_manifest.json"
    try:
        dvm = json.loads((data_dir / "manifests" / "dataset_version.json").read_text())
        if dvm.get("status") != "validated":
            raise RuntimeError(f"data version {a.data_version} is not validated")
        manifest.update(
            annotation_version=dvm["annotation_version"], preprocess_config_hash=dvm["preprocess_config_hash"],
            split_manifest=str(data_dir / "manifests" / "split.json"),
            split_sha256=sha256_file(data_dir / "manifests" / "split.json"),
            dataset_counts=dvm["counts"],
        )
        if a.model == "fastrcnn":
            return train_fastrcnn(a, data_dir, run_dir, manifest, manifest_path)
        cand["weights_sha256"] = sha256_file(cand["weights_path"])
        args = build_train_args(
            str(data_dir.resolve() / "data.yaml"), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience,
            batch=a.batch, seed=a.seed, workers=a.workers, device=int(a.device) if a.device.isdigit() else a.device,
            project=str((run_dir).resolve()), name="train", smoke=a.smoke,
        )
        assert_no_builtin_aug(args)
        manifest.update(hyperparams=args, status="training",
                        selection_rule="Ultralytics best.pt: max val fitness (0.9*AP50-95+0.1*AP50); test split unused")
        write_manifest(manifest_path, manifest)

        from ultralytics import YOLO

        model = YOLO(cand["weights_path"])
        batch = a.batch
        while True:
            try:
                args["batch"] = batch
                res = model.train(**args)
                break
            except RuntimeError as e:  # CUDA OOM -> halve batch, record
                if "out of memory" not in str(e).lower() or batch <= 2:
                    raise
                manifest.setdefault("oom_retries", []).append({"batch": batch, "error": str(e)[:200]})
                batch //= 2
                args["name"] = f"train_b{batch}"
                import torch; torch.cuda.empty_cache()
                model = YOLO(cand["weights_path"])
        save_dir = Path(res.save_dir) if hasattr(res, "save_dir") else Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        manifest.update(
            status="trained", effective_args={k: getattr(model.trainer.args, k, None) for k in args},
            selected_weights={"path": str(best), "sha256": sha256_file(best)}, train_dir=str(save_dir),
            val_metrics_last={k: float(v) for k, v in (getattr(model.trainer, "metrics", {}) or {}).items()},
            nondeterminism_note="deterministic=True requested; GPU kernels may still differ across runs/devices.",
        )
        write_manifest(manifest_path, manifest)
        print(json.dumps({"run_id": run_id, "best": str(best)}))
        return 0
    except BaseException as e:  # keep failure record (Constitution II)
        manifest.update(status="failed", failure_reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        write_manifest(manifest_path, manifest)
        print(f"FAILED {run_id}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

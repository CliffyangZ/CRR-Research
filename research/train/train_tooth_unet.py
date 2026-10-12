"""Train the U-Net baseline on single-tooth ROIs (256x256, CLAHE, target-only masks).

Validation Dice (per-ROI original-resolution frame, pixel-accumulated) selects the checkpoint.
Example:
  python -m research.train.train_tooth_unet --data-version roi_xxx \
    --config configs/segmentation/unet_baseline_v1.yaml --device cuda:0 [--smoke]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research.models.unet import build_unet
from src.segmentation.roi_config import load_roi_train_config
from src.segmentation.roi_train import build_roi_datasets, evaluate_roi_split
from src.segmentation.train import run_training


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-version", required=True, help="roi_<hash> directory name under --derived")
    ap.add_argument("--derived", default="data/derived")
    ap.add_argument("--config", default="configs/segmentation/roi_unet_baseline_v1.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="reports/segmentation")
    ap.add_argument("--batch-size", type=int, default=None, help="override (recorded in the manifest)")
    ap.add_argument("--smoke", action="store_true", help="16 train ROIs, 8 val ROIs, 2 epochs")
    args = ap.parse_args(argv)

    cfg = load_roi_train_config(args.config)
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    model = build_unet(cfg.model)
    try:
        out = run_training(model, cfg, Path(args.derived) / args.data_version, args.output, args.device, args.smoke,
                           sys.argv if argv is None else ["train_tooth_roi_unet", *argv], args.seed, args.config,
                           build_datasets=build_roi_datasets, evaluate=evaluate_roi_split, run_prefix="roiunet")
    except BaseException as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

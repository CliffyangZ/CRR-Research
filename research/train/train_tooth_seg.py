"""Train one architecture (unet | unetpp | attention_unet) on single-tooth ROIs for the architecture comparison.

Validation Dice (per-ROI original-resolution frame, pixel-accumulated) selects the checkpoint; test is never read.
With --protocol the run must belong to a frozen comparison protocol (config hash, seed, data/split all checked).
Example:
  python -m research.train.train_tooth_seg --model unetpp --seed 20261012 --data-version roi_2a0c9441396f \
    --protocol configs/segmentation/compare_protocol_v1.yaml --device cuda:0 [--smoke]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research.models.registry import MODELS, build_model
from src.segmentation.protocol import ProtocolError, load_protocol, require_frozen
from src.segmentation.roi_config import load_roi_train_config
from src.segmentation.roi_train import build_roi_datasets, evaluate_roi_split
from src.segmentation.train import run_training


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--data-version", required=True, help="roi_<hash> directory name under --derived")
    ap.add_argument("--derived", default="data/derived")
    ap.add_argument("--config", default=None, help="default: configs/segmentation/roi_<model>_v1.yaml")
    ap.add_argument("--protocol", default=None, help="frozen comparison protocol yaml; omit for an exploratory run")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="reports/segmentation")
    ap.add_argument("--batch-size", type=int, default=None, help="protocol batch_fallback only (needs --override-reason)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--override-reason", default=None)
    ap.add_argument("--smoke", action="store_true", help="16 train ROIs, 8 val ROIs, 2 epochs; never part of a protocol")
    args = ap.parse_args(argv)

    config_path = args.config or f"configs/segmentation/roi_{args.model}_v1.yaml"
    cfg = load_roi_train_config(config_path)
    if cfg.model.name != args.model:
        print(f"config model.name {cfg.model.name!r} != --model {args.model!r}", file=sys.stderr)
        return 2
    protocol_info = None
    if args.protocol and not args.smoke:
        try:
            p = load_protocol(args.protocol)
            rec = require_frozen(p)
            if cfg.hash() != rec["candidates"].get(args.model, {}).get("config_hash"):
                raise ProtocolError(f"{config_path} is not the frozen config of {args.model}")
            if args.seed not in p.seeds:
                raise ProtocolError(f"seed {args.seed} not in protocol seeds {p.seeds}")
            if args.data_version != p.data_version or str(Path(args.derived)) != str(Path(p.derived_root)):
                raise ProtocolError("data version/root differs from the protocol")
            fb = p.batch_fallback
            changed = args.batch_size is not None or args.grad_accum != 1
            if changed and ((args.batch_size, args.grad_accum) != (fb["batch_size"], fb["grad_accum"]) or not args.override_reason):
                raise ProtocolError(f"batch override must be exactly {fb} and carry --override-reason")
            protocol_info = {"version": p.protocol_version, "hash": rec["protocol_hash"]}
        except ProtocolError as e:
            print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
            return 2
    if args.batch_size:
        cfg.train.batch_size = args.batch_size

    model = build_model(cfg.model)
    try:
        out = run_training(model, cfg, Path(args.derived) / args.data_version, args.output, args.device, args.smoke,
                           sys.argv if argv is None else ["train_tooth_seg", *argv], args.seed, config_path,
                           build_datasets=build_roi_datasets, evaluate=evaluate_roi_split, run_prefix=args.model,
                           grad_accum=args.grad_accum, candidate=args.model, protocol=protocol_info,
                           override_reason=args.override_reason)
    except BaseException as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

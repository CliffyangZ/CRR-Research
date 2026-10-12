"""Training loop, original-coordinate evaluation, run manifest. Model is injected (no research import)."""

from __future__ import annotations

import csv
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detection.provenance import base_manifest, new_run_id, sha256_file, write_manifest

from .config import SegConfig
from .dataset import ToothSegDataset, collate
from .losses import bce_dice_loss
from .metrics import aggregate, evaluate_pair
from .transform import restore_to_original

import cv2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_original(model, ds: ToothSegDataset, device, batch_size: int = 4, amp: bool = True):
    """Yield (id, prob_original float32 HxW, gt_original uint8 HxW) for every sample."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    for b in dl:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(b["image"].to(device))
        probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for iid, rec, p in zip(b["id"], b["rec"], probs):
            _, gt = ds.load_original(iid)
            yield iid, restore_to_original(p, rec), gt


def evaluate_split(model, ds, device, threshold: float, amp: bool = True) -> dict:
    rows = []
    for iid, prob, gt in predict_original(model, ds, device, amp=amp):
        r = evaluate_pair((prob >= threshold).astype(np.uint8), gt)
        r["image_id"] = iid
        rows.append(r)
    out = aggregate(rows)
    out["per_image"] = rows
    return out


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _default_datasets(derived: Path, cfg, seed: int, smoke: bool):
    tr = ToothSegDataset(derived, "train", cfg, seed=seed)
    va = ToothSegDataset(derived, "val", cfg)
    if smoke:
        tr.ids, va.ids = tr.ids[:8], va.ids[:4]
    return tr, va


def run_training(model: torch.nn.Module, cfg, derived_dir: str | Path, output_root: str | Path,
                 device: str = "cuda:0", smoke: bool = False, command: list[str] | None = None,
                 seed: int | None = None, config_path: str | None = None, *,
                 build_datasets=_default_datasets, evaluate=None, run_prefix: str = "unet") -> Path:
    """Generic fit loop. `build_datasets(derived, cfg, seed, smoke) -> (train_ds, val_ds)` and
    `evaluate(model, val_ds, device, threshold, amp) -> dict` default to the whole-image segmentation setup."""
    evaluate = evaluate or evaluate_split
    t = cfg.train
    seed = t.seed if seed is None else seed
    derived = Path(derived_dir)
    run_id = new_run_id("smoke" if smoke else run_prefix)
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=False)
    weights_dir = Path("models/segmentation") / run_id
    weights_dir.mkdir(parents=True, exist_ok=False)
    dm = json.loads((derived / "dataset_manifest.json").read_text())
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    max_epochs = 2 if smoke else t.max_epochs
    manifest = base_manifest(
        run_id, command, status="running", failure_reason=None, seed=seed, smoke=smoke,
        config=cfg.model_dump(mode="json"), config_path=config_path, config_hash=cfg.hash(),
        data_version=dm["data_version"], split_sha256=sha256_file(derived / "splits.json"),
        annotation_review=dm["annotation_review"], init="random_kaiming", pretrained=None,
        selection_rule="max val_dice (original coords, threshold %.2f), tie -> earlier epoch" % t.threshold,
        output_dir=str(out), weights_dir=str(weights_dir), device=str(dev),
        nondeterminism_note="cuDNN/AMP kernels and DataLoader ordering are not bit-reproducible; single seed only.",
    )
    manifest["exploratory"] = True
    mpath = out / "run_manifest.json"
    write_manifest(mpath, manifest)
    try:
        seed_everything(seed)
        tr, va = build_datasets(derived, cfg, seed, smoke)
        g = torch.Generator().manual_seed(seed)
        dl = DataLoader(tr, batch_size=t.batch_size, shuffle=True, drop_last=len(tr) > t.batch_size,
                        collate_fn=collate, num_workers=2, generator=g, worker_init_fn=lambda w: np.random.seed(seed + w))
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=t.optimizer.lr, weight_decay=t.optimizer.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=t.scheduler.min_lr)
        scaler = torch.amp.GradScaler("cuda", enabled=t.amp and dev.type == "cuda")
        best, best_epoch, bad, history, stop_reason = -1.0, -1, 0, [], "max_epochs"
        n_params = sum(p.numel() for p in model.parameters())
        manifest["n_parameters"] = n_params
        for epoch in range(1, max_epochs + 1):
            tr.epoch = epoch
            model.train()
            t0, losses, geo = time.time(), [], {"applied": 0, "cancelled": 0, "not_selected": 0}
            for b in dl:
                x, y = b["image"].to(dev), b["mask"].to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, enabled=t.amp and dev.type == "cuda"):
                    logits = model(x)
                loss = bce_dice_loss(logits.float(), y, t.loss.bce, t.loss.dice)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
                for a in b["aug"]:
                    geo[a["geometry"]] += 1
            sched.step()
            ev = evaluate(model, va, dev, t.threshold, t.amp)
            val_dice = ev["aggregate"]["dice"]
            val_dice = -1.0 if val_dice is None else val_dice
            # validation loss on the letterboxed canvas for the learning curve
            model.eval()
            vl = []
            with torch.no_grad():
                for b in DataLoader(va, batch_size=t.batch_size, collate_fn=collate):
                    vl.append(bce_dice_loss(model(b["image"].to(dev)).float(), b["mask"].to(dev),
                                            t.loss.bce, t.loss.dice).item())
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vl)),
                   "val_dice": val_dice, "val_iou": ev["aggregate"]["iou"], "val_precision": ev["aggregate"]["precision"],
                   "val_recall": ev["aggregate"]["recall"], "lr": opt.param_groups[0]["lr"],
                   "geo_applied": geo["applied"], "geo_cancelled": geo["cancelled"], "seconds": round(time.time() - t0, 1)}
            history.append(row)
            _write_history(out / "history.csv", history)
            if val_dice > best:       # strict '>' keeps the earlier epoch on ties
                best, best_epoch, bad = val_dice, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, weights_dir / "best.pt")
            else:
                bad += 1
            print(f"epoch {epoch:3d} train {row['train_loss']:.4f} val_loss {row['val_loss']:.4f} "
                  f"val_dice {val_dice:.4f} best {best:.4f}@{best_epoch}", flush=True)
            if bad >= t.early_stop.patience:
                stop_reason = f"early_stop: no val_dice gain for {t.early_stop.patience} epochs"
                break
        torch.save({"model": model.state_dict(), "epoch": epoch}, weights_dir / "last.pt")
        manifest.update(
            status="succeeded",
            checkpoints={
                "best": {"path": str(weights_dir / "best.pt"), "sha256": sha256_file(weights_dir / "best.pt"),
                         "epoch": best_epoch, "val_dice": best},
                "last": {"path": str(weights_dir / "last.pt"), "sha256": sha256_file(weights_dir / "last.pt"), "epoch": epoch},
            },
            stopping={"epochs_run": epoch, "reason": stop_reason, "max_epochs": max_epochs},
        )
        write_manifest(mpath, manifest)
        print(f"best weights (epoch {best_epoch}, val_dice {best:.4f}): {weights_dir / 'best.pt'}", flush=True)
        print(f"last weights (epoch {epoch}): {weights_dir / 'last.pt'}", flush=True)
        return out
    except BaseException as e:  # keep the failure record, never mark as success
        manifest.update(status="failed", failure_reason=f"{type(e).__name__}: {e}",
                        traceback=traceback.format_exc()[-2000:])
        write_manifest(mpath, manifest)
        raise

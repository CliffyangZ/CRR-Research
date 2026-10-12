"""Fast R-CNN training loop (SGD, step LR, early stopping on validation AP@[.5:.95]).

Hyper-parameters follow the reference solver (lr 1e-3, momentum 0.9, wd 5e-4, x0.1 step decay),
adapted to a small dataset: 2 images / mini-batch, 64 RoIs per image (128 per batch), 25% foreground.
Validation split only is used for model selection; the test split is never read here.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..evaluate import evaluate
from .boxes import sample_rois
from .data import ToothSplit
from .model import FastRCNN, fast_rcnn_loss
from .predict import detect

DEFAULTS = dict(lr=1e-3, momentum=0.9, weight_decay=5e-4, images_per_batch=2, rois_per_image=64,
                fg_fraction=0.25, fg_thresh=0.5, bg_hi=0.5, bg_lo=0.1, lr_step_frac=2 / 3, lr_gamma=0.1,
                test_nms_iou=0.3)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def val_metrics(model, split: ToothSplit, device) -> dict:
    data = {}
    for i in range(len(split)):
        n, img, gt, props = split[i]
        b, s = detect(model, img, props, device, nms_iou=DEFAULTS["test_nms_iou"])
        data[n] = {"gt": gt.numpy(), "pred": b, "scores": s}
    r = evaluate(data, None)  # letterbox is a uniform scale+shift, so IoU-based AP is identical to original space
    return {"AP": r["AP"], "AP50": r["AP50"]}


def train_fast_rcnn(data_dir: str | Path, out_dir: str | Path, *, epochs: int = 100, patience: int = 30, seed: int = 0,
                    device="cuda:0", smoke: bool = False, hp: dict | None = None, log=print) -> dict:
    hp = {**DEFAULTS, **(hp or {})}
    seed_everything(seed)
    out = Path(out_dir)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    train, val = ToothSplit(data_dir, "train"), ToothSplit(data_dir, "val")
    if smoke:
        epochs = 2
        train.names = train.names[:8]
        val.names = val.names[:4]
    dev = torch.device(device)
    model = FastRCNN(1, pretrained=True).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, int(epochs * hp["lr_step_frac"])), gamma=hp["lr_gamma"])
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, rows = -1.0, -1, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(train), generator=gen).tolist()
        tot = n_it = 0.0
        for k in range(0, len(order), hp["images_per_batch"]):
            imgs, rois, labels, targets = [], [], [], []
            for i in order[k:k + hp["images_per_batch"]]:
                _, img, gt, props = train[i]
                r, l, t = sample_rois(props, gt, hp["rois_per_image"], hp["fg_fraction"], hp["fg_thresh"], hp["bg_hi"], hp["bg_lo"], gen)
                imgs.append(img); rois.append(r.to(dev)); labels.append(l); targets.append(t)
            cls, deltas = model(torch.stack(imgs).to(dev), rois)
            loss, lc, lb = fast_rcnn_loss(cls, deltas[:, :4], torch.cat(labels).to(dev), torch.cat(targets).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); n_it += 1
        sched.step()
        m = val_metrics(model, val, dev)
        fit = m["AP"] if m["AP"] is not None else -1.0
        rows.append({"epoch": ep + 1, "train_loss": tot / max(1, n_it), "val_AP": m["AP"], "val_AP50": m["AP50"],
                     "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)})
        log(f"epoch {ep + 1}/{epochs} loss {rows[-1]['train_loss']:.4f} val AP {fit:.4f} AP50 {m['AP50']}")
        if fit > best:
            best, best_epoch = fit, ep + 1
            torch.save({"state_dict": model.state_dict(), "num_fg_classes": 1, "epoch": ep + 1, "val_AP": fit, "hp": hp},
                       out / "weights" / "best.pt")
        if ep + 1 - best_epoch >= patience:
            log(f"early stop at epoch {ep + 1} (best {best_epoch})")
            break
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return {"best_epoch": best_epoch, "best_val_AP": best, "epochs_run": len(rows), "hp": hp,
            "weights": str(out / "weights" / "best.pt")}

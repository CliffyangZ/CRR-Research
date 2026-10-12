"""Detection metrics for a single class, original-image pixel xyxy boxes.

Protocol (specs/001-train-tooth-detection/contracts/evaluation-protocol.md):
 - predictions sorted by score; each GT matched at most once (best-IoU unmatched GT >= thr)
 - AP: COCO-style 101-point interpolation, mean over IoU 0.50:0.05:0.95; AP50 auxiliary
 - P/R/F1 at IoU 0.5 for predictions with score >= tau
 - zero denominators -> None (reported as "undefined"), never 0
 - errors @ IoU 0.5: missed, duplicate (extra hit on an already-matched GT),
   localization (best IoU in [0.1, 0.5)), false_positive (best IoU < 0.1)
"""

from __future__ import annotations

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
LOC_LOW = 0.1


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match_image(gt: np.ndarray, pred: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    """Greedy score-ordered matching. Returns arrays aligned to the *sorted* predictions."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    pred = np.asarray(pred, float).reshape(-1, 4)[order]
    sc = np.asarray(scores, float)[order]
    ious = iou_matrix(pred, gt)
    matched = np.zeros(len(gt), bool)
    tp = np.zeros(len(pred), bool)
    kind = np.empty(len(pred), object)
    for i in range(len(pred)):
        best_iou = ious[i].max() if len(gt) else 0.0
        cand = np.where(~matched & (ious[i] >= thr))[0] if len(gt) else np.array([], int)
        if len(cand):
            j = cand[np.argmax(ious[i][cand])]
            matched[j], tp[i], kind[i] = True, True, "tp"
        elif best_iou >= thr:
            kind[i] = "duplicate"
        elif best_iou >= LOC_LOW:
            kind[i] = "localization"
        else:
            kind[i] = "false_positive"
    return {"scores": sc, "tp": tp, "kind": kind, "n_gt": len(gt), "gt_matched": matched}


def _ap_from(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float | None:
    if n_gt == 0:
        return None
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="stable")
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    for k in range(len(prec) - 2, -1, -1):
        prec[k] = max(prec[k], prec[k + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side="left")
    q = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
    return float(q.mean())


def average_precision(data: dict, thr: float) -> float | None:
    s, t, n = [], [], 0
    for d in data.values():
        m = match_image(d["gt"], d["pred"], d["scores"], thr)
        s.append(m["scores"]); t.append(m["tp"]); n += m["n_gt"]
    return _ap_from(np.concatenate(s) if s else np.zeros(0), np.concatenate(t) if t else np.zeros(0, bool), n)


def prf_at(data: dict, tau: float, thr: float = 0.5) -> dict:
    tp = fp = n_gt = 0
    errors = {"missed": 0, "false_positive": 0, "duplicate": 0, "localization": 0}
    for d in data.values():
        keep = np.asarray(d["scores"]) >= tau
        m = match_image(d["gt"], np.asarray(d["pred"]).reshape(-1, 4)[keep], np.asarray(d["scores"])[keep], thr)
        tp += int(m["tp"].sum())
        fp += int((~m["tp"]).sum())
        n_gt += m["n_gt"]
        errors["missed"] += int((~m["gt_matched"]).sum())
        for k in ("false_positive", "duplicate", "localization"):
            errors[k] += int((m["kind"] == k).sum())
    P = tp / (tp + fp) if tp + fp else None
    R = tp / n_gt if n_gt else None
    F1 = 2 * P * R / (P + R) if P is not None and R is not None and (P + R) > 0 else (0.0 if P is not None and R is not None else None)
    return {"tau": tau, "precision": P, "recall": R, "f1": F1, "tp": tp, "fp": fp, "n_gt": n_gt, "errors": errors}


def choose_threshold(val: dict, thr: float = 0.5) -> dict:
    """tau maximizing validation F1 over observed scores; ties -> higher tau."""
    cands = sorted({float(s) for d in val.values() for s in np.asarray(d["scores"]).ravel()})
    if not cands:
        return {"tau": None, "reason": "no predictions on validation set"}
    best = None
    for tau in cands:
        r = prf_at(val, tau, thr)
        f = r["f1"] if r["f1"] is not None else -1
        if best is None or f > best[0] or (f == best[0] and tau > best[1]):
            best = (f, tau)
    return {"tau": best[1], "val_f1": best[0] if best[0] >= 0 else None}


def evaluate(data: dict, tau: float | None) -> dict:
    aps = {float(t): average_precision(data, float(t)) for t in IOU_THRS}
    vals = [v for v in aps.values() if v is not None]
    n_gt = sum(len(d["gt"]) for d in data.values())
    out = {
        "population": {"n_images": len(data), "n_boxes": n_gt},
        "AP": float(np.mean(vals)) if len(vals) == len(aps) else None,
        "AP50": aps[0.5],
        "AP_per_iou": aps,
        "tau": tau,
    }
    if n_gt == 0:
        out["undefined_reason"] = "no ground-truth boxes in this population"
    if tau is None:
        out.update(precision=None, recall=None, f1=None, errors=None,
                   undefined_reason=out.get("undefined_reason", "no working threshold (no validation predictions)"))
    else:
        r = prf_at(data, tau, 0.5)
        out.update({k: r[k] for k in ("precision", "recall", "f1", "tp", "fp", "errors")})
        for k, why in (("precision", "no predictions at working threshold"), ("recall", "no ground-truth boxes")):
            if out[k] is None:
                out.setdefault("undefined_reasons", {})[k] = why
    return out

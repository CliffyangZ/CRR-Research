"""Evaluate one trained run: choose tau on VAL (F1-max), lock it, then evaluate TEST once.

  python -m research.evaluation.evaluate_tooth_yolo --run reports/detection/runs/<run_id> --device cpu

Protocol: specs/001-train-tooth-detection/contracts/evaluation-protocol.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.detection.evaluate import choose_threshold, evaluate, match_image
from src.detection.letterbox import LetterboxRecord, boxes_inverse
from src.detection.provenance import sha256_file, write_manifest


def yolo_infer(model, data_dir: Path, device, imgsz: int):
    def infer(split: str, iid: str):
        res = model.predict(str(data_dir / "images" / split / f"{iid}.png"), conf=0.001, iou=0.7, max_det=300,
                            imgsz=imgsz, device=device, verbose=False)[0]
        if not len(res.boxes):
            return np.zeros((0, 4)), np.zeros(0)
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
    return infer


def fastrcnn_infer(model, data_dir: Path, device):
    """Fast R-CNN on cached selective-search proposals; score>=0.001, NMS 0.3, max 300 detections."""
    from src.detection.fast_rcnn.data import ToothSplit
    from src.detection.fast_rcnn.predict import detect

    cache: dict = {}

    def infer(split: str, iid: str):
        ds = cache.setdefault(split, ToothSplit(data_dir, split))
        _, img, _, props = ds[ds.names.index(iid)]
        return detect(model, img, props, device)
    return infer


def predict_split(infer, ann: dict, split: str) -> dict:
    """`infer(split, image_id) -> (xyxy boxes, scores)` in letterbox space."""
    out = {}
    for iid in [i for i, a in ann.items() if a["split"] == split]:
        b, sc = infer(split, iid)
        rec = LetterboxRecord(**ann[iid]["letterbox"])
        out[iid] = {"gt": np.array(ann[iid]["orig_boxes_xyxy"]).reshape(-1, 4),
                    "pred": boxes_inverse(b, rec) if len(b) else np.zeros((0, 4)),
                    "scores": sc, "letterbox_pred": b}
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def write_overlays(out_dir: Path, data_dir: Path, ann: dict, data: dict, tau: float, split: str, max_err=6, max_ok=2):
    from src.detection.letterbox import boxes_forward

    out_dir.mkdir(parents=True, exist_ok=True)
    err, ok = [], []
    for iid, d in data.items():
        keep = d["scores"] >= tau
        m = match_image(d["gt"], d["pred"][keep], d["scores"][keep], 0.5)
        n_err = int((~m["tp"]).sum() + (~m["gt_matched"]).sum())
        (err if n_err else ok).append((n_err, iid, keep))
    for tag, lst, cap in (("error", sorted(err, reverse=True), max_err), ("ok", ok, max_ok)):
        for n, iid, keep in lst[:cap]:
            img = cv2.imread(str(data_dir / "images" / split / f"{iid}.png"))
            rec = LetterboxRecord(**ann[iid]["letterbox"])
            for g in boxes_forward(data[iid]["gt"], rec).round().astype(int):
                cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (0, 255, 0), 2)
            for p in data[iid]["letterbox_pred"][keep].round().astype(int):
                cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / f"{tag}_{iid}.jpg"), img)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data/derived")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-retest", action="store_true", help="re-evaluate TEST again; recorded as test reuse")
    ap.add_argument("--skip-test", action="store_true", help="only choose tau on val (e.g. smoke checks)")
    a = ap.parse_args(argv)

    run = Path(a.run)
    rm = json.loads((run / "run_manifest.json").read_text())
    if rm["status"] != "trained":
        print(f"run {rm['run_id']} status={rm['status']}; nothing to evaluate", file=sys.stderr)
        return 1
    out = Path(a.output) if a.output else run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data_root) / rm["data_version"]
    ann = json.loads((data_dir / "manifests" / "annotations.json").read_text())
    weights = rm["selected_weights"]["path"]
    assert sha256_file(weights) == rm["selected_weights"]["sha256"], "weights changed since training"
    dev = int(a.device) if a.device.isdigit() else a.device
    imgsz = rm["hyperparams"]["imgsz"]
    if rm["model"] == "fastrcnn":
        from src.detection.fast_rcnn.predict import load_fast_rcnn

        tdev = f"cuda:{a.device}" if a.device.isdigit() else a.device
        infer = fastrcnn_infer(load_fast_rcnn(weights, tdev), data_dir, tdev)
        predict_info = {"score_min": 0.001, "nms_iou": 0.3, "max_det": 300, "proposals": "selective search (cached)"}
    else:
        from ultralytics import YOLO

        infer = yolo_infer(YOLO(weights), data_dir, dev, imgsz)
        predict_info = {"conf": 0.001, "iou_nms": 0.7, "max_det": 300}

    val = predict_split(infer, ann, "val")
    tau = choose_threshold(val)
    lock = out / "threshold_lock.json"
    if not lock.exists():  # tau is fixed once, before test is touched
        lock.write_text(json.dumps({"tau": tau["tau"], "val_f1": tau.get("val_f1"), "chosen_on": "val",
                                    "rule": "max F1 @IoU0.5; ties -> higher tau", "weights_sha256": rm["selected_weights"]["sha256"]}, indent=1))
    tau_v = json.loads(lock.read_text())["tau"]
    val_res = evaluate(val, tau_v)
    meta = {"run_id": rm["run_id"], "model": rm["model"], "data_version": rm["data_version"], "split_sha256": rm.get("split_sha256"),
            "weights_sha256": rm["selected_weights"]["sha256"], "annotation_review_status": rm["annotation_version"]["review_status"],
            "exploratory": True, "seed_variance": "未量測 (single run per model)",
            "predict": {**predict_info, "imgsz": imgsz, "device": str(a.device)}}
    (out / "val_metrics.json").write_text(json.dumps(_jsonable({**meta, "split": "val", **val_res}), indent=1, ensure_ascii=False))
    if a.skip_test:
        print(json.dumps({"tau": tau_v, "val_AP": val_res["AP"]}))
        return 0

    tm = out / "test_metrics.json"
    reuse = tm.exists()
    if reuse and not a.allow_retest:
        print("TEST already evaluated for this run; pass --allow-retest to repeat (recorded as test reuse)", file=sys.stderr)
        return 2
    test = predict_split(infer, ann, "test")
    res = evaluate(test, tau_v)
    errors = res["errors"]
    (out / "test_metrics.json").write_text(json.dumps(_jsonable({**meta, "split": "test", "test_reuse": reuse, **res}), indent=1, ensure_ascii=False))
    (out / "errors.json").write_text(json.dumps(_jsonable({"run_id": rm["run_id"], "iou": 0.5, "tau": tau_v, "counts": errors}), indent=1))
    write_overlays(out / "overlays", data_dir, ann, test, tau_v, "test")
    print(json.dumps(_jsonable({"run_id": rm["run_id"], "tau": tau_v, "AP": res["AP"], "AP50": res["AP50"], "P": res["precision"],
                                "R": res["recall"], "F1": res["f1"], "errors": errors})))
    return 0


if __name__ == "__main__":
    sys.exit(main())

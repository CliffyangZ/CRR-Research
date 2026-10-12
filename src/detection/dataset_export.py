"""Build the derived Ultralytics dataset: preprocess -> letterbox -> (train only) augment.

Inputs are plain records so this module has no dependency on any research entry:
  record = {image_id, source_path, split, boxes (orig-pixel xyxy, Nx4)}
Originals are only read. Outputs under <out_dir>:
  images/{split}/<id>.png, labels/{split}/<id>.txt      (letterboxed space, normalized)
  images/train/<id>_aug<k>.png + labels                  (augmented copies, train only)
  manifests/{annotations,augmentation}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .augment import augment_once
from .config import PreprocessConfig
from .inventory import to_gray_u8
from .letterbox import boxes_forward, letterbox_image
from .preprocess import preprocess_gray, to_model_rgb
from .yolo_io import write_yolo_labels


def _write_sample(out_dir: Path, split: str, name: str, img: np.ndarray, boxes: np.ndarray, size: int):
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "images" / split / f"{name}.png"), img)
    write_yolo_labels(out_dir / "labels" / split / f"{name}.txt", np.zeros(len(boxes), int), boxes, size, size)


def export_dataset(records: list[dict], cfg: PreprocessConfig, out_dir: str | Path, overlay_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir)
    lb = cfg.letterbox
    annotations: dict = {}
    aug_log: list[dict] = []
    for rec in records:
        iid, split = rec["image_id"], rec["split"]
        img = cv2.imread(rec["source_path"], cv2.IMREAD_UNCHANGED)
        gray = preprocess_gray(to_gray_u8(img), cfg)
        canvas, lrec = letterbox_image(to_model_rgb(gray), lb.size, lb.pad_value, lb.interp_down, lb.interp_up)
        fboxes = boxes_forward(rec["boxes"], lrec) if len(rec["boxes"]) else np.zeros((0, 4))
        _write_sample(out_dir, split, iid, canvas, fboxes, lb.size)
        annotations[iid] = {
            "split": split,
            "source_path": rec["source_path"],
            "letterbox": lrec.to_dict(),
            "orig_boxes_xyxy": np.asarray(rec["boxes"]).reshape(-1, 4).tolist(),
        }
        if split == "train":
            for k in range(cfg.augment.copies_per_train_image):
                rng = np.random.default_rng([cfg.augment.seed, _stable_int(iid), k])
                r = augment_once(canvas, fboxes, cfg, rng, lb.pad_value)
                _write_sample(out_dir, "train", f"{iid}_aug{k}", r.image, r.boxes, lb.size)
                aug_log.append({
                    "aug_id": f"{iid}_aug{k}", "source_image_id": iid, "seed": [cfg.augment.seed, _stable_int(iid), k],
                    "photometric_applied": r.photometric_applied, "geometry": r.geometry,
                    "cancel_reason": r.cancel_reason, "params": r.params,
                })
        if overlay_dir is not None:
            _overlay(Path(overlay_dir), iid, canvas, fboxes)
    man = out_dir / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    (man / "annotations.json").write_text(json.dumps(annotations, indent=1))
    (man / "augmentation.json").write_text(json.dumps(aug_log, indent=1))
    data_yaml = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                 "test": "images/test", "names": {0: "tooth"}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return {"n_images": len(records), "n_aug": len(aug_log), "annotations": annotations, "augmentation": aug_log}


def _stable_int(s: str) -> int:
    import zlib

    return zlib.crc32(s.encode())


def _overlay(overlay_dir: Path, name: str, img: np.ndarray, boxes: np.ndarray) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    o = img.copy()
    for x1, y1, x2, y2 in boxes.round().astype(int):
        cv2.rectangle(o, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(overlay_dir / f"{name}.jpg"), o)


def overlay_augmented(out_dir: str | Path, overlay_dir: str | Path, names: list[str]) -> None:
    """Overlay boxes from the written label files on written augmented images (alignment check)."""
    from .yolo_io import read_yolo_labels

    out_dir = Path(out_dir)
    for n in names:
        im = cv2.imread(str(out_dir / "images" / "train" / f"{n}.png"))
        _, b = read_yolo_labels(out_dir / "labels" / "train" / f"{n}.txt", im.shape[1], im.shape[0])
        _overlay(Path(overlay_dir), n, im, b)

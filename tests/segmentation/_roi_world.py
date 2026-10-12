"""Synthetic detection + segmentation sources for ROI pipeline tests."""

import json
from pathlib import Path

import cv2
import numpy as np

W, H = 200, 300


def textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 4)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def make_world(root: Path, images: list[dict]) -> tuple[Path, Path]:
    """images: [{id, subset, seed, det_subset?, boxes:[(x0,y0,x1,y1)], polys:[(x0,y0,x1,y1)]}]"""
    det, seg = root / "det", root / "seg"
    coco = {s: {"categories": [{"id": 1, "name": "tooth"}, {"id": 2, "name": "Metal"}], "images": [], "annotations": []}
            for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        (det / "images" / s).mkdir(parents=True)
        (det / "labels" / s).mkdir(parents=True)
        (seg / s / "images").mkdir(parents=True)
        (seg / s / "annotations").mkdir(parents=True)
    aid = 0
    for k, im in enumerate(images):
        name = f"{im['id']}.png"
        img = textured(im["seed"])
        cv2.imwrite(str(seg / im["subset"] / "images" / name), img)
        cv2.imwrite(str(det / "images" / im.get("det_subset", im["subset"]) / name), img)
        coco[im["subset"]]["images"].append({"id": k, "file_name": name, "width": W, "height": H})
        for (x0, y0, x1, y1) in im["polys"]:
            aid += 1
            coco[im["subset"]]["annotations"].append({"id": aid, "image_id": k, "category_id": 1,
                                                       "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]})
        lines = [f"0 {(x0 + x1) / 2 / W} {(y0 + y1) / 2 / H} {(x1 - x0) / W} {(y1 - y0) / H}" for (x0, y0, x1, y1) in im["boxes"]]
        (det / "labels" / im.get("det_subset", im["subset"]) / f"{im['id']}.txt").write_text("\n".join(lines) + "\n")
    for s, c in coco.items():
        (seg / s / "annotations" / "instances_default.json").write_text(json.dumps(c))
    return det, seg


def write_cfg(root: Path, det: Path, seg: Path) -> Path:
    import yaml

    from src.segmentation.roi_config import RoiConfig

    d = RoiConfig().model_dump(mode="json")
    d["data"]["detection_root"], d["data"]["seg_root"] = str(det), str(seg)
    p = root / "cfg.yaml"
    p.write_text(yaml.safe_dump(d))
    return p

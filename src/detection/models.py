"""Candidate YOLO families (all Ultralytics, size 's'). Weights are downloaded by the caller."""

from __future__ import annotations

from pathlib import Path

CANDIDATES = {
    "v8": {"family": "YOLOv8", "weights_name": "yolov8s.pt", "size": "s"},
    "v11": {"family": "YOLO11", "weights_name": "yolo11s.pt", "size": "s"},
    "v26": {"family": "YOLO26", "weights_name": "yolo26s.pt", "size": "s"},
    "fastrcnn": {"family": "Fast R-CNN (VGG16)", "weights_name": None, "size": "vgg16"},
}
COMMON = {
    "source": "https://github.com/ultralytics/ultralytics (assets release v8.4.0)",
    "license": "AGPL-3.0 (Ultralytics); enterprise license needed for closed-source/commercial use",
    "pretrained": "COCO-pretrained detection weights (generic objects, not dental); fine-tuned for single class 'tooth'",
}
FASTRCNN_COMMON = {
    "source": "arXiv:1504.08083; reference architecture github.com/rbgirshick/fast-rcnn (MIT); PyTorch re-implementation in src/detection/fast_rcnn",
    "license": "MIT (reference code); torchvision VGG16 ImageNet weights (BSD-3-Clause)",
    "pretrained": "ImageNet-1k VGG16 (torchvision IMAGENET1K_V1) backbone; heads randomly initialised; selective-search proposals",
}


def candidate(key: str, weights_dir: str | Path = "models/pretrained") -> dict:
    if key not in CANDIDATES:
        raise ValueError(f"unknown model {key!r}; choose from {sorted(CANDIDATES)}")
    if key == "fastrcnn":
        return dict(CANDIDATES[key], **FASTRCNN_COMMON, weights_path=None)
    c = dict(CANDIDATES[key], **COMMON)
    c["weights_path"] = str(Path(weights_dir) / c["weights_name"])
    return c

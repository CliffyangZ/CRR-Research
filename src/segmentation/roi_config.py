"""Registered configuration for the single-tooth ROI segmentation dataset (CLAHE-only preprocessing)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from src.detection import config as dcfg

from .config import Augment, Eval, Model, Train


class RoiData(BaseModel):
    detection_root: str = "data/detection_dataset"
    seg_root: str = "data/seg_dataset"
    tooth_category: str = "tooth"
    ignore_categories: list[str] = ["Metal"]


class Pairing(BaseModel):
    min_iou: float = 0.5
    second_margin: float = 0.1
    min_mask_frac: float = 0.05


class RoiGeom(BaseModel):
    expand_frac_per_side: float = 0.10
    size: int = 256
    image_pad: int = 114
    mask_pad: int = 0
    interp_down: str = "area"
    interp_up: str = "linear"
    mask_interp: str = "nearest"


class Split(BaseModel):
    source: str = "existing"
    phash_dist: int = 6
    seed: int = 42


class RoiConfig(BaseModel):
    data: RoiData = RoiData()
    preprocess: dict = Field(default_factory=lambda: {
        "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8]},
        "input_mapping": {"type": "linear_uint8", "divisor": 255},
    })
    pairing: Pairing = Pairing()
    roi: RoiGeom = RoiGeom()
    split: Split = Split()
    augment: Augment = Augment()

    @model_validator(mode="after")
    def _clahe_only(self):
        extra = set(self.preprocess) - {"clahe", "input_mapping"}
        if extra:
            raise ValueError(f"only CLAHE preprocessing is allowed; found {sorted(extra)}")
        return self

    def preprocess_config(self) -> dcfg.PreprocessConfig:
        return dcfg.PreprocessConfig(**self.preprocess)

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_roi_config(path: str | Path) -> RoiConfig:
    with open(path) as f:
        return RoiConfig(**yaml.safe_load(f))


class RoiTrainConfig(RoiConfig):
    """ROI dataset settings + U-Net model and training settings for the ROI baseline."""

    model: Model = Model()
    train: Train = Train(batch_size=16)
    eval: Eval = Eval()


def load_roi_train_config(path: str | Path) -> RoiTrainConfig:
    with open(path) as f:
        return RoiTrainConfig(**yaml.safe_load(f))

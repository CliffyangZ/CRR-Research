"""Registered configuration for the U-Net baseline (frozen before formal training)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.detection import config as dcfg

REQUIRED_DISABLED = {"scale", "flip", "mosaic", "mixup", "crop", "elastic"}


class Data(BaseModel):
    root: str = "data/seg_dataset"
    derived: str = "data/derived"
    tooth_category: str = "tooth"
    ignore_categories: list[str] = ["Metal"]


class Resize(BaseModel):
    size: int = 512
    image_pad: int = 114
    mask_pad: int = 0
    interp_down: str = "area"
    interp_up: str = "linear"
    mask_interp: str = "nearest"


class Truncation(BaseModel):
    tolerance: float = 0.005
    margin_frac: float = 0.25


class Augment(BaseModel):
    prob: float = 0.5
    order: list[str] = ["brightness_contrast", "rotate", "translate"]
    brightness: float = 0.10
    contrast: float = 0.10
    rotate_deg: float = 5
    translate_frac: float = 0.05
    truncation: Truncation = Truncation()
    disabled: list[str] = Field(default_factory=lambda: sorted(REQUIRED_DISABLED))

    @model_validator(mode="after")
    def _forbidden_stay_disabled(self):
        if not REQUIRED_DISABLED <= set(self.disabled):
            raise ValueError(f"augmentations must stay disabled: {sorted(REQUIRED_DISABLED)}")
        return self


class Model(BaseModel):
    name: Literal["unet", "unetpp", "attention_unet"] = "unet"
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    norm: str = "batch"
    upsample: str = "transpose"
    init: str = "kaiming"


class Optimizer(BaseModel):
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4


class Scheduler(BaseModel):
    name: str = "cosine"
    min_lr: float = 1e-5


class Loss(BaseModel):
    bce: float = 1.0
    dice: float = 1.0


class EarlyStop(BaseModel):
    metric: str = "val_dice"
    patience: int = 30


class Selection(BaseModel):
    metric: str = "val_dice"
    mode: str = "max"
    tie: str = "earlier_epoch"


class Train(BaseModel):
    seed: int = 20261012
    batch_size: int = 4
    max_epochs: int = 150
    optimizer: Optimizer = Optimizer()
    scheduler: Scheduler = Scheduler()
    loss: Loss = Loss()
    amp: bool = True
    early_stop: EarlyStop = EarlyStop()
    selection: Selection = Selection()
    threshold: float = 0.5


class ErrorCases(BaseModel):
    miss_min_frac: float = 0.02
    false_min_frac: float = 0.001


class Eval(BaseModel):
    error_cases: ErrorCases = ErrorCases()


class SegConfig(BaseModel):
    data: Data = Data()
    preprocess: dict = Field(default_factory=dict)
    resize: Resize = Resize()
    augment: Augment = Augment()
    model: Model = Model()
    train: Train = Train()
    eval: Eval = Eval()

    def preprocess_config(self) -> dcfg.PreprocessConfig:
        """Fixed NLM/CLAHE/USM settings as the shared detection PreprocessConfig."""
        return dcfg.PreprocessConfig(**self.preprocess)

    def preprocess_hash(self) -> str:
        return self.preprocess_config().hash()

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path) -> SegConfig:
    with open(path) as f:
        return SegConfig(**yaml.safe_load(f))

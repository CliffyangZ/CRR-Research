"""Preprocessing / augmentation configuration (pydantic) with a stable hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class NLM(BaseModel):
    h: float = 10
    template_window: int = 7
    search_window: int = 21


class CLAHE(BaseModel):
    clip_limit: float = 2.0
    tile_grid: tuple[int, int] = (8, 8)


class USM(BaseModel):
    sigma: float = 1.0
    amount: float = 0.5
    threshold: int = 0


class InputMapping(BaseModel):
    type: str = "linear_uint8"
    divisor: float = 255


class Letterbox(BaseModel):
    size: int = 640
    pad_value: int = 114
    interp_down: str = "area"
    interp_up: str = "linear"


class Augment(BaseModel):
    copies_per_train_image: int = 3
    seed: int = 20261009
    prob: float = 0.5
    order: list[str] = Field(
        default_factory=lambda: ["brightness_contrast", "rotate", "translate"]
    )
    brightness: float = 0.10
    contrast: float = 0.10
    rotate_deg: float = 5
    translate_frac: float = 0.05
    disabled: list[str] = Field(default_factory=list)


class PreprocessConfig(BaseModel):
    nlm: NLM = NLM()
    clahe: CLAHE = CLAHE()
    usm: USM = USM()
    input_mapping: InputMapping = InputMapping()
    letterbox: Letterbox = Letterbox()
    augment: Augment = Augment()

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path) -> PreprocessConfig:
    with open(path) as f:
        return PreprocessConfig(**yaml.safe_load(f))

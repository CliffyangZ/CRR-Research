"""Torch dataset over a built ROI dataset. Random augmentation is training-only (same policy as Spec1)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.detection.letterbox import LetterboxRecord  # noqa: F401  (re-export for collate users)

from .augment import augment_once
from .dataset import collate  # noqa: F401
from .roi_config import RoiConfig


class RoiSegDataset(Dataset):
    """split in {train,val,test}; augment defaults to True only for train."""

    def __init__(self, derived_dir: str | Path, split: str, cfg: RoiConfig, augment: bool | None = None,
                 seed: int = 0, ids: list[str] | None = None):
        self.dir = Path(derived_dir)
        self.cfg = cfg
        index = json.loads((self.dir / "roi_index.json").read_text())
        rows = [r for r in index if r["subset"] == split]
        if ids is not None:
            rows = [r for r in rows if r["roi_id"] in set(ids)]
        self.rows = sorted(rows, key=lambda r: r["roi_id"])
        self.augment = (split == "train") if augment is None else augment
        if self.augment and split != "train":
            raise ValueError("augmentation is training-only")
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def ids(self) -> list[str]:
        return [r["roi_id"] for r in self.rows]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = cv2.imread(str(self.dir / row["files"]["image"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.dir / row["files"]["mask"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)          # accept 0/1 and 0/255 label files alike
        aug = {"geometry": "not_selected", "photometric": False}
        if self.augment:
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            res = augment_once(img, mask, [mask], self.cfg.augment, rng, self.cfg.roi.image_pad)
            img, mask = res.image, res.mask
            aug = {"geometry": res.geometry, "photometric": res.photometric_applied,
                   "params": res.params, "cancel_reason": res.cancel_reason}
        div = self.cfg.preprocess_config().input_mapping.divisor
        return {"image": torch.from_numpy(img.astype(np.float32) / div)[None],
                "mask": torch.from_numpy(mask.astype(np.float32))[None],
                "id": row["roi_id"], "rec": row["letterbox"], "aug": aug}

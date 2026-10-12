"""Torch dataset: cached fixed-preprocessed images + masks, letterbox, train-only augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.detection.letterbox import LetterboxRecord

from .augment import augment_once
from .config import SegConfig
from .masks import instance_masks
from .transform import letterbox_pair


class ToothSegDataset(Dataset):
    """split in {train,val,test}. Only 'train' with augment=True applies random augmentation."""

    def __init__(self, derived_dir: str | Path, split: str, cfg: SegConfig,
                 augment: bool | None = None, seed: int = 0, ids: list[str] | None = None):
        self.dir = Path(derived_dir)
        self.cfg = cfg
        splits = json.loads((self.dir / "splits.json").read_text())["assignments"]
        self.ids = ids if ids is not None else sorted(i for i, s in splits.items() if s == split)
        self.augment = (split == "train") if augment is None else augment
        if self.augment and split != "train":
            raise ValueError("augmentation is training-only")
        self.seed = seed
        self.epoch = 0
        self.instances_src = (
            json.loads((self.dir / "tooth_annotations.json").read_text()) if self.augment else {}
        )

    def __len__(self) -> int:
        return len(self.ids)

    def load_original(self, iid: str) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(self.dir / "images" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.dir / "masks" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, idx: int) -> dict:
        iid = self.ids[idx]
        img, mask = self.load_original(iid)
        r = self.cfg.resize
        li, lm, rec = letterbox_pair(img, mask, r.size, r.image_pad, r.mask_pad, r.interp_down, r.interp_up)
        aug_info = {"geometry": "not_selected", "photometric": False}
        if self.augment:
            h, w = img.shape
            insts = []
            for m in instance_masks(self.instances_src[iid], {1}, w, h):
                _, lmi, _ = letterbox_pair(img, m, r.size, r.image_pad, r.mask_pad, r.interp_down, r.interp_up)
                insts.append(lmi)
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            res = augment_once(li, lm, insts, self.cfg.augment, rng, r.image_pad)
            li, lm = res.image, res.mask
            aug_info = {"geometry": res.geometry, "photometric": res.photometric_applied,
                        "params": res.params, "cancel_reason": res.cancel_reason}
        x = torch.from_numpy(li.astype(np.float32) / self.cfg.preprocess_config().input_mapping.divisor)[None]
        y = torch.from_numpy(lm.astype(np.float32))[None]
        return {"image": x, "mask": y, "id": iid, "rec": rec.to_dict(), "aug": aug_info}


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "id": [b["id"] for b in batch],
        "rec": [LetterboxRecord(**b["rec"]) for b in batch],
        "aug": [b["aug"] for b in batch],
    }

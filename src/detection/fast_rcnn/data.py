"""Dataset over the derived (letterboxed 640x640) tooth data version."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..yolo_io import read_yolo_labels
from .proposals import proposal_path


class ToothSplit:
    """Items: (name, image CHW float32 RGB [0,1], gt xyxy, proposals xyxy). No augmentation here
    (conservative augmented copies were already written offline into the train split)."""

    def __init__(self, data_dir: str | Path, split: str, with_proposals: bool = True):
        self.dir, self.split = Path(data_dir), split
        self.names = sorted(p.stem for p in (self.dir / "images" / split).glob("*.png"))
        self.with_proposals = with_proposals

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i: int):
        n = self.names[i]
        img = cv2.imread(str(self.dir / "images" / self.split / f"{n}.png"))
        h, w = img.shape[:2]
        _, gt = read_yolo_labels(self.dir / "labels" / self.split / f"{n}.txt", w, h)
        props = np.load(proposal_path(self.dir, self.split, n)) if self.with_proposals else np.zeros((0, 4), np.float32)
        t = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1).float() / 255.0
        return n, t, torch.from_numpy(gt).float().reshape(-1, 4), torch.from_numpy(props).float().reshape(-1, 4)

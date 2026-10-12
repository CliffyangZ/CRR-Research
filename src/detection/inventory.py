"""Per-file image inventory: decode check, bit depth, channels, intensity range, hashes.

Status rules (see data-model.md DatasetVersion):
  - decode failure                      -> excluded   (reason: corrupt)
  - non-uint8 dtype                     -> quarantined (reason: non_8bit)
  - 3/4-channel with mean channel
    deviation > gray_mean_tol           -> quarantined (reason: color_content)
  - byte-identical to an earlier file   -> excluded   (reason: duplicate_of:<id>)
Everything else is "included".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from .provenance import sha256_file


@dataclass
class InventoryItem:
    image_id: str
    source_path: str
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    dtype: str | None = None
    bit_depth: int | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    channel_mean_dev: float | None = None
    phash: str | None = None
    status: str = "included"
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """uint8 H x W [x C] (BGR) -> uint8 H x W gray."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def inspect_image(
    path: str | Path, image_id: str | None = None, gray_mean_tol: float = 1.0
) -> InventoryItem:
    path = Path(path)
    item = InventoryItem(image_id=image_id or path.stem, source_path=str(path))
    try:
        item.sha256 = sha256_file(path)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except OSError as e:
        item.status, item.reason = "excluded", f"corrupt: {e}"
        return item
    if img is None or img.size == 0:
        item.status, item.reason = "excluded", "corrupt: decode failed"
        return item
    item.height, item.width = img.shape[:2]
    item.channels = 1 if img.ndim == 2 else img.shape[2]
    item.dtype = str(img.dtype)
    item.bit_depth = img.dtype.itemsize * 8
    item.intensity_min, item.intensity_max = float(img.min()), float(img.max())
    if img.dtype != np.uint8:
        item.status, item.reason = "quarantined", "non_8bit"
        return item
    if item.channels >= 3:
        c = img[..., :3].astype(np.int16)
        item.channel_mean_dev = float(np.abs(c - c.mean(2, keepdims=True)).mean())
        if item.channel_mean_dev > gray_mean_tol:
            item.status, item.reason = "quarantined", "color_content"
            return item
    item.phash = str(imagehash.phash(Image.fromarray(to_gray_u8(img))))
    return item


def mark_duplicates(items: list[InventoryItem]) -> None:
    """Byte-identical later files are excluded and point at the first one."""
    seen: dict[str, str] = {}
    for it in items:
        if not it.sha256 or it.status == "excluded":
            continue
        if it.sha256 in seen:
            it.status, it.reason = "excluded", f"duplicate_of:{seen[it.sha256]}"
        else:
            seen[it.sha256] = it.image_id

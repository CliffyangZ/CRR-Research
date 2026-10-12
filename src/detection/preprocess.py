"""Fixed intensity preprocessing: NLM -> CLAHE -> USM (8-bit in, 8-bit out).

Image size and box coordinates are never changed. The final linear mapping to
the model input range (uint8 / 255) is applied by the model loader exactly once;
no per-image min-max normalization is done anywhere.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig


def nlm(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    p = cfg.nlm
    return cv2.fastNlMeansDenoising(gray, None, p.h, p.template_window, p.search_window)


def clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    p = cfg.clahe
    return cv2.createCLAHE(clipLimit=p.clip_limit, tileGridSize=tuple(p.tile_grid)).apply(gray)


def usm(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    p = cfg.usm
    blur = cv2.GaussianBlur(gray, (0, 0), p.sigma)
    g = gray.astype(np.float32)
    detail = g - blur.astype(np.float32)
    if p.threshold > 0:
        detail[np.abs(detail) < p.threshold] = 0
    return np.clip(g + p.amount * detail, 0, 255).round().astype(np.uint8)


def preprocess_gray(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """uint8 gray -> uint8 gray, order NLM -> CLAHE -> USM."""
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("preprocess_gray expects a 2-D uint8 image (non-8-bit files are quarantined)")
    return usm(clahe(nlm(gray, cfg), cfg), cfg)


def to_model_rgb(gray_u8: np.ndarray) -> np.ndarray:
    """Replicate to 3 channels (BGR order irrelevant for gray)."""
    return np.repeat(gray_u8[..., None], 3, axis=2)

import numpy as np
import pytest

from src.detection import preprocess as P
from src.detection.config import PreprocessConfig


def _img():
    rng = np.random.RandomState(0)
    base = np.tile(np.linspace(40, 200, 96, dtype=np.float32), (80, 1))
    return np.clip(base + rng.randn(80, 96) * 8, 0, 255).astype(np.uint8)


def test_order_and_shape():
    cfg, g = PreprocessConfig(), _img()
    out = P.preprocess_gray(g, cfg)
    assert out.shape == g.shape and out.dtype == np.uint8
    manual = P.usm(P.clahe(P.nlm(g, cfg), cfg), cfg)
    assert np.array_equal(out, manual)
    wrong = P.nlm(P.usm(P.clahe(g, cfg), cfg), cfg)  # different order must differ
    assert not np.array_equal(out, wrong)


def test_deterministic_and_no_per_image_minmax():
    cfg = PreprocessConfig()
    g = _img()
    assert np.array_equal(P.preprocess_gray(g, cfg), P.preprocess_gray(g.copy(), cfg))
    dark = (g // 4).astype(np.uint8)  # low-range image must NOT be stretched to 0..255 by min-max
    out = P.preprocess_gray(dark, cfg)
    assert out.max() < 255 * 0.9 or out.min() > 0  # CLAHE may widen but no forced 0..255 stretch rule
    assert P.to_model_rgb(out).shape == (80, 96, 3)


def test_rejects_non_8bit():
    with pytest.raises(ValueError):
        P.preprocess_gray(np.zeros((8, 8), np.uint16), PreprocessConfig())

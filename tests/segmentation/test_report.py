import numpy as np

from src.segmentation.report import classify_errors, hd95, overlay


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8); m[y0:y1, x0:x1] = 1
    return m


def test_miss_false_boundary_and_zero_cases():
    gt = box(100, 100, 10, 60, 10, 60)
    perfect = classify_errors(gt, gt)
    assert not perfect["miss"] and not perfect["false"] and perfect["hd95"] == 0.0
    missed = classify_errors(box(100, 100, 10, 60, 10, 30), gt)          # 60% of tooth missed
    assert missed["miss"] and not missed["false"]
    spurious = classify_errors(gt | box(100, 100, 80, 95, 80, 95), gt)   # 225 px blob outside
    assert spurious["false"] and not spurious["miss"]
    shifted = classify_errors(box(100, 100, 10, 60, 14, 64), gt)
    assert 3 <= shifted["hd95"] <= 5
    assert classify_errors(np.zeros_like(gt), gt)["miss"]


def test_empty_masks_and_overlay_shape():
    z = np.zeros((20, 20), np.uint8)
    r = classify_errors(z, z)
    assert not r["miss"] and not r["false"] and r["hd95"] is None
    assert hd95(z, box(20, 20, 1, 5, 1, 5)) is None
    assert overlay(np.zeros((20, 20), np.uint8), z, z).shape == (20, 20, 3)

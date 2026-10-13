import numpy as np

from src.segmentation.boundary import boundary_f1, hd95


def box(x0, x1, shape=(60, 60)):
    m = np.zeros(shape, bool)
    m[15:45, x0:x1] = True
    return m


def test_identical_masks():
    m = box(10, 40)
    assert hd95(m, m)[0] == 0
    assert boundary_f1(m, m)[0] == 1.0


def test_shifted_box_hd95_about_shift():
    d, _ = hd95(box(13, 43), box(10, 40))
    assert 2.5 <= d <= 3.5
    f1, _ = boundary_f1(box(13, 43), box(10, 40), tol=2)
    assert 0.0 < f1 < 1.0                      # top/bottom edges match within 2px, side edges do not


def test_empty_gives_none_and_reason():
    m = box(10, 40)
    z = np.zeros_like(m)
    assert hd95(z, m) == (None, "pred_empty")
    assert boundary_f1(m, z) == (None, "gt_empty")
    assert hd95(z, z) == (None, "both_empty")


def test_invalid_region_errors_do_not_count():
    gt = box(10, 40)
    pred = gt.copy()
    pred[:, 50:58] = True                       # extra blob entirely in the invalid area
    valid = np.ones_like(gt)
    valid[:, 48:] = False
    assert hd95(pred, gt, valid)[0] == 0
    assert hd95(pred, gt)[0] > 0

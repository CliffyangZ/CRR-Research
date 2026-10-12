import numpy as np

from src.segmentation.augment import augment_once
from src.segmentation.config import Augment

CFG = Augment()
S = 128


def canvas(boxes):
    img = np.full((S, S), 100, np.uint8)
    insts, full = [], np.zeros((S, S), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m = np.zeros((S, S), np.uint8); m[y0:y1, x0:x1] = 1
        insts.append(m); full |= m
    return img, full, insts


def run(boxes, geo, photo=(0.05, 0.05)):
    img, mask, insts = canvas(boxes)
    return augment_once(img, mask, insts, CFG, np.random.default_rng(0),
                        force={"photometric": photo, "geometry": geo}), mask


def test_multi_tooth_legal_transform_applied_and_labels_discrete():
    r, mask = run([(40, 40, 60, 90), (70, 40, 90, 90)], (3.0, 4.0, -3.0))
    assert r.geometry == "applied" and set(np.unique(r.mask)) <= {0, 1}
    assert r.mask.sum() > 0.97 * mask.sum()
    assert r.photometric_applied and r.image.dtype == np.uint8


def test_no_tooth_image_applies():
    r, _ = run([], (3.0, 4.0, 4.0))
    assert r.geometry == "applied" and r.mask.sum() == 0


def test_new_truncation_cancels_whole_geometry_keeps_photometric():
    r, mask = run([(2, 40, 30, 90)], (0.0, -6.0, 0.0))   # tooth near left edge pushed out
    assert r.geometry == "cancelled" and "new_truncation" in r.cancel_reason
    assert np.array_equal(r.mask, mask)                    # mask unchanged, nothing deleted
    assert r.photometric_applied and (r.image != 100).any()


def test_source_truncated_tooth_ok_if_not_cut_further_else_cancel():
    box = [(0, 40, 30, 90)]                                # already touches left border
    ok, _ = run(box, (0.0, 4.0, 0.0))                      # moved inward -> no extra loss
    assert ok.geometry == "applied"
    bad, _ = run(box, (0.0, -4.0, 0.0))                    # moved outward -> extra loss
    assert bad.geometry == "cancelled"


def test_no_geometry_selected():
    r, mask = run([(40, 40, 60, 90)], None, photo=None)
    assert r.geometry == "not_selected" and not r.photometric_applied
    assert np.array_equal(r.mask, mask)


def test_forbidden_ops_cannot_be_enabled():
    import pytest
    with pytest.raises(ValueError):
        Augment(disabled=["flip"])

import numpy as np

from src.detection.augment import augment_once, boxes_inside, transform_boxes
from src.detection.config import PreprocessConfig

import cv2


def _cfg(prob=1.0, rot=5, tr=0.05):
    c = PreprocessConfig()
    c.augment.prob, c.augment.rotate_deg, c.augment.translate_frac = prob, rot, tr
    return c


def _img():
    return np.full((640, 640, 3), 128, np.uint8)


def test_inside_boxes_allow_geometry():
    boxes = np.array([[250, 250, 350, 400]], float)  # far from borders
    for s in range(20):
        r = augment_once(_img(), boxes, _cfg(), np.random.default_rng(s))
        assert r.geometry == "applied"
        assert boxes_inside(r.boxes, 640, 640)
        # boxes follow the same affine as the image
        assert (r.boxes[0, 2] > r.boxes[0, 0]) and (r.boxes[0, 3] > r.boxes[0, 1])


def test_out_of_bounds_cancels_whole_geometry_keeps_photometric():
    boxes = np.array([[300, 300, 640, 640]], float)  # flush with corner, any rot/trans outward breaks
    cancelled = 0
    for s in range(40):
        r = augment_once(_img(), boxes, _cfg(), np.random.default_rng(s))
        if r.geometry == "cancelled":
            cancelled += 1
            assert np.array_equal(r.boxes, boxes)  # restored exactly
            assert r.photometric_applied and "brightness_beta" in r.params
            assert r.cancel_reason == "box_out_of_bounds"
        assert boxes_inside(r.boxes, 640, 640)
    assert cancelled > 0


def test_cancelled_image_has_no_geometric_change():
    img = np.random.RandomState(0).randint(0, 255, (640, 640, 3)).astype(np.uint8)
    boxes = np.array([[0, 0, 640, 640]], float)
    cfg = _cfg()
    cfg.augment.brightness = cfg.augment.contrast = 0.0
    r = augment_once(img, boxes, cfg, np.random.default_rng(1))
    assert r.geometry == "cancelled"
    assert np.array_equal(r.image, img)


def test_originally_truncated_tooth_kept_not_dropped():
    # box already touching border (originally truncated): it is kept, never removed
    boxes = np.array([[0, 100, 80, 300], [300, 100, 400, 300]], float)
    for s in range(30):
        r = augment_once(_img(), boxes, _cfg(), np.random.default_rng(s))
        assert len(r.boxes) == 2
        assert boxes_inside(r.boxes, 640, 640)


def test_seed_reproducible_and_no_scale():
    boxes = np.array([[250, 250, 350, 400]], float)
    a = augment_once(_img(), boxes, _cfg(), np.random.default_rng(7))
    b = augment_once(_img(), boxes, _cfg(), np.random.default_rng(7))
    assert np.array_equal(a.image, b.image) and np.allclose(a.boxes, b.boxes)
    m = cv2.getRotationMatrix2D((320, 320), 5, 1.0)
    t = transform_boxes(boxes, m)
    assert t[0, 2] - t[0, 0] < 100 * 1.2  # rotation envelope only, no scaling

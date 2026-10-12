import numpy as np
import pytest

from src.detection.letterbox import (
    boxes_forward,
    boxes_inverse,
    compute_record,
    letterbox_image,
)


@pytest.mark.parametrize("w,h", [(1200, 825), (825, 1200), (1842, 1324), (640, 640), (2400, 1720), (100, 37)])
def test_record_invariants(w, h):
    r = compute_record(w, h, 640)
    assert r.new_w + r.pad_left + r.pad_right == 640
    assert r.new_h + r.pad_top + r.pad_bottom == 640
    assert max(r.new_w, r.new_h) == 640 or max(w, h) < 1  # long side fills


@pytest.mark.parametrize("w,h", [(1200, 825), (825, 1200), (1842, 1324), (2400, 1720)])
def test_roundtrip_error_within_1px(w, h):
    boxes = np.array([[0, 0, w, h], [10.5, 20.25, 300.7, 400.1], [w - 50, h - 60, w, h]])
    r = compute_record(w, h, 640)
    back = boxes_inverse(boxes_forward(boxes, r), r)
    assert np.abs(back - boxes).max() <= 1.0


def test_image_shape_and_content_preserved():
    img = np.full((825, 1200, 3), 200, np.uint8)
    out, r = letterbox_image(img, 640, 114)
    assert out.shape == (640, 640, 3)
    assert out[0, 0, 0] == 114  # padding
    assert out[320, 320, 0] == 200  # content
    # full-image box maps onto the content region exactly
    fb = boxes_forward(np.array([[0, 0, 1200, 825]]), r)[0]
    assert fb[0] == r.pad_left and fb[2] == r.pad_left + r.new_w


def test_grayscale_2d():
    out, _ = letterbox_image(np.zeros((100, 200), np.uint8), 640, 114)
    assert out.shape == (640, 640)

import numpy as np

from src.segmentation.roi import (build_roi, crop_with_pad, expanded_rect, original_points_to_roi,
                                  roi_points_to_original)
from src.segmentation.roi_config import RoiConfig


def poly(i, x0, y0, x1, y1):
    return {"id": i, "category_id": 1, "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]}


def scene(w=600, h=800):
    img = np.random.default_rng(0).integers(0, 255, (h, w), dtype=np.uint8)
    return img


CFG = RoiConfig()


def test_expansion_is_per_side_and_crop_pads_outside():
    assert expanded_rect([100, 200, 200, 400], 0.10) == [90, 180, 210, 420]
    img = scene()
    crop, outside = crop_with_pad(img, [-20, 10, 50, 30], 114)
    assert crop.shape == (20, 70) and outside == {"left": 20, "top": 0, "right": 0, "bottom": 0}
    assert (crop[:, :20] == 114).all() and (crop[:, 20:] == img[10:30, 0:50]).all()


def test_inside_box_roi_mask_and_label_values():
    t, n = poly(1, 200, 200, 300, 500), poly(2, 330, 200, 430, 500)
    r = build_roi(scene(), [195, 195, 305, 505], t, [n], CFG)
    assert r["image"].shape == (256, 256) and r["mask"].shape == (256, 256)
    assert set(np.unique(r["mask"])) == {0, 1}
    assert r["outside_image_px"] == {"left": 0, "top": 0, "right": 0, "bottom": 0}
    assert not r["target_truncated_by_roi"] and not r["truncated_at_source"]
    assert r["neighbor_in_roi_frac"] == 0.0


def test_neighbour_and_metal_are_background_but_counted():
    t, n = poly(1, 200, 200, 300, 500), poly(2, 290, 200, 400, 500)
    r = build_roi(scene(), [195, 195, 305, 505], t, [n], CFG)
    assert r["neighbor_in_roi_frac"] > 0
    lb = r["letterbox"]
    xs = np.where(r["mask"].any(0))[0]
    # neighbour overlap with the ROI must not appear in the mask: mask ends at the target's right edge
    right_target = (300 - r["expanded_xyxy"][0]) * lb["new_w"] / (r["expanded_xyxy"][2] - r["expanded_xyxy"][0]) + lb["pad_left"]
    assert xs.max() <= right_target + 1


def test_outside_image_padded_and_source_truncation_flagged():
    t = poly(1, 0, 200, 100, 500)                       # touches the left image border
    r = build_roi(scene(), [0, 195, 105, 505], t, [], CFG)
    assert r["outside_image_px"]["left"] > 0 and r["truncated_at_source"]
    assert not r["target_truncated_by_roi"]             # padding gives room on the left


def test_target_cut_by_small_box_is_flagged():
    t = poly(1, 200, 200, 300, 500)
    r = build_roi(scene(), [210, 250, 290, 450], t, [], CFG)   # box far smaller than the tooth
    assert r["target_truncated_by_roi"]


def test_tiny_target_quarantined():
    t = poly(1, 200, 200, 210, 210)
    r = build_roi(scene(), [100, 100, 400, 600], t, [], CFG)
    assert r["quarantine"] == "tiny_target"


def test_control_points_round_trip_within_one_pixel():
    t = poly(1, 200, 200, 300, 500)
    r = build_roi(scene(1200, 825), [195, 195, 305, 505], t, [], CFG)
    pts = np.array([[200, 200], [300, 500], [250.5, 333.3], [180, 480]], dtype=np.float64)
    back = roi_points_to_original(original_points_to_roi(pts, r["letterbox"], r["expanded_xyxy"]),
                                  r["letterbox"], r["expanded_xyxy"])
    assert np.abs(back - pts).max() <= 1.0

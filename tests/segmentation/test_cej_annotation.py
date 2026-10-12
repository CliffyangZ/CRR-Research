import json

import numpy as np
import pytest

from src.segmentation.cej_annotation import (CejSession, CejStore, build_record, cej_line, order_left_right,
                                              render, skip_record)
from src.segmentation.roi import original_points_to_roi

LB = {"orig_w": 300, "orig_h": 400, "scale": 0.64, "new_w": 192, "new_h": 256, "pad_left": 32, "pad_top": 0,
      "pad_right": 32, "pad_bottom": 0, "size": 256}
ROW = {"roi_id": "7_00", "image_id": "7", "subset": "val", "letterbox": LB, "expanded_xyxy": [100, 50, 400, 450]}


def test_points_are_ordered_left_to_right_and_line_derived():
    pts = order_left_right([[180, 90], [70, 100]])
    assert pts[0][0] == 70
    line = cej_line([[70, 100], [180, 90]])
    assert line["midpoint"] == [125.0, 95.0] and abs(line["length_px"] - np.hypot(110, 10)) < 1e-9
    assert -6 < line["angle_deg_vs_x_axis"] < -4


def test_record_stores_roi_and_original_coordinates_consistently():
    rec = build_record([[180, 90], [70, 100]], ROW, "tester")
    assert rec["status"] == "annotated" and rec["points_roi"][0][0] == 70
    back = original_points_to_roi(rec["points_original"], LB, ROW["expanded_xyxy"])
    assert np.abs(back - np.array(rec["points_roi"])).max() < 1e-9
    assert rec["points_original"][0][0] > 100            # inside the expanded crop of the source image


@pytest.mark.parametrize("pts", [[[10, 10]], [[10, 10], [10.5, 10.5]], [[10, 10], [300, 20]], [[-1, 5], [20, 20]]])
def test_invalid_clicks_are_rejected(pts):
    with pytest.raises(ValueError):
        build_record(pts, ROW, "tester")


def test_session_click_undo_and_display_scaling():
    s = CejSession(scale=3.0)
    assert s.click(30, 60) and s.points == [[10.0, 20.0]]
    assert s.click(300, 90) and s.ready and not s.click(100, 100)     # third click ignored
    assert not CejSession(3.0).click(900, 10)                         # outside the canvas
    s.undo()
    assert len(s.points) == 1
    s.reset()
    assert s.points == []


def test_store_is_resumable_atomic_and_version_checked(tmp_path):
    p = tmp_path / "cej.json"
    st = CejStore(p, "roi_x", "tester", "researcher")
    st.set("7_00", build_record([[70, 100], [180, 90]], ROW, "tester"))
    st.set("8_00", skip_record("cej_unclear", "tester", {**ROW, "roi_id": "8_00", "image_id": "8"}))
    doc = json.loads(p.read_text())
    assert doc["counts"] == {"annotated": 1, "skipped": 1} and doc["annotator_role"] == "researcher"
    assert "expert confirmation not asserted" in doc["annotation_status"] and not p.with_suffix(".json.tmp").exists()
    again = CejStore(p, "roi_x", "tester")
    assert again.done() == {"7_00", "8_00"}
    with pytest.raises(ValueError):
        CejStore(p, "roi_other", "tester")
    again.remove("8_00")
    assert json.loads(p.read_text())["counts"]["skipped"] == 0


def test_render_frame_layout_without_a_display():
    img = np.random.default_rng(0).integers(0, 255, (256, 256), dtype=np.uint8)
    mask = np.zeros((256, 256), np.uint8); mask[60:200, 80:170] = 1
    s = CejSession(3.0); s.click(240, 300); s.click(450, 290)
    frame = render(img, mask, s, "7_00 [val] 1/1")
    assert frame.shape == (256 * 3 + 92, 256 * 3, 3) and frame.dtype == np.uint8

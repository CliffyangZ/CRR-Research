from src.segmentation.pairing import bbox_iou, pair_boxes, read_yolo_boxes


def poly(i, x0, y0, x1, y1):
    return {"id": i, "category_id": 1, "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]}


def box(k, x0, y0, x1, y1, flag=""):
    return {"box_index": k, "xyxy": [x0, y0, x1, y1], "flag": flag}


def by_idx(rows):
    return {r["box_index"]: r for r in rows}


def test_exact_and_offset_boxes_pair():
    polys = [poly(1, 100, 100, 200, 300), poly(2, 400, 100, 500, 300)]
    rows, orphans = pair_boxes([box(0, 100, 100, 200, 300), box(1, 410, 110, 510, 310)], polys)
    r = by_idx(rows)
    assert r[0]["status"] == "paired" and r[0]["polygon_ann_id"] == 1 and r[0]["iou"] == 1.0
    assert r[1]["status"] == "paired" and r[1]["polygon_ann_id"] == 2 and r[1]["iou"] > 0.5
    assert orphans == []


def test_adjacent_teeth_ambiguous_second_best_quarantined():
    polys = [poly(1, 100, 100, 200, 300), poly(2, 120, 100, 220, 300)]   # heavily overlapping neighbours
    rows, _ = pair_boxes([box(0, 110, 100, 210, 300)], polys)
    assert rows[0]["status"] == "quarantined" and rows[0]["reason"] == "ambiguous_second_best"
    assert rows[0]["second_iou"] > 0


def test_no_polygon_low_iou_degenerate_and_orphans():
    rows, orphans = pair_boxes([box(0, 0, 0, 50, 50), box(1, 1000, 1000, 1100, 1100), box(2, 5, 5, 5, 5, "degenerate_box")],
                               [poly(7, 900, 900, 1000, 1000)])
    r = by_idx(rows)
    assert r[0]["reason"] == "low_iou" and r[0]["polygon_ann_id"] is None
    assert r[2]["reason"] == "degenerate_box"
    assert r[1]["reason"] in {"low_iou"}
    assert [o["polygon_ann_id"] for o in orphans] == [7]
    rows, _ = pair_boxes([box(0, 0, 0, 50, 50)], [])
    assert rows[0]["reason"] == "no_polygon"


def test_contested_polygon_leader_keeps_only_if_clear():
    polys = [poly(1, 100, 100, 200, 300)]
    rows, _ = pair_boxes([box(0, 100, 100, 200, 300), box(1, 100, 100, 200, 400)], polys)
    r = by_idx(rows)
    assert r[0]["status"] == "paired" and r[1]["reason"] == "polygon_contested"
    # near-tie: both quarantined
    rows, _ = pair_boxes([box(0, 100, 100, 200, 310), box(1, 100, 100, 200, 290)], polys)
    assert all(x["status"] == "quarantined" and x["reason"] == "polygon_contested" for x in rows)


def test_one_polygon_never_shared():
    polys = [poly(1, 100, 100, 200, 300), poly(2, 210, 100, 310, 300)]
    rows, _ = pair_boxes([box(0, 100, 100, 200, 300), box(1, 100, 100, 200, 300), box(2, 210, 100, 310, 300)], polys)
    ids = [r["polygon_ann_id"] for r in rows if r["status"] == "paired"]
    assert len(ids) == len(set(ids))


def test_yolo_conversion_and_flags(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n0 0.95 0.5 0.3 0.4\n0 0.5 0.5 0.0 0.4\n")
    b = read_yolo_boxes(p, 1000, 500)
    assert b[0]["xyxy"] == [400.0, 150.0, 600.0, 350.0] and b[0]["flag"] == ""
    assert b[1]["flag"] == "box_out_of_image" and b[2]["flag"] == "degenerate_box"
    assert abs(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) - 25 / 175) < 1e-12

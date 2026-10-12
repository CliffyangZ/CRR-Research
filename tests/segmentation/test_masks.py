import numpy as np

from src.segmentation.masks import instance_masks, polygon_issue, rasterize_union

TOOTH, METAL = 1, 2


def sq(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def ann(i, cat, poly):
    return {"id": i, "category_id": cat, "segmentation": [poly]}


def test_multi_tooth_union():
    a = [ann(1, TOOTH, sq(5, 5, 15, 15)), ann(2, TOOTH, sq(12, 12, 25, 25))]
    m, issues = rasterize_union(a, {TOOTH}, 40, 40)
    assert not issues and set(np.unique(m)) == {0, 1}
    assert m[10, 10] == 1 and m[20, 20] == 1 and m[30, 30] == 0
    assert m[13, 13] == 1  # overlap counted once, still foreground


def test_metal_overlap_keeps_tooth_and_metal_alone_is_background():
    a = [ann(1, TOOTH, sq(5, 5, 20, 20)), ann(2, METAL, sq(10, 10, 30, 30))]
    m, _ = rasterize_union(a, {TOOTH}, 40, 40)
    assert m[12, 12] == 1      # tooth ∩ metal stays tooth
    assert m[25, 25] == 0      # metal only -> background


def test_invalid_polygons_reported():
    assert polygon_issue([1, 1, 2, 2], 40, 40).startswith("degenerate")
    assert polygon_issue(sq(0, 0, 0, 0), 40, 40).startswith("degenerate")
    assert polygon_issue([0, 0, 10, 0, 5, 0.0], 40, 40).startswith("degenerate")  # collinear
    assert polygon_issue(sq(10, 10, 100, 100), 40, 40) == "out_of_bounds"
    a = [ann(1, TOOTH, [1, 1, 2, 2]), ann(2, TOOTH, sq(5, 5, 10, 10))]
    m, issues = rasterize_union(a, {TOOTH}, 40, 40)
    assert len(issues) == 1 and m.sum() > 0


def test_instance_masks_only_tooth():
    a = [ann(1, TOOTH, sq(5, 5, 10, 10)), ann(2, METAL, sq(20, 20, 30, 30)), ann(3, TOOTH, sq(15, 15, 18, 18))]
    assert len(instance_masks(a, {TOOTH}, 40, 40)) == 2


def test_inventory_excludes_with_reasons(tmp_path):
    import cv2, json
    from src.segmentation.config import SegConfig
    from src.segmentation.inventory import build_inventory

    for s in ("train", "val", "test"):
        (tmp_path / s / "images").mkdir(parents=True)
        (tmp_path / s / "annotations").mkdir()
        coco = {"categories": [{"id": 1, "name": "tooth"}, {"id": 2, "name": "Metal"}], "images": [], "annotations": []}
        if s == "train":
            for k, anns in enumerate([[(1, sq(5, 5, 20, 20))], [], [(1, [1, 1, 2, 2])], [(2, sq(5, 5, 20, 20))]]):
                name = f"{k}.jpg"
                cv2.imwrite(str(tmp_path / s / "images" / name), (np.random.rand(40, 40) * 255).astype(np.uint8))
                coco["images"].append({"id": k, "file_name": name, "width": 40, "height": 40})
                for j, (c, p) in enumerate(anns):
                    coco["annotations"].append({"id": k * 10 + j, "image_id": k, "category_id": c, "segmentation": [p]})
        (tmp_path / s / "annotations" / "instances_default.json").write_text(json.dumps(coco))
    items, _ = build_inventory(tmp_path, SegConfig())
    by = {i["image_id"]: i for i in items}
    assert by["0"]["status"] == "included"
    assert by["1"]["status"] == "excluded" and "empty_tooth_annotation_unconfirmed" in by["1"]["exclude_reason"]
    assert "invalid_polygon" in by["2"]["exclude_reason"]
    assert "empty_tooth_annotation_unconfirmed" in by["3"]["exclude_reason"]   # Metal-only is not tooth

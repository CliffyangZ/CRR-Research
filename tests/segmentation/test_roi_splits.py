import json

from research.data.prepare_tooth_roi_dataset import main

from ._roi_world import make_world, write_cfg

TOOTH = (50, 60, 120, 240)


def run(tmp_path, images, *extra):
    det, seg = make_world(tmp_path, images)
    cfg = write_cfg(tmp_path, det, seg)
    rc = main(["--config", str(cfg), "--output", str(tmp_path / "out"), "--report", str(tmp_path / "rep"), *extra])
    d = next((tmp_path / "out").glob("roi_*"))
    return rc, d


def img(i, subset, seed, **kw):
    return {"id": str(i), "subset": subset, "seed": seed, "boxes": [TOOTH, (140, 10, 190, 60)], "polys": [TOOTH], **kw}


def test_roi_inherits_image_subset_and_unmatched_box_quarantined(tmp_path):
    rc, d = run(tmp_path, [img(1, "train", 1), img(2, "val", 2), img(3, "test", 3)])
    assert rc == 0
    rois = json.loads((d / "roi_index.json").read_text())
    assert {r["image_id"]: r["subset"] for r in rois} == {"1": "train", "2": "val", "3": "test"}
    q = json.loads((d / "quarantine.json").read_text())
    assert len(q["boxes"]) == 3 and all(b["reason"] == "low_iou" for b in q["boxes"])
    m = json.loads((d / "dataset_manifest.json").read_text())
    assert m["status"] == "succeeded" and m["split"]["leakage_check"]["cross_split_groups"] == 0
    assert m["annotation_review"]["boxes"]["status"] == "unknown"
    assert (tmp_path / "rep" / m["data_version"] / "README.md").exists()


def test_near_duplicate_across_subsets_is_regrouped_and_old_split_saved(tmp_path):
    rc, d = run(tmp_path, [img(1, "train", 7), img(2, "test", 7), img(3, "val", 3), img(4, "train", 4), img(5, "train", 5)])
    assert rc == 0
    sp = json.loads((d / "splits.json").read_text())
    assert sp["assignment_source"] == "regrouped" and sp["previous_assignments"]["2"] == "test"
    assert sp["assignments"]["1"] == sp["assignments"]["2"]
    assert sp["leakage_check"]["cross_split_groups"] == 0 and sp["frozen"]
    rois = json.loads((d / "roi_index.json").read_text())
    by_img = {r["image_id"]: r["subset"] for r in rois}
    assert by_img["1"] == by_img["2"]


def test_subset_mismatch_quarantines_all_boxes_of_that_image(tmp_path):
    rc, d = run(tmp_path, [img(1, "train", 1), img(2, "val", 2, det_subset="test")])
    assert rc == 0
    q = json.loads((d / "quarantine.json").read_text())
    assert {b["reason"] for b in q["boxes"] if b["image_id"] == "2"} == {"subset_mismatch"}
    assert {r["image_id"] for r in json.loads((d / "roi_index.json").read_text())} == {"1"}


def test_existing_output_is_not_overwritten(tmp_path):
    rc, d = run(tmp_path, [img(1, "train", 1)])
    assert rc == 0
    cfg = tmp_path / "cfg.yaml"
    assert main(["--config", str(cfg), "--output", str(tmp_path / "out"), "--report", str(tmp_path / "rep")]) == 2


def test_output_is_laid_out_per_subset(tmp_path):
    rc, d = run(tmp_path, [img(1, "train", 1), img(2, "val", 2), img(3, "test", 3)])
    assert rc == 0
    for sub, iid in (("train", "1"), ("val", "2"), ("test", "3")):
        assert (d / sub / "images" / f"{iid}_00.png").exists() and (d / sub / "masks" / f"{iid}_00.png").exists()
        idx = json.loads((d / sub / "roi_index.json").read_text())
        assert [r["roi_id"] for r in idx] == [f"{iid}_00"] and all(r["subset"] == sub for r in idx)
    import cv2
    vis = cv2.imread(str(d / "train" / "masks_vis" / "1_00.png"), 0)
    lab = cv2.imread(str(d / "train" / "masks" / "1_00.png"), 0)
    assert set(vis.flatten()) == {0, 255} and ((vis > 0) == (lab > 0)).all() and set(lab.flatten()) == {0, 1}
    assert not (d / "images").exists()
    rep = next((tmp_path / "rep").glob("roi_*"))
    assert (rep / "boxes_on_seg" / "train" / "1.jpg").exists()

import numpy as np
import pytest

from src.detection.evaluate import choose_threshold, evaluate, iou_matrix, match_image, prf_at

B = lambda *r: np.array(r, float).reshape(-1, 4)


def img(gt, pred, sc):
    return {"gt": B(*gt) if len(gt) else np.zeros((0, 4)), "pred": B(*pred) if len(pred) else np.zeros((0, 4)),
            "scores": np.array(sc, float)}


GT = [(0, 0, 100, 100), (200, 200, 300, 300)]


def test_perfect():
    r = evaluate({"a": img(GT, GT, [0.9, 0.8])}, 0.5)
    assert r["AP"] == pytest.approx(1.0) and r["AP50"] == 1.0
    assert (r["precision"], r["recall"], r["f1"]) == (1, 1, 1)
    assert r["errors"] == {"missed": 0, "false_positive": 0, "duplicate": 0, "localization": 0}


def test_all_missed():
    r = evaluate({"a": img(GT, [], [])}, 0.5)
    assert r["AP"] == 0 and r["recall"] == 0 and r["errors"]["missed"] == 2
    assert r["precision"] is None  # zero denominator -> undefined, not 0
    assert "precision" in r["undefined_reasons"]


def test_no_gt_recall_undefined_and_ap_none():
    r = evaluate({"a": img([], [(0, 0, 10, 10)], [0.9])}, 0.5)
    assert r["recall"] is None and r["AP"] is None and "undefined_reason" in r
    assert r["errors"]["false_positive"] == 1


def test_duplicate_and_localization_and_fp():
    pred = [(0, 0, 100, 100), (2, 2, 102, 102), (200, 240, 300, 340), (500, 500, 600, 600)]
    # 2nd duplicates GT0; 3rd: IoU with GT1 = 60*100/(2*10000-6000)=0.43 -> localization; 4th far -> FP
    r = evaluate({"a": img(GT, pred, [0.9, 0.8, 0.7, 0.6])}, 0.0)
    e = r["errors"]
    assert e == {"missed": 1, "false_positive": 1, "duplicate": 1, "localization": 1}
    assert r["tp"] == 1 and r["fp"] == 3


def test_each_gt_matched_once():
    m = match_image(B(*GT[:1]), B((0, 0, 100, 100), (1, 1, 99, 99)), np.array([0.9, 0.8]), 0.5)
    assert m["tp"].tolist() == [True, False] and m["kind"][1] == "duplicate"


def test_threshold_ties_pick_higher():
    d = {"a": img(GT, GT, [0.6, 0.9])}
    t = choose_threshold(d)
    assert t["tau"] == 0.9 or t["val_f1"] == 1.0
    d2 = {"a": img(GT[:1], GT[:1], [0.5])}
    assert choose_threshold(d2)["tau"] == 0.5
    d3 = {"a": img(GT, [(0, 0, 100, 100), (200, 200, 300, 300)], [0.9, 0.9])}
    assert choose_threshold(d3)["tau"] == 0.9
    assert choose_threshold({"a": img(GT, [], [])})["tau"] is None


def test_matches_pycocotools():
    pycoco = pytest.importorskip("pycocotools.coco")
    from pycocotools.cocoeval import COCOeval
    import contextlib, io

    rng = np.random.RandomState(0)
    data, images, anns, dets = {}, [], [], []
    aid = 1
    for i in range(15):
        gt = []
        for _ in range(rng.randint(1, 4)):
            x, y = rng.randint(0, 400, 2); w, h = rng.randint(40, 150, 2)
            gt.append((x, y, x + w, y + h))
        pred, sc = [], []
        for g in gt:
            if rng.rand() < 0.8:
                j = rng.randint(-12, 12, 4)
                pred.append(tuple(np.array(g) + j)); sc.append(rng.rand())
        for _ in range(rng.randint(0, 3)):
            x, y = rng.randint(0, 400, 2); pred.append((x, y, x + 60, y + 60)); sc.append(rng.rand() * 0.5)
        data[i] = img(gt, pred, sc)
        images.append({"id": i, "width": 600, "height": 600})
        for g in gt:
            anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [g[0], g[1], g[2] - g[0], g[3] - g[1]],
                         "area": (g[2] - g[0]) * (g[3] - g[1]), "iscrowd": 0}); aid += 1
        for p, s in zip(pred, sc):
            dets.append({"image_id": i, "category_id": 1, "bbox": [float(p[0]), float(p[1]), float(p[2] - p[0]), float(p[3] - p[1])], "score": float(s)})
    with contextlib.redirect_stdout(io.StringIO()):
        gtc = pycoco.COCO(); gtc.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1}]}; gtc.createIndex()
        dtc = gtc.loadRes(dets)
        ev = COCOeval(gtc, dtc, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    mine = evaluate(data, 0.3)
    assert mine["AP"] == pytest.approx(ev.stats[0], abs=0.02)
    assert mine["AP50"] == pytest.approx(ev.stats[1], abs=0.02)

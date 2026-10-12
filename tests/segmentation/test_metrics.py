import numpy as np

from src.segmentation.metrics import aggregate, evaluate_pair


def test_known_pair():
    gt = np.zeros((10, 10), np.uint8); gt[:5] = 1       # 50 px
    pr = np.zeros((10, 10), np.uint8); pr[:, :5] = 1    # 50 px, overlap 25
    r = evaluate_pair(pr, gt)
    assert r["counts"] == {"tp": 25, "fp": 25, "fn": 25, "tn": 25}
    assert r["metrics"]["dice"] == 0.5
    assert abs(r["metrics"]["iou"] - 25 / 75) < 1e-12
    assert r["metrics"]["precision"] == 0.5 and r["metrics"]["recall"] == 0.5


def test_zero_denominators_undefined():
    z = np.zeros((4, 4), np.uint8)
    r = evaluate_pair(z, z)
    assert r["metrics"]["dice"] is None and r["undefined"]["dice"] == "gt_empty_and_pred_empty"
    one = np.ones((4, 4), np.uint8)
    r = evaluate_pair(z, one)
    assert r["metrics"]["precision"] is None and r["metrics"]["recall"] == 0.0
    r = evaluate_pair(one, z)
    assert r["metrics"]["recall"] is None and r["metrics"]["dice"] == 0.0


def test_aggregate_is_pixel_accumulated_not_mean_of_images():
    big_gt = np.ones((10, 10), np.uint8); big_pr = big_gt.copy()          # dice 1, 100 px
    small_gt = np.zeros((10, 10), np.uint8); small_gt[0, 0] = 1
    small_pr = np.zeros((10, 10), np.uint8)                               # dice 0, 1 px
    empty = np.zeros((10, 10), np.uint8)
    empty_pr = np.zeros((10, 10), np.uint8); empty_pr[5, 5] = 1           # all-bg gt, 1 FP
    rows = [evaluate_pair(p, g) for p, g in [(big_pr, big_gt), (small_pr, small_gt), (empty_pr, empty)]]
    a = aggregate(rows)
    assert abs(a["aggregate"]["dice"] - 200 / 202) < 1e-12   # TP=100, FP=1, FN=1
    assert abs(a["per_image_mean"]["dice"] - 1 / 3) < 1e-12 and a["per_image_n_valid"]["dice"] == 3
    assert a["n_all_background_gt"] == 1 and a["fp_pixels_on_all_background"] == 1
    assert a["n_valid_pixels"] == 300

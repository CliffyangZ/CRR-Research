import numpy as np

from src.segmentation.transform import (letterbox_pair, points_forward, points_inverse,
                                        restore_to_original)


def test_aspect_ratios_and_label_values():
    for w, h in [(1200, 825), (1720, 2400), (300, 300)]:
        img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        mask = np.zeros((h, w), np.uint8)
        mask[h // 4 : h // 2, w // 4 : w // 2] = 1
        li, lm, rec = letterbox_pair(img, mask, 512)
        assert li.shape == lm.shape == (512, 512)
        assert set(np.unique(lm)) <= {0, 1}
        assert lm[: rec.pad_top].sum() == 0 and lm[:, : rec.pad_left].sum() == 0
        assert (li[: rec.pad_top] == 114).all() if rec.pad_top else True


def test_control_point_round_trip_within_one_pixel():
    for w, h in [(1200, 825), (1720, 2400)]:
        _, _, rec = letterbox_pair(np.zeros((h, w), np.uint8), None, 512)
        pts = np.array([[0, 0], [w - 1, h - 1], [w / 2, h / 3], [17.5, h - 3]])
        back = points_inverse(points_forward(pts, rec), rec)
        assert np.abs(back - pts).max() <= 1.0


def test_restore_shape_and_mask_overlap():
    h, w = 825, 1200
    mask = np.zeros((h, w), np.uint8)
    mask[200:600, 300:900] = 1
    _, lm, rec = letterbox_pair(np.zeros((h, w), np.uint8), mask, 512)
    back = restore_to_original(lm.astype(np.float32), rec)
    assert back.shape == (h, w)
    pred = back > 0.5
    inter = (pred & (mask > 0)).sum()
    assert 2 * inter / (pred.sum() + mask.sum()) > 0.98

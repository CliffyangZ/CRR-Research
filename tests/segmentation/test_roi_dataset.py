import json

import cv2
import numpy as np

from research.data.prepare_tooth_roi_dataset import main
from src.segmentation.roi_config import load_roi_config
from src.segmentation.roi_dataset import RoiSegDataset, collate

from ._roi_world import make_world, write_cfg

TOOTH = (50, 60, 120, 240)


def build(tmp_path, n=3):
    imgs = [{"id": str(i), "subset": s, "seed": i, "boxes": [TOOTH], "polys": [TOOTH]}
            for i, s in enumerate((["train"] * n) + ["val", "test"])]
    det, seg = make_world(tmp_path, imgs)
    cfgp = write_cfg(tmp_path, det, seg)
    assert main(["--config", str(cfgp), "--output", str(tmp_path / "out"), "--report", str(tmp_path / "rep")]) == 0
    return load_roi_config(cfgp), next((tmp_path / "out").glob("roi_*"))


def test_only_train_is_augmented(tmp_path):
    cfg, d = build(tmp_path)
    for split in ("val", "test"):
        ds = RoiSegDataset(d, split, cfg)
        assert not ds.augment and all(ds[i]["aug"]["geometry"] == "not_selected" and not ds[i]["aug"]["photometric"]
                                      for i in range(len(ds)))
    import pytest
    with pytest.raises(ValueError):
        RoiSegDataset(d, "val", cfg, augment=True)
    tr = RoiSegDataset(d, "train", cfg, seed=1)
    seen = set()
    for ep in range(12):
        tr.epoch = ep
        for i in range(len(tr)):
            s = tr[i]
            assert set(np.unique(s["mask"].numpy())) <= {0.0, 1.0} and s["image"].shape == (1, 256, 256)
            seen.add(s["aug"]["geometry"])
    assert {"applied", "not_selected"} <= seen


def test_deterministic_given_seed_epoch_idx_and_collate(tmp_path):
    cfg, d = build(tmp_path)
    a, b = RoiSegDataset(d, "train", cfg, seed=3), RoiSegDataset(d, "train", cfg, seed=3)
    a.epoch = b.epoch = 4
    assert np.array_equal(a[1]["image"].numpy(), b[1]["image"].numpy())
    batch = collate([a[0], a[1]])
    assert batch["image"].shape == (2, 1, 256, 256) and len(batch["rec"]) == 2


def test_new_truncation_cancels_geometry_keeps_photometric(tmp_path):
    cfg, d = build(tmp_path)
    ds = RoiSegDataset(d, "train", cfg)
    row = ds.rows[0]
    mask = np.zeros((256, 256), np.uint8); mask[60:200, 0:40] = 1       # tooth already against the left edge
    cv2.imwrite(str(d / row["files"]["mask"]), mask)
    cfg.augment.prob = 1.0
    cfg.augment.translate_frac = 0.05
    from src.segmentation.augment import augment_once
    img = cv2.imread(str(d / row["files"]["image"]), cv2.IMREAD_GRAYSCALE)
    res = augment_once(img, mask, [mask], cfg.augment, np.random.default_rng(0), 114,
                       force={"photometric": (0.05, 0.05), "geometry": (0.0, -8.0, 0.0)})
    assert res.geometry == "cancelled" and res.photometric_applied and np.array_equal(res.mask, mask)


def test_rebuild_is_deterministic(tmp_path):
    outs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        outs.append(build(tmp_path / name)[1])
    d1, d2 = outs
    assert (d1 / "roi_index.json").read_text().replace(str(tmp_path / "a"), "") == \
        (d2 / "roi_index.json").read_text().replace(str(tmp_path / "b"), "")
    assert json.loads((d1 / "pairs.json").read_text()) == json.loads((d2 / "pairs.json").read_text())
    assert (d1 / "train" / "images" / "0_00.png").read_bytes() == (d2 / "train" / "images" / "0_00.png").read_bytes()


def test_masks_stored_as_0_255_are_binarised(tmp_path):
    cfg, d = build(tmp_path)
    ds = RoiSegDataset(d, "val", cfg)
    f = d / ds.rows[0]["files"]["mask"]
    m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(f), m * 255)
    assert set(np.unique(ds[0]["mask"].numpy())) <= {0.0, 1.0} and ds[0]["mask"].sum() == (m > 0).sum()

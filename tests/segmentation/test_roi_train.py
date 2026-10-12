import cv2
import numpy as np
import yaml
import torch

from research.data.prepare_tooth_roi_dataset import main
from src.segmentation.roi_config import RoiTrainConfig
from src.segmentation.roi_dataset import RoiSegDataset
from src.segmentation.roi_train import GroundTruth, evaluate_roi_split

from ._roi_world import make_world, write_cfg

TOOTH = (0, 60, 70, 240)          # touches the left image border, so the expanded crop leaves the image


def _build(tmp_path):
    imgs = [{"id": str(i), "subset": s, "seed": i, "boxes": [TOOTH], "polys": [TOOTH]}
            for i, s in enumerate(["train", "train", "val", "test"])]
    det, seg = make_world(tmp_path, imgs)
    cfgp = write_cfg(tmp_path, det, seg)
    assert main(["--config", str(cfgp), "--output", str(tmp_path / "out"), "--report", str(tmp_path / "rep")]) == 0
    cfg = RoiTrainConfig(**yaml.safe_load(open(cfgp)))
    return cfg, next((tmp_path / "out").glob("roi_*")), seg


def test_ground_truth_is_source_resolution_and_valid_excludes_outside_image(tmp_path):
    cfg, d, seg = _build(tmp_path)
    ds = RoiSegDataset(d, "val", cfg)
    row = ds.rows[0]
    gt, valid = GroundTruth(seg).crop(row, "val")
    x0, y0, x1, y1 = row["expanded_xyxy"]
    assert gt.shape == valid.shape == (y1 - y0, x1 - x0)
    assert x0 < 0 and not valid[:, : -x0].any() and valid[:, -x0:].all()      # left padding excluded
    assert gt[~valid].sum() == 0 and gt.sum() > 0


def test_perfect_logits_give_dice_one_in_roi_frame(tmp_path):
    cfg, d, _ = _build(tmp_path)
    ds = RoiSegDataset(d, "val", cfg)

    class Oracle(torch.nn.Module):          # returns the (binarised) label as a strongly signed logit
        def __init__(self, ds):
            super().__init__()
            self.masks = {r["roi_id"]: torch.from_numpy((cv2.imread(str(ds.dir / r["files"]["mask"]), 0) > 0).astype(np.float32)) for r in ds.rows}
            self.ds, self.i = ds, 0

        def forward(self, x):
            out = []
            for _ in range(x.shape[0]):
                out.append(self.masks[self.ds.rows[self.i]["roi_id"]] * 20 - 10)
                self.i += 1
            return torch.stack(out)[:, None].to(x.device)

    res = evaluate_roi_split(Oracle(ds), ds, torch.device("cpu"), 0.5, amp=False)
    assert res["aggregate"]["dice"] > 0.97 and res["n_images"] == len(ds)

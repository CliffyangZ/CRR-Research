import pytest

from src.segmentation.roi_config import RoiConfig, load_roi_config


def test_repo_config_loads_and_is_clahe_only():
    cfg = load_roi_config("configs/segmentation/roi_dataset_v1.yaml")
    assert set(cfg.preprocess) == {"clahe", "input_mapping"}
    assert cfg.roi.size == 256 and cfg.roi.expand_frac_per_side == 0.10
    assert cfg.pairing.min_iou == 0.5


@pytest.mark.parametrize("key", ["nlm", "usm"])
def test_nlm_usm_rejected(key):
    with pytest.raises(ValueError):
        RoiConfig(preprocess={"clahe": {}, key: {}})


def test_forbidden_aug_cannot_be_enabled():
    with pytest.raises(ValueError):
        RoiConfig(augment={"disabled": ["flip"]})


def test_hash_stable_and_sensitive():
    a, b = RoiConfig(), RoiConfig()
    assert a.hash() == b.hash()
    assert RoiConfig(pairing={"min_iou": 0.6}).hash() != a.hash()

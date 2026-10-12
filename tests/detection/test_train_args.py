import pytest

from src.detection.train_args import DISABLED_AUG, assert_no_builtin_aug, build_train_args
from src.detection.models import candidate


def test_all_builtin_aug_disabled():
    a = build_train_args("d.yaml", project="p", name="n")
    assert_no_builtin_aug(a)
    for k, v in DISABLED_AUG.items():
        assert a[k] == v
    for k in ("mosaic", "mixup", "cutmix", "copy_paste", "fliplr", "flipud", "scale", "shear",
              "perspective", "degrees", "translate", "hsv_h", "hsv_s", "hsv_v", "erasing"):
        assert a[k] == 0


def test_tampering_detected():
    a = build_train_args("d.yaml", project="p", name="n")
    a["mosaic"] = 1.0
    with pytest.raises(ValueError):
        assert_no_builtin_aug(a)


def test_smoke_and_candidates():
    assert build_train_args("d", project="p", name="n", smoke=True)["epochs"] == 2
    assert candidate("v26")["weights_name"] == "yolo26s.pt"
    with pytest.raises(ValueError):
        candidate("v9")

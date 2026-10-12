"""Unified Ultralytics training arguments with ALL built-in augmentation disabled.

Conservative augmentation (brightness/contrast, small rotation/translation with the
cancel-on-out-of-bounds rule) is done offline in dataset_export.augment; the training
framework must therefore not augment anything on its own.
"""

from __future__ import annotations

DISABLED_AUG = {
    "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "fliplr": 0.0, "flipud": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
    "degrees": 0.0, "translate": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "erasing": 0.0, "close_mosaic": 0, "multi_scale": 0.0,
}


def build_train_args(
    data_yaml: str, *, imgsz: int = 640, epochs: int = 100, patience: int = 30,
    batch: int = 8, seed: int = 0, device: str | int = 0, workers: int = 4,
    project: str, name: str, smoke: bool = False,
) -> dict:
    args = dict(
        data=data_yaml, imgsz=imgsz, epochs=2 if smoke else epochs, patience=patience,
        batch=batch, seed=seed, deterministic=True, device=device, workers=workers,
        cache=False, rect=False, single_cls=True, project=project, name=name,
        exist_ok=False, plots=True, val=True, pretrained=True, amp=True,
    )
    args.update(DISABLED_AUG)
    return args


def assert_no_builtin_aug(args: dict) -> None:
    bad = {k: args.get(k) for k, v in DISABLED_AUG.items() if args.get(k) != v}
    if bad:
        raise ValueError(f"built-in augmentation not disabled: {bad}")
    try:
        import albumentations  # noqa: F401

        raise RuntimeError("albumentations is installed: Ultralytics would add random Albumentations transforms; uninstall it")
    except ImportError:
        pass

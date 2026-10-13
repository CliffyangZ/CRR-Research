import json

import pytest
import torch

from research.models.unet import UNet
from src.segmentation.roi_train import build_roi_datasets, evaluate_roi_split
from src.segmentation.train import run_training

from .test_roi_train import _build


def _run(tmp_path, monkeypatch, **kw):
    cfg, d, _ = _build(tmp_path)
    monkeypatch.chdir(tmp_path)               # weights go to ./models/segmentation/<run_id>
    cfg.train.batch_size = kw.pop("batch_size", 1)
    cfg.train.amp = False
    torch.manual_seed(0)
    out = run_training(UNet(base_channels=4, depth=2), cfg, d, tmp_path / "runs", "cpu", True, ["test"], 1, None,
                       build_datasets=build_roi_datasets, evaluate=evaluate_roi_split, run_prefix="t", **kw)
    return json.loads((out / "run_manifest.json").read_text())


def test_new_fields_and_exploratory_without_protocol(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, candidate="unet")
    assert m["status"] == "succeeded" and m["exploratory"] is True
    assert m["candidate"] == "unet" and m["protocol_hash"] is None
    assert m["train_seconds"] >= 0 and "peak_gpu_mem_mb" in m
    assert m["batch"] == {"size": 1, "accum": 1, "effective": 1, "override_reason": None}
    assert "single seed only" not in m["nondeterminism_note"]


def test_protocol_run_is_not_exploratory(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, protocol={"version": "v1", "hash": "abc"})
    assert m["exploratory"] is False and m["protocol_version"] == "v1" and m["protocol_hash"] == "abc"


def test_grad_accum_halves_optimizer_steps(tmp_path, monkeypatch):
    steps = []
    orig = torch.optim.AdamW.step

    def counting(self, *a, **k):
        steps.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting)
    _run(tmp_path, monkeypatch, grad_accum=1)
    n1 = len(steps)
    steps.clear()
    (tmp_path / "b").mkdir()
    m = _run(tmp_path / "b", monkeypatch, grad_accum=2, override_reason="OOM")
    assert n1 == 4 and len(steps) == 2           # 2 train ROIs x 2 smoke epochs
    assert m["batch"]["effective"] == 2 and m["batch"]["override_reason"] == "OOM"


def test_failure_is_recorded_and_not_overwritten(tmp_path, monkeypatch):
    cfg, d, _ = _build(tmp_path)
    monkeypatch.chdir(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("no data")

    with pytest.raises(RuntimeError):
        run_training(UNet(base_channels=4, depth=2), cfg, d, tmp_path / "runs", "cpu", True, ["test"], 1, None,
                     build_datasets=boom, evaluate=evaluate_roi_split, run_prefix="t")
    (run,) = (tmp_path / "runs").iterdir()
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["status"] == "failed" and "no data" in m["failure_reason"]

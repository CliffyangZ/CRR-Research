import pytest
import torch

from research.models.attention_unet import AttentionGate
from research.models.registry import MODELS, build_model
from src.segmentation.config import Model

BASELINE_PARAMS = 7_762_465


def _cfg(name, base=8):
    return Model(name=name, base_channels=base)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_forward_backward_shape(name):
    m = build_model(_cfg(name))
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    y.mean().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_unetpp_node_count_and_dense_inputs():
    m = build_model(_cfg("unetpp"))
    assert len(m.nodes) == 4 + 3 + 2 + 1          # X[i][j], j >= 1
    # node X[0][4] fuses 4 earlier nodes + upsampled one
    assert m.nodes["0_4"][0].in_channels == 8 * 4 + 8


def test_attention_gate_range_and_params():
    g = AttentionGate(8, 8, 4)
    g.eval()
    out = g(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    a = g.last_alpha
    assert a.min() >= 0 and a.max() <= 1 and out.shape == (1, 8, 16, 16)
    m = build_model(_cfg("attention_unet"))
    assert sum(isinstance(x, AttentionGate) for x in m.modules()) == 4


def test_param_counts_reported_and_same_seed_same_weights():
    counts = {}
    for name in MODELS:
        torch.manual_seed(0)
        a = build_model(_cfg(name, 32))
        counts[name] = sum(p.numel() for p in a.parameters())
        torch.manual_seed(0)
        b = build_model(_cfg(name, 32))
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    print(counts)
    assert counts["unet"] == BASELINE_PARAMS
    assert counts["unetpp"] > counts["unet"] and counts["attention_unet"] > counts["unet"]


def test_unknown_model_rejected():
    class C:
        name = "nope"
    with pytest.raises(ValueError):
        build_model(C())

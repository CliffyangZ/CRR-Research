import imagehash
import numpy as np

from src.detection.splits import assign_groups, build_groups, leakage_check, make_split


def _hash(seed, flip=0):
    bits = np.random.RandomState(seed).rand(8, 8) > 0.5
    bits = bits.copy()
    for k in range(flip):
        bits.flat[k] = ~bits.flat[k]
    return str(imagehash.ImageHash(bits))


def _data():
    h = {}
    for i in range(30):
        h[f"a{i}"] = _hash(i)
    h["a0_dup"] = _hash(0, flip=2)  # near duplicate of a0
    h["a1_dup"] = _hash(1, flip=1)
    return h


def test_near_duplicates_grouped_and_never_cross():
    h = _data()
    g = build_groups(h, 6)
    assert g["a0"] == g["a0_dup"] and g["a1"] == g["a1_dup"]
    for seed in range(10):
        a = assign_groups(g, seed=seed)
        assert leakage_check(a, g)["cross_split_groups"] == 0
        assert set(a.values()) == {"train", "val", "test"}


def test_existing_split_with_leak_is_regrouped():
    h = _data()
    existing = {k: "train" for k in h}
    existing["a0_dup"] = "test"
    r = make_split(h, existing, seed=42)
    assert r["assignment_source"] == "regrouped"
    assert r["leakage_check"]["cross_split_groups"] == 0
    assert r["assignments"]["a0"] == r["assignments"]["a0_dup"]


def test_clean_existing_split_kept_and_risk_recorded():
    h = _data()
    g = build_groups(h)
    existing = assign_groups(g, seed=1)
    r = make_split(h, existing)
    assert r["assignment_source"] == "existing"
    assert "NOT excluded" in r["residual_risk"]

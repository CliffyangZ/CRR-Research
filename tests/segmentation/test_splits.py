import json
from pathlib import Path

from src.detection.splits import build_groups, leakage_check, make_split


def h(bits_on: int) -> str:
    return f"{(1 << bits_on) - 1:016x}"


def test_near_duplicates_grouped_and_regrouped_without_leakage():
    ph = {"a": h(0), "b": h(2), "c": h(40), "d": h(60), "e": h(20), "f": h(50)}  # a~b only
    groups = build_groups(ph, phash_dist=6)
    assert groups["a"] == groups["b"] and groups["c"] != groups["a"]
    leaking = {"a": "train", "b": "test", "c": "train", "d": "val", "e": "val", "f": "test"}
    assert leakage_check(leaking, groups)["cross_split_groups"] == 1
    res = make_split(ph, leaking, seed=1, phash_dist=6)
    assert res["assignment_source"] == "regrouped"
    assert res["leakage_check"]["cross_split_groups"] == 0
    assert "NOT excluded" in res["residual_risk"]


def test_clean_existing_split_kept():
    ph = {"a": h(0), "b": h(2), "c": h(40), "d": h(60)}
    existing = {"a": "train", "b": "train", "c": "val", "d": "test"}
    res = make_split(ph, existing, phash_dist=6)
    assert res["assignment_source"] == "existing" and res["assignments"] == existing

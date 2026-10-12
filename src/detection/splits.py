"""Grouped train/val/test split with near-duplicate grouping and leakage checks.

No patient identifiers exist in the data, so groups are a *surrogate*: images whose
perceptual hashes are within `phash_dist` bits are tied together (union-find). The
residual risk (same patient, visually different images) is recorded, not removed.
"""

from __future__ import annotations

import random
from collections import defaultdict

import imagehash

SPLITS = ("train", "val", "test")


def build_groups(phashes: dict[str, str], phash_dist: int = 6) -> dict[str, str]:
    """image_id -> group_id (smallest member id in the group)."""
    ids = sorted(phashes)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    hashes = {i: imagehash.hex_to_hash(phashes[i]) for i in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if hashes[ids[a]] - hashes[ids[b]] <= phash_dist:
                ra, rb = find(ids[a]), find(ids[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}


def leakage_check(assignments: dict[str, str], groups: dict[str, str]) -> dict:
    by_group: dict[str, set] = defaultdict(set)
    for i, s in assignments.items():
        by_group[groups.get(i, i)].add(s)
    crossing = {g: sorted(s) for g, s in by_group.items() if len(s) > 1}
    return {"cross_split_groups": len(crossing), "details": crossing}


def assign_groups(
    groups: dict[str, str],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole groups to splits, largest groups first, shuffled by seed."""
    members: dict[str, list] = defaultdict(list)
    for i, g in groups.items():
        members[g].append(i)
    n = len(groups)
    target = {s: r * n for s, r in zip(SPLITS, ratios)}
    count = {s: 0 for s in SPLITS}
    rng = random.Random(seed)
    order = sorted(members, key=lambda g: (-len(members[g]), rng.random()))
    out: dict[str, str] = {}
    for g in order:
        s = max(SPLITS, key=lambda k: target[k] - count[k])
        for i in members[g]:
            out[i] = s
        count[s] += len(members[g])
    return out


def make_split(
    phashes: dict[str, str],
    existing: dict[str, str] | None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    phash_dist: int = 6,
) -> dict:
    """Keep `existing` assignments if they have no cross-split groups, else regroup."""
    groups = build_groups(phashes, phash_dist)
    source = "existing"
    assignments = None
    if existing:
        assignments = {i: existing[i] for i in groups if i in existing}
        if len(assignments) != len(groups) or leakage_check(assignments, groups)["cross_split_groups"]:
            assignments = None
    if assignments is None:
        assignments = assign_groups(groups, ratios, seed)
        source = "regrouped"
    return {
        "seed": seed,
        "ratios": list(ratios),
        "grouping_rule": f"surrogate: pHash hamming<={phash_dist} near-duplicate union (no patient IDs available)",
        "assignment_source": source,
        "assignments": assignments,
        "groups": groups,
        "leakage_check": leakage_check(assignments, groups),
        "residual_risk": (
            "No patient identifiers: images of the same patient that are not "
            "perceptual near-duplicates may still be spread across subsets. "
            "Patient-level leakage is NOT excluded."
        ),
    }

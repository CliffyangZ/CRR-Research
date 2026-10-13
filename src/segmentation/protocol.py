"""Comparison protocol: load, hash, validate, freeze, and decide whether a run belongs to it.

  python -m src.segmentation.protocol check  --protocol configs/segmentation/compare_protocol_v1.yaml
  python -m src.segmentation.protocol freeze --protocol configs/segmentation/compare_protocol_v1.yaml

Frozen state lives in `<reports_dir>/compare_<version>/protocol_freeze.json`; the YAML itself never changes
after freezing (a change means a new protocol version plus a CHANGELOG entry).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.detection.provenance import git_state, sha256_file, write_manifest

from .roi_config import RoiTrainConfig, load_roi_train_config


class Candidate(BaseModel):
    name: str
    config: str
    role: str = "primary"


class Statistics(BaseModel):
    unit: str = "source_image"
    method: str = "paired_cluster_bootstrap"
    B: int = 10000
    seed: int = 20261012
    ci: float = 0.95
    evidence_rule: str = "ci_excludes_zero AND abs(mean_diff) > max(seed_sd_baseline, seed_sd_candidate)"


class Protocol(BaseModel):
    protocol_version: str
    data_version: str
    derived_root: str = "data/derived"
    reports_dir: str = "reports/segmentation"
    split_sha256: str
    seeds: list[int]
    candidates: list[Candidate]
    threshold: float = 0.5
    batch_fallback: dict = {"batch_size": 8, "grad_accum": 2}
    metrics: dict = {}
    statistics: Statistics = Statistics()

    @property
    def baseline(self) -> str:
        return "unet"

    def compare_dir(self) -> Path:
        return Path(self.reports_dir) / f"compare_{self.protocol_version}"

    def freeze_path(self) -> Path:
        return self.compare_dir() / "protocol_freeze.json"


class ProtocolError(ValueError):
    pass


def load_protocol(path: str | Path) -> Protocol:
    with open(path) as f:
        return Protocol(**yaml.safe_load(f))


def recipe_hash(cfg: RoiTrainConfig) -> str:
    """Hash of everything except the `model` block (the only thing candidates may differ in)."""
    d = cfg.model_dump(mode="json")
    d.pop("model")
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _derived(p: Protocol) -> Path:
    return Path(p.derived_root) / p.data_version


def validate(p: Protocol) -> dict:
    """Raise ProtocolError on any inconsistency; return {candidate: config_hash, ...} details."""
    errors, cands = [], {}
    d = _derived(p)
    sp = d / "splits.json"
    if not sp.exists():
        errors.append(f"missing {sp}")
    elif sha256_file(sp) != p.split_sha256:
        errors.append(f"split_sha256 mismatch: protocol {p.split_sha256[:12]} vs data {sha256_file(sp)[:12]}")
    if len(p.seeds) < 3 or len(set(p.seeds)) != len(p.seeds):
        errors.append("need >= 3 distinct seeds")
    names = [c.name for c in p.candidates]
    if p.baseline not in names or len(set(names)) != len(names):
        errors.append("candidates must be unique and include the baseline 'unet'")
    recipes = set()
    for c in p.candidates:
        cfg = load_roi_train_config(c.config)
        if cfg.model.name != c.name:
            errors.append(f"{c.config}: model.name {cfg.model.name!r} != candidate {c.name!r}")
        if cfg.train.threshold != p.threshold:
            errors.append(f"{c.config}: threshold {cfg.train.threshold} != protocol {p.threshold}")
        recipes.add(recipe_hash(cfg))
        cands[c.name] = {"config": c.config, "config_hash": cfg.hash(), "recipe_hash": recipe_hash(cfg)}
    if len(recipes) > 1:
        errors.append("candidate configs differ outside the `model` block")
    if errors:
        raise ProtocolError("; ".join(errors))
    return cands


def protocol_hash(p: Protocol, details: dict) -> str:
    blob = json.dumps({"protocol": p.model_dump(mode="json"), "candidates": details}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def freeze(p: Protocol, now: datetime | None = None) -> dict:
    details = validate(p)
    path = p.freeze_path()
    if path.exists():
        raise ProtocolError(f"already frozen: {path} (create a new protocol version instead)")
    rec = {
        "protocol_version": p.protocol_version, "protocol_hash": protocol_hash(p, details),
        "frozen_at": (now or datetime.now(timezone.utc)).isoformat(), "git": git_state(),
        "candidates": details, "recipe_hash": next(iter(details.values()))["recipe_hash"],
        "data_version": p.data_version, "split_sha256": p.split_sha256, "seeds": p.seeds,
    }
    write_manifest(path, rec)
    return rec


def load_freeze(p: Protocol) -> dict | None:
    path = p.freeze_path()
    return json.loads(path.read_text()) if path.exists() else None


def require_frozen(p: Protocol) -> dict:
    """The freeze record, verified against the current YAML and configs (detects edits after freezing)."""
    rec = load_freeze(p)
    if rec is None:
        raise ProtocolError(f"protocol {p.protocol_version} is not frozen ({p.freeze_path()} missing)")
    if protocol_hash(p, validate(p)) != rec["protocol_hash"]:
        raise ProtocolError("protocol or candidate configs changed after freezing; use a new protocol version")
    return rec


def check_run(manifest: dict, p: Protocol, freeze_rec: dict) -> list[str]:
    """Reasons a run manifest does not belong to the frozen protocol (empty list = belongs)."""
    why = []
    cand = freeze_rec["candidates"].get(manifest.get("candidate"))
    if cand is None:
        why.append(f"unknown candidate {manifest.get('candidate')!r}")
    elif manifest.get("config_hash") != cand["config_hash"]:
        why.append("config_hash differs from the frozen candidate config")
    if manifest.get("protocol_hash") != freeze_rec["protocol_hash"]:
        why.append("protocol_hash differs")
    if manifest.get("seed") not in p.seeds:
        why.append(f"seed {manifest.get('seed')} not in protocol seeds")
    if manifest.get("data_version") != p.data_version or manifest.get("split_sha256") != p.split_sha256:
        why.append("data version or split differs")
    if manifest.get("smoke"):
        why.append("smoke run")
    return why


def build_runset(p: Protocol, reports_dir: str | Path | None = None) -> dict:
    """Map (candidate, seed) -> adopted succeeded run, listing every other attempt. Missing cells are None.

    More than one succeeded run for a cell is an error (the choice must not be made silently)."""
    rec = require_frozen(p)
    root = Path(reports_dir or p.reports_dir)
    cells: dict[tuple[str, int], dict] = {(c.name, s): {"adopted": None, "other_attempts": []}
                                         for c in p.candidates for s in p.seeds}
    for mp in sorted(root.glob("*/run_manifest.json")):
        m = json.loads(mp.read_text())
        if m.get("protocol_hash") != rec["protocol_hash"]:
            continue
        key = (m.get("candidate"), m.get("seed"))
        if key not in cells:
            continue
        cell = cells[key]
        if m["status"] == "succeeded" and not check_run(m, p, rec):
            if cell["adopted"] is not None:
                raise ProtocolError(f"two succeeded runs for {key}: {cell['adopted']} and {m['run_id']}")
            cell["adopted"] = m["run_id"]
        else:
            cell["other_attempts"].append({"run_id": m["run_id"], "status": m["status"],
                                           "failure_reason": m.get("failure_reason")})
    return {"protocol_hash": rec["protocol_hash"],
            "runs": [{"candidate": c, "seed": s, **v} for (c, s), v in cells.items()],
            "complete": all(v["adopted"] for v in cells.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "freeze"])
    ap.add_argument("--protocol", required=True)
    a = ap.parse_args(argv)
    try:
        p = load_protocol(a.protocol)
        if a.cmd == "check":
            d = validate(p)
            print(f"OK: {len(d)} candidates share one recipe; hash {protocol_hash(p, d)}")
            if (rec := load_freeze(p)):
                print("frozen:", "consistent" if require_frozen(p) else "")
                print(f"  freeze hash {rec['protocol_hash']} at {rec['frozen_at']}")
        else:
            rec = freeze(p)
            print(f"frozen {rec['protocol_version']} hash {rec['protocol_hash']} -> {p.freeze_path()}")
            if rec["git"]["dirty"]:
                print("WARNING: git tree has uncommitted changes; recorded in the freeze file.", file=sys.stderr)
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

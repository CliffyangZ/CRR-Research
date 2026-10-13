"""One-time test-set lock for a comparison protocol.

`test_lock.json` in the protocol's compare directory records which weights were evaluated on test and when.
A second evaluation is refused unless a violation is recorded explicitly, which invalidates confirmatory claims.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .protocol import Protocol, ProtocolError, build_runset, require_frozen


class TestAlreadyUsed(RuntimeError):
    __test__ = False                       # not a pytest class


def lock_path(p: Protocol) -> Path:
    return p.compare_dir() / "test_lock.json"


def load_lock(p: Protocol) -> dict | None:
    lp = lock_path(p)
    return json.loads(lp.read_text()) if lp.exists() else None


def check_ready(p: Protocol, run_dirs: list[str | Path]) -> list[dict]:
    """Verify the frozen protocol, that the given run dirs are exactly the adopted 9 runs, and the lock is unused.
    Returns the manifests, ordered by (candidate, seed)."""
    require_frozen(p)
    rs = build_runset(p)
    if not rs["complete"]:
        missing = [(r["candidate"], r["seed"]) for r in rs["runs"] if not r["adopted"]]
        raise ProtocolError(f"incomplete run set, missing {missing}")
    adopted = {r["adopted"] for r in rs["runs"]}
    manifests = {}
    for d in run_dirs:
        m = json.loads((Path(d) / "run_manifest.json").read_text())
        manifests[m["run_id"]] = m
    if set(manifests) != adopted:
        raise ProtocolError(f"--runs must be exactly the adopted runs; differ by {sorted(set(manifests) ^ adopted)}")
    if load_lock(p) is not None:
        raise TestAlreadyUsed(f"test already evaluated under protocol {p.protocol_version} ({lock_path(p)})")
    return sorted(manifests.values(), key=lambda m: (m["candidate"], m["seed"]))


def acquire(p: Protocol, evaluated: list[dict]) -> dict:
    """Write the lock after all weights were evaluated. `evaluated` rows: {run_id, weights_sha256}."""
    if load_lock(p) is not None:
        raise TestAlreadyUsed("lock already exists")
    rec = {"protocol_hash": require_frozen(p)["protocol_hash"], "evaluated_runs": evaluated,
           "timestamp": datetime.now(timezone.utc).isoformat(), "violations": []}
    lock_path(p).parent.mkdir(parents=True, exist_ok=True)
    lock_path(p).write_text(json.dumps(rec, indent=2))
    return rec


def record_violation(p: Protocol, reason: str) -> dict:
    rec = load_lock(p)
    if rec is None:
        raise ProtocolError("no lock to record a violation against")
    rec["violations"].append({"time": datetime.now(timezone.utc).isoformat(), "reason": reason})
    lock_path(p).write_text(json.dumps(rec, indent=2))
    return rec


def confirmatory_valid(p: Protocol) -> bool:
    rec = load_lock(p)
    return rec is not None and not rec["violations"]

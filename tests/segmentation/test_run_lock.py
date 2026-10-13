import json

import pytest

from src.segmentation import protocol as P
from src.segmentation import run_lock as L

from .test_protocol import _manifest, world  # noqa: F401  (fixture)


def _populate(tmp, p, rec, skip=None):
    dirs = []
    for c in p.candidates:
        for s in p.seeds:
            if (c.name, s) == skip:
                continue
            m = _manifest(rec, p, c.name, s)
            d = tmp / "reports" / m["run_id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "run_manifest.json").write_text(json.dumps(m))
            dirs.append(d)
    return dirs


def test_lock_lifecycle(world):  # noqa: F811
    tmp, path, _ = world
    p = P.load_protocol(path)
    rec = P.freeze(p)
    dirs = _populate(tmp, p, rec)
    ms = L.check_ready(p, dirs)
    assert len(ms) == 9
    L.acquire(p, [{"run_id": m["run_id"], "weights_sha256": "x"} for m in ms])
    assert L.confirmatory_valid(p)
    with pytest.raises(L.TestAlreadyUsed):
        L.check_ready(p, dirs)
    L.record_violation(p, "re-evaluated after tweak")
    assert not L.confirmatory_valid(p)


def test_not_ready_cases(world):  # noqa: F811
    tmp, path, _ = world
    p = P.load_protocol(path)
    with pytest.raises(P.ProtocolError, match="not frozen"):
        L.check_ready(p, [])
    rec = P.freeze(p)
    dirs = _populate(tmp, p, rec, skip=("unetpp", 2))
    with pytest.raises(P.ProtocolError, match="incomplete"):
        L.check_ready(p, dirs)


def test_wrong_run_set_rejected(world):  # noqa: F811
    tmp, path, _ = world
    p = P.load_protocol(path)
    rec = P.freeze(p)
    dirs = _populate(tmp, p, rec)
    with pytest.raises(P.ProtocolError, match="exactly the adopted"):
        L.check_ready(p, dirs[:-1])

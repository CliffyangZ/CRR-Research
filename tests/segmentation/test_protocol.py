import json
import shutil
from pathlib import Path

import pytest
import yaml

from research.train.train_tooth_seg import main as train_main
from src.segmentation import protocol as P

CFG = Path("configs/segmentation")


@pytest.fixture
def world(tmp_path):
    d = tmp_path / "derived" / "roi_x"
    d.mkdir(parents=True)
    (d / "splits.json").write_text('{"assignments": {"1": "train"}}')
    cfgs = {}
    for n in ("unet", "unetpp", "attention_unet"):
        dst = tmp_path / f"roi_{n}.yaml"
        shutil.copy(CFG / f"roi_{n}_v1.yaml", dst)
        cfgs[n] = dst
    proto = {
        "protocol_version": "vt", "data_version": "roi_x", "derived_root": str(tmp_path / "derived"),
        "reports_dir": str(tmp_path / "reports"), "split_sha256": P.sha256_file(d / "splits.json"),
        "seeds": [1, 2, 3], "candidates": [{"name": n, "config": str(c)} for n, c in cfgs.items()],
    }
    path = tmp_path / "proto.yaml"
    path.write_text(yaml.safe_dump(proto))
    return tmp_path, path, cfgs


def test_hash_stable_under_key_order(world):
    _, path, _ = world
    p = P.load_protocol(path)
    h1 = P.protocol_hash(p, P.validate(p))
    raw = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump(dict(reversed(list(raw.items()))), sort_keys=False))
    p2 = P.load_protocol(path)
    assert P.protocol_hash(p2, P.validate(p2)) == h1


def test_non_model_config_difference_rejected(world):
    _, path, cfgs = world
    cfgs["unetpp"].write_text(cfgs["unetpp"].read_text().replace("max_epochs: 150", "max_epochs: 100"))
    with pytest.raises(P.ProtocolError, match="differ outside"):
        P.validate(P.load_protocol(path))


def test_split_hash_mismatch_rejected(world):
    tmp, path, _ = world
    (tmp / "derived/roi_x/splits.json").write_text("{}")
    with pytest.raises(P.ProtocolError, match="split_sha256"):
        P.validate(P.load_protocol(path))


def test_freeze_once_and_detect_later_edit(world):
    _, path, cfgs = world
    p = P.load_protocol(path)
    rec = P.freeze(p)
    assert P.require_frozen(p)["protocol_hash"] == rec["protocol_hash"]
    with pytest.raises(P.ProtocolError, match="already frozen"):
        P.freeze(p)
    cfgs["unet"].write_text(cfgs["unet"].read_text().replace("lr: 1.0e-3", "lr: 2.0e-3"))
    with pytest.raises(P.ProtocolError):       # recipe now differs from the other candidates / frozen hash
        P.require_frozen(p)


def test_not_frozen_rejected(world):
    _, path, _ = world
    with pytest.raises(P.ProtocolError, match="not frozen"):
        P.require_frozen(P.load_protocol(path))


def _manifest(rec, p, cand, seed, status="succeeded", run_id=None, **kw):
    m = {"run_id": run_id or f"{cand}-{seed}", "candidate": cand, "seed": seed, "status": status, "smoke": False,
         "config_hash": rec["candidates"][cand]["config_hash"], "protocol_hash": rec["protocol_hash"],
         "data_version": p.data_version, "split_sha256": p.split_sha256}
    m.update(kw)
    return m


def test_check_run_and_runset(world):
    tmp, path, _ = world
    p = P.load_protocol(path)
    rec = P.freeze(p)
    ok = _manifest(rec, p, "unet", 1)
    assert P.check_run(ok, p, rec) == []
    assert P.check_run({**ok, "config_hash": "zzz"}, p, rec)
    assert P.check_run({**ok, "seed": 99}, p, rec)
    assert P.check_run({**ok, "smoke": True}, p, rec)
    assert P.check_run({**ok, "protocol_hash": None}, p, rec)

    def put(m):
        d = tmp / "reports" / m["run_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_manifest.json").write_text(json.dumps(m))

    put(_manifest(rec, p, "unet", 1, status="failed", run_id="f1", failure_reason="oom"))
    put(ok)
    rs = P.build_runset(p)
    cell = next(r for r in rs["runs"] if (r["candidate"], r["seed"]) == ("unet", 1))
    assert cell["adopted"] == "unet-1" and cell["other_attempts"][0]["run_id"] == "f1"
    assert rs["complete"] is False                                     # 8 of 9 cells missing
    put(_manifest(rec, p, "unet", 1, run_id="dup"))
    with pytest.raises(P.ProtocolError, match="two succeeded"):
        P.build_runset(p)


def test_train_entry_refuses_unfrozen_protocol(world, capsys):
    tmp, path, cfgs = world
    rc = train_main(["--model", "unet", "--seed", "1", "--data-version", "roi_x", "--derived", str(tmp / "derived"),
                     "--config", str(cfgs["unet"]), "--protocol", str(path), "--device", "cpu"])
    assert rc == 2 and "not frozen" in capsys.readouterr().err
    assert not (tmp / "reports").exists() or not list((tmp / "reports").glob("unet-*"))


def test_train_entry_refuses_bad_seed_and_batch(world, capsys):
    tmp, path, cfgs = world
    P.freeze(P.load_protocol(path))
    base = ["--model", "unet", "--data-version", "roi_x", "--derived", str(tmp / "derived"),
            "--config", str(cfgs["unet"]), "--protocol", str(path), "--device", "cpu"]
    assert train_main([*base, "--seed", "99"]) == 2
    assert "not in protocol seeds" in capsys.readouterr().err
    assert train_main([*base, "--seed", "1", "--batch-size", "4", "--grad-accum", "4", "--override-reason", "x"]) == 2
    assert "batch override" in capsys.readouterr().err

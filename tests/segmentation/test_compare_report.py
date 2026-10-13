import numpy as np

from research.evaluation.compare_seg_models import build_comparison, render
from src.segmentation.protocol import Protocol

from .test_compare_stats import make

SEEDS = (1, 2, 3)


def fake_run(cand, seed, counts, ids, images, flags):
    rows = []
    for i, (c, rid, img) in enumerate(zip(counts, ids, images)):
        tp, fp, fn = (int(x) for x in c)
        rows.append({"roi_id": rid, "source_image_id": img, "tp": tp, "fp": fp, "fn": fn, "dice": 2 * tp / (2 * tp + fp + fn),
                     "has_neighbor": str(flags[i][0]), "truncated": str(flags[i][1]),
                     "fn_frac_of_gt": fn / (tp + fn)})
    a = {"dice": float(2 * counts[:, 0].sum() / (2 * counts[:, 0].sum() + counts[:, 1].sum() + counts[:, 2].sum())),
         "iou": 0.5, "precision": 0.9, "recall": 0.9}
    return {"m": {"run_id": f"{cand}-{seed}", "n_parameters": 10, "train_seconds": 5.0, "peak_gpu_mem_mb": 100.0,
                  "stopping": {"epochs_run": 7}, "batch": {"size": 16}},
            "metrics": {"aggregate": a, "per_image_mean": {"dice": a["dice"]}, "hd95_mean": 3.0, "boundary_f1_mean": 0.8,
                        "inference": {"ms_per_roi_mean": 2.0}, "weights_sha256": "ab" * 32},
            "rows": rows}


def test_build_and_render_three_way_conclusions():
    p = Protocol(protocol_version="vt", data_version="d", split_sha256="s", seeds=list(SEEDS),
                 candidates=[{"name": n, "config": "x"} for n in ("unet", "good", "same")])
    base, images = make(n_img=12, per_img=3, rng_seed=5)
    good, _ = make(n_img=12, per_img=3, rng_seed=5, shift=0.06)
    ids = [f"r{i}" for i in range(len(images))]
    flags = [(i % 5 == 0, i % 7 == 0) for i in range(len(ids))]
    runs = {n: {s: fake_run(n, s, c[s], ids, images, flags) for s in SEEDS}
            for n, c in (("unet", base), ("good", good), ("same", base))}
    comp = build_comparison(p, runs, dry_run=False)
    assert comp["paired"]["good"]["verdict"] == "evidence_better"
    assert comp["paired"]["same"]["verdict"] == "no_demonstrable_difference"
    cats = comp["error_cases"]
    assert [c["n_rois"] for c in cats.values()] == [len([f for f in flags if f[0]]), len([f for f in flags if f[1]]), 10]
    comp["_sha"] = {r["m"]["run_id"]: r["metrics"]["weights_sha256"] for by in runs.values() for r in by.values()}
    text = render(p, comp, None, None)
    assert "有證據優於 baseline" in text and "未見可證實差異" in text and "未量測" in text and "先驗揭露" in text
    assert "臨床診斷能力" in text

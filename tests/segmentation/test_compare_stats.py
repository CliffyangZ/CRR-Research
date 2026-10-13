import numpy as np

from src.segmentation import compare_stats as S


def make(n_img=20, per_img=3, p_hit=0.9, seeds=(1, 2, 3), rng_seed=0, shift=0.0):
    """Counts with ~1000 gt px per ROI; `p_hit` controls tp share; image-level random effect drives clustering."""
    rng = np.random.default_rng(rng_seed)
    ids = [f"img{i}" for i in range(n_img) for _ in range(per_img)]
    eff = np.repeat(rng.normal(0, 0.03, n_img), per_img)
    out = {}
    for s in seeds:
        hit = np.clip(p_hit + shift + eff + rng.normal(0, 0.005, len(ids)), 0.05, 0.99)
        tp = (1000 * hit).astype(int)
        out[s] = np.stack([tp, 1000 - tp, 1000 - tp], axis=1)       # fp = fn = misses
    return out, ids


def test_pixel_dice_and_undefined():
    assert S.pixel_dice(np.array([[2, 1, 1], [3, 0, 0]])) == 2 * 5 / (2 * 5 + 1 + 1)
    assert S.pixel_dice(np.zeros((2, 3))) is None


def test_seed_summary_sample_sd():
    s = S.seed_summary([0.9, 0.92, 0.94])
    assert abs(s["mean"] - 0.92) < 1e-12 and abs(s["sd"] - 0.02) < 1e-12


def test_clearly_better_candidate():
    base, ids = make(rng_seed=1)
    cand, _ = make(rng_seed=1, shift=0.05)
    r = S.paired_cluster_bootstrap(base, cand, ids, B=500)
    sd = S.seed_summary([S.pixel_dice(c) for c in base.values()])["sd"]
    assert r["ci"][0] > 0 and r["mean_diff"] > sd
    assert S.verdict(r["ci"], r["mean_diff"], sd, sd) == S.EVIDENCE_BETTER
    worse = S.paired_cluster_bootstrap(cand, base, ids, B=500)
    assert S.verdict(worse["ci"], worse["mean_diff"], sd, sd) == S.EVIDENCE_WORSE


def test_identical_models_no_difference():
    base, ids = make(rng_seed=2)
    r = S.paired_cluster_bootstrap(base, base, ids, B=300)
    assert r["mean_diff"] == 0 and r["ci"] == [0.0, 0.0]
    assert S.verdict(r["ci"], 0.0, 0.01, 0.01) == S.NO_DIFFERENCE


def test_ci_excludes_zero_but_below_seed_sd_is_not_evidence():
    assert S.verdict([0.001, 0.003], 0.002, 0.01, 0.01) == S.NO_DIFFERENCE
    assert S.verdict([-0.01, 0.02], 0.005, 0.001, 0.001) == S.NO_DIFFERENCE
    assert S.verdict([0.001, 0.003], 0.002, None, None) == S.EVIDENCE_BETTER


def test_resampling_unit_is_the_image():
    """Clustered data must give a wider CI than treating ROIs as independent would; here we check clusters stay whole."""
    base, ids = make(n_img=10, per_img=5, rng_seed=3)
    cand, _ = make(n_img=10, per_img=5, rng_seed=3, shift=0.02)
    r = S.paired_cluster_bootstrap(base, cand, ids, B=200)
    assert r["n_images"] == 10 and r["n_rois"] == 50 and r["unit"] == "source_image"
    # one ROI per image can only ever resample 50 clusters instead -> narrower interval than the clustered one
    ids_single = [f"u{i}" for i in range(50)]
    r2 = S.paired_cluster_bootstrap(base, cand, ids_single, B=200)
    assert (r["ci"][1] - r["ci"][0]) > (r2["ci"][1] - r2["ci"][0])


def test_reproducible_and_roi_wins():
    base, ids = make(rng_seed=4)
    cand, _ = make(rng_seed=4, shift=0.03)
    assert S.paired_cluster_bootstrap(base, cand, ids, B=100, seed=7) == S.paired_cluster_bootstrap(base, cand, ids, B=100, seed=7)
    w = S.roi_wins(base, cand)
    assert w["wins"] + w["losses"] + w["ties"] == w["n_defined"] == len(ids) and w["wins"] > w["losses"]

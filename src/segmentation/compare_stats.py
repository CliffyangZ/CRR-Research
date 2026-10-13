"""Statistics for the architecture comparison.

Counts arrays have shape (n_roi, 3) = per-ROI (tp, fp, fn) in the ROI's original-resolution frame. ROIs of the same
source image are not independent, so the bootstrap resamples whole source images (clusters).
"""

from __future__ import annotations

import numpy as np

EVIDENCE_BETTER, NO_DIFFERENCE, EVIDENCE_WORSE = "evidence_better", "no_demonstrable_difference", "evidence_worse"


def pixel_dice(counts: np.ndarray) -> float | None:
    """Pixel-accumulated Dice over all rows; None when 2TP+FP+FN == 0."""
    tp, fp, fn = np.asarray(counts, dtype=np.int64).sum(axis=0)
    den = 2 * tp + fp + fn
    return None if den == 0 else float(2 * tp / den)


def seed_summary(values: list[float]) -> dict:
    """Mean and sample SD (ddof=1) over seeds; SD is None for fewer than 2 seeds."""
    v = np.asarray(values, dtype=float)
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "values": [float(x) for x in v]}


def _group(image_ids: list[str]) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(image_ids), return_inverse=True)
    return inv, int(inv.max()) + 1


def _per_image(counts_by_seed: dict, inv: np.ndarray, n_img: int) -> np.ndarray:
    """-> (n_seeds, n_img, 3) summed counts per source image, seeds in sorted order."""
    out = np.zeros((len(counts_by_seed), n_img, 3), dtype=np.int64)
    for k, s in enumerate(sorted(counts_by_seed)):
        np.add.at(out[k], inv, np.asarray(counts_by_seed[s], dtype=np.int64))
    return out


def _dice(tot: np.ndarray) -> np.ndarray:
    """tot (..., 3) -> Dice with nan where undefined."""
    tp, fp, fn = tot[..., 0], tot[..., 1], tot[..., 2]
    den = 2.0 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2.0 * tp / den, np.nan)


def paired_cluster_bootstrap(base: dict, cand: dict, image_ids: list[str], B: int = 10000, seed: int = 20261012,
                             ci: float = 0.95) -> dict:
    """Seed-averaged pixel-Dice difference (candidate - baseline) with an image-level bootstrap CI.

    `base`/`cand`: {seed: (n_roi, 3) counts}, rows aligned with `image_ids`. Within each resample the same images
    are used for every seed and both models; the Dice difference is averaged over seeds."""
    inv, n_img = _group(image_ids)
    cb, cc = _per_image(base, inv, n_img), _per_image(cand, inv, n_img)
    point = float(np.nanmean(_dice(cc.sum(axis=1))) - np.nanmean(_dice(cb.sum(axis=1))))
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, n_img, n_img), minlength=n_img) for _ in range(B)]).astype(np.int64)
    # (B, n_img) @ (n_img, 3) per seed
    tb = np.einsum("bi,sic->bsc", w, cb)
    tc = np.einsum("bi,sic->bsc", w, cc)
    diffs = np.nanmean(_dice(tc), axis=1) - np.nanmean(_dice(tb), axis=1)
    lo, hi = np.nanpercentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"mean_diff": point, "ci": [float(lo), float(hi)], "B": B, "seed": seed, "unit": "source_image",
            "n_images": n_img, "n_rois": len(image_ids), "n_seeds": cb.shape[0]}


def roi_wins(base: dict, cand: dict, tol: float = 1e-9) -> dict:
    """Descriptive per-ROI win/loss/tie of candidate vs baseline (seed-averaged per-ROI Dice; undefined ROIs skipped)."""
    def per_roi(by_seed):
        return np.nanmean([_dice(np.asarray(c, dtype=np.int64)) for c in by_seed.values()], axis=0)

    d = per_roi(cand) - per_roi(base)
    d = d[~np.isnan(d)]
    return {"wins": int((d > tol).sum()), "losses": int((d < -tol).sum()), "ties": int((np.abs(d) <= tol).sum()),
            "n_defined": int(len(d))}


def verdict(ci: list[float], mean_diff: float, sd_base: float | None, sd_cand: float | None) -> str:
    """Three-way conclusion. Evidence requires the CI to exclude zero AND |diff| to exceed the larger seed SD."""
    margin = max(x for x in (sd_base, sd_cand, 0.0) if x is not None)
    if ci[0] > 0 and mean_diff > margin:
        return EVIDENCE_BETTER
    if ci[1] < 0 and -mean_diff > margin:
        return EVIDENCE_WORSE
    return NO_DIFFERENCE

"""Clinical crown/root reference from the alveolar bone crest, found on a contrast-adjusted radiograph.

Clinical definition used here: the reference line joins the alveolar crest on the two sides of one
tooth. Crown = from the crown tip A to the point O where that line meets the tooth axis; root = from
O to the apex B; CRR = |AO| / |OB|. (This is NOT the anatomical CEJ ratio.)

Method, per tooth, on a source-resolution crop:
  1. contrast adjustment: percentile normalisation over valid pixels, then CLAHE;
  2. tooth axis by PCA of the mask, oriented crown -> root (narrower end = root);
  3. warp image/masks into the axis frame (axis vertical, crown up, x' = lateral offset);
  4. per side, sample a lateral band just outside the tooth contour (neighbouring teeth, padding and
     the tooth itself are excluded) -> 1-D "bone brightness" profile along the axis;
  5. the crest is the most coronal strong step "darker above -> brighter below" of that profile;
  6. map points back to the crop, intersect the crest line with the axis, report lengths and CRR.
Nothing here is validated against a clinician yet; outputs are candidates with explicit warnings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .roi import polygons_mask_in_crop


@dataclass
class AlveolarConfig:
    p_low: float = 1.0
    p_high: float = 99.0
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    margin_frac: float = 0.10          # frame margin above A / below B, as a fraction of axis length
    end_percentile: float = 1.0        # A, B = 1% / 99% of the mask projection on the axis
    end_segment_frac: float = 0.25     # segments used to decide which end is the root
    ambiguity: float = 0.85            # width ratio above which the orientation is "unclear"
    band_gap_frac: float = 0.02        # gap between contour and sampling band (axis-length fraction)
    band_width_frac: float = 0.12      # band width (axis-length fraction)
    min_band_valid_frac: float = 0.5   # share of band pixels that must be usable
    smooth_sigma_frac: float = 0.01    # profile smoothing (axis-length fraction)
    window_fracs: tuple = (0.03, 0.05, 0.08)   # half-windows of the step detector
    t_range: tuple = (0.20, 0.95)      # search range along the axis (0 = A, 1 = B)
    coronal_frac: float = 0.60         # accept peaks >= this fraction of the strongest, take the most coronal
    min_step: float = 0.10             # minimum brightness step (contrast-adjusted image, 0-1 units)
    min_line_axis_angle_deg: float = 25.0
    plausible_crr: tuple = (0.1, 3.0)
    min_root_frac: float = 0.05
    pair_mode: str = "independent"     # "independent": best step per side | "weighted": multi-parameter pair score
    top_k: int = 4                     # candidates kept per side in weighted mode
    pair_weights: tuple = (("step", 0.40), ("level", 0.15), ("symmetry", 0.10), ("angle", 0.15), ("anatomy", 0.20))
    level_sigma_frac: float = 0.12     # level-difference tolerance (axis-length fraction)
    anatomy_center: float = 0.50       # prior position of the crest along the axis (0 = A, 1 = B)
    anatomy_sigma: float = 0.20


def contrast_image(gray: np.ndarray, valid: np.ndarray, cfg: AlveolarConfig) -> np.ndarray:
    """uint8 -> float32 [0,1]: percentile normalisation over valid pixels, then CLAHE."""
    v = gray[valid] if valid.any() else gray.ravel()
    lo, hi = np.percentile(v, [cfg.p_low, cfg.p_high])
    n = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    u8 = (n * 255).round().astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)).apply(u8)
    return c.astype(np.float32) / 255.0


def tooth_axis(mask: np.ndarray, cfg: AlveolarConfig) -> dict:
    """PCA axis. v points crown -> root (root = narrower end); A and B are the axis end points."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
    v = vecs[:, -1]
    elongation = float(vals[-1] / max(vals[0], 1e-9))
    t = (pts - c) @ v
    u = np.array([v[1], -v[0]])
    s = (pts - c) @ u
    t_lo, t_hi = np.percentile(t, [cfg.end_percentile, 100 - cfg.end_percentile])
    L = t_hi - t_lo
    seg = cfg.end_segment_frac * L
    w_lo = _width(s[t <= t_lo + seg])
    w_hi = _width(s[t >= t_hi - seg])
    warnings = []
    if w_lo <= w_hi:                      # the t_lo end is narrower = root: flip so that v runs crown -> root
        v = -v
        t_lo, t_hi = -t_hi, -t_lo
        w_lo, w_hi = w_hi, w_lo
    ratio = min(w_lo, w_hi) / max(w_lo, w_hi, 1e-9)
    if ratio > cfg.ambiguity:
        warnings.append("orientation_unclear")
    if elongation < 1.5:
        warnings.append("axis_unreliable_low_elongation")
    u = np.array([v[1], -v[0]])
    return {"origin": c, "v": v, "u": u, "t_top": float(t_lo), "t_bot": float(t_hi), "length": float(L),
            "width_ratio": float(ratio), "elongation": elongation, "warnings": warnings}


def scan_profile(mask: np.ndarray, axis: dict, cfg: AlveolarConfig, n_levels: int = 40) -> dict:
    """Scan lines perpendicular to the PCA axis from A to B: distance from the axis to the contour on
    each side, and the contour points / axis feet in crop coordinates (for drawing the scan arrows)."""
    M, (wd, hd) = axis_warp(axis, mask.shape, cfg, mask)
    Mw = _warp((mask > 0).astype(np.uint8), M, (wd, hd), cv2.INTER_NEAREST)
    inv = cv2.invertAffineTransform(M)
    L = axis["length"]
    yA = cfg.margin_frac * L
    xc = wd / 2.0
    out = {"t_frac": [], "left_dist": [], "right_dist": [], "left_pts": [], "right_pts": [], "feet": []}
    for t in np.linspace(0.0, 1.0, n_levels):
        y = int(round(yA + t * L))
        xs = np.nonzero(Mw[min(max(y, 0), hd - 1)])[0]
        if len(xs) == 0:
            continue
        el, er = float(xs.min()), float(xs.max())
        out["t_frac"].append(float(t))
        out["left_dist"].append(xc - el)
        out["right_dist"].append(er - xc)
        out["left_pts"].append((inv @ np.array([el, y, 1.0])).tolist())
        out["right_pts"].append((inv @ np.array([er, y, 1.0])).tolist())
        out["feet"].append((inv @ np.array([xc, y, 1.0])).tolist())
    return {k: np.array(v) for k, v in out.items()}


def _width(s: np.ndarray) -> float:
    return float(np.percentile(s, 98) - np.percentile(s, 2)) if len(s) else 0.0


def axis_warp(axis: dict, shape: tuple, cfg: AlveolarConfig, mask: np.ndarray) -> tuple[np.ndarray, tuple]:
    """2x3 matrix: crop -> axis frame (x' = lateral, y' = along crown->root, A at y' = margin)."""
    L, m = axis["length"], cfg.margin_frac * axis["length"]
    c, v, u = axis["origin"], axis["v"], axis["u"]
    ys, xs = np.nonzero(mask)
    half = max(np.abs((np.stack([xs, ys], 1) - c) @ u).max() + 0.6 * L, 0.5 * L)
    wd, hd = int(np.ceil(2 * half)), int(np.ceil(L + 2 * m))
    M = np.array([[u[0], u[1], -(u @ c) + wd / 2], [v[0], v[1], -(v @ c) - axis["t_top"] + m]], dtype=np.float64)
    return M, (wd, hd)


def _warp(img: np.ndarray, M: np.ndarray, size: tuple, interp: int, border=0) -> np.ndarray:
    return cv2.warpAffine(img, M, size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def side_profile(I: np.ndarray, tooth: np.ndarray, usable: np.ndarray, side: str, y0: int, y1: int,
                 L: float, cfg: AlveolarConfig) -> tuple[np.ndarray, np.ndarray]:
    """-> (brightness profile, contour x) per row; NaN where the tooth or the band is not usable."""
    h, w = I.shape
    prof = np.full(h, np.nan, dtype=np.float32)
    edge = np.full(h, np.nan, dtype=np.float32)
    gap, bw = max(1, int(round(cfg.band_gap_frac * L))), max(2, int(round(cfg.band_width_frac * L)))
    for y in range(max(0, y0), min(h, y1 + 1)):
        xs = np.nonzero(tooth[y])[0]
        if len(xs) == 0:
            continue
        e = xs.min() if side == "left" else xs.max()
        edge[y] = e
        a, b = (e - gap - bw, e - gap) if side == "left" else (e + gap + 1, e + gap + 1 + bw)
        a, b = max(a, 0), min(b, w)
        if b - a < 1:
            continue
        ok = usable[y, a:b]
        if ok.sum() >= cfg.min_band_valid_frac * bw:
            prof[y] = np.median(I[y, a:b][ok])
    return prof, edge


def _fill(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def step_scores(prof: np.ndarray, L: float, cfg: AlveolarConfig) -> np.ndarray:
    """Mean(brightness below) - mean(brightness above), averaged over windows (0-1 brightness units)."""
    if np.isfinite(prof).sum() < 3:
        return np.full_like(prof, np.nan)
    x = _fill(prof)
    sig = max(1.0, cfg.smooth_sigma_frac * L)
    xs = cv2.GaussianBlur(x.reshape(-1, 1), (0, 0), sigmaX=0.01, sigmaY=sig).ravel()
    out = np.zeros_like(xs)
    for f in cfg.window_fracs:
        k = max(2, int(round(f * L)))
        cs = np.concatenate([[0], np.cumsum(xs)])
        below = np.full_like(xs, np.nan)
        above = np.full_like(xs, np.nan)
        for y in range(k, len(xs) - k):
            below[y] = (cs[y + 1 + k] - cs[y + 1]) / k
            above[y] = (cs[y] - cs[y - k]) / k
        out += np.nan_to_num(below - above)
    return out / len(cfg.window_fracs)


def pick_crest(score: np.ndarray, lo: int, hi: int, cfg: AlveolarConfig) -> tuple[int | None, float]:
    """Most coronal local maximum whose score is >= coronal_frac of the strongest in [lo, hi]."""
    seg = score[lo : hi + 1]
    if len(seg) < 3 or not np.isfinite(seg).any():
        return None, 0.0
    top = np.nanmax(seg)
    if top < cfg.min_step:
        return None, float(max(top, 0.0))
    for i in range(1, len(seg) - 1):
        if seg[i] >= cfg.coronal_frac * top and seg[i] >= cfg.min_step and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            return lo + i, float(seg[i] / top) * float(min(1.0, seg[i] / (2 * cfg.min_step)))
    return None, 0.0


def candidates(score: np.ndarray, lo: int, hi: int, cfg: AlveolarConfig) -> list[tuple[int, float]]:
    """Local maxima of the step score in [lo, hi] with score >= min_step; the top_k strongest, as (y, score)."""
    seg = score[lo : hi + 1]
    out = []
    for i in range(1, len(seg) - 1):
        if np.isfinite(seg[i]) and seg[i] >= cfg.min_step and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            out.append((lo + i, float(seg[i])))
    return sorted(out, key=lambda c: -c[1])[: cfg.top_k]


def pair_terms(left: tuple, right: tuple, ctx: dict, cfg: AlveolarConfig) -> dict:
    """Terms of the pair score in [0, 1]. left/right = (y, step, edge_x); ctx has L, yA, xc, max steps."""
    (yl, sl, xl), (yr, sr, xr) = left, right
    L = ctx["L"]
    hl, hr = ctx["xc"] - xl, xr - ctx["xc"]
    dx = max(abs(xr - xl), 1e-9)
    return {
        "step": 0.5 * (sl / ctx["max_l"] + sr / ctx["max_r"]),
        "level": float(np.exp(-(((yl - yr) / L) / cfg.level_sigma_frac) ** 2)),
        "symmetry": float(min(hl, hr) / max(hl, hr, 1e-9)) if min(hl, hr) > 0 else 0.0,
        "angle": float(dx / np.hypot(dx, yr - yl)),                      # 1 = the line is perpendicular to the axis
        "anatomy": float(np.exp(-((((yl + yr) / 2 - ctx["yA"]) / L - cfg.anatomy_center) / cfg.anatomy_sigma) ** 2)),
    }


def best_pair(cl: list, cr: list, ctx: dict, cfg: AlveolarConfig) -> tuple[tuple, tuple, float, dict] | None:
    w = dict(cfg.pair_weights)
    best = None
    for a in cl:
        for b in cr:
            t = pair_terms(a, b, ctx, cfg)
            sc = sum(w[k] * t[k] for k in w)
            if best is None or sc > best[2]:
                best = (a, b, sc, t)
    return best


def locate_alveolar_crest(gray: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None,
                          others: np.ndarray | None = None, cfg: AlveolarConfig | None = None,
                          truncated: bool = False) -> dict:
    """Run the full method on one tooth crop. `gray` uint8, `mask` {0,1} of the target tooth,
    `valid` = pixels that are real image (not padding), `others` = other teeth/metal to exclude."""
    cfg = cfg or AlveolarConfig()
    valid = np.ones_like(mask, dtype=bool) if valid is None else valid.astype(bool)
    others = np.zeros_like(mask) if others is None else others
    res: dict = {"warnings": [], "status": "failed"}
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() < 50:
        res["warnings"].append("tiny_mask")
        return res
    I = contrast_image(gray, valid, cfg)
    ax = tooth_axis(mask, cfg)
    res["warnings"] += ax["warnings"]
    if truncated:
        res["warnings"].append("tooth_truncated_apex_or_crown_unreliable")
    M, (wd, hd) = axis_warp(ax, mask.shape, cfg, mask)
    Iw = _warp(I, M, (wd, hd), cv2.INTER_LINEAR)
    Mw = _warp(mask, M, (wd, hd), cv2.INTER_NEAREST)
    Ow = _warp((others > 0).astype(np.uint8), M, (wd, hd), cv2.INTER_NEAREST)
    Vw = _warp(valid.astype(np.uint8), M, (wd, hd), cv2.INTER_NEAREST) > 0
    usable = Vw & (Ow == 0) & (Mw == 0)
    L = ax["length"]
    yA = cfg.margin_frac * L
    yB = yA + L
    lo, hi = int(round(yA + cfg.t_range[0] * L)), int(round(yA + cfg.t_range[1] * L))
    xc = wd / 2.0
    inv = cv2.invertAffineTransform(M)
    pts, per = {}, {}
    cands = {}
    for side in ("left", "right"):
        prof, edge = side_profile(Iw, Mw, usable, side, int(yA), int(yB), L, cfg)
        sc = step_scores(prof, L, cfg)
        per[side] = {"profile": prof, "edge": edge, "score": sc, "y": None, "confidence": 0.0}
        cands[side] = [(y, st, float(edge[y])) for y, st in candidates(sc, lo, hi, cfg) if np.isfinite(edge[y])]
        if cfg.pair_mode == "independent":
            y, conf = pick_crest(sc, lo, hi, cfg)
            per[side].update(y=y, confidence=conf)
    pair_info = None
    if cfg.pair_mode == "weighted" and cands["left"] and cands["right"]:
        ctx = {"L": L, "yA": yA, "xc": xc, "max_l": max(c[1] for c in cands["left"]), "max_r": max(c[1] for c in cands["right"])}
        a, b, sc, terms = best_pair(cands["left"], cands["right"], ctx, cfg)
        per["left"].update(y=a[0], confidence=min(1.0, a[1] / (2 * cfg.min_step)))
        per["right"].update(y=b[0], confidence=min(1.0, b[1] / (2 * cfg.min_step)))
        pair_info = {"score": float(sc), "terms": terms}
    elif cfg.pair_mode == "weighted":                       # one side only: fall back to its strongest candidate
        for side in ("left", "right"):
            if cands[side]:
                y, st, _ = max(cands[side], key=lambda c: c[1])
                per[side].update(y=y, confidence=min(1.0, st / (2 * cfg.min_step)))
    for side in ("left", "right"):
        y = per[side]["y"]
        if y is None or not np.isfinite(per[side]["edge"][y]):
            res["warnings"].append(f"no_crest_{side}")
        else:
            pts[side] = (float(per[side]["edge"][y]), float(y))
    res.update(axis=ax, frame=dict(M=M, size=(wd, hd), yA=yA, yB=yB, xc=xc), contrast=I, per_side=per, candidates=cands, pair=pair_info,
               warped=dict(image=Iw, mask=Mw, others=Ow, usable=usable), crest_frame=pts)
    res["crest_crop"] = {k: (inv @ np.array([x, y, 1.0])).tolist() for k, (x, y) in pts.items()}
    if not pts:
        return res
    # reference level on the axis
    if len(pts) == 2:
        (xl, yl), (xr, yr) = pts["left"], pts["right"]
        ang = np.degrees(np.arctan2(abs(yr - yl), max(abs(xr - xl), 1e-9)))     # line vs the perpendicular
        yO = yl + (yr - yl) * (xc - xl) / (xr - xl) if xr - xl > 1e-6 else (yl + yr) / 2
        if 90 - ang < cfg.min_line_axis_angle_deg:
            res["warnings"].append("crest_line_nearly_parallel_to_axis")
            yO = (yl + yr) / 2
        if not (yA < yO < yB):
            res["warnings"].append("O_outside_tooth_used_midpoint")
            yO = min(max((yl + yr) / 2, yA), yB)
    else:
        res["warnings"].append("single_side_horizontal_reference")
        yO = next(iter(pts.values()))[1]
    crown, root = yO - yA, yB - yO
    res.update(O_frame=(xc, float(yO)), O_crop=(inv @ np.array([xc, yO, 1.0])).tolist(),
               A_crop=(inv @ np.array([xc, yA, 1.0])).tolist(), B_crop=(inv @ np.array([xc, yB, 1.0])).tolist(),
               crown_px=float(crown), root_px=float(root))
    if root < cfg.min_root_frac * L:
        res["warnings"].append("root_too_short")
    else:
        crr = crown / root
        res["crr"] = float(crr)
        if not (cfg.plausible_crr[0] <= crr <= cfg.plausible_crr[1]):
            res["warnings"].append("crr_implausible")
    confs = [per[s]["confidence"] for s in pts]
    res["confidence"] = float(np.mean(confs) * (0.75 ** len(res["warnings"])))
    res["status"] = "ok" if "crr" in res else "failed"
    return res


def load_crop(row: dict, seg_root: str | Path) -> dict:
    """Source-resolution crop of one ROI row: gray, target mask, other-teeth mask, valid (in-image) mask."""
    seg_root = Path(seg_root)
    sub = row.get("source_subset") or row["subset"]
    coco = json.loads((seg_root / sub / "annotations" / "instances_default.json").read_text())
    anns = {a["id"]: a for a in coco["annotations"]}
    target = anns[row["polygon_ann_id"]]
    im = next(i for i in coco["images"] if i["id"] == target["image_id"])
    path = next((seg_root / sub / "images").glob(f"{row['image_id']}.*"))
    full = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = row["expanded_xyxy"]
    crop = np.full((y1 - y0, x1 - x0), 114, dtype=np.uint8)
    valid = np.zeros(crop.shape, dtype=bool)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, im["width"]), min(y1, im["height"])
    crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = full[sy0:sy1, sx0:sx1]
    valid[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = True
    others = [a for a in coco["annotations"] if a["image_id"] == target["image_id"] and a["id"] != target["id"]]
    return {"gray": crop, "mask": polygons_mask_in_crop([target], row["expanded_xyxy"]),
            "others": polygons_mask_in_crop(others, row["expanded_xyxy"]) if others else np.zeros_like(crop),
            "valid": valid}


def reference_from_points(res: dict, points_crop: list[list[float]], cfg: AlveolarConfig | None = None) -> dict | None:
    """Clinician reference: two crest points (crop coordinates) -> level on the tooth axis and CRR,
    using the same axis and A/B as the algorithm run `res`. None if the run has no axis frame."""
    cfg = cfg or AlveolarConfig()
    if "frame" not in res or len(points_crop) != 2:
        return None
    M, f = res["frame"]["M"], res["frame"]
    p = [(M @ np.array([x, y, 1.0])).tolist() for x, y in points_crop]
    (xl, yl), (xr, yr) = sorted(p, key=lambda q: q[0])
    xc, yA, yB = f["xc"], f["yA"], f["yB"]
    yO = yl + (yr - yl) * (xc - xl) / (xr - xl) if xr - xl > 1e-6 else (yl + yr) / 2
    yO = min(max(yO, yA), yB)
    return {"points_frame": [[xl, yl], [xr, yr]], "yO": float(yO), "crown_px": float(yO - yA),
            "root_px": float(yB - yO), "crr": float((yO - yA) / (yB - yO)) if yB - yO > 1e-6 else None}


def cosine_similarity(a, b) -> float | None:
    """cos of the angle between two vectors; None if either has zero length."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na < 1e-9 or nb < 1e-9 else float(a @ b / (na * nb))


def compare_with_reference(res: dict, ref: dict) -> dict:
    """Errors of a run against the clinician reference, normalised by axis length (no mm calibration)."""
    L = res["frame"]["yB"] - res["frame"]["yA"]
    out = {"axis_level_error_frac": abs(res["O_frame"][1] - ref["yO"]) / L if "O_frame" in res else None,
           "crr_abs_error": abs(res["crr"] - ref["crr"]) if "crr" in res and ref["crr"] is not None else None}
    if len(res.get("crest_frame", {})) == 2:
        pred = np.subtract(res["crest_frame"]["right"], res["crest_frame"]["left"])     # left -> right direction
        gt = np.subtract(ref["points_frame"][1], ref["points_frame"][0])
        cs = cosine_similarity(pred, gt)
        out["line_cosine"] = cs                                                          # both vectors run left -> right
        out["line_angle_deg"] = None if cs is None else float(np.degrees(np.arccos(np.clip(cs, -1, 1))))
    for k, side in enumerate(("left", "right")):
        if side in res["crest_frame"]:
            gx, gy = ref["points_frame"][k]
            x, y = res["crest_frame"][side]
            out[f"{side}_axial_error_frac"] = abs(y - gy) / L
            out[f"{side}_point_error_frac"] = float(np.hypot(x - gx, y - gy)) / L
    return out

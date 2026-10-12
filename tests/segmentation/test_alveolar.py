import cv2
import numpy as np

from src.segmentation.alveolar import (AlveolarConfig, contrast_image, locate_alveolar_crest, pick_crest,
                                       side_profile, step_scores, tooth_axis)

H, W = 400, 300
Y_A, Y_B, Y_BONE = 60, 340, 240          # crown tip, apex, true bone crest


def scene(neighbours=True, bone_left=Y_BONE, bone_right=Y_BONE, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 60, np.float32)
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    img = np.where((xx < 150) & (yy >= bone_left), 150, img)           # bone, left of the tooth
    img = np.where((xx >= 150) & (yy >= bone_right), 150, img)         # bone, right of the tooth
    mask = np.zeros((H, W), np.uint8)
    poly = np.array([[110, Y_A], [190, Y_A], [190, 200], [158, Y_B], [142, Y_B], [110, 200]], np.int32)
    cv2.fillPoly(mask, [poly], 1)
    others = np.zeros((H, W), np.uint8)
    if neighbours:
        cv2.rectangle(others, (20, Y_A), (100, min(bone_left, bone_right) - 8), 1, -1)            # neighbouring crowns: very bright
        cv2.rectangle(others, (200, Y_A), (280, min(bone_left, bone_right) - 8), 1, -1)
        img[others > 0] = 235
    img[mask > 0] = 200
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask, others


def rotate(arrs, deg):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return [cv2.warpAffine(a, m, (W, H), flags=cv2.INTER_LINEAR if a.dtype == np.float32 or a.max() > 1 else cv2.INTER_NEAREST,
                           borderValue=0 if a.max() <= 1 else 60) for a in arrs]


def test_axis_points_from_crown_to_root_whatever_the_rotation():
    _, mask, _ = scene()
    for deg in (0, 35, 140, -70):
        m = cv2.warpAffine(mask, cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0), (W, H), flags=cv2.INTER_NEAREST)
        ax = tooth_axis(m, AlveolarConfig())
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], 1)
        wide = pts[((pts - ax["origin"]) @ ax["v"]) < ax["t_top"] + 0.25 * ax["length"]]       # crown end
        narrow = pts[((pts - ax["origin"]) @ ax["v"]) > ax["t_bot"] - 0.25 * ax["length"]]
        u = ax["u"]
        spread = lambda p: np.ptp((p - ax["origin"]) @ u)
        assert spread(wide) > spread(narrow)
        assert abs(ax["length"] - (Y_B - Y_A)) < 15          # 1%/99% percentiles trim the tapering apex


def test_crest_found_on_both_sides_and_crr_matches_truth():
    gray, mask, others = scene()
    r = locate_alveolar_crest(gray, mask, others=others)
    assert r["status"] == "ok", r["warnings"]
    true_crr = (Y_BONE - Y_A) / (Y_B - Y_BONE)                       # 1.8
    assert abs(r["crr"] - true_crr) / true_crr < 0.12
    for side in ("left", "right"):
        assert abs(r["crest_crop"][side][1] - Y_BONE) < 12


def test_result_is_stable_under_rotation():
    gray, mask, others = scene()
    g, m, o = rotate([gray.astype(np.float32), mask, others], 28)
    r = locate_alveolar_crest(np.clip(g, 0, 255).astype(np.uint8), m, others=o)
    assert r["status"] == "ok", r["warnings"]
    assert abs(r["crr"] - 1.8) / 1.8 < 0.15


def test_oblique_crest_line_gives_intermediate_level():
    gray, mask, others = scene(bone_left=Y_BONE - 30, bone_right=Y_BONE + 30)
    r = locate_alveolar_crest(gray, mask, others=others)
    assert r["status"] == "ok"
    assert r["crest_crop"]["left"][1] < r["crest_crop"]["right"][1] - 30     # tilted line recovered
    assert abs(r["O_crop"][1] - Y_BONE) < 15                                  # axis level ~ the mean


def test_neighbour_pixels_are_excluded_from_the_band():
    gray, mask, others = scene()
    cfg = AlveolarConfig()
    I = contrast_image(gray, np.ones_like(mask, bool), cfg)
    usable_all = np.ones_like(mask, bool) & (mask == 0)
    usable_excl = usable_all & (others == 0)
    pa, _ = side_profile(I, mask, usable_all, "left", 70, 230, 280, cfg)
    pe, _ = side_profile(I, mask, usable_excl, "left", 70, 230, 280, cfg)
    assert not np.isnan(pa[100:200]).any() and np.nanmean(pa[100:200]) > 0.6      # neighbour crown leaks in
    assert np.isnan(pe[100:200]).all()                                           # ...unless it is excluded


def test_flat_background_has_no_crest_and_no_crr():
    gray = np.full((H, W), 60, np.uint8)
    _, mask, _ = scene()
    gray[mask > 0] = 200
    r = locate_alveolar_crest(gray, mask)
    assert "crr" not in r and r["status"] == "failed" and any(w.startswith("no_crest") for w in r["warnings"])


def test_single_side_and_truncation_are_flagged():
    gray, mask, others = scene(bone_right=H + 100)               # no bone on the right at all
    r = locate_alveolar_crest(gray, mask, others=others, truncated=True)
    assert "tooth_truncated_apex_or_crown_unreliable" in r["warnings"]
    assert "no_crest_right" in r["warnings"] and "single_side_horizontal_reference" in r["warnings"]


def test_pick_crest_prefers_most_coronal_strong_peak():
    s = np.zeros(100); s[30] = 0.15; s[70] = 0.30
    cfg = AlveolarConfig()
    assert pick_crest(s, 10, 90, cfg)[0] == 70                    # 0.15 < 0.6*0.30 -> the deeper, stronger peak wins
    s[30] = 0.20
    assert pick_crest(s, 10, 90, cfg)[0] == 30                    # 0.20 >= 0.18 -> the more coronal one wins
    assert pick_crest(np.zeros(50), 5, 45, cfg)[0] is None


def test_reference_from_points_matches_algorithm_geometry_and_errors():
    from src.segmentation.alveolar import compare_with_reference, reference_from_points
    gray, mask, others = scene()
    r = locate_alveolar_crest(gray, mask, others=others)
    gt = reference_from_points(r, [[190, Y_BONE], [110, Y_BONE]])               # order-insensitive
    assert abs(gt["crr"] - 1.8) / 1.8 < 0.12
    cmp = compare_with_reference(r, gt)
    assert cmp["axis_level_error_frac"] < 0.06 and cmp["crr_abs_error"] < 0.4
    assert reference_from_points({"warnings": []}, [[1, 1], [2, 2]]) is None


def test_cosine_similarity_of_crest_lines():
    from src.segmentation.alveolar import compare_with_reference, cosine_similarity, reference_from_points
    assert cosine_similarity([1, 0], [2, 0]) == 1.0 and abs(cosine_similarity([1, 0], [0, 3])) < 1e-12
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0 and cosine_similarity([0, 0], [1, 1]) is None
    gray, mask, others = scene(bone_left=Y_BONE - 30, bone_right=Y_BONE + 30)       # tilted true crest line
    r = locate_alveolar_crest(gray, mask, others=others)
    tilted = reference_from_points(r, [[110, Y_BONE - 30], [190, Y_BONE + 30]])
    flat = reference_from_points(r, [[110, Y_BONE], [190, Y_BONE]])
    c_good, c_flat = compare_with_reference(r, tilted), compare_with_reference(r, flat)
    assert c_good["line_cosine"] > 0.99 and c_good["line_angle_deg"] < 8
    assert c_flat["line_cosine"] < c_good["line_cosine"] and c_flat["line_angle_deg"] > 15


def test_scan_profile_measures_half_widths_perpendicular_to_the_axis():
    from src.segmentation.alveolar import axis_warp, scan_profile
    _, mask, _ = scene()
    cfg = AlveolarConfig()
    ax = tooth_axis(mask, cfg)
    assert ax["elongation"] > 2 and "axis_unreliable_low_elongation" not in ax["warnings"]
    sp = scan_profile(mask, ax, cfg, n_levels=21)
    assert len(sp["t_frac"]) >= 19
    crown = sp["t_frac"] < 0.4
    assert abs(sp["left_dist"][crown].mean() - 40) < 4 and abs(sp["right_dist"][crown].mean() - 40) < 4     # crown half-width 40 px
    assert sp["left_dist"][-3] < sp["left_dist"][2] / 2                                                  # root is narrower
    # arrows run from the contour to the axis: the foot of each scan line is on the PCA axis
    c, v = ax["origin"], ax["v"]
    off = [abs(v[0] * (f - c)[1] - v[1] * (f - c)[0]) for f in sp["feet"]]
    assert max(off) < 1.5


def test_flat_blob_gets_a_low_elongation_warning():
    blob = np.zeros((200, 200), np.uint8); cv2.circle(blob, (100, 100), 60, 1, -1)
    assert "axis_unreliable_low_elongation" in tooth_axis(blob, AlveolarConfig())["warnings"]


def test_extreme_end_points_follow_max_projection_from_the_centroid():
    _, mask, _ = scene()
    ax = tooth_axis(mask, AlveolarConfig(end_percentile=0.0))
    ys, xs = np.nonzero(mask)
    t = (np.stack([xs, ys], 1) - ax["origin"]) @ ax["v"]
    assert abs(ax["t_top"] - t.min()) < 1e-9 and abs(ax["t_bot"] - t.max()) < 1e-9
    assert abs(ax["origin"][0] - xs.mean()) < 1e-9 and abs(ax["origin"][1] - ys.mean()) < 1e-9      # C' = centroid
    assert abs(ax["length"] - (Y_B - Y_A)) < 4


def spurious_right_step_scene():
    gray, mask, others = scene()
    yy, xx = np.mgrid[:H, :W]
    g = gray.astype(np.float32)
    g[(xx >= 150) & (yy >= 200) & (yy < Y_BONE) & (mask == 0) & (others == 0)] = 115     # extra step well above the true crest, right side only
    return np.clip(g, 0, 255).astype(np.uint8), mask, others


def test_weighted_pairing_prefers_a_level_consistent_pair_over_a_stronger_coronal_distractor():
    gray, mask, others = spurious_right_step_scene()
    ind = locate_alveolar_crest(gray, mask, others=others, cfg=AlveolarConfig(pair_mode="independent"))
    wtd = locate_alveolar_crest(gray, mask, others=others, cfg=AlveolarConfig(pair_mode="weighted"))
    assert wtd["status"] == "ok" and wtd["pair"]["terms"]["level"] > 0.8
    e_w = abs(wtd["crest_crop"]["right"][1] - Y_BONE)
    e_i = abs(ind["crest_crop"]["right"][1] - Y_BONE) if "right" in ind["crest_crop"] else 999
    assert e_w < 15 and e_w <= e_i
    assert set(wtd["pair"]["terms"]) == {"step", "level", "symmetry", "angle", "anatomy"}


def test_weighted_mode_matches_independent_on_a_clean_scene():
    gray, mask, others = scene()
    a = locate_alveolar_crest(gray, mask, others=others, cfg=AlveolarConfig(pair_mode="independent"))
    b = locate_alveolar_crest(gray, mask, others=others, cfg=AlveolarConfig(pair_mode="weighted"))
    assert abs(a["crr"] - b["crr"]) < 0.15

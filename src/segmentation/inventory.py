"""COCO polygon inventory for data/seg_dataset: per-file checks, annotation validity, review status."""

from __future__ import annotations

import json
from pathlib import Path

from src.detection.inventory import inspect_image
from src.detection.provenance import sha256_file

from .config import SegConfig
from .masks import polygon_issue

SUBSETS = ("train", "val", "test")

# User confirmation (2026-10-12): all tooth annotations in the current data version are
# human-made and expert-confirmed. Metal status was never stated -> unknown.
def review_status(data_version: str) -> dict:
    return {
        "tooth": {
            "status": "human_expert_confirmed",
            "source": "user confirmation in spec clarification session",
            "date": "2026-10-12",
            "applies_to": data_version,
        },
        "metal": {"status": "unknown", "note": "not inferred from tooth status"},
    }


def load_coco(root: Path, subset: str) -> dict:
    return json.loads((root / subset / "annotations" / "instances_default.json").read_text())


def coco_hashes(root: Path) -> dict[str, str]:
    return {s: sha256_file(root / s / "annotations" / "instances_default.json") for s in SUBSETS}


def build_inventory(root: str | Path, cfg: SegConfig, limit: int | None = None) -> tuple[list[dict], dict]:
    """-> (items, annotations_by_image_id). Every image gets included/excluded + reason."""
    root = Path(root)
    items: list[dict] = []
    anns_by_id: dict[str, list[dict]] = {}
    seen_ids: dict[str, str] = {}
    for subset in SUBSETS:
        coco = load_coco(root, subset)
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        tooth_ids = {i for i, n in cats.items() if n == cfg.data.tooth_category}
        known = tooth_ids | {i for i, n in cats.items() if n in cfg.data.ignore_categories}
        by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        listed = {im["file_name"] for im in coco["images"]}
        imgs = sorted(coco["images"], key=lambda im: im["file_name"])
        if limit:
            imgs = imgs[: max(1, limit // 3)]
        for im in imgs:
            iid = Path(im["file_name"]).stem
            path = root / subset / "images" / im["file_name"]
            reasons: list[str] = []
            if iid in seen_ids:
                reasons.append(f"duplicate_image_id:{seen_ids[iid]}")
            seen_ids[iid] = subset
            if not path.exists():
                reasons.append("missing_image_file")
                base = {"image_id": iid, "source_path": str(path), "sha256": None, "width": None,
                        "height": None, "channels": None, "bit_depth": None, "phash": None}
                inv = base
                status_img = "excluded"
            else:
                it = inspect_image(path, iid)
                inv = it.to_dict()
                status_img = it.status
                if it.status != "included":
                    reasons.append(it.reason)
                elif (it.width, it.height) != (im["width"], im["height"]):
                    reasons.append("size_mismatch_with_coco")
            anns = by_img.get(im["id"], [])
            bad_cat = [a["id"] for a in anns if a["category_id"] not in known]
            if bad_cat:
                reasons.append(f"unknown_category:{bad_cat}")
            tooth = [a for a in anns if a["category_id"] in tooth_ids]
            metal = [a for a in anns if a["category_id"] not in tooth_ids and a["category_id"] in known]
            invalid = []
            for a in tooth:
                segs = a.get("segmentation")
                if not isinstance(segs, list) or not segs:
                    invalid.append({"ann_id": a["id"], "reason": "no_polygon"})
                    continue
                for poly in segs:
                    r = polygon_issue(poly, im["width"], im["height"])
                    if r:
                        invalid.append({"ann_id": a["id"], "reason": r})
            if invalid:
                reasons.append("invalid_polygon:" + ",".join(sorted({i["reason"] for i in invalid})))
            if not tooth and not bad_cat:
                reasons.append("empty_tooth_annotation_unconfirmed")
            inv.update(
                subset=subset, coco_image_id=im["id"], n_tooth=len(tooth), n_metal=len(metal),
                invalid_polygons=invalid, empty_confirmed=False,
                status="excluded" if (reasons or status_img != "included") else "included",
                exclude_reason="; ".join(reasons),
            )
            items.append(inv)
            anns_by_id[iid] = anns
        for fn in sorted({p.name for p in (root / subset / "images").glob("*")} - listed):
            items.append({"image_id": Path(fn).stem, "subset": subset, "status": "excluded",
                          "exclude_reason": "image_without_coco_entry", "source_path": str(root / subset / "images" / fn)})
    return items, anns_by_id

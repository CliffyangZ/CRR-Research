"""Pair YOLO tooth boxes with COCO tooth polygons one-to-one; quarantine anything uncertain.

IoU is computed between the detection box and the polygon's axis-aligned bounding box.
A box is accepted iff best IoU >= min_iou and best - second >= margin; when several boxes
choose the same polygon only a leader ahead by >= margin keeps it. Nothing is guessed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_yolo_boxes(label_path: str | Path, width: int, height: int, tol: float = 1.0) -> list[dict]:
    """YOLO normalised cx cy w h -> original-pixel xyxy; flags degenerate / out-of-image boxes."""
    out = []
    p = Path(label_path)
    lines = p.read_text().splitlines() if p.exists() else []
    for k, line in enumerate(l for l in lines if l.strip()):
        parts = line.split()
        _, cx, cy, w, h = parts[0], *map(float, parts[1:5])
        xyxy = [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
        flag = ""
        if w <= 0 or h <= 0 or xyxy[2] - xyxy[0] < 1 or xyxy[3] - xyxy[1] < 1:
            flag = "degenerate_box"
        elif xyxy[0] < -tol or xyxy[1] < -tol or xyxy[2] > width + tol or xyxy[3] > height + tol:
            flag = "box_out_of_image"
        out.append({"box_index": k, "xyxy": xyxy, "flag": flag})
    return out


def polygon_bbox(ann: dict) -> list[float]:
    pts = np.concatenate([np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in ann["segmentation"]])
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def bbox_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pair_boxes(boxes: list[dict], polygons: list[dict], min_iou: float = 0.5, margin: float = 0.1
               ) -> tuple[list[dict], list[dict]]:
    """-> (pairs, orphan_polygons). One pair row per box, in box order.

    row: box_index, polygon_ann_id|None, iou, second_iou, status(paired|quarantined), reason.
    """
    pboxes = [polygon_bbox(a) for a in polygons]
    rows: list[dict] = []
    for b in boxes:
        row = {"box_index": b["box_index"], "polygon_ann_id": None, "iou": 0.0, "second_iou": 0.0,
               "status": "quarantined", "reason": "", "_poly": None}
        if b["flag"]:
            row["reason"] = b["flag"]
        elif not polygons:
            row["reason"] = "no_polygon"
        else:
            ious = np.array([bbox_iou(b["xyxy"], pb) for pb in pboxes])
            order = np.argsort(-ious, kind="stable")
            best = float(ious[order[0]])
            second = float(ious[order[1]]) if len(order) > 1 else 0.0
            row.update(iou=best, second_iou=second, polygon_ann_id=polygons[order[0]]["id"], _poly=int(order[0]))
            if best < min_iou:
                row["reason"], row["polygon_ann_id"] = "low_iou", None
            elif best - second < margin:
                row["reason"] = "ambiguous_second_best"
            else:
                row["status"] = "candidate"
        rows.append(row)

    by_poly: dict[int, list[dict]] = {}
    for r in rows:
        if r["status"] == "candidate":
            by_poly.setdefault(r["_poly"], []).append(r)
    for rs in by_poly.values():
        rs.sort(key=lambda r: -r["iou"])
        leader_ok = len(rs) == 1 or rs[0]["iou"] - rs[1]["iou"] >= margin
        for i, r in enumerate(rs):
            if i == 0 and leader_ok:
                r["status"] = "paired"
            else:
                r["status"], r["reason"] = "quarantined", "polygon_contested"
    paired_ids = {r["polygon_ann_id"] for r in rows if r["status"] == "paired"}
    for r in rows:
        r.pop("_poly")
    orphans = [{"polygon_ann_id": a["id"], "polygon_bbox": pb, "reason": "orphan_polygon"}
               for a, pb in zip(polygons, pboxes) if a["id"] not in paired_ids]
    return rows, orphans


def collect_sources(det_root: str | Path, seg_root: str | Path, seg_cfg, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    """Join the two sources per image. -> (records, det_only_images).

    record: item (seg inventory row), anns (all COCO anns), tooth_anns, boxes, det_subset, det_sha256,
    image_issue ('' if usable). An image is unusable if the inventory excluded it, the subsets differ,
    or the detection copy of the image is not byte-identical to the segmentation copy.
    """
    from src.detection.provenance import sha256_file

    from .inventory import build_inventory, load_coco

    det_root, seg_root = Path(det_root), Path(seg_root)
    items, anns_by_id = build_inventory(seg_root, seg_cfg, limit)
    tooth_ids = {c["id"] for c in load_coco(seg_root, "train")["categories"] if c["name"] == seg_cfg.data.tooth_category}
    det_files = {p.stem: p for s in ("train", "val", "test") for p in (det_root / "images" / s).glob("*")}
    det_subset = {p.stem: p.parent.name for p in det_files.values()}
    records = []
    for it in items:
        iid = it["image_id"]
        if iid not in anns_by_id:          # file listed without a COCO entry
            continue
        issue = ""
        if it["status"] != "included":
            issue = "image_excluded: " + it["exclude_reason"]
        elif iid not in det_files:
            issue = "missing_in_detection_dataset"
        elif det_subset[iid] != it["subset"]:
            issue = f"subset_mismatch: det={det_subset[iid]} seg={it['subset']}"
        elif sha256_file(det_files[iid]) != it["sha256"]:
            issue = "image_content_mismatch"
        boxes = []
        if iid in det_files:
            boxes = read_yolo_boxes(det_root / "labels" / det_subset[iid] / f"{iid}.txt", it["width"] or 0, it["height"] or 0) \
                if it.get("width") else []
        records.append({"item": it, "anns": anns_by_id[iid],
                        "tooth_anns": [a for a in anns_by_id[iid] if a["category_id"] in tooth_ids],
                        "boxes": boxes, "det_subset": det_subset.get(iid), "image_issue": issue})
    seg_ids = {it["image_id"] for it in items}
    det_only = [{"image_id": i, "subset": det_subset[i]} for i in sorted(set(det_files) - seg_ids)] if not limit else []
    return records, det_only

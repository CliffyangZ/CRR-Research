"""Statistics, overlays and the Markdown data report for the ROI dataset (unit: ROI, population: subset)."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

SUBSETS = ("train", "val", "test")


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    q = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return dict(zip(["min", "p5", "p25", "median", "p75", "p95", "max"], [round(float(x), 4) for x in q]))


def compute_stats(pairs: list[dict], rois: list[dict], orphans: list[dict], n_polygons: dict, image_issues: list[dict]) -> dict:
    out: dict = {"per_subset": {}, "orphan_polygons": len(orphans), "image_issues": image_issues}
    for s in SUBSETS:
        ps = [p for p in pairs if p["subset"] == s]
        rs = [r for r in rois if r["subset"] == s]
        quar = Counter(p["reason"] for p in ps if p["status"] == "quarantined")
        out["per_subset"][s] = {
            "boxes": len(ps), "polygons": n_polygons.get(s, 0), "rois": len(rs),
            "quarantined": sum(quar.values()), "quarantined_by_reason": dict(quar),
            "iou_paired": quantiles([p["iou"] for p in ps if p["status"] == "paired"]),
            "iou_all_best": quantiles([p["iou"] for p in ps if p["iou"] > 0]),
            "second_iou_paired": quantiles([p["second_iou"] for p in ps if p["status"] == "paired"]),
            "mask_frac_in_expanded": quantiles([r["mask_frac_in_expanded"] for r in rs]),
            "n_target_truncated_by_roi": sum(r["target_truncated_by_roi"] for r in rs),
            "n_truncated_at_source": sum(r["truncated_at_source"] for r in rs),
            "n_neighbor_in_roi": sum(r["neighbor_in_roi_frac"] > 0 for r in rs),
            "neighbor_in_roi_frac": quantiles([r["neighbor_in_roi_frac"] for r in rs if r["neighbor_in_roi_frac"] > 0]),
            "n_outside_image": sum(any(v > 0 for v in r["outside_image_px"].values()) for r in rs),
        }
    return out


def overlay_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    layer = base.copy()
    layer[mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(layer, 0.4, base, 0.6, 0)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, (0, 255, 255), 1)
    return out


def overlay_pairing(gray: np.ndarray, box, polygons: list[dict], target_id, label: str, max_side: int = 640) -> np.ndarray:
    """Original image with the detection box (white) and tooth polygons (green = chosen, red = others)."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for a in polygons:
        col = (0, 220, 0) if a["id"] == target_id else (0, 0, 230)
        for poly in a["segmentation"]:
            cv2.polylines(img, [np.round(np.asarray(poly).reshape(-1, 2)).astype(np.int32)], True, col, max(2, gray.shape[1] // 400))
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), max(2, gray.shape[1] // 300))
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def overlay_all(gray: np.ndarray, anns: list[dict], boxes: list[dict], rows: list[dict], max_side: int = 900) -> np.ndarray:
    """All polygons (tooth green, other categories yellow) plus every detection box (paired white, quarantined red)."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = max(2, gray.shape[1] // 400)
    for a in anns:
        col = (0, 220, 0) if a["category_id"] == 1 else (0, 220, 230)
        for poly in a["segmentation"]:
            cv2.polylines(img, [np.round(np.asarray(poly).reshape(-1, 2)).astype(np.int32)], True, col, t)
    for b, r in zip(boxes, rows):
        x0, y0, x1, y1 = [int(round(v)) for v in b["xyxy"]]
        col = (255, 255, 255) if r["status"] == "paired" else (0, 0, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, t + 1)
        cv2.putText(img, f"{r['iou']:.2f}", (x0 + 4, y0 + 24 + 8 * t), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, gray.shape[1] / 1200), col, t)
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def render_markdown(manifest: dict, stats: dict, overlays: dict[str, str]) -> str:
    L = [f"# ROI 資料集驗證報告：`{manifest['data_version']}`", "",
         f"- 建置：`{manifest['run_id']}`；設定 hash `{manifest['config_hash']}`；命令 `{manifest['command']}`",
         f"- 單位＝ROI（一個偵測框），母體＝各子集；本報告僅含資料統計，**不含模型表現**。",
         f"- 多邊形標註審閱：{manifest['annotation_review']['polygons']['status']}"
         f"（使用者 {manifest['annotation_review']['polygons']['date']} 確認）；偵測框審閱：{manifest['annotation_review']['boxes']['status']}；Metal：{manifest['annotation_review']['metal']['status']}。",
         f"- 配對規則：IoU≥{manifest['config']['pairing']['min_iou']}、與次佳差≥{manifest['config']['pairing']['second_margin']}、"
         f"目標牙占擴張矩形≥{manifest['config']['pairing']['min_mask_frac']}；ROI＝每側外擴 {manifest['config']['roi']['expand_frac_per_side']:.0%}，letterbox {manifest['config']['roi']['size']}²；前處理僅 CLAHE。",
         "", "## 子集統計", "",
         "| 子集 | 框 | 多邊形 | ROI | 隔離 | 越界補邊 ROI | 目標被 ROI 切到 | 原圖已截斷 | 含鄰牙 |", "|---|---|---|---|---|---|---|---|---|"]
    for s, d in stats["per_subset"].items():
        L.append(f"| {s} | {d['boxes']} | {d['polygons']} | {d['rois']} | {d['quarantined']} | {d['n_outside_image']} | "
                 f"{d['n_target_truncated_by_roi']} | {d['n_truncated_at_source']} | {d['n_neighbor_in_roi']} |")
    L += ["", "## 隔離原因", ""]
    for s, d in stats["per_subset"].items():
        L.append(f"- {s}: " + (", ".join(f"{k}={v}" for k, v in sorted(d["quarantined_by_reason"].items())) or "無"))
    L += [f"- 未被配對的多邊形（orphan）：{stats['orphan_polygons']}", "", "## IoU 與遮罩占比分布（分位數）", ""]
    for s, d in stats["per_subset"].items():
        L += [f"- **{s}** 已配對 IoU：{d['iou_paired']}；次佳 IoU：{d['second_iou_paired']}；目標占擴張矩形：{d['mask_frac_in_expanded']}；鄰牙占比（含鄰牙者）：{d['neighbor_in_roi_frac']}"]
    L += ["", "## 影像層級問題", "", (", ".join(f"{i['image_id']}({i['issue']})" for i in stats["image_issues"]) or "無"), "",
          "## 切分與洩漏", "",
          f"- 來源：{manifest['split']['assignment_source']}（{manifest['split']['reason']}）；跨子集近重複組：{manifest['split']['leakage_check']['cross_split_groups']}。",
          f"- 殘餘風險：{manifest['split']['residual_risk']}", "", "## 疊圖", ""]
    for k, v in overlays.items():
        L.append(f"- {k}: `{v}`")
    L += ["", "## 限制與未量測", "",
          "- 無病人識別資訊，病人層級洩漏未排除；偵測框審閱狀態未知，標籤品質受兩來源不一致影響。",
          "- 被隔離的框可能集中於難例（相鄰牙、偏差大的框），造成選擇偏差；隔離比例見上表。",
          "- 單一資料版本；門檻（IoU、次佳差、最小遮罩占比）為登錄值，若曾調整須見 manifest 理由欄。",
          "- 未訓練模型：任何模型表現、穩健性、臨床能力均**未量測**。"]
    return "\n".join(L) + "\n"

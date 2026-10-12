# Data Model

## SourceVersion（`dataset_manifest.json`）
`data_version`（`roi_<12hex>`）、`detection_root`、`seg_root`、`label_files_sha256`、`coco_sha256{train,val,test}`、`config_hash`、`annotation_review`：
`{polygons:{status:"human_expert_confirmed", source:"user confirmation", date:"2026-10-12", applies_to:<seg data version>}, boxes:{status:"unknown", source_note:"data/tooth_detection/labels (per detection_dataset/README.md)"}, metal:{status:"unknown"}}`。

## ImageItem
`image_id`（檔名主幹）、`subset_det`、`subset_seg`（必須相等，否則 `subset_mismatch` 並隔離整張）、`sha256`、`width,height,channels,bit_depth`、`phash`、`group_id`、`status`（included／excluded）、`exclude_reason`、`n_boxes`、`n_polygons`。

## BoxPolygonPair（`pairs.json`，每個框一列）
| 欄位 | 說明 |
|---|---|
| `roi_id` | `<image_id>_<box_index:02d>`，box_index 為標籤檔列序 |
| `box_xyxy` | 原圖像素座標（由 YOLO 正規化換算） |
| `polygon_ann_id` / `polygon_bbox` | 配對的 COCO 標註；未配對為 null |
| `iou`、`second_iou` | 最佳／次佳 IoU |
| `status` | `paired` \| `quarantined` |
| `reason` | 見 research R4 與 R5；`paired` 時為空 |

不變量：同一 `polygon_ann_id` 至多出現在一個 `paired` 列；`quarantined` 列不產生 ROI。

## RoiSample（`roi_index.json` 一列，僅 `paired` 且通過品質檢查者）
`roi_id`、`image_id`、`subset`、`box_xyxy`、`expanded_xyxy`（可超出原圖）、`outside_image_px{left,top,right,bottom}`、`letterbox`（`LetterboxRecord`）、`polygon_ann_id`、`mask_area_px`（ROI 256 上）、`target_truncated_by_roi`、`truncated_at_source`、`neighbor_in_roi_frac`、`files{image,mask}`（相對資料集根目錄，`<subset>/images/<roi_id>.png`、`<subset>/masks/<roi_id>.png`）。

**目錄結構**：`data/derived/roi_<dv>/{train,val,test}/{images,masks,masks_vis,roi_index.json}（masks 值為 0／1；masks_vis 為 0／255 檢視副本）`，根目錄另有 `roi_index.json`（全部 ROI）、`pairs.json`、`quarantine.json`、`splits.json`、`stats.json`、`dataset_manifest.json`。
遮罩值域 {0,1}；補邊 0；影像為 CLAHE 後 uint8 灰階 PNG。

## Quarantine（`quarantine.json`）
`pairs.json` 中 `quarantined` 列加上 `orphan_polygons[]`（未被配對的多邊形）與影像層級排除，皆含原因。

## SplitManifest（`splits.json`）
沿用 spec 002 格式：`assignments{image_id→subset}`、`groups`、`leakage_check`、`assignment_source(existing|regrouped)`、`reason`、`residual_risk`、`frozen`；ROI 子集＝來源影像子集。

## BuildManifest
`build_id`、`status(running|succeeded|failed)`、`failure_reason`、`command`、`git`、`env`、`seed`、`config`（內嵌）、`split_sha256`、`counts{subset→{boxes,polygons,paired,quarantined_by_reason,rois}}`、`output_dir`、`exploratory:false`（資料建置為確定性）。

## 狀態轉換
Build：`running → succeeded | failed`（終態不可覆寫）。Split：`draft → frozen`（報告產生後）。

# Data Model

## DatasetVersion（`data/derived/seg_<dv>/dataset_manifest.json`）
- `data_version`：由輸入 COCO 檔 sha256 + 前處理設定 hash 產生
- `source_root`：`data/seg_dataset`；`coco_sha256{train,val,test}`
- `annotation_review`：`{tooth:{status:"human_expert_confirmed", source:"user confirmation", date:"2026-10-12", applies_to:<data_version>}, metal:{status:"unknown"}}`
- `preprocess_config_hash`、`items[]`

## ImageItem（`inventory.json` 一列）
| 欄位 | 說明 |
|---|---|
| `image_id`, `file_name`, `subset` | COCO id、檔名、原始子集 |
| `sha256`, `phash`, `group_id` | 追溯與近重複分組 |
| `width,height,channels,bit_depth` | 逐檔檢查結果 |
| `n_tooth`, `n_metal`, `n_invalid_polygons` | 標註統計 |
| `status` | `included` \| `excluded` |
| `exclude_reason` | 損毀／缺配對／未知類別／退化多邊形／越界／非 8-bit 未處理…；`status=excluded` 時必填 |
| `empty_confirmed` | 無牙齒且經確認才可作背景樣本 |
| `truncated_at_source` | 原圖已有截斷牙齒 |

驗證：included ⇒ exclude_reason 為空；缺標註不得視為背景；同一 `group_id` 僅出現在一個 subset。

## SplitManifest（`splits.json`，凍結）
`{train:[image_id], val:[...], test:[...], source:"existing_dirs"|"resplit", seed, reason, leakage_check:{cross_subset_groups:0}, group_basis:"phash+filename_proximity", residual_risk:"no patient id"}`；`sha256` 凍結後不可改，修改須產生新版本並保存舊檔。

## SegSample（執行時）
`image(512×512 uint8)`, `mask(512×512 uint8 {0,1})`, `LetterboxRecord`（沿用 `src.detection.letterbox`：orig_w/h, scale, new_w/h, pad_*）, `aug{photometric, geometry: applied|cancelled|not_selected, params, cancel_reason}`。
遮罩值域恆為 {0,1}；補邊為 0。

## RunManifest（`reports/segmentation/<run_id>/run_manifest.json`）
`run_id`, `status`(`running|succeeded|failed`), `failure_reason`, `command`, `git{commit,dirty,diff_hash}`, `env{python,torch,cuda,device,cudnn_deterministic,amp}`, `seed`, `config`(完整 YAML 內嵌), `data_version`, `split_sha256`, `init:"random_kaiming"`, `checkpoints{best:{path,sha256,epoch,val_dice},last:{...}}`, `selection_rule`, `stopping{epochs_run,reason}`, `exploratory:true`, `nondeterminism_note`。

## EvaluationReport（`eval/metrics.json` + `per_image.csv`）
欄位見 [contracts/evaluation-protocol.md](contracts/evaluation-protocol.md)；含 `run_id`, `weights_sha256`, `subset`, `n_images`, `n_valid_pixels`, `threshold`, `annotation_review`, `aggregate{dice,iou,precision,recall}`（像素累計）, `per_image_mean{...,n_valid}`, `n_all_background_gt`, `fp_pixels_on_all_background`, `undefined_reasons`。

## 狀態轉換
Run：`running → succeeded | failed`（終態不可覆寫）。Split：`draft → frozen`（正式訓練前）。評估只對 `frozen` split 與 `succeeded` run 執行。

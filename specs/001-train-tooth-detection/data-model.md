# Data Model

所有實體以 JSON／YAML 檔保存於 `data/derived/<data_version>/manifests/` 或 `reports/detection/<run_id>/`。

## DatasetVersion（資料版本）
| 欄位 | 說明 |
|---|---|
| data_version | 例 `dv-YYYYMMDD-<hash8>` |
| items[] | `image_id`, `source_path`, `sha256`, `width`, `height`, `mode`, `bit_depth`, `channels`, `intensity_min/max`, `group_id`, `phash`, `status`(included/excluded/quarantined), `reason` |
| preprocess_config_hash | 指向 PreprocessConfig |

驗證：sha256 唯一（重複視為近重複並同組）；非 8-bit 或通道不一致 → status=quarantined；每個 excluded 必須有 reason。

## AnnotationVersion
`annotation_version`, `source`(manual/pseudo/derived-from-seg), `review_status`(confirmed/unreviewed/pending), `boxes[]`: `image_id`, `class=tooth(0)`, `xyxy`(原圖像素), `truncated`(bool), `flag`(待審閱)。驗證：框在影像內、寬高 >0；無標註影像不得視為無牙。

## PreprocessConfig
`nlm{h,template,search}`, `clahe{clip,tile}`, `usm{sigma,amount,threshold}`, `input_mapping`(uint8/255), `letterbox{size,pad_value,interp}`, `augment{brightness,contrast,rotate_deg,translate_frac,prob,order,copies,seed}`；序列化後取 hash，登錄於訓練前。

## LetterboxRecord
`image_id`, `orig_w/h`, `scale`, `new_w/h`, `pad_left/top/right/bottom`。不變式：`new_w + pad_left + pad_right = size`；往返框誤差 ≤1 px。

## SplitManifest
`split_id`, `seed`, `ratios`, `grouping_rule`, `assignments{image_id→train|val|test}`, `leakage_check{cross_split_groups:0}`, `residual_risk`(文字，缺病人資訊必填), `locked_at`。測試集鎖定後不可變。

## AugmentedSample
`aug_id`, `source_image_id`, `seed`, `photometric_applied`, `geometry`{applied|cancelled, params, cancel_reason}, `boxes`。僅 train 子集產生。

## ModelCandidate
`family`(v8/v11/v26), `ultralytics_version`, `weights_name`, `weights_sha256`, `size`(s), `license`, `source_url`, `fit_notes`, `vram_estimate`。

## ExperimentRun
`run_id`, `model`, `status`(success/failed), `failure_reason`, `command`, `git_commit`, `git_dirty`(+diff hash), `env`(python, torch, cuda, driver, gpu), `seed`, `data_version`, `annotation_version`, `split_id`, `preprocess_config_hash`, `hyperparams`, `selection_rule`, `selected_weights{path,sha256,epoch}`, `exploratory=true`。

## EvaluationReport
`run_id`, `population`(split, n_images, n_boxes), `iou_thresholds`, `conf_threshold`(來自 val), `metrics{AP, AP50, P, R, F1}`(值或 `undefined_reason`), `errors{missed, false_positive, duplicate, localization}` 計數 + 例子, `seed_variance="未量測"`, `limitations`。

## 狀態轉換
`DatasetVersion: draft → validated → frozen`；`SplitManifest: proposed → checked → locked`；`ExperimentRun: created → training → trained|failed → selected → evaluated`。

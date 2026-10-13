# Data Model: 牙齒分割模型架構比較

## Candidate（候選架構，登錄於協議）
| 欄位 | 說明 |
|---|---|
| name | `unet` \| `unetpp` \| `attention_unet` |
| design_orientation | 設計取向（baseline／巢狀密集跳接／跳接注意力閘門） |
| implementation | 檔案路徑與 git commit |
| source_license | 方法論文與「專案自有實作」 |
| config_path / config_hash | `configs/segmentation/roi_<name>_v1.yaml` 與其 hash |
| n_parameters | 實測值（run manifest 填入，登錄時為空） |
| init | 固定 `random_kaiming`；`pretrained: null` |
| role | `primary`；等容量變體為 `secondary`（未執行則標未量測） |

驗證：三個設定檔除 `model` 區塊外必須逐欄相同（以 hash 比對去除 model 後的內容）。

## Protocol（比較協議）
`protocol_version`、`frozen_at`、`frozen_git_commit/dirty`、`data_version`（roi_2a0c9441396f）、`split_sha256`、
`recipe`（優化器、排程、損失、AMP、batch、max_epochs、早停、選模、閾值 0.5）、`batch_fallback`（R5）、
`seeds`（[20261012, 20261013, 20261014]）、`candidates[]`、`metrics`（含零分母規則）、
`statistics`（bootstrap B、種子、單位＝原圖、證據門檻）、`protocol_hash`。
狀態：`draft → frozen`；frozen 後唯讀，變更＝新版本＋CHANGELOG。

## Run（實驗執行，一個 candidate × seed）
`run_id`、`candidate`、`seed`、`protocol_version/hash`（凍結前為 null 並 `exploratory: true`）、`status`（running／succeeded／failed／incomplete）、
`failure_reason`、`command`、`git{commit,dirty}`、`environment{python,torch,cuda,device}`、`config_hash`、`data_version`、`split_sha256`、
`init`、`n_parameters`、`batch{size,accum,override_reason}`、`train_seconds`、`peak_gpu_mem_mb`、`stopping{epochs_run,reason}`、
`checkpoints{best,last}{path,sha256,epoch,val_dice}`、`nondeterminism_note`。
狀態轉換：`running → succeeded | failed`；失敗 run 不覆寫、不刪，由新 run_id 重跑並於 protocol 的 run 清單標註替代關係（同 seed 只計一個 succeeded，且須列出所有嘗試）。

## RunSet
協議下 9 個計畫 run 的清單（candidate、seed → 採用的 run_id，及被取代的失敗 run_id）。比較前必須完整，缺者報告列「未完成」。

## TestLock
`protocol_hash`、`evaluated_runs[{run_id, weights_sha256}]`、`timestamp`、`violations[{time, reason}]`。存在即表示 test 已用；`violations` 非空使確認性結論失效。

## EvaluationResult（每 run 一份，`eval/`）
- `per_roi.csv`：roi_id、source_image_id、subset、tp、fp、fn、tn(valid)、dice、iou、precision、recall、hd95、boundary_f1、undefined 原因、truncated、has_neighbor。
- `metrics.json`：像素累計主要指標、逐 ROI 平均、n_rois、n_images、n_valid_pixels、閾值、weights sha256、`inference_ms_per_roi`。
- 邊界指標在 pred 或 gt 為空時 `null` 並附原因。

## Comparison（`compare_<protocol_version>/`）
`seed_table`（模型×種子×指標）、`model_summary`（平均、SD、n）、`paired_differences[candidate vs baseline]`
（`mean_diff`、`ci95`、`B`、`unit=source_image`、`n_images`、`seed_sd{baseline,candidate}`、`verdict` ∈ {`evidence_better`,`no_demonstrable_difference`,`evidence_worse`}、`roi_wins/losses/ties`）、
`cost_table`、`error_cases`（三類×模型）、`limitations`、`unmeasured`。

## 關係
Protocol 1—N Candidate；Protocol 1—1 RunSet 1—N Run；Run 1—1 EvaluationResult（test）；Comparison 彙整 EvaluationResult 並引用 TestLock。

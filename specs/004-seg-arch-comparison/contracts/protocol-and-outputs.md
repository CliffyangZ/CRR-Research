# 協議與輸出檔契約

## `configs/segmentation/compare_protocol_v1.yaml`（節錄）
```yaml
protocol_version: v1
data_version: roi_2a0c9441396f
split_sha256: <from splits.json>
seeds: [20261012, 20261013, 20261014]
candidates:
  - {name: unet,           config: configs/segmentation/roi_unet_v1.yaml,           role: primary}
  - {name: unetpp,         config: configs/segmentation/roi_unetpp_v1.yaml,         role: primary}
  - {name: attention_unet, config: configs/segmentation/roi_attention_unet_v1.yaml, role: primary}
threshold: 0.5
batch_fallback: {batch_size: 8, grad_accum: 2}
metrics: {primary: pixel_accumulated_dice, secondary: [iou, precision, recall, hd95_px, boundary_f1_tol2px]}
statistics: {unit: source_image, method: paired_cluster_bootstrap, B: 10000, seed: 20261012, ci: 0.95,
             evidence_rule: "ci_excludes_zero AND abs(mean_diff) > max(seed_sd_baseline, seed_sd_candidate)"}
```
## 輸出佈局
```
reports/segmentation/<run_id>/            run_manifest.json  history.csv  eval/{metrics.json,per_roi.csv,eval_manifest.json}
models/segmentation/<run_id>/             best.pt  last.pt                 (不入版本庫)
reports/segmentation/compare_<ver>/       protocol_freeze.json  test_lock.json  runset.json
                                          comparison.json  report.md  tables/  error_cases/
```
- `report.md` 必含：協議、候選與取捨（含參數量）、各模型各種子表、平均±SD、配對差異＋CI＋verdict、成本表、錯誤案例、限制、未量測、baseline 先前已見 test 之揭露、標註審閱狀態、資料版本與 run_id 清單。
- `error_cases/` 疊圖含原圖，僅存本機並標示**不得公開**；公開文字僅含 roi_id 與數值。
- `verdict` 只允許三個列舉值；措辭不得超出之。

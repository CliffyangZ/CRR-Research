# CLI 契約

所有入口：路徑與參數可配置、`--help` 有說明、非零結束碼代表失敗；皆自 repo 根目錄以 `python -m` 執行。

## `research.train.train_tooth_seg`
```
python -m research.train.train_tooth_seg --model {unet,unetpp,attention_unet} --data-version roi_<hash>
  [--derived data/derived] [--config configs/segmentation/roi_<model>_v1.yaml]
  [--protocol configs/segmentation/compare_protocol_v1.yaml] --seed INT
  [--device cuda:0] [--output reports/segmentation] [--batch-size N] [--grad-accum K] [--smoke]
```
- `--smoke`：16 train／8 val ROI、2 epoch；run_id 前綴 `smoke`，不計入協議。
- 帶 `--protocol` 時：協議須為 frozen、config hash 與協議一致、seed 在清單內，否則結束碼 2 並不建立 run。
- 不帶 `--protocol`：manifest 標 `exploratory: true`。
- 失敗：manifest `status=failed` + `failure_reason`，結束碼 1，既有 run 不被覆寫。
- `--batch-size/--grad-accum` 僅在協議 `batch_fallback` 允許的組合下接受，並寫入 `batch.override_reason`（需 `--override-reason`）。

## `research.evaluation.evaluate_seg_models`
```
python -m research.evaluation.evaluate_seg_models --protocol <yaml> --runs <run_dir> ... [--split test]
  [--derived data/derived] [--device cuda:0] [--output reports/segmentation/compare_<ver>] [--record-violation "原因"]
```
- 檢查：協議 frozen；9 個 run 完整且 succeeded 且屬於協議；TestLock 不存在。通過後評估全部 `best.pt`，寫每 run `eval/` 與 TestLock。
- `--split val` 為演練（不寫鎖、輸出標 `dry_run`）。
- TestLock 已存在且無 `--record-violation`：結束碼 3，不評估。

## `research.evaluation.compare_seg_models`
```
python -m research.evaluation.compare_seg_models --protocol <yaml> --eval-dir reports/segmentation/compare_<ver>
  [--bootstrap-b 10000] [--output <同上>]
```
- 只讀已存在的 `eval/` 與 TestLock，不載入模型、不碰 test 原始資料以外的任何內容；輸出 `comparison.json`、`report.md`、`tables/*.csv`、`error_cases/`。

## 協議工具
```
python -m src.segmentation.protocol freeze --protocol <yaml>   # 寫 protocol_freeze.json，並印 hash
python -m src.segmentation.protocol check  --protocol <yaml>   # 驗證 hash／設定一致性
```

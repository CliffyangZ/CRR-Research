# Quickstart：驗證 U-Net Baseline 流程

前置：`uv sync --extra dev --extra unet`；GPU 可用（`python -c "import torch;print(torch.cuda.is_available())"`）。

1. **單元測試**：`pytest tests/segmentation -q` — 預期遮罩 Metal 不入前景、letterbox 還原誤差 ≤1 px、新增截斷取消、零分母標 undefined、洩漏檢查通過。
2. **資料檢查（小樣本）**：`python -m research.data.prepare_tooth_seg_dataset --data data/seg_dataset --config configs/segmentation/unet_baseline_v1.yaml --output data/derived --limit 8` — 預期產生 inventory／masks／splits，排除項皆有原因。
3. **完整資料檢查**：同上去掉 `--limit` — 預期 101 張盤點、跨子集重複組 = 0（否則加 `--resplit` 並檢視理由）。
4. **Smoke 訓練**：`python -m research.train.train_tooth_unet --data-version <dv> --config ... --device cuda:0 --output reports/segmentation --smoke` — 預期 2 epochs 完成、`best.pt`／`last.pt` 載入成功。
5. **正式訓練**：去掉 `--smoke` — 預期 `run_manifest.json` `status=succeeded`、`history.csv`、選模依據。
6. **測試評估**：`python -m research.evaluation.evaluate_tooth_unet --run reports/segmentation/<run_id> --weights best --output .../eval` — 預期 `metrics.json`（四指標含樣本數與定義）、`per_image.csv`、疊圖與三類錯誤案例（或零例標示）。
7. **失敗路徑**：以過大 batch 強制 OOM — 預期 manifest `status=failed` 且不產生成功標記。

格式細節見 [contracts/](contracts/)，欄位見 [data-model.md](data-model.md)。

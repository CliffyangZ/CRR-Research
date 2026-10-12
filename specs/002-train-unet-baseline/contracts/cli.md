# Contract: CLI Entrypoints

所有路徑、種子、裝置由參數提供，無硬編碼私人路徑。

```bash
# 1. 資料檢查：盤點、洩漏檢查、前處理、遮罩、凍結切分（--limit 供小樣本）
python -m research.data.prepare_tooth_seg_dataset \
  --data data/seg_dataset --config configs/segmentation/unet_baseline_v1.yaml \
  --output data/derived [--limit N] [--resplit]

# 2. 訓練（--smoke：≤8 張、2 epochs）
python -m research.train.train_tooth_unet \
  --data-version <dv> --config configs/segmentation/unet_baseline_v1.yaml \
  --seed 20261012 --device cuda:0 --output reports/segmentation [--smoke]

# 3. 固定權重測試評估與報告
python -m research.evaluation.evaluate_tooth_unet \
  --run reports/segmentation/<run_id> --weights best --output reports/segmentation/<run_id>/eval
```

退出碼：0 成功；非 0 失敗，且 `run_manifest.json` 寫入 `status=failed` 與原因（含 OOM／中斷）。輸出目錄已存在 → 拒絕覆寫。
`--resplit` 僅於洩漏檢查發現跨組時使用，會保存新舊清單與理由。

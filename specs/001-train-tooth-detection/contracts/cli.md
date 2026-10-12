# Contract: CLI Entrypoints

所有路徑與裝置由參數提供，無硬編碼私人路徑。

```bash
# 1. 盤點、前處理、切分、匯出（新 data_version）
python -m research.data.prepare_tooth_dataset \
  --images data/original --annotations data/detection_annotations \
  --manifest labeled_images.json --config configs/detection/preprocess_v1.yaml \
  --output data/derived --limit N           # --limit 供小樣本驗證

# 2. 訓練單一模型
python -m research.train.train_tooth_yolo \
  --data-version <dv> --model {v8,v11,v26} --size s \
  --seed 0 --device 0 --output reports/detection [--smoke]

# 3. 評估（驗證集選 τ → 鎖定 → 測試集）
python -m research.evaluation.evaluate_tooth_yolo \
  --run reports/detection/<run_id> --output reports/detection/<run_id>/eval

# 4. 比較報告
python -m research.evaluation.compare_runs --runs <run_id>... --output reports/detection/comparison
```

退出碼：0 成功；非 0 失敗且 `run_manifest.json` 仍寫入 `status=failed` 與原因。`--smoke` 使用 ≤8 張影像、2 epochs。

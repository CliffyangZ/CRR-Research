# Quickstart：驗證比較流程

前置：Python 3.12、`pip install -e ".[unet,dev]"`（含 scipy）、`data/derived/roi_2a0c9441396f/` 存在、單張 GPU（CPU 可跑 smoke）。

## 1. 結構與參數量檢查（無需資料）
```
pytest tests/segmentation/test_models_arch.py -q
```
預期：三個模型輸入 (B,1,256,256) → 輸出 (B,1,256,256)，可反傳；參數量印出並與 baseline 7,762,465 對照。

## 2. 小樣本煙霧測試（每個候選）
```
for m in unet unetpp attention_unet; do
  python -m research.train.train_tooth_seg --model $m --data-version roi_2a0c9441396f --seed 1 --smoke --device cuda:0
done
```
預期：各產生 `smoke-*` run，manifest 含 `peak_gpu_mem_mb`、`n_parameters`、best／last 權重雜湊；無 OOM（U-Net++ 峰值需 <8 GB）。

## 3. 協議凍結與檢查
```
python -m src.segmentation.protocol check  --protocol configs/segmentation/compare_protocol_v1.yaml
python -m src.segmentation.protocol freeze --protocol configs/segmentation/compare_protocol_v1.yaml
```
預期：設定檔除 model 外一致；寫出 `protocol_freeze.json`。凍結前不得開始正式訓練。

## 4. 正式訓練（9 次，可分批）
```
for m in unet unetpp attention_unet; do for s in 20261012 20261013 20261014; do
  python -m research.train.train_tooth_seg --model $m --seed $s --protocol configs/segmentation/compare_protocol_v1.yaml \
    --data-version roi_2a0c9441396f
done; done
```
預期：9 個 succeeded run（失敗者保留紀錄、重跑另得新 run_id）；每個 manifest 含協議 hash、非 exploratory。

## 5. 一次性 test 評估
```
python -m research.evaluation.evaluate_seg_models --protocol configs/segmentation/compare_protocol_v1.yaml --runs <9 個 run 目錄>
python -m research.evaluation.evaluate_seg_models --protocol ... --runs ...     # 第二次必須以結束碼 3 拒絕
```
預期：產生 `test_lock.json`；重跑被拒。

## 6. 比較報告
```
python -m research.evaluation.compare_seg_models --protocol configs/segmentation/compare_protocol_v1.yaml \
  --eval-dir reports/segmentation/compare_v1
```
預期：`report.md` 含每候選 vs baseline 的差異、CI、verdict；用合成資料的單元測試 `tests/segmentation/test_compare_stats.py` 已驗證 CI 與 verdict 邏輯。

## 7. 回歸
```
pytest tests/segmentation -q
```
預期：既有測試維持通過（若有既有失效測試，須逐一記錄，不宣稱通過）。

# Smoke 驗證與候選登錄（T012／T013）

日期：2026-10-12　裝置：NVIDIA GeForce RTX 5060 Laptop（8151 MiB）　torch 2.13.0+cu130　`--smoke`（16 train／8 val ROI、2 epoch、batch 16、AMP）

| 候選 | 設計取向 | 實作 | 來源／授權 | 參數量（實測） | 峰值 GPU 記憶體 | smoke run_id |
|---|---|---|---|---|---|---|
| unet（baseline） | 編碼器—解碼器＋直接跳接 | `research/models/unet.py` | 專案自有；Ronneberger et al. 2015 | 7,762,465 | 1,620 MiB | smoke-20261012T144542Z-a1dbbd |
| unetpp | 巢狀密集跳接，無 deep supervision | `research/models/unetpp.py` | 專案自有；Zhou et al. 2018 | 9,045,249（+16.5%） | 4,147 MiB | smoke-20261012T144533Z-01d4cf |
| attention_unet | 跳接處加性注意力閘門 | `research/models/attention_unet.py` | 專案自有；Oktay et al. 2018 | 7,850,713（+1.1%） | 1,922 MiB | smoke-20261012T144546Z-4c5b04 |

- 三者皆隨機初始化（Kaiming）、無預訓練權重，batch 16 可在 8 GB 內訓練（U-Net++ 約一半）；不需要 batch 降級。
- 峰值記憶體是訓練（含驗證前向）全程的 `torch.cuda.max_memory_allocated`，smoke 用的是完整 batch 16。
- 注意力閘門以「上採樣後的解碼特徵」為 gating 訊號（與 skip 同解析度），是論文跨步閘門的常見簡化。
- smoke 的 Dice 只用 16 個訓練 ROI 訓練 2 輪，數字無意義，不可用。
- 基準比對：`configs/segmentation/roi_unet_v1.yaml` 的 `RoiTrainConfig.hash()` = `5f548d7c898f91e6`，與既有 baseline run `roiunet-20261012T125011Z-8e067f` 的 `config_hash` 相同。

## 既有失效測試（不屬本功能）
`tests/test_download_sam_weights.py`、`tests/test_yolo_dataset.py` 在收集階段 `ModuleNotFoundError: No module named 'src.models'`（舊文件所述的 `src.models` 已不存在）；本功能前即如此（stash 驗證）。`pytest tests/segmentation tests/detection`：156 passed。

## T013 使用者確認
（待確認：候選登錄表是否接受。確認前不得凍結協議或開始正式訓練。）

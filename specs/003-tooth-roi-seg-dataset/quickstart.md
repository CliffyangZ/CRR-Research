# Quickstart：驗證 ROI 資料集

前置：`uv sync --extra dev --extra unet`（或現有環境）；`data/detection_dataset/` 與 `data/seg_dataset/` 存在。

1. **單元測試**：`pytest tests/segmentation -q` — 預期配對（七類案例）、裁切越界補背景、還原 ≤1 px、單牙遮罩（鄰牙／Metal 為背景）、增強取消、同圖 ROI 同子集皆通過。
2. **小樣本**：`python -m research.data.prepare_tooth_roi_dataset --config configs/segmentation/roi_dataset_v1.yaml --limit 9` — 預期產生 roi_index／pairs／quarantine，隔離項皆有原因。
3. **完整建置**：同上去掉 `--limit` — 預期 `dataset_manifest.json status=succeeded`；對照 spec 估計（約 232 個框可配對）檢視實際數字；跨子集 ROI 與近重複組皆為 0。
4. **檢視報告**：開啟 `reports/segmentation/roi_<dv>/README.md`、`overlays/` 與 `boxes_on_seg/`（偵測框疊在分割多邊形上）；確認資料夾為 `train/val/test` 結構 — 預期含各子集統計、IoU 分布、隔離原因、截斷／鄰牙比例、疊圖。
5. **增強檢查**：以 `RoiSegDataset(..., "train")` 取樣多個 epoch，統計 `geometry` 分布（applied／cancelled／not_selected）；val／test 的 `augment` 必為 not_selected。
6. **重現性**：以相同輸入與種子再建一次到新目錄，`roi_index.json` 內容（除 build_id 與時間）應一致。

欄位見 [data-model.md](data-model.md)，規則見 [contracts/](contracts/)。

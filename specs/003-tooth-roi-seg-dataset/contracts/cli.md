# Contract: CLI

```bash
python -m research.data.prepare_tooth_roi_dataset \
  --config configs/segmentation/roi_dataset_v1.yaml \
  --output data/derived --report reports/segmentation \
  [--limit N] [--resplit] [--seed 42]
```

- 輸出：`data/derived/roi_<dv>/{train,val,test}/{images,masks,masks_vis,roi_index.json}`；報告 `reports/segmentation/roi_<dv>/`（含 `boxes_on_seg/<subset>/<image_id>.jpg`：偵測框疊在分割多邊形上的全量審閱圖）。
- `--limit N`：小樣本（約 N 張影像，各子集均分）。
- `--resplit`：僅在洩漏檢查發現跨組時使用，保存新舊清單與理由。
- 輸出目錄 `data/derived/roi_<data_version>/` 已存在 → 拒絕覆寫（退出碼 2）。
- 失敗：退出碼 1，`dataset_manifest.json` 寫 `status=failed` 與原因。成功：退出碼 0，印出納入／隔離統計。

載入資料集（供後續訓練，非本功能入口）：

```python
from src.segmentation.roi_dataset import RoiSegDataset
ds = RoiSegDataset("data/derived/roi_<dv>", "train", cfg, seed=0)   # train 才啟用增強
```

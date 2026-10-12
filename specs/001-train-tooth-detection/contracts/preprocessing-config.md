# Contract: Preprocessing & Augmentation Config

`configs/detection/preprocess_v1.yaml`（路徑可由參數覆寫）。欄位定義見 [data-model.md](../data-model.md) PreprocessConfig。

```yaml
nlm:      {h: 10, template_window: 7, search_window: 21}
clahe:    {clip_limit: 2.0, tile_grid: [8, 8]}
usm:      {sigma: 1.0, amount: 0.5, threshold: 0}
input_mapping: {type: linear_uint8, divisor: 255}
letterbox: {size: 640, pad_value: 114, interp_down: area, interp_up: linear}
augment:
  copies_per_train_image: 3
  seed: 20261009
  prob: 0.5
  order: [brightness_contrast, rotate, translate]
  brightness: 0.10
  contrast: 0.10
  rotate_deg: 5
  translate_frac: 0.05
  disabled: [scale, flip, mosaic, mixup, crop, elastic]
```

規則：
1. 處理順序固定 NLM → CLAHE → USM → letterbox → （僅 train）增強；輸入映射在最後一次進模型前恰好一次。
2. 8-bit 以外或通道不一致的檔案 → quarantined，不套用此設定。
3. 幾何增強後任一框越界 → 取消整次幾何，保留亮度／對比。
4. 設定一經訓練使用即凍結；變更須新版本檔名。

# Contract: 登錄設定 `configs/segmentation/unet_baseline_v1.yaml`

訓練前凍結；run_manifest 內嵌完整副本。

```yaml
data: {root: data/seg_dataset, derived: data/derived, tooth_category: tooth, ignore_categories: [Metal]}
preprocess:                       # 沿用 configs/detection/preprocess_v1.yaml 的值
  nlm: {h: 10, template_window: 7, search_window: 21}
  clahe: {clip_limit: 2.0, tile_grid: [8, 8]}
  usm: {sigma: 1.0, amount: 0.5, threshold: 0}
  input_mapping: {type: linear_uint8, divisor: 255}
resize: {size: 512, image_pad: 114, mask_pad: 0, interp_down: area, interp_up: linear, mask_interp: nearest}
augment:
  prob: 0.5
  order: [brightness_contrast, rotate, translate]
  brightness: 0.10
  contrast: 0.10
  rotate_deg: 5
  translate_frac: 0.05
  truncation: {tolerance: 0.005, margin_frac: 0.25}
  disabled: [scale, flip, mosaic, mixup, crop, elastic]
model: {name: unet, in_channels: 1, out_channels: 1, base_channels: 32, depth: 4, norm: batch, upsample: transpose, init: kaiming}
train:
  seed: 20261012
  batch_size: 4
  max_epochs: 150
  optimizer: {name: adamw, lr: 1.0e-3, weight_decay: 1.0e-4}
  scheduler: {name: cosine, min_lr: 1.0e-5}
  loss: {bce: 1.0, dice: 1.0}
  amp: true
  early_stop: {metric: val_dice, patience: 30}
  selection: {metric: val_dice, mode: max, tie: earlier_epoch}
  threshold: 0.5            # 僅可由驗證集調整，且須於測試前凍結
eval: {error_cases: {miss_min_frac: 0.02, false_min_frac: 0.001}}
```

驗證規則：`disabled` 內項目若被啟用 → 載入失敗；`threshold` 修改須同步記錄驗證依據。

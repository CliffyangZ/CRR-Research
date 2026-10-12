# Contract: `configs/segmentation/roi_dataset_v1.yaml`

建置前凍結；manifest 內嵌副本。

```yaml
data:
  detection_root: data/detection_dataset     # images/{train,val,test}, labels/{train,val,test}
  seg_root: data/seg_dataset                 # {train,val,test}/annotations/instances_default.json
  tooth_category: tooth
  ignore_categories: [Metal]
preprocess:                                  # 只有 CLAHE；不得出現 nlm / usm
  clahe: {clip_limit: 2.0, tile_grid: [8, 8]}
  input_mapping: {type: linear_uint8, divisor: 255}
pairing: {min_iou: 0.5, second_margin: 0.1, min_mask_frac: 0.05}
roi:
  expand_frac_per_side: 0.10
  size: 256
  image_pad: 114
  mask_pad: 0
  interp_down: area
  interp_up: linear
  mask_interp: nearest
split: {source: existing, phash_dist: 6, seed: 42}
augment:                                     # 同 Spec1／spec 002
  prob: 0.5
  order: [brightness_contrast, rotate, translate]
  brightness: 0.10
  contrast: 0.10
  rotate_deg: 5
  translate_frac: 0.05
  truncation: {tolerance: 0.005, margin_frac: 0.25}
  disabled: [scale, flip, mosaic, mixup, crop, elastic]
```

驗證：`preprocess` 含 `nlm` 或 `usm` → 載入失敗；`augment.disabled` 必須包含六項；`min_iou`、`second_margin`、`min_mask_frac` 修改須在 manifest 記錄理由且不得參考 test。

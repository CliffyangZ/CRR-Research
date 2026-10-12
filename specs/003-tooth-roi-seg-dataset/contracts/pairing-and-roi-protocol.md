# Contract: 配對與 ROI 協議

**配對**（每張影像）：IoU(box, polygon 外接框)。接受 ⇔ `best ≥ min_iou` 且 `best − second ≥ second_margin`；多框競爭同一多邊形時，領先者需領先 ≥ `second_margin` 才保留，其餘隔離，否則全隔離。未配對多邊形為 `orphan_polygon`。不使用任何 test 結果調整。

**ROI 幾何**：`expanded = box ± expand_frac_per_side × (w, h)` → 裁切（越界以 `image_pad`／遮罩 0 補足）→ letterbox 至 `roi.size`。還原：ROI 座標 → 去補邊 → 縮放回擴張矩形 → 平移加回原圖座標；控制點誤差每軸 ≤ 1 原圖像素。

**遮罩**：配對多邊形光柵化，值域 {0,1}；其他 tooth 與 Metal 為背景。

**品質旗標**：`target_truncated_by_roi`、`truncated_at_source`、`neighbor_in_roi_frac`、`outside_image_px`；`tiny_target`（目標占擴張矩形 < `min_mask_frac`）隔離。

**增強**（僅 train）：同 Spec1；instances＝目標牙單一遮罩；新增截斷（loss > tolerance）取消整次幾何、保留亮度／對比。

**報告統計**：單位＝ROI，母體＝各子集；列框數、多邊形數、納入／隔離（依原因）、IoU 分布（分位數）、截斷與含鄰牙比例、orphan 多邊形數、疊圖範例（合法、各類隔離、截斷）。

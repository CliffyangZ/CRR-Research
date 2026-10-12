# Phase 0 Research: 單顆牙齒 ROI 分割資料集

## R1 「外擴 10%」的定義
- **Decision**: 框的每一側各向外擴框寬／框高的 10%（總邊長 ×1.2），超出原圖的部分以固定灰值 114 補足（遮罩補 0），並記錄 `outside_image_px` 與四側補足量。設定鍵 `expand_frac_per_side: 0.10`。
- **Rationale**: 規格只說「向外擴 10%」；採每側 10% 以保留牙根／牙冠周邊上下文且避免偏小的框切掉牙。若使用者意指總邊長 ×1.1，只需改設定值為 0.05。
- **Alternatives**: 總邊長 ×1.1（每側 5%）→ 偏小框更易截斷；固定像素外擴 → 不隨牙大小縮放。

## R2 補背景與 letterbox
- **Decision**: 先裁出擴張後矩形（越界補 114），再以 `src.detection.letterbox` 等比例縮放補邊至 256×256；影像縮小用 area、放大用 linear，補邊 114；遮罩 nearest、補邊 0。ROI 紀錄含原圖座標矩形、補足量、縮放、補邊（`LetterboxRecord`）。CLAHE 在**完整原圖**上套用後再裁切（避免補邊影響局部直方圖），然後 8-bit 線性 /255。
- **Rationale**: 補邊與越界區在 CLAHE 之後加入，不污染統計；與 spec 002 一致。
- **Alternatives**: 先裁再 CLAHE → 補邊區影響 tile；反射補邊 → 製造不存在的牙結構。

## R3 前處理：CLAHE only
- **Decision**: `clip_limit 2.0, tile_grid 8×8`（同 Spec1 值），無 NLM、USM；僅 8-bit 灰階輸入（3 通道且通道相同者轉灰階）。
- **Rationale**: 使用者指定。非 8-bit 或有色彩內容者隔離（沿用 `inspect_image`）。

## R4 配對規則（一對一、不猜）
- **Decision**: 每張影像獨立處理。以框與多邊形外接框的 IoU 建矩陣；對每個框取最佳多邊形 p* 與次佳 IoU。接受條件：`iou* ≥ 0.5` 且 `iou* − 次佳 ≥ 0.1`。若多個框的 p* 相同：IoU 最高者若領先其他競爭框 ≥ 0.1 則保留，其餘隔離，否則全部隔離。隔離原因：`no_polygon`、`low_iou`、`ambiguous_second_best`、`polygon_contested`、`degenerate_box`、`box_out_of_image`。未被任何框配對的多邊形記為 `orphan_polygon`（不入資料集，列入報告）。
- **Rationale**: 全資料估計 232/273 個框 IoU≥0.5 且最佳與次佳差距均足夠；規則可解釋、可重現。門檻只能由訓練／驗證資料或合成案例決定，不看 test。
- **Alternatives**: 匈牙利演算法 → 會強迫配對低品質框；以質心距離 → 對大小不一的牙不穩。

## R5 單牙遮罩與 ROI 標記
- **Decision**: 遮罩＝配對多邊形於原圖光柵化（`cv2.fillPoly`），再與 ROI 同步裁切／letterbox。同圖其他 tooth 多邊形與 Metal 為背景；記錄 `neighbor_in_roi_frac`（其他 tooth 多邊形落入 ROI 的像素占比）。`target_truncated_by_roi`＝目標牙遮罩接觸擴張後 ROI 邊界（含因越界外而被原圖邊切）；`truncated_at_source`＝目標牙多邊形接觸原圖邊界。最小遮罩：目標牙占 ROI 擴張矩形面積 < 5% → 隔離 `tiny_target`。
- **Rationale**: 對應 spec 邊界案例；5% 為保守登錄值，建置報告列出實際分布，若不合理須在凍結前依 train／val 調整。

## R6 增強
- **Decision**: 完全沿用 Spec1／spec 002 的數值：prob 0.5、亮度 ±0.10、對比 ±0.10、旋轉 ±5°、平移 ±5%、順序 brightness_contrast → rotate → translate；停用 scale／flip／mosaic／mixup／crop／elastic。直接重用 `src/segmentation/augment.py`（以 256 為畫布、instances＝目標牙單一遮罩、pad 114）。僅 train 啟用；線上增強，不離線展開。
- **Rationale**: 使用者指定；spec 002 已驗證截斷判定。補邊區視為畫布內，目標牙在 ROI 邊界被切掉者只要不再損失即通過。
- **Alternatives**: 離線展開 3 倍 → 佔空間且不利於調整，本功能只提供資料集，Dataset 於載入時增強。

## R7 切分與洩漏
- **Decision**: 驗證 `detection_dataset` 與 `seg_dataset` 的影像與子集一致；沿用該切分，對含納入 ROI 的影像計算 pHash（距離 ≤6 分組，重用 `src.detection.splits`），ROI 以其來源影像的子集為準；若發現跨子集近重複，以組為單位重切並保存新舊清單與理由。
- **Rationale**: FR-007；spec 002 結果顯示 101 張皆為單獨一組，預期無洩漏，但仍須檢查。

## R8 偵測框審閱狀態
- **Decision**: 框來源記為 `data/tooth_detection/labels/`（`detection_dataset/README.md` 所述複製來源），審閱狀態 `unknown`；manifest 與報告並列兩種狀態。若日後確認，可在 manifest 的 `annotation_review.boxes` 更新並產生新資料版本。
- **Rationale**: 憲章 IV：不推定。

## R9 重現性
- **Decision**: 重用 `src.detection.provenance`（build_id、git、環境、sha256）；ROI 以 `(image_id, box_index)` 排序，無隨機性（增強隨機性只在 Dataset 載入時，以 `(seed, epoch, idx)` 種子）；輸出目錄存在則拒絕覆寫；`data_version` 由來源檔 hash + 設定 hash 產生。

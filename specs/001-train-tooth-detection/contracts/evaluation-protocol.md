# Contract: Evaluation Protocol

- **Population**: 指定 split 的全部 included 影像與其確認標註；報告須列影像數與框數。
- **座標**: 預測框先以 LetterboxRecord 還原至原圖像素後再與 GT 比對。
- **預測**: conf ≥ 0.001、max_det = 300、NMS IoU = 0.7（三模型一致；NMS-free 的 v26 若不適用則記錄差異）。
- **匹配**: 預測依信心降序，每個 GT 至多匹配一個預測（IoU 最大且 ≥ 閾值）；其餘預測為誤檢，重複框（與已匹配 GT IoU ≥ 閾值）另計為 duplicate。
- **AP**: COCO 101 點插值，IoU ∈ {0.50, 0.55, …, 0.95} 平均；AP50 輔助。
- **P/R/F1**: IoU 0.5、工作信心閾值 τ。τ = 驗證集 F1 最大者（同值取較高），在測試前寫入檔案並鎖定。
- **零分母**: 無預測 → precision 為 `undefined`；無 GT → recall 為 `undefined`；不填 0。
- **錯誤分類**: missed（未匹配 GT）、false_positive（IoU < 0.1 的未匹配預測）、duplicate、localization（0.1 ≤ IoU < 0.5）。
- **輸出**: `metrics.json`、`errors.json`、疊圖樣例；所有檔案含 run_id、split_id、n_images、n_boxes。
- **選模**: 僅用 val AP@[.5:.95]；測試集只在 τ 與權重鎖定後評估一次。

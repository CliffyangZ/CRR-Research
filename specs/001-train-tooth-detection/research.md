# Phase 0 Research

## R1 — YOLO v8 / v11 / v26 框架
- **Decision**: 三者皆用 Ultralytics 同一套 API；權重 `yolov8s.pt`、`yolo11s.pt`、`yolo26s.pt`（小型 s 規模，統一規模以利比較）。
- **Rationale**: 同框架使資料格式、訓練迴圈與評估一致；s 規模在 8 GB VRAM、imgsz 640 可訓練。
- **Alternatives**: n 規模（更省資源、容量低）；m 以上（8 GB 風險高）。
- **待實作時驗證（記入 manifest）**: 安裝的 ultralytics 版本確實提供 `yolo26*.pt` 並可下載；否則升級 `ultralytics` 下限並記錄精確版本。授權：Ultralytics 為 AGPL-3.0 / 企業授權，研究使用須記錄於模型選擇紀錄。

## R2 — 前處理參數（訓練前登錄、不得用測試集調整）
- **Decision**: 灰階化（RGB 影像三通道相同則取單通道；不同則記錄並隔離）→ NLM（`cv2.fastNlMeansDenoising`, h=10, templateWindowSize=7, searchWindowSize=21）→ CLAHE（clipLimit=2.0, tileGridSize=(8,8)）→ USM（Gaussian σ=1.0, amount=0.5, threshold=0）→ 固定線性映射 uint8/255 → [0,1]，輸出複製為 3 通道。
- **Rationale**: 沿用現有 `src/data/preprocess.py` 的 CLAHE 參數；NLM／USM 取常見保守值。參數為預設，若需變動須在訓練前改動並新版本化，不用測試集決定。
- **Alternatives**: 逐張 min–max（spec 禁止）；先 CLAHE 後 NLM（違反規定次序）。
- **離線預計算**: 處理結果存為 PNG（8-bit）到衍生資料夾，避免訓練時重複 NLM；USM 後 clip 到 0–255。

## R3 — Letterbox
- **Decision**: 目標 640×640、等比縮放（INTER_AREA 縮小／INTER_LINEAR 放大）、補邊值 114（Ultralytics 慣例）置中；以 `LetterboxRecord{orig_w,orig_h,scale,new_w,new_h,pad_left,pad_top,pad_right,pad_bottom}` 記錄；反向還原誤差 ≤1 原圖像素。
- **Rationale**: spec 要求固定 letterbox；訓練、驗證、測試與推論共用同一函式。離線 letterbox 後，Ultralytics 以 `rect=False` 且 imgsz=640 載入時不會再縮放（影像已為 640×640）。
- **Alternatives**: 讓 Ultralytics 內部做 letterbox（難以保證補邊記錄與一致性）。

## R4 — 增強與關閉預設增強
- **Decision**: 不使用 Ultralytics 內建增強做幾何與色彩，改在自訂資料管線離線／線上實作：亮度／對比 ±10%、旋轉 ±5°、平移 ±5%（機率 0.5、順序：亮度／對比 → 旋轉 → 平移）；幾何後若任何框超出邊界（容差 0 px）則整次取消。Ultralytics 引數明確設為 `mosaic=0, mixup=0, cutmix=0, copy_paste=0, fliplr=0, flipud=0, scale=0, shear=0, perspective=0, degrees=0, translate=0, hsv_h=0, hsv_s=0, hsv_v=0, erasing=0, auto_augment=None, close_mosaic=0`。
- **Rationale**: Ultralytics 內建的旋轉／平移會直接裁掉或截斷框，無法實作「超出就取消」。因此以自訂訓練期增強（見 R4 實作方式）替代。
- **實作方式**: 預設採「離線預生成固定份數增強副本」（每張訓練影像 K=3 份，種子固定、記錄於清單）；優點是增強結果可審閱、可重現，且三模型使用完全相同資料。Alternatives：自訂 Ultralytics Dataset / callback（動態、較貼近標準訓練，但三版本 API 差異風險高）。
- **邊緣原始截斷牙**: 框與影像邊界重合視為「界內」；判定「額外截斷」以變換後框是否超出 [0,W]×[0,H] 為準（框 clip 前）。

## R5 — 切分與病人分群
- **Decision**: 沿用現有 `labeled_images.json` 的 61/20/20 切分作為候選，但先做洩漏檢查：以感知 hash（pHash 距離）找近重複，並以檔名序號鄰近與尺寸群組為替代分群標記；若發現跨子集群組則重新分群切分，仍記錄種子（seed=42）。無病人資訊 → 明載「未排除病人層級洩漏」。
- **Rationale**: 保留既有切分可追溯，但憲章 III 要求檢查。測試集 20 張很小，指標變異大，須在報告揭露。
- **Alternatives**: 完全重切；K-fold（不在本階段範圍）。

## R6 — 評估規約
- **Decision**: 自行以 `src/detection/evaluate.py` 計算（COCO-style AP@[.5:.05:.95]、AP50、IoU0.5 的 P/R/F1），輸入為相同的 GT 與預測 JSON，三模型走同一程式；預測用 conf=0.001、max_det=300 取得 PR 曲線，工作閾值由驗證集 F1 最大選定（同值取較高）。以 pycocotools 結果交叉檢驗 AP（若可安裝）。
- **Rationale**: 避免不同 Ultralytics 版本內部指標差異；規約由 FR-010 固定。
- **Alternatives**: 直接用 `model.val()` 數值（便於但三版本實作細節可能不同）。仍保留作為對照紀錄。

## R7 — 選模與訓練設定
- **Decision**: epochs=100、patience=30、imgsz=640、batch=8（OOM 則 4）、AdamW/auto、seed=0、deterministic=True、workers=4、`cache=False`；選模以驗證集 AP@[.5:.95] 最高 epoch（`best.pt`）。
- **Rationale**: 小資料集需足夠 epochs 並早停；8 GB 顯存限制 batch。三模型統一超參數，差異記錄。

## R8 — 資料標註來源
- **Decision**: 現有 `data/detection_annotations` 與 `data/detection_dataset/labels` 由 `seg_annotations`（CVAT）推導；實作前須在盤點中確認其是否為人工確認並登錄 annotation version 的審閱狀態。若為偽標註則報告標示「對偽標註的結果」，不宣稱對人工 ground truth。
- **Risk**: 未確認前，所有指標標示探索性。

## 結論
所有技術未知項已決定或登錄為實作期驗證項（yolo26 權重可用性、標註審閱狀態、病人分群缺口）。

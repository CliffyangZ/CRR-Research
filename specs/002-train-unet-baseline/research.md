# Phase 0 Research: U-Net Baseline

格式：Decision／Rationale／Alternatives。

## R1 架構與初始化
- **Decision**: 標準 2D U-Net（4 次下採樣，通道 32-64-128-256-512，雙卷積 Conv-BN-ReLU，轉置卷積上採樣，skip concat），1 通道輸入、1 通道 logit 輸出，隨機初始化（Kaiming），無預訓練權重。
- **Rationale**: 「baseline」要求最單純、可比較的起點；通道 32 起算約 7.8M 參數，512×512 batch 4 + AMP 適合 8 GB。無預訓練避免權重來源與授權不明。
- **Alternatives**: ImageNet 預訓練 encoder（Attention/ResNet-UNet）→ 屬後續比較實驗；通道 64 起算（31M）→ 8 GB 下 batch 受限。

## R2 輸入尺寸與前處理
- **Decision**: 離線固定前處理 NLM(h10,7,21)→CLAHE(2.0,8×8)→USM(σ1,0.5)，8-bit 線性 /255 映射（沿用 [configs/detection/preprocess_v1.yaml](../../configs/detection/preprocess_v1.yaml)，重用 `src.detection.preprocess`）；等比 letterbox 至 512×512，影像補邊 114、遮罩補邊 0；影像縮小用 area、放大用 linear，遮罩一律 nearest。
- **Rationale**: 規格要求沿用 spec1 並避免逐張 min–max；影像尺寸有 1200×825 與 1720×2400 等，512 在 8 GB 下能保留牙齒邊界；NLM 昂貴，離線快取。先盤點位元深度，非 8-bit 者先隔離而不套映射。
- **Alternatives**: 640（spec1 值）→ 8 GB 下 U-Net 的 batch 過小；center-crop → 違反保留完整內容。

## R3 遮罩建立
- **Decision**: 於原圖解析度以 `cv2.fillPoly` 光柵化每個 tooth 多邊形（category_id=1）並取聯集；Metal（id=2）不繪製，故與 tooth 重疊仍為 tooth、僅 Metal 處為背景。儲存 0/1 uint8 PNG 至 `data/derived/`。多邊形合法性：≥3 點、面積>0、座標落在影像範圍內（容差 1 px，超出者裁切並記錄、嚴重越界者隔離）。
- **Rationale**: 符合 FR-002；不依賴 pycocotools（少一個依賴）。
- **Alternatives**: pycocotools `annToMask` → 功能相同但多依賴。

## R4 切分與洩漏
- **Decision**: 沿用既有 train/val/test 目錄切分作為候選，對全 101 張計算 pHash（重用 `src.detection.splits.build_groups`，距離 ≤6 同組）+ 檔名序號鄰近作替代分群，執行 `leakage_check`。零跨子集組 → 凍結清單（含 sha256）；有跨組 → 以固定種子重切（組為單位）並保存新舊清單與理由。無病人資訊，報告揭露殘餘風險。
- **Rationale**: FR-003；spec1 已有同方法實作可重用。
- **Alternatives**: 重新隨機切 → 破壞使用者既有切分且無必要。

## R5 增強數值與截斷判定
- **Decision**: 沿用 spec1：每張訓練影像線上抽樣；prob 0.5；亮度 ±0.10、對比 ±0.10（僅影像）；旋轉 ±5°、平移 ±5%（影像 bilinear、遮罩 nearest，同一 2×3 仿射）；順序 brightness_contrast → rotate → translate（幾何合成一個仿射）。停用 scale/flip/mosaic/mixup/crop/elastic。
  **新增截斷判定**：對每個 tooth 多邊形單獨光柵化於 letterbox 畫布；以同一仿射在擴大畫布（各邊 +25%）上變換，計算「畫布內像素數／擴大畫布像素數」；若變換後 in-canvas 比例低於變換前比例 − 0.5%（即新增被畫布邊界切掉的牙齒像素）→ 判新增截斷，取消整次幾何變換並保留亮度／對比。原圖已截斷的牙齒其變換前比例 <1，只要不再損失即通過。補邊區視為畫布內（不構成截斷）。
- **Rationale**: 以每牙獨立像素保留比判定，與規格「有效牙齒可見區域新增截斷」一致且不刪標註；擴大畫布避免「已被切掉」的像素無法比較。
- **Alternatives**: 以 bbox 界內判定（spec1）→ 對原本截斷的牙齒會永遠取消；彈性修剪標註 → 違反規格。
- **訓練資料量**: 線上增強（每 epoch 重抽），不離線展開；不需要 `copies_per_train_image`。

## R6 損失、最佳化、輪數、停止條件
- **Decision**: 損失 = BCEWithLogits + soft Dice（1:1）；AdamW lr 1e-3、wd 1e-4、cosine 退火至 1e-5；batch 4、AMP(fp16)；最多 150 epochs；每 epoch 於驗證集評估牙齒 Dice（閾值 0.5，原圖座標計算以確保與最終指標一致）；早停 patience 30（以驗證 Dice 無提升計）；選模 = 驗證 Dice 最高、同值選較早輪；保存 `best.pt`、`last.pt`、`history.csv`。種子 20261012。
- **Rationale**: Dice+BCE 對小前景不平衡穩定；patience 30 / 150 epochs 在 61 張資料下單次訓練約數十分鐘級（待實測，不設門檻）。
- **Alternatives**: 純 Dice / focal → 無證據需要；ReduceLROnPlateau → 與早停交互複雜。

## R7 驗證階段評估座標
- **Decision**: 驗證與測試預測皆 sigmoid → 去除補邊 → 以 bilinear 還原至原圖尺寸（對機率）→ 閾值 0.5，與標註原圖遮罩比較。訓練中每 epoch 的驗證指標也走此路徑。
- **Rationale**: FR-011：補邊不納入評估、原圖座標。

## R8 重現性
- **Decision**: 重用 `src.detection.provenance`（run_id、git 狀態、環境、sha256）；設定 `torch.manual_seed`、numpy、DataLoader worker seed；記錄 cuDNN／AMP 非確定性限制。失敗以 try/finally 寫 `status=failed` 與原因；輸出目錄以 run_id 區隔、已存在則拒絕覆寫。
- **Rationale**: 憲章 II、FR-010。

## R9 錯誤案例分類
- **Decision**: 漏分＝FN 連通區域面積占 GT 前景 ≥ 2% 之影像；誤分＝FP 連通區域面積 ≥ 影像面積 0.1%；邊界偏差＝兩遮罩邊界之 95% Hausdorff（HD95）距離。每類取指標最差者為首選案例，未發生則標示零例。
- **Alternatives**: 人工挑選 → 難以重現。

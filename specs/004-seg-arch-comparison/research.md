# Phase 0 Research: 牙齒分割模型架構比較

所有 NEEDS CLARIFICATION 已在此解決。格式：Decision / Rationale / Alternatives。

## R1 候選實作來源

- **Decision**: 在 `research/models/` 自行實作 `UNetPlusPlus` 與 `AttentionUNet`，重用 baseline 的 `double_conv`（Conv-BN-ReLU×2）、MaxPool 下採樣、Kaiming 初始化與 1×1 輸出頭；單通道輸入、單 logit 輸出。授權：專案自有程式碼；方法來源註明 Zhou et al. 2018（UNet++）、Oktay et al. 2018（Attention U-Net）。
- **Rationale**: 編碼器、正規化、初始化、上採樣方式與 baseline 完全相同，唯一變因是跳接結構，符合「差異來自架構」；無新依賴、無預訓練權重、無外部授權問題；結構簡單可由 pytest 核對。
- **Alternatives**: `segmentation_models_pytorch`（MIT，有 UnetPlusPlus）——其 Unet++ 預設搭配 ResNet 類編碼器，編碼器差異會混淆比較，且 Attention U-Net 並無對應；`MIC-DKFZ/nnUNet`、`LeeJunHyun/Image_Segmentation`（AttU_Net，無明確授權）——引入框架或授權不明，否決。

## R2 結構與規模

- **Decision**:
  - **U-Net++**：深度 4、基礎通道 32（通道 32/64/128/256/512），節點 X(i,j) 以 `double_conv` 融合同層先前所有節點與下一層上採樣（轉置卷積，同 baseline）；**不用 deep supervision、不做剪枝**，只用最頂層 X(0,4) 輸出，使損失（BCE+Dice）與 baseline 完全相同。
  - **Attention U-Net**：與 baseline 同編碼／解碼，每個跳接處加性注意力閘門（gating 訊號＝下一層解碼特徵，1×1 卷積降到 C/2，加總 → ReLU → 1×1 → sigmoid，對 skip 逐像素加權後再串接）；基礎通道 32。
  - 參數量於實作時以 `sum(p.numel())` 實測並寫入 manifest；預估（待實測）：baseline 7.76M（已實測）、Attention U-Net 約 +0.3–1M、U-Net++ 約 9–10M。
- **Rationale**: 256×256、batch 16、AMP 下 U-Net++ 的密集節點記憶體最高；估計仍在 8 GB 內，會以 smoke 實測峰值記憶體確認（FR-002 的 8 GB 要求），不足時走 R5 的預先登錄降級。
- **Alternatives**: deep supervision 版本——會改變損失與選模定義，留待後續；基礎通道 64——8 GB 風險高。

## R3 容量控制

- **Decision**: 主要比較固定「基礎通道 32」（同配方、不同跳接），並報告參數量、FLOPs 以外的實測成本（峰值記憶體、時間）。預先登錄一組**次要**等容量變體：調整 U-Net++／Attention U-Net 基礎通道使參數量落在 baseline ±5%，**僅在主要比較完成且使用者同意後**執行，單獨成表，不與主要組混合排名。本次計畫的 tasks 將其列為可選。
- **Rationale**: spec Assumptions 要求容量差異不得當成架構優劣證據；先固定主要比較避免通道數成為額外自由度。
- **Alternatives**: 全部等參數量當主要比較——需要為每個模型挑通道數，增加設計者自由度；否決作為主要組。

## R4 訓練協議（凍結內容）

- **Decision**: 全部模型逐項沿用 baseline 配方：AdamW lr 1e-3、wd 1e-4；餘弦排程 min_lr 1e-5（T_max＝max_epochs）；BCE+Dice（1:1）；AMP；batch 16；max_epochs 150；早停 patience 30（val Dice）；選模＝val Dice（ROI 原解析度、像素累計、閾值 0.5）最高、同值取較早輪；增強沿用 spec 003（亮度對比、±5° 旋轉、5% 平移，目標牙截斷取消幾何）；不做逐模型調參。baseline 的配方來源為 `configs/segmentation/unet_baseline_v1.yaml` 與已跑 manifest（batch 16）；複製為 `roi_unet_v1.yaml` 並鎖定 hash，**不修改**原檔。
- **Rationale**: 與既有 baseline 同尺，單次結果可作參考；固定配方比較架構本身。
- **Alternatives**: 每模型各自調學習率——超出本次範圍且 46 個 ROI 的 test 與 48 個 val 太小，會放大選擇偏差。

## R5 資源不足的預先登錄處理

- **Decision**: 預設 batch 16。若 OOM：改 batch 8 + 梯度累積 2（有效 batch 16，BN 統計仍變），在 manifest 記 `batch_override` 與原因，並在報告標註；仍失敗則該候選列「未完成」並請使用者決定。需在 `run_training` 加入 `grad_accum`（預設 1，行為不變）。
- **Rationale**: spec Edge Case 允許協議內預先登錄的方式；保持有效 batch 讓配方差異最小。
- **Alternatives**: 降低通道數——改變模型規模，違反「同通道規模」。

## R6 種子

- **Decision**: 3 個種子，固定清單 `[20261012, 20261013, 20261014]`（第一個等於 baseline 現有種子，便於與既有單次結果並列）；每模型每種子獨立訓練。種子同時控制初始化、DataLoader 順序與增強；資料切分固定不隨種子變。不足 3 個成功種子的模型在報告列為不完整。
- **Rationale**: spec FR-004 最低 3；資源允許時可附加種子（協議版本升級並記錄）。
- **Alternatives**: 5 個種子——時間成本約 1.7 倍，列為後續建議，並說明 3 個種子下變異估計本身不穩（n=3 的 SD 很粗）。

## R7 評估與邊界指標

- **Decision**: 主要指標沿用 `src/segmentation/metrics.py`（像素累計 Dice 為主；IoU、precision、recall；逐 ROI 平均另列；ROI 原解析度、補邊不計）。新增邊界指標 **HD95**（95 百分位 Hausdorff 距離，像素，ROI 原解析度）與 **boundary F1（容差 2 px）**，在 valid 區域內以距離轉換計算（scipy.ndimage）；pred 或 gt 為空時回報 `None` 並記原因，不補 0。
- **Rationale**: 邊界誤差對牙根／CEJ 後續使用較有意義；HD95 比 HD 對單點離群穩健。
- **Alternatives**: 僅 Dice——spec 要求至少一項邊界指標；surface Dice 同類，不另加。

## R8 統計方法

- **Decision**: 
  1. 每個模型每個種子：test 的逐 ROI 計數（TP/FP/FN）→ 該種子的像素累計 Dice。報告各種子值、平均、SD（n=3，樣本 SD）。
  2. 配對差異：對每個候選 vs baseline，以**原圖為單位**的 cluster bootstrap（B=10000，固定 bootstrap 種子 `20261012`），每次重抽原圖（含其所有 ROI），在**同一重抽**下對每個種子計算 Dice 差，再對種子取平均 → 得差異分布，取 2.5／97.5 百分位為 95% CI。逐 ROI 配對也輸出。
  3. 證據門檻（寫入協議）：「有證據優於」需 CI 下界>0 **且** 平均差異>兩模型種子 SD 的較大者；「有證據較差」對稱；其餘一律「未見可證實差異」。
  4. 另輸出各 ROI 勝負計數作描述，不做檢定。
- **Rationale**: 同圖 ROI 不獨立；對種子平均後再 bootstrap，使結論反映「架構」而非單一種子；與 spec FR-010、US3 一致。
- **Alternatives**: 配對 t 檢定／Wilcoxon 以 ROI 為單位——違反獨立性；以種子為單位的檢定——n=3 無檢定力。報告明示約 20 張原圖時檢定力有限。

## R9 test 一次性與 baseline 已看過 test

- **Decision**: `src/segmentation/run_lock.py` 寫入 `reports/segmentation/compare_<protocol_version>/test_lock.json`（協議 hash、被評估的 run_id 與權重 sha256、時間）。`evaluate_seg_models` 開始前檢查：協議須為凍結狀態、所有 9 個 run 皆 `succeeded` 且屬於該協議、鎖不存在；一次評估全部 9 個權重後才寫鎖。若鎖已存在，預設拒絕；`--record-violation "<原因>"` 才允許重跑，並把違規寫入鎖並使報告的確認性結論失效。
- baseline 既有 test 結果（探索性、協議前）只作參考列在報告，不計入協議內 test 次數；但報告須揭露「設計者在協議前已見 baseline test」。
- **Rationale**: SC-002 要求 test 使用次數可查證。
- **Alternatives**: 純文件約定——不可查證。

## R10 協議凍結機制

- **Decision**: `compare_protocol_v1.yaml` 含資料版本、切分 sha256、訓練配方、種子、候選設定檔 hash、閾值、統計方法、證據門檻。`protocol.freeze()` 計算 hash 寫入 `protocol_freeze.json`（時間、git commit／dirty）。訓練入口帶 `--protocol` 時檢查：設定檔 hash 與協議登錄一致，否則拒絕。凍結後修改須建立 `v2` 並寫 `CHANGELOG`（理由、新舊 hash）。
- **Rationale**: US1 驗收 2。
- **Alternatives**: git tag——不便程式檢查。

## R11 baseline 重現與「單次結果」並列

- **Decision**: 正式的 baseline 3 種子就是 US1 的重跑；報告並列既有單次結果（種子 20261012、val/test 數字取自既有 eval），並檢查新 seed 1 是否落在合理範圍（定義：與既有值差異<種子 SD 與 0.01 之大者；僅描述性，不當閘門）。因 GPU/AMP 非確定性，不要求逐位元相同，manifest 記錄此限制。
- **Rationale**: 避免把「不可重現」誤判為錯誤，同時保持可追溯。

## R12 計時與記憶體量測

- **Decision**: 訓練時間＝`run_training` 的總 wall-clock（含驗證）與每 epoch 秒數（history.csv 已有）；峰值記憶體＝`torch.cuda.max_memory_allocated` 於訓練全程重設後讀取；推論時間＝固定權重在 test 上 batch 1、AMP 與訓練相同、warm-up 5 次後逐 ROI 平均（毫秒）；皆記裝置、驅動與 torch 版本。僅在同一台機器上可互比。
- **Alternatives**: 理論 FLOPs——需新套件，且不反映實測；留作未量測。

## R13 錯誤案例分類

- **Decision**: 沿用 `src/segmentation/report.classify_errors`（漏分／誤判門檻 0.02／0.001，來自 `Eval.error_cases`）。三類案例：
  1. **含鄰牙**、**貼近邊界或被截斷**：直接使用 spec 003 的 ROI 標記。
  2. **牙根漏分**：資料沒有牙位／上下顎資訊，無法精準定義「牙根側」。本次以近似定義：漏分像素占目標牙 GT 的比例最高的前 10 個 ROI（以所有模型種子平均排序，排序用 val 以外之 test 僅作展示，不回饋調參）；報告明示為近似。
  各模型在同一批 ROI 並列；某類無案例則標示零例。
- **Rationale**: 誠實標示近似，避免宣稱未經牙位資訊支持的解剖定義。
- **Alternatives**: 以影像下方三分之一當牙根——上顎牙會錯，否決。

# Research：實驗流程

這裡負責 CRR（Crown-to-Root Ratio，牙冠／牙根比）研究的實驗編排：
資料處理、特徵工程、模型比較、訓練與評估。實驗程式必須簡潔，讓讀者能看懂
「在驗證什麼假設、使用哪些資料、如何比較結果」。複雜或可重用的函式放在
[`src/`](../src/README.md)，實驗結果寫到 [`reports/`](../reports/README.md)。

## 目錄與責任

```text
research/
├── README.md
├── data/         # 資料品質、切分、前處理與特徵比較的實驗入口
├── models/       # 候選模型、模型設定與比較方案
├── train/        # 訓練與參數搜尋的實驗入口
└── evaluation/  # 指標計算流程、模型比較與失敗案例分析
```

目前四個子目錄尚無實驗程式；以上定義後續放置位置，並非已完成的訓練流程。
`research/models/` 放研究用模型組合與設定；大型權重不放在這裡。
Notebook 的互動探索放在根目錄 `notebooks/`，可重現的正式實驗入口放在這裡。
一次性資料轉換工具放在根目錄 `scripts/`。

## 實驗怎麼寫

1. 先寫下研究問題、假設與 baseline，定義比較指標及固定的資料切分。
2. 入口只編排「載入設定 → 準備資料 → 執行模型 → 評估 → 儲存結果」。
3. 複雜演算法、影像幾何、格式驗證與共用前處理抽成 `src/` 函式。
4. 記錄實際執行命令、程式版本、環境、隨機種子、資料版本與模型權重來源。
5. 每次執行使用獨立的輸出目錄，保留原結果；中斷或失敗也要記錄原因。

所有命令以專案根目錄為工作目錄。新入口必須提供參數說明與小樣本執行方式，
不得依賴開發者個人的絕對路徑。現有分割入口見 [`src/README.md`](../src/README.md)。

## 牙齒語意分割 U-Net baseline（spec 002，已實作）

- `research/models/unet.py`：U-Net 架構（使用者指定位置）。
- `research/data/prepare_tooth_seg_dataset.py`：盤點 → 洩漏檢查 → 固定前處理 → tooth 聯集遮罩 → 凍結切分。
- `research/train/train_tooth_unet.py`：訓練與驗證選模（`--smoke` 小樣本）。
- `research/evaluation/evaluate_tooth_unet.py`：固定權重測試評估、疊圖與錯誤案例。
- `research/data/prepare_tooth_roi_dataset.py`（spec 003）：偵測框＋分割多邊形 → 單顆牙 ROI 資料集（一對一配對、CLAHE、256 letterbox、單牙遮罩、凍結切分、驗證報告）；設定 `configs/segmentation/roi_dataset_v1.yaml`，輸出 `data/derived/roi_<data_version>/{train,val,test}/`，報告 `reports/segmentation/roi_<data_version>/`。偵測框審閱狀態為「未知」；僅產生資料集，未訓練模型。
- `notebooks/find_cej.ipynb`＋`src/segmentation/alveolar.py`：以齒槽骨頂（臨床定義）計算冠根比的候選演算法；`research/data/annotate_cej.py` 標註左右骨頂供評估。結果為候選，未經醫師真值驗證。
- 設定：`configs/segmentation/unet_baseline_v1.yaml`；使用步驟見 [`specs/002-train-unet-baseline/quickstart.md`](../specs/002-train-unet-baseline/quickstart.md)；證據在 `reports/segmentation/`。
- 權重寫入 `models/segmentation/<run_id>/`（不入版本庫，由訓練命令重建）；`data/` 與 `reports/*` 亦被 git 忽略。

## 分割架構比較（spec 004，實作中：協議凍結前）

比較 U-Net++、Attention U-Net 與 baseline U-Net（同資料、同訓練配方、3 種子、隨機初始化）；候選設定與協議見
[`specs/004-seg-arch-comparison/`](../specs/004-seg-arch-comparison/plan.md)。**尚未凍結協議、尚未做正式訓練與 test 評估。**

- `research/models/{unetpp,attention_unet,registry}.py`：兩個候選（專案內自行實作，無預訓練）與 `model.name → builder`。
- `research/train/train_tooth_seg.py`：`--model {unet,unetpp,attention_unet} --seed N [--protocol ...] [--smoke]`；帶 `--protocol` 時檢查協議已凍結、設定 hash、種子。
- `research/evaluation/evaluate_seg_models.py`：九個採用的 run 在 test 一次性評估並寫 `test_lock.json`（重跑被拒）；`--split val` 為演練。
- `research/evaluation/compare_seg_models.py`：以原圖為單位的配對 bootstrap、成本表、錯誤案例與 `report.md`。
- 共用邏輯：`src/segmentation/{protocol,run_lock,boundary,compare_stats,compare_eval}.py`；協議 `configs/segmentation/compare_protocol_v1.yaml`。
- 凍結：`python -m src.segmentation.protocol check|freeze --protocol configs/segmentation/compare_protocol_v1.yaml`。

## 資料與評估規則

- 原始影像保持不變；處理後資料保留來源影像與轉換紀錄。
- 以病人為優先切分單位，並將近重複影像放在同一子集；缺少病人識別資訊時，
  記錄替代分群方法與無法排除的洩漏風險。
- 需從資料估計參數的轉換只能用訓練集擬合；增強只用於訓練資料。
- 驗證集用於調參，測試集不參與模型選擇；不同方法使用相同切分比較。
- SAM／YOLO 輸出的 pseudo-label 必須保留來源與審閱狀態，經專家校正後才可當作
  人工確認的 ground truth。模型分數不能當作對人工標註的實測 IoU。
- 分割實驗可記錄 Dice／IoU；CRR 實驗須先定義 landmark、長度與比值的計算方式，
  再選用 MAE 等指標。不得將尚未實作或未量測的結果寫成既有能力。

## 每次實驗最低紀錄

| 項目 | 內容 |
| --- | --- |
| 目的 | 假設、baseline、比較變因 |
| 資料 | 版本或識別碼、來源、標註狀態、各子集數量與分群方式 |
| 執行 | 命令、程式版本、環境、種子、參數、權重識別碼 |
| 評估 | 指標定義、評估子集、樣本數、排除條件 |
| 結果 | 指標、圖表、失敗案例、限制與下一步 |

範例紀錄格式見 [`reports/README.md`](../reports/README.md)。
專案原則見 [constitution](../.specify/memory/constitution.md)；
本次目錄分工規格見 [spec](../specs/001-document-ml-structure/spec.md)。

## 牙齒偵測訓練（specs/001-train-tooth-detection）

```bash
python -m research.data.prepare_tooth_dataset [--limit 12]          # 盤點→前處理→切分→增強→匯出 data/derived/<dv>
python -m research.train.train_tooth_detection --data-version <dv> --model {v8,v11,v26} [--smoke] --device 0
python -m research.evaluation.evaluate_tooth_detection --run reports/detection/runs/<run_id> --device 0
python -m research.evaluation.compare_runs --runs <run_dir>... --output reports/detection/comparison
```

驗證：`python -m pytest tests/detection`。既有 `tests/` 中部分舊測試仍引用已移除的 `src/models/`，未納入此驗證。

訓練目錄命名：`v8_dv-20261009-01_1009-170021`（模型、資料版本、台北時間）。
可加 `--experiment baseline` 標記實驗；完整參數保存在 `run_manifest.json`。
名稱中的 batch 等參數不再列出；既有實驗目錄保留原名。同一秒重複啟動相同名稱會拒絕覆寫。

`train_tooth_yolo --model fastrcnn` 訓練 Fast R-CNN（VGG16、RoIPool、selective search 提案；實作在 `src/detection/fast_rcnn/`）。
第一次執行會為資料版本建立並快取提案（`data/derived/<dv>/proposals/`），並下載 torchvision VGG16 ImageNet 權重（約 528 MB）。
`evaluate_tooth_yolo` 會依 run manifest 的 model 自動使用對應推論。

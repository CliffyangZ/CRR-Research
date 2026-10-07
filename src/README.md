# Src：可重用的實作

這裡放置 CRR 研究使用的共用函式、影像處理、模型推論及資料格式驗證。
[`research/`](../research/README.md) 負責實驗編排，`src/` 負責具體實作，
[`reports/`](../reports/README.md) 保存輸出與證據。

## 現有目錄

```text
src/
├── README.md
├── core/
│   ├── README.md
│   ├── testing.py
│   └── segmentation/
│       ├── cvat_coco.py      # COCO polygon 驗證、CVAT 封裝與審閱判斷
│       ├── sam_infer.py      # SAM box／point prompt 分割與輸出
│       └── yolo_sam_coco.py  # YOLO 偵測框引導 SAM，輸出 pseudo-label
└── data/                   # 保留給共用資料處理，目前沒有實作
```

`core/testing.py` 目前只有說明字串，不是測試框架。
目前沒有訓練入口或 CRR 計算模組；舊文件中的 `src/models/` 與
`src/data/preprocess.py` 等路徑不能視為現有可執行功能。

## 實作原則

- 函式責任單一；明確說明輸入／輸出、影像尺寸、座標系、資料型別與錯誤條件。
- 核心計算與檔案讀寫、命令列解析分開；可重用函式不得在 import 時執行實驗。
- `research/` 可以呼叫 `src/`；`src/` 不得依賴特定 `research/` 實驗或既有報告檔案。
- 資料路徑、裝置、權重與門檻由參數提供；新增入口不得硬編碼私人路徑。
- 改動座標轉換、標註格式、資料切分或指標時，必須以小型 fixture 驗證相關邊界情況。
- 輸出保存來源、參數與審閱狀態；偽標註不得自動升格成人工標註。

## 現有推論入口

在專案根目錄執行，使用符合 `pyproject.toml` 的 Python 3.12 環境。
SAM 需要 `segmentation` 選配依賴；YOLO＋SAM 另需要 `yolo` 選配依賴，
並確認 PyTorch、裝置與權重相容。以下命令需先準備影像與模型權重，
路徑是示例，並不表示倉庫已附帶這些檔案。

```bash
python -m src.core.segmentation.sam_infer \
  --input data/processed \
  --checkpoint models/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --output reports/models/sam_run_001

python -m src.core.segmentation.yolo_sam_coco \
  --input data/processed \
  --yolo-model models/tooth_detector.pt \
  --sam-checkpoint models/sam_vit_b_01ec64.pth \
  --device auto \
  --limit 5 \
  --output reports/models/yolo_sam_run_001
```

兩個入口輸出 `coco_annotations.json`、`cvat_coco_annotations.zip` 與
`metadata.json`；完整影像 ZIP 需額外指定 `--package-cvat-dataset`。
YOLO＋SAM 的 `--limit` 可做小樣本檢查。使用新的輸出目錄保存每次執行；
YOLO＋SAM 的 `--overwrite` 會替換既有輸出，使用前須確認舊結果已保留。
目前程式的部分預設權重路徑仍指向舊位置，因此範例明確指定權重參數。

## 驗證與維護

相關測試位於根目錄 `tests/`。例如只驗證 COCO 格式工具：

```bash
python -m pytest tests/test_cvat_coco.py
```

現有部分測試仍引用已移除的 `src/models/` 模組，不能宣稱完整測試套件已通過。
每次改動依受影響功能執行相關測試，並在結果紀錄中說明未執行或被阻擋的檢查。
變更模組位置時，同步更新 import、命令、測試與 README。

詳細核心模組說明見 [`core/README.md`](core/README.md)，
開發原則見 [constitution](../.specify/memory/constitution.md)。

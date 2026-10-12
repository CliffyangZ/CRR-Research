# Tasks: 牙齒 Detection 模型訓練與實驗評估

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: 依憲章 V，對座標幾何、增強、切分、指標加 pytest（`tests/detection/`）。

**Format**: `- [ ] ID [P] [Story] 描述（含路徑）`；[P] = 可平行（不同檔案、無未完成相依）。

## Phase 1: Setup

- [X] T001 建立目錄與套件檔：`src/detection/__init__.py`、`tests/detection/__init__.py`、`configs/detection/`、`reports/detection/.gitkeep`、`data/derived/`（並確認 `.gitignore` 排除 `data/derived/`、`models/*.pt`、`reports/detection/*/weights`）
- [X] T002 在 `pyproject.toml` 的 `yolo` 選配加入 `pyyaml`、`pycocotools`、`imagehash`；確認 `ultralytics` 下限提供 `yolo26*.pt`（`pip download`／`python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"` 驗證），調整版本限制並記錄精確版本到 `reports/detection/env_check.md`
- [X] T003 [P] 安裝環境並驗證 CUDA：`pip install -e ".[dev,yolo]"`，確認 PyTorch 支援 RTX 5060（sm_120），結果寫入 `reports/detection/env_check.md`
- [X] T004 [P] 依 [contracts/preprocessing-config.md](contracts/preprocessing-config.md) 建立 `configs/detection/preprocess_v1.yaml`

## Phase 2: Foundational（阻擋所有 User Story）

- [X] T005 [P] 實作 `src/detection/provenance.py`：產生 run_id、讀取 git commit／dirty（含 diff hash）、環境（python/torch/cuda/GPU）、sha256 工具、寫 `run_manifest.json`（失敗亦寫入 status=failed）
- [X] T006 [P] 實作 `src/detection/config.py`：以 pydantic 載入／驗證 PreprocessConfig 並輸出 config hash（對應 data-model）
- [X] T007 [P] 實作 `src/detection/letterbox.py`：`letterbox_image`、`LetterboxRecord`、`boxes_forward`、`boxes_inverse`（640、pad 114、INTER_AREA／LINEAR）
- [X] T008 [P] 測試 `tests/detection/test_letterbox.py`：橫／直／方形與邊緣框，往返誤差每座標 ≤1 px，`new_w+pad_l+pad_r=size`
- [X] T009 [P] 實作 `src/detection/yolo_io.py`：YOLO txt ↔ xyxy 原圖像素轉換與框驗證（越界、零面積）

## Phase 3: US1 — 建立可信的訓練資料（P1）🎯 MVP

**Goal**: 可追溯、無洩漏、前處理與增強符合規格的資料版本。
**Independent Test**: 用含正常／重複／損毀／錯誤標註的小 fixture 跑資料準備，檢查清單、品質報告、切分與疊圖。

- [X] T010 [P] [US1] 測試 `tests/detection/test_inventory.py`：損毀檔、非 8-bit、通道不一致、重複 sha256 → quarantined／excluded 且有 reason
- [X] T011 [P] [US1] 測試 `tests/detection/test_preprocess.py`：順序 NLM→CLAHE→USM、尺寸與框不變、固定線性映射（非逐張 min–max）、8-bit 以外拒絕
- [X] T012 [P] [US1] 測試 `tests/detection/test_augment.py`：三情境（框皆界內→允許；越界→取消幾何、保留亮度／對比；原已截斷牙→保留且不視為額外截斷）、種子可重現
- [X] T013 [P] [US1] 測試 `tests/detection/test_splits.py`：近重複同組不跨子集、洩漏檢查為零、缺病人資訊時 residual_risk 必填
- [X] T014 [US1] 實作 `src/detection/inventory.py`：逐檔解碼，記錄 mode／bit depth／channels／強度範圍／sha256／pHash，輸出 DatasetVersion items
- [X] T015 [US1] 實作 `src/detection/preprocess.py`：灰階化→`fastNlMeansDenoising`→CLAHE→USM→uint8/255 映射→複製 3 通道；各階段參數來自 config
- [X] T016 [US1] 實作 `src/detection/splits.py`：pHash+檔名序號+尺寸替代分群、沿用 `labeled_images.json` 61/20/20 並做洩漏檢查，必要時重分群（seed=42）、鎖定 test、輸出 SplitManifest
- [X] T017 [US1] 實作 `src/detection/augment.py`：亮度／對比→旋轉→平移，框同步變換；任一框越界即取消整次幾何；每份記錄 AugmentedSample（僅 train，K=3，固定種子）
- [X] T018 [US1] 實作 `src/detection/dataset_export.py`：輸出 Ultralytics 資料集（`data/derived/<dv>/{images,labels}/{train,val,test}` 與 `data.yaml`，使用相對路徑）、LetterboxRecord 與 manifests，原始影像不被修改
- [X] T019 [US1] 在 `src/detection/dataset_export.py` 加疊圖輸出（前處理後影像+框，含增強樣例）至 `reports/detection/<dv>/overlays/`，供對齊目視檢查
- [X] T020 [US1] 實作 `research/data/prepare_tooth_dataset.py`（依 [contracts/cli.md](contracts/cli.md)，支援 `--limit`）：盤點→標註驗證→前處理→切分→增強→匯出；產出品質／分布摘要與排除清單 `reports/detection/<dv>/data_report.md`
- [X] T021 [US1] 確認標註來源與審閱狀態（`data/seg_annotations`→`data/detection_annotations`），寫入 AnnotationVersion `review_status`；若非人工確認則在報告標示為偽標註
- [X] T022 [US1] 以 `--limit 12` 小樣本執行 T020，確認 sha256 原檔不變、疊圖對齊、`pytest tests/detection` 通過；再對全部 101 張執行並凍結 `data_version`

**Checkpoint**: 資料版本與切分已凍結，可獨立驗收 SC-001。

## Phase 4: US2 — 選擇模型並完成可重現實驗（P2）

**Goal**: v8、v11、v26 各一次可重現訓練，僅用驗證集選模。
**Independent Test**: 用 `--smoke`（≤8 張、2 epochs）跑三模型，檢查 manifest 欄位與失敗紀錄。

- [X] T023 [P] [US2] 建立 `reports/detection/model_candidates.md` 與 `src/detection/models.py`：登錄 v8/v11/v26 的 s 規模、來源、精確版本、授權（AGPL-3.0）、權重 sha256、PA 適配性、VRAM 需求
- [X] T024 [P] [US2] 實作 `src/detection/train_args.py`：產生統一訓練參數（imgsz 640、epochs 100、patience 30、batch 8、seed 0、deterministic）並強制關閉全部內建增強（mosaic/mixup/cutmix/copy_paste/fliplr/flipud/scale/shear/perspective/degrees/translate/hsv_*/erasing/auto_augment）；測試 `tests/detection/test_train_args.py` 驗證每個關閉項
- [X] T025 [US2] 實作 `research/train/train_tooth_yolo.py`（`--model {v8,v11,v26} --smoke --data-version --seed --device`）：凍結 data_version 檢查、登錄設定、呼叫 Ultralytics、OOM 時降 batch 並記錄、例外時寫 failed manifest
- [X] T026 [US2] 訓練前列印並保存實際生效的 Ultralytics 增強參數至 `run_manifest.json`，確認無預設增強遺漏
- [X] T027 [US2] 對 v8、v11、v26 各跑 `--smoke`，確認 manifest 完整（FR-008 欄位）
- [X] T028 [US2] 正式訓練 v8，保存 `best.pt`（依驗證集 AP@[.5:.95]）、sha256、選模依據
- [X] T029 [US2] 正式訓練 v11（同上）
- [X] T030 [US2] 正式訓練 v26（同上）；若有不可一致設定（如 NMS-free）記錄於 manifest

**Checkpoint**: 三組權重與 manifest 就緒（SC-002）。

## Phase 5: US3 — 確定評估與分析結果（P3）

**Goal**: 固定規約下的指標、錯誤分析與比較報告。
**Independent Test**: 以人工構造的 GT／預測 fixture 核對指標與錯誤分類。

- [X] T031 [P] [US3] 測試 `tests/detection/test_evaluate.py`：完美預測、全漏檢、重複框、定位偏差、無預測／無 GT 零分母 → `undefined` 而非 0；AP 與 pycocotools 交叉檢驗
- [X] T032 [US3] 實作 `src/detection/evaluate.py`：框還原到原圖、匹配、AP@[.5:.95]（101 點）、AP50、P/R/F1、錯誤分類（missed／false_positive／duplicate／localization）
- [X] T033 [US3] 實作 `research/evaluation/evaluate_tooth_yolo.py`：以 conf=0.001 在 val 預測→選 τ（F1 最大，同值取高）→鎖定寫檔→測試集評估一次；輸出 `metrics.json`、`errors.json`、錯誤與成功案例疊圖
- [X] T034 [US3] 對三個 run 執行評估，確認每項指標有值或明確 `undefined_reason`，並列出 n_images／n_boxes
- [X] T035 [US3] 實作 `research/evaluation/compare_runs.py` 並產出 `reports/detection/comparison/report.md`：三模型指標表、錯誤計數與代表案例、分組指標（可用者，否則標「未量測」）、種子變異「未量測」、探索性聲明、限制（101 張、無病人資訊、標註審閱狀態）、至少一項由錯誤分析支持的後續建議
- [X] T036 [US3] 報告檢查：每個結論連回 run_id／data_version／split_id／權重 sha256；無可識別病人資料

## Phase 6: Polish

- [X] T037 [P] 更新 `src/README.md`、`research/README.md`、`reports/README.md`，區分已實作與規劃；記錄既有失效測試（引用已移除 `src/models/`）
- [X] T038 [P] （部分：僅於現有 .venv 驗證，未乾淨環境重跑）依 [quickstart.md](quickstart.md) 從乾淨環境完整走一遍，修正命令與路徑
- [X] T039 對照 SC-001～SC-005 逐項勾選，結果寫入 `reports/detection/acceptance.md`

## Dependencies

- Phase 1 → Phase 2 → US1 → US2 → US3 → Polish（US2 需凍結資料；US3 需訓練權重）。
- US1 內：T010–T013（測試）先於 T014–T018；T020 依賴 T014–T019；T022 依賴 T020–T021。
- US2 內：T023、T024 → T025 → T027 → T028–T030（三者可依序或在資源許可下分開跑；單 8 GB GPU 建議序列）。
- US3 內：T031 → T032 → T033 → T034 → T035 → T036。

## Parallel Examples

- Phase 2：T005、T006、T007、T009 同時；T008 在 T007 後。
- US1：T010–T013 四個測試檔同時。
- US2：T023、T024 同時。

## Implementation Strategy

- **MVP = US1**（凍結資料版本與切分）：可獨立驗收並檢視資料品質與標註狀態，這也是最大風險所在（標註審閱狀態、病人分群）。
- 之後 US2 先以 `--smoke` 驗證三模型管線，再正式訓練；最後 US3 評估與比較。
- 所有結果標示探索性。

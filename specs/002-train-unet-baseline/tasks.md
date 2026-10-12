# Tasks: 牙齒語意分割 U-Net Baseline 訓練

**Input**: `/specs/002-train-unet-baseline/`（plan.md、spec.md、research.md、data-model.md、contracts/、quickstart.md）

**Prerequisites**: plan.md、spec.md

**Tests**: 依憲章 V，遮罩光柵化、letterbox 還原、增強取消、切分洩漏、指標皆附 pytest（置於 `tests/segmentation/`）。

**Organization**: 依使用者故事分組（US1 資料 → US2 訓練 → US3 評估）。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可平行（不同檔案、無未完成依賴）
- **[Story]**: US1／US2／US3

## Phase 1: Setup

- [X] T001 在 pyproject.toml 新增選配依賴群組 `unet = ["torch", "pyyaml>=6", "imagehash>=4.3"]`
- [X] T002 [P] 建立套件骨架 src/segmentation/__init__.py、tests/segmentation/__init__.py、research/models/__init__.py
- [X] T003 [P] 依 contracts/train-config.md 建立 configs/segmentation/unet_baseline_v1.yaml
- [X] T004 [P] 在 .gitignore 加入 data/derived/seg_*/ 與 models/segmentation/（大型產物不入版本庫）

---

## Phase 2: Foundational（阻擋所有故事）

- [X] T005 依 contracts/train-config.md 以 pydantic 實作 SegConfig 與 load_config（停用項啟用即報錯）於 src/segmentation/config.py
- [X] T006 [P] 實作 letterbox 512 影像／遮罩轉換與預測還原原圖（重用 src/detection/letterbox.py 的 compute_record；遮罩 nearest、補邊 0、去補邊後 bilinear 還原）於 src/segmentation/transform.py
- [X] T007 [P] 實作 tooth 多邊形 cv2.fillPoly 聯集光柵化，忽略 Metal，含合法性檢查（≥3 點、面積>0、越界容差 1 px）於 src/segmentation/masks.py
- [X] T008 [P] 實作像素累計 Dice/IoU/precision/recall、逐圖平均、零分母 undefined 原因、全背景統計於 src/segmentation/metrics.py
- [X] T009 [P] 實作 BCEWithLogits + soft Dice 損失於 src/segmentation/losses.py
- [X] T010 [P] 在 tests/segmentation/test_masks.py 驗證多牙聯集、Metal 重疊保留 tooth、僅 Metal 為背景、退化／越界多邊形
- [X] T011 [P] 在 tests/segmentation/test_transform.py 驗證不同長寬比 letterbox、遮罩無新值、控制點還原誤差每軸 ≤1 原圖像素
- [X] T012 [P] 在 tests/segmentation/test_metrics.py 驗證已知遮罩配對、零分母 undefined、補邊不計、累計與逐圖平均不混用

**Checkpoint**: 共用基礎完成

---

## Phase 3: User Story 1 - 可信且對齊的分割資料 (P1) 🎯 MVP

**Goal**: 盤點、洩漏檢查、固定前處理、遮罩、凍結切分、訓練專用同步增強。

**Independent Test**: 以多牙、Metal 重疊、無牙齒、不同長寬比、合法變換、新增截斷、原圖截斷七類案例核對遮罩與增強規則（SC-002）。

- [X] T013 [P] [US1] 在 tests/segmentation/test_augment.py 撰寫七類案例：合法變換套用、新增截斷取消整次幾何並保留亮度／對比、原圖已截斷牙齒不再損失則通過、遮罩標籤仍為 {0,1}、不刪標註
- [X] T014 [P] [US1] 在 tests/segmentation/test_splits.py 驗證近重複組不跨子集與 leakage_check 輸出
- [X] T015 [US1] 實作 COCO 盤點（可解碼性、位元深度、通道、尺寸、配對、類別、sha256、pHash、排除原因，Metal 狀態記為 unknown，tooth 審閱狀態含使用者 2026-10-12 確認）於 src/segmentation/inventory.py（重用 src/detection/inventory.py）
- [X] T016 [US1] 實作影像＋遮罩同步增強，含逐牙像素保留比截斷判定（容差 0.005、擴大畫布 25%）與取消邏輯於 src/segmentation/augment.py
- [X] T017 [US1] 實作 torch Dataset（離線前處理快取、train 線上增強、val/test 僅固定處理、回傳 LetterboxRecord 與增強紀錄）於 src/segmentation/dataset.py
- [X] T018 [US1] 實作資料準備入口：盤點 → 重用 src/detection/splits.py 分群與洩漏檢查 → NLM/CLAHE/USM 快取 → 遮罩 → 凍結 splits.json 與 dataset_manifest.json（--limit、--resplit，保存新舊清單與理由）於 research/data/prepare_tooth_seg_dataset.py
- [X] T019 [US1] 以 --limit 8 與完整資料執行 T018，確認排除項皆有原因、跨子集重複組為零，並將盤點摘要寫入 reports/segmentation/data_audit/README.md

**Checkpoint**: US1 可獨立驗證

---

## Phase 4: User Story 2 - 可追溯的基準訓練 (P2)

**Goal**: 完成一次正式 U-Net 訓練，驗證集選模，保留 best/last 與完整紀錄。

**Independent Test**: smoke（≤8 張、2 epochs）完成並重載權重；OOM／中斷時 manifest 為 failed。

- [X] T020 [P] [US2] 實作 U-Net（通道 32-64-128-256-512、depth 4、BN、轉置卷積、Kaiming 初始化）於 research/models/unet.py
- [X] T021 [P] [US2] 在 tests/segmentation/test_unet_smoke.py 驗證輸出形狀 (B,1,512,512)、單步前向／反向、權重存取重載
- [X] T022 [US2] 於 src/segmentation/train.py 重用 src/detection/provenance.py 實作 RunManifest（run_id、命令、git dirty、環境、種子、設定、資料與切分 sha256、權重 sha256、不覆寫既有目錄）
- [X] T023 [US2] 於 src/segmentation/train.py 實作訓練迴圈：AdamW + cosine、AMP、每 epoch 於原圖座標計算驗證 Dice、早停 patience 30、同值取較早輪選模、保存 best.pt／last.pt／history.csv；try/finally 寫 failed 與原因
- [X] T024 [US2] 實作訓練入口（--data-version、--config、--seed、--device、--output、--smoke；model 以參數注入）於 research/train/train_tooth_unet.py
- [X] T025 [US2] 執行 --smoke，確認 best/last 可載入；再以過大 batch 強制 OOM，確認 status=failed 且無成功標記
- [X] T026 [US2] 執行正式訓練一次，確認 run_manifest.json status=succeeded、history.csv 與選模依據完整，並在 reports/segmentation/<run_id>/ 保存

**Checkpoint**: US2 可獨立驗證

---

## Phase 5: User Story 3 - 固定測試集表現 (P3)

**Goal**: 在原圖座標產出指標、疊圖與三類錯誤案例及改善建議。

**Independent Test**: 已知遮罩配對核對指標；以保存權重對固定 test 清單產出報告。

- [X] T027 [P] [US3] 實作疊圖與錯誤案例分類（漏分 FN 連通區 ≥ GT 2%、誤分 FP 連通區 ≥ 影像 0.1%、HD95 邊界偏差；未發生標零例）於 src/segmentation/report.py
- [X] T028 [P] [US3] 在 tests/segmentation/test_report.py 以合成遮罩驗證三類錯誤分類與零例標示
- [X] T029 [US3] 實作評估入口：載入 best.pt（驗證 sha256）→ 還原原圖 → 閾值 0.5 → metrics.json、per_image.csv、疊圖，內含母體、有效像素、審閱狀態與未量測聲明，於 research/evaluation/evaluate_tooth_unet.py
- [X] T030 [US3] 對固定 test 清單執行 T029，撰寫 reports/segmentation/<run_id>/eval/README.md（指標定義、洩漏限制、單種子探索性、至少一項有案例支持的改善建議）

**Checkpoint**: 全部故事完成

---

## Phase 6: Polish

- [X] T031 [P] 更新 research/README.md、src/README.md、reports/README.md，區分既有功能與新增 segmentation 能力
- [X] T032 [P] 新增 configs/segmentation/README.md 或於 research/README.md 記錄資料、權重取得方式與授權待盤點事項
- [X] T033 依 quickstart.md 逐步執行，確認命令與路徑正確；記錄既有失效測試（若有）而不宣稱通過
- [X] T034 執行 pytest tests/segmentation -q 並保存輸出至 reports/segmentation/<run_id>/tests.txt

---

## Dependencies & Order

- Setup → Foundational → US1 → US2 → US3 → Polish；US2 依賴 US1 的凍結切分與快取，US3 依賴 US2 的權重。
- T020、T021（模型）僅依賴 Setup，可與 US1 平行。

## Parallel Examples

- Foundational：T006–T012 可同時進行。
- US1：T013、T014 可平行；US2：T020、T021 可平行；US3：T027、T028 可平行。

## Implementation Strategy

1. 完成 Setup + Foundational。
2. **MVP = US1**：資料可信、遮罩對齊、增強規則通過七類案例後停下驗證。
3. 再做 US2（smoke → 正式訓練），最後 US3（評估與報告）。

# Tasks: 單顆牙齒 ROI 分割資料集

**Input**: `/specs/003-tooth-roi-seg-dataset/`（plan.md、spec.md、research.md、data-model.md、contracts/、quickstart.md）

**Prerequisites**: plan.md、spec.md；spec 002 的 `src/segmentation/`（augment、masks、transform、config）已存在

**Tests**: 依憲章 V，配對、裁切／還原、單牙遮罩、增強取消、切分洩漏皆附 pytest（`tests/segmentation/`）。

**Organization**: US1 配對 → US2 ROI 影像與遮罩 → US3 切分、增強與報告。

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 依 contracts/roi-config.md 建立 configs/segmentation/roi_dataset_v1.yaml（每側外擴 0.10、size 256、pad 114、min_iou 0.5、second_margin 0.1、min_mask_frac 0.05、CLAHE only、增強同 Spec1）
- [X] T002 [P] 建立空檔 src/segmentation/roi_config.py、pairing.py、roi.py、roi_dataset.py、roi_report.py 與 research/data/prepare_tooth_roi_dataset.py（含模組 docstring）

---

## Phase 2: Foundational

- [X] T003 以 pydantic 實作 RoiConfig 與 load_roi_config（含 hash；`preprocess` 出現 nlm／usm 或 augment.disabled 缺六項即報錯；重用 src/segmentation/config.py 的 Augment）於 src/segmentation/roi_config.py
- [X] T004 [P] 在 tests/segmentation/test_roi_config.py 驗證 NLM／USM 被拒、停用項不可啟用、config hash 穩定

**Checkpoint**: 設定可載入

---

## Phase 3: User Story 1 - 可信的框—多邊形配對 (P1) 🎯 MVP

**Goal**: 每個偵測框一對一配對 tooth 多邊形，不確定者隔離並記錄原因。

**Independent Test**: 框遮罩一致、框偏移、相鄰多牙、無對應多邊形、多框競爭五類合成案例（SC-002 前五類）。

- [X] T005 [P] [US1] 在 tests/segmentation/test_pairing.py 撰寫合成案例：一致配對、偏移但 IoU≥0.5 通過、相鄰牙最佳與次佳差<0.1 隔離、無多邊形（no_polygon）、低 IoU、兩框搶同一多邊形（領先≥0.1 保留一個；否則全隔離）、退化框、orphan 多邊形、YOLO 正規化座標換算
- [X] T006 [US1] 實作 YOLO 標籤讀取（正規化 → 原圖像素 xyxy，退化／越界框標記）、IoU 矩陣、一對一配對與隔離原因（no_polygon、low_iou、ambiguous_second_best、polygon_contested、degenerate_box）、orphan 多邊形於 src/segmentation/pairing.py
- [X] T007 [US1] 在 src/segmentation/pairing.py 實作來源盤點：兩來源影像與子集一一對應檢查（subset_mismatch／缺檔隔離）、各子集框數／多邊形數統計，重用 src/segmentation/inventory.py 與 src/detection/inventory.py 的逐檔檢查
- [X] T008 [US1] 在 research/data/prepare_tooth_roi_dataset.py 實作盤點與配對階段（--config、--output、--limit），輸出 data/derived/roi_<dv>/pairs.json、quarantine.json 與 dataset_manifest.json 的 counts 與 annotation_review（polygons＝人工且專家確認、boxes＝unknown），輸出已存在則拒絕覆寫
- [X] T009 [US1] 以 --limit 9 與完整資料執行 T008，檢視配對統計（對照 spec 估計約 232/273），確認每個隔離項有原因、無多邊形被兩個框共用；若 5% 或 IoU 門檻需調整，只能依 train／val 並記錄理由

**Checkpoint**: US1 可獨立驗證

---

## Phase 4: User Story 2 - ROI 影像與單牙遮罩 (P2)

**Goal**: 每個已配對框產生 256×256 CLAHE ROI 與只含目標牙的遮罩，可還原回原圖座標。

**Independent Test**: 控制點還原每軸 ≤1 原圖像素；遮罩僅 {0,1} 且只含目標牙。

- [X] T010 [P] [US2] 在 tests/segmentation/test_roi.py 撰寫案例：框在影像內、框外擴越界補 114／遮罩補 0 並記錄補足量、原圖已截斷、鄰牙為背景並計 neighbor_in_roi_frac、Metal 重疊仍為背景、tiny_target 隔離、控制點還原 ≤1 px、遮罩值域 {0,1}
- [X] T011 [US2] 實作每側外擴、越界補足裁切、重用 src/segmentation/transform.letterbox_pair 縮放至 256、單牙遮罩光柵化（重用 masks.rasterize_union 對單一標註）、座標還原函式、target_truncated_by_roi／truncated_at_source／neighbor_in_roi_frac／tiny_target 判定於 src/segmentation/roi.py
- [X] T012 [US2] 在 research/data/prepare_tooth_roi_dataset.py 加入 ROI 階段：完整原圖套用 src/detection/preprocess.clahe（僅 CLAHE）→ 裁切／letterbox → 寫入 images/<roi_id>.png、masks/<roi_id>.png 與 roi_index.json（欄位依 data-model.md）
- [X] T013 [US2] 完整建置後抽樣檢視：確認 train/val/test ROI 數、越界補足量統計、遮罩值域、隨機 10 個 ROI 還原誤差 ≤1 px，記錄於 reports/segmentation/roi_<dv>/README.md 草稿

**Checkpoint**: US2 可獨立驗證

---

## Phase 5: User Story 3 - 防洩漏切分、增強與報告 (P3)

**Goal**: 沿用切分並檢查洩漏；訓練專用增強；產出資料驗證報告。

**Independent Test**: 同圖 ROI 同子集、跨子集近重複為零；增強兩類（合法、新增截斷）與 val／test 無增強。

- [X] T014 [P] [US3] 在 tests/segmentation/test_roi_splits.py 驗證 ROI 繼承來源影像子集、同圖 ROI 不跨子集、近重複跨子集時重切且保存新舊清單
- [X] T015 [P] [US3] 在 tests/segmentation/test_roi_dataset.py 驗證：val／test 的 augment 皆為 not_selected、train 增強同步影像與遮罩、遮罩仍為 {0,1}、目標牙新增截斷取消幾何且保留亮度／對比、相同 (seed, epoch, idx) 結果一致
- [X] T016 [US3] 在 research/data/prepare_tooth_roi_dataset.py 加入切分階段：確認兩來源子集一致，對納入影像計算 pHash 並重用 src/detection/splits.make_split（existing 優先，--resplit、--seed），ROI 繼承子集，凍結 splits.json（含 residual_risk）
- [X] T017 [US3] 實作 RoiSegDataset（讀 roi_index.json，train 才啟用增強，重用 src/segmentation/augment.augment_once，instances＝目標牙單一遮罩，pad 114，種子 (seed, epoch, idx)，回傳 image／mask／roi 紀錄）於 src/segmentation/roi_dataset.py
- [X] T018 [US3] 實作報告：各子集框數、多邊形數、納入／隔離（依原因）、IoU 與次佳 IoU 分位數、截斷與含鄰牙比例、orphan 多邊形數、疊圖（合法、各類隔離、截斷）、限制與未量測項於 src/segmentation/roi_report.py，並由 research/data/prepare_tooth_roi_dataset.py 輸出 reports/segmentation/roi_<dv>/README.md 與 overlays/
- [X] T019 [US3] 以增強取樣多 epoch 統計 geometry 分布（applied／cancelled／not_selected），寫入報告；並以相同輸入與種子重建一次，確認 roi_index.json 內容一致（SC-005）

**Checkpoint**: 全部故事完成

---

## Phase 6: Polish

- [X] T020 [P] 更新 research/README.md、src/README.md、reports/README.md，區分既有功能與新增 ROI 資料集能力，並記錄偵測框審閱狀態為未知、資料授權待盤點
- [X] T021 依 quickstart.md 逐步執行並修正命令與路徑；記錄既有失效測試（若有）而不宣稱通過
- [X] T022 執行 pytest tests/segmentation -q 並保存輸出至 reports/segmentation/roi_<dv>/tests.txt

---

## Dependencies & Order

- Setup → Foundational → US1 → US2 → US3 → Polish；US2 依賴 US1 的配對，US3 的 Dataset 依賴 US2 的 roi_index。
- T016（切分）只依賴 T008，可與 US2 平行；其餘依序。

## Parallel Examples

- US1：T005 與 T006 前可先寫測試；US2：T010 與 T011 的測試／實作分檔；US3：T014、T015 可同時撰寫。

## Implementation Strategy

1. 完成 Setup + Foundational。
2. **MVP = US1**：配對統計可信、隔離原因完整後停下檢視門檻（尤其 min_mask_frac 實際分布）。
3. 再做 US2（ROI 與遮罩），最後 US3（切分、增強、報告）。

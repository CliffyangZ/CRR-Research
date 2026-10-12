# Implementation Plan: 牙齒 Detection 模型訓練與實驗評估

**Branch**: `001-train-tooth-detection` | **Date**: 2026-10-09 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-train-tooth-detection/spec.md`

## Summary

以 PA 單類別 tooth 偵測為範圍，建立「固定前處理（NLM → CLAHE → USM）→ 固定 letterbox → 訓練專用保守增強」的資料管線，並用 Ultralytics 訓練 YOLO v8、v11、v26 各一次，在同一份固定切分與評估規約下比較。技術做法：共用邏輯放 `src/`（前處理、letterbox 與座標還原、增強保護、資料驗證、指標），實驗編排放 `research/`，證據放 `reports/`。增強以自訂「全框界內否則取消幾何變換」邏輯實作，並關閉 Ultralytics 預設增強。細節見 [research.md](research.md)、[data-model.md](data-model.md)、[contracts/](contracts/)、[quickstart.md](quickstart.md)。

## Technical Context

**Language/Version**: Python 3.12（`pyproject.toml` 限定 `>=3.12,<3.13`）

**Primary Dependencies**: ultralytics（`yolo` 選配，需確認支援 YOLO26 的最小版本，現限 `>=8.3.0,<9`）、PyTorch（CUDA，需支援 RTX 5060 / Blackwell）、opencv-python（NLM、CLAHE、USM）、numpy、pydantic、PyYAML

**Storage**: 檔案系統；原始影像 `data/original/`（不可變）；衍生資料 `data/derived/<data_version>/`；權重 `models/`（不入版本庫）；證據 `reports/`

**Testing**: pytest；小型 fixture 驗證 letterbox 還原、增強取消邏輯、切分洩漏與指標

**Target Platform**: Linux，單 GPU（NVIDIA RTX 5060 Laptop，8 GB VRAM）

**Project Type**: 研究管線（library + CLI 入口），非服務

**Performance Goals**: 無效能門檻（FR-011）；三模型各完成一次可重現訓練與評估

**Constraints**: 8 GB VRAM → 小規模模型（n/s）、imgsz 640、小 batch；資料僅 101 張已標註影像（61/20/20），結果標示探索性；NLM 離線預計算以免訓練時重複耗時

**Scale/Scope**: 101 張影像、PA 單類別；3 個模型 × 1 次訓練；`data/original/` 共 282 張，其中僅 101 張有標註

**已知缺口（NEEDS CLARIFICATION 已於 research.md 解決或登錄為風險）**: 無病人識別資訊（以替代分群處理）；現有 `data/detection_dataset` 為 1200×825 等多尺寸 RGB／L JPG，需逐檔盤點。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] I. `src/detection/` 放可重用實作，`research/` 放編排與入口，`reports/detection/` 放證據；`src/` 不依賴 research。README 區分現有與新增。
- [x] II. 每次 run 有 run_id；`run_manifest.json` 記錄命令、git commit/dirty、環境、種子、資料／切分版本、權重 hash（見 data-model）；失敗亦保存。單次結果標示探索性。
- [x] III. 原始影像唯讀；衍生資料帶處理設定；切分以替代分群（近重複 + 檔名序號鄰近）處理，無病人資訊則明載殘餘風險；增強僅訓練集；測試集鎖定於調參前，閾值與選模僅用驗證集。
- [x] IV. 標註來源與審閱狀態寫入 annotation version（現有標註來自 CVAT / seg 標註推導，須確認是否人工確認）；指標定義見 contracts/evaluation-protocol.md；未量測標示。
- [x] V. 座標幾何、增強、切分、指標皆附 pytest fixture；文件變更檢查路徑與命令。

**Post-design 重新檢查**: 通過，無需 Complexity Tracking 例外。唯一需注意：`tests/` 既有部分失效測試須記錄而非宣稱通過。

## Project Structure

### Documentation (this feature)

```text
specs/001-train-tooth-detection/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── preprocessing-config.md
│   ├── evaluation-protocol.md
│   └── cli.md
└── tasks.md             # /speckit-tasks 產出，本命令不建立
```

### Source Code (repository root)

```text
src/detection/                 # 新增：可重用實作
├── inventory.py               # 影像盤點（位元深度、通道、尺寸、hash）
├── preprocess.py              # NLM → CLAHE → USM → 固定線性映射
├── letterbox.py               # 等比縮放補邊 + 框前向／反向轉換
├── splits.py                  # 分群切分與洩漏檢查
├── augment.py                 # 保守增強 + 越界則取消幾何變換
├── dataset_export.py          # 輸出 Ultralytics YOLO 資料集
├── evaluate.py                # AP / P / R / F1、錯誤分類
└── provenance.py              # run manifest（git、環境、hash）

research/
├── data/prepare_tooth_dataset.py     # 編排：盤點→前處理→切分→匯出
├── train/train_tooth_yolo.py         # 編排：單模型訓練（--model v8|v11|v26）
└── evaluation/evaluate_tooth_yolo.py # 編排：選閾值（val）→ 測試評估 → 報告

reports/detection/<run_id>/    # 證據：manifest、指標、錯誤案例、疊圖
tests/detection/               # pytest fixtures
```

**Structure Decision**: 遵循憲章 I 的 `research/`（編排）、`src/`（實作）、`reports/`（證據）分工；新增 `src/detection/` 與 `tests/detection/`，不改動既有 segmentation 模組。

## Complexity Tracking

無例外。

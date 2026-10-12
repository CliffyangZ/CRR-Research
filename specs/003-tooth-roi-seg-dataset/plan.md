# Implementation Plan: 單顆牙齒 ROI 分割資料集

**Branch**: 沿用目前分支（`main`；setup 腳本回報 `003-tooth-roi-seg-dataset`） | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-tooth-roi-seg-dataset/spec.md`

## Summary

以 `data/detection_dataset/`（YOLO 偵測框）定義單顆牙 ROI，以 `data/seg_dataset/`（COCO tooth 多邊形）定義標籤：
「盤點兩來源 → 框—多邊形一對一配對（IoU≥0.5、與次佳差≥0.1，不合者隔離）→ 框每側外擴 10% 裁切（超出原圖補背景）→
固定 CLAHE → 等比例 letterbox 256×256 → 單牙遮罩（鄰牙／Metal 為背景）→ 沿用既有切分並做洩漏檢查 → 驗證報告」，
並提供訓練專用增強（同 Spec1，目標牙新增截斷取消幾何變換）。本功能只產出資料集與報告，不訓練模型。
共用邏輯放 `src/segmentation/`（擴充 spec 002 的模組，重用 `src/detection/` 的 CLAHE、inventory、splits、provenance），
入口放 `research/data/`，證據放 `reports/segmentation/roi_*`。細節見 [research.md](research.md)、[data-model.md](data-model.md)、
[contracts/](contracts/)、[quickstart.md](quickstart.md)。

## Technical Context

**Language/Version**: Python 3.12（`pyproject.toml` 限定 `>=3.12,<3.13`）

**Primary Dependencies**: opencv-python（CLAHE、fillPoly、warpAffine）、numpy、pydantic、PyYAML、imagehash（近重複）、PyTorch（僅 ROI Dataset 的張量輸出，已在 `unet` 選配群組）；不新增依賴

**Storage**: 檔案系統；原始資料 `data/detection_dataset/`、`data/seg_dataset/` 唯讀；衍生資料 `data/derived/roi_<data_version>/`；證據 `reports/segmentation/roi_<data_version>/`

**Testing**: pytest（`tests/segmentation/`），以合成框／多邊形驗證配對、裁切、還原與增強取消

**Target Platform**: Linux（無 GPU 需求）

**Project Type**: 研究資料管線（library + CLI 入口）

**Performance Goals**: 無；全資料建置預期在數十秒至數分鐘內完成（CLAHE 僅 273 個 ROI）

**Constraints**: 101 張影像、273 個偵測框、233 個 tooth 多邊形；約一半框外擴後超出原圖（train 86/162、val 26/58、test 26/53），需補背景並標記

**Scale/Scope**: 約 230 個 ROI（估計），單類 tooth；不訓練模型

**NEEDS CLARIFICATION 狀態**: 全部於 [research.md](research.md) 解決（外擴定義、補邊值、配對規則、最小遮罩、ROI 截斷與鄰牙標記、切分來源）。已揭露風險：無病人資訊、偵測框審閱狀態未知、被隔離框的選擇偏差。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] I. 入口 `research/data/prepare_tooth_roi_dataset.py`；可重用邏輯放 `src/segmentation/`（`roi_config.py`、`pairing.py`、`roi.py`、`roi_dataset.py`、`roi_report.py`）；證據放 `reports/segmentation/`；`src` 不依賴 research。README 區分既有與新增。
- [x] II. 每次建置有 `build_id`／`dataset_manifest.json`（命令、git commit／dirty、環境、種子、設定與其 hash、來源檔 sha256、輸出路徑）；失敗保存原因；輸出目錄存在則拒絕覆寫；相同輸入與種子得相同清單（確定性排序）。
- [x] III. 原始資料唯讀；ROI 保留來源框與多邊形 id 與轉換紀錄；沿用既有切分並以 pHash 近重複查核，同圖 ROI 同子集；增強僅訓練集；CLAHE 為固定參數、不從資料擬合；配對門檻不得參考測試集；報告不含可辨識病人資訊。
- [x] IV. 多邊形標註＝人工且專家確認（使用者 2026-10-12）；偵測框審閱狀態記為「未知」，兩者分開記錄；報告為資料統計（單位＝ROI，母體＝子集），不宣稱模型表現；未量測項標示。
- [x] V. 配對、裁切／還原（≤1 px）、遮罩單牙化、增強取消、切分洩漏皆附 pytest；文件同步更新 README；既有失效測試不宣稱通過。

**Post-design 重新檢查**: 通過，無需 Complexity Tracking。

## Project Structure

### Documentation (this feature)

```text
specs/003-tooth-roi-seg-dataset/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   ├── roi-config.md
│   └── pairing-and-roi-protocol.md
└── tasks.md             # /speckit-tasks 產出，本命令不建立
```

### Source Code (repository root)

```text
research/data/prepare_tooth_roi_dataset.py   # 編排：盤點 → 配對 → ROI → 切分 → 報告

src/segmentation/                            # 擴充 spec 002 的套件
├── roi_config.py     # RoiConfig（pydantic，含 CLAHE-only 前處理、配對、ROI、增強）
├── pairing.py        # YOLO 框讀取、IoU、一對一配對、隔離原因
├── roi.py            # 外擴裁切（越界補背景）、letterbox 256、單牙遮罩、座標還原、截斷／鄰牙標記
├── roi_dataset.py    # torch Dataset：ROI 影像＋單牙遮罩，訓練專用增強（重用 augment.py）
└── roi_report.py     # 統計、IoU 分布、疊圖與 Markdown 報告

tests/segmentation/  # test_pairing / test_roi / test_roi_dataset / test_roi_splits

configs/segmentation/roi_dataset_v1.yaml
data/derived/roi_<dv>/{images,masks}/、roi_index.json、pairs.json、quarantine.json、splits.json、dataset_manifest.json
reports/segmentation/roi_<dv>/README.md、overlays/
```

**Structure Decision**: 延用 spec 002 的 `src/segmentation/` 與 `research/data/` 分工；`augment.py`、`masks.py`、`transform.py` 直接重用，新增檔案只處理框—多邊形配對與 ROI 幾何。CLAHE 呼叫 `src.detection.preprocess.clahe`。

## Complexity Tracking

無違反項目。

# Implementation Plan: 牙齒語意分割 U-Net Baseline 訓練

**Branch**: 沿用目前分支（`main`；spec 未建立分支，setup 腳本回報 `002-train-unet-baseline`） | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-train-unet-baseline/spec.md`

## Summary

以 `data/seg_dataset/`（COCO 多邊形，train 61／val 20／test 20）建立牙齒／背景二元語意分割 baseline：
tooth 多邊形聯集為前景，Metal 不獨立預測。流程為「盤點與洩漏檢查 → 離線固定前處理（NLM → CLAHE → USM，沿用 spec1）→
固定 letterbox 512 → 訓練專用保守增強（影像＋遮罩同步，任一牙齒新增截斷即取消整次幾何變換）→ U-Net 訓練（驗證 Dice 選模）→
原圖座標固定測試評估與錯誤案例報告」。U-Net 架構放 `research/models/unet.py`，共用邏輯放新增 `src/segmentation/`
（重用 `src/detection/` 的前處理、letterbox、provenance、splits），入口放 `research/`，證據放 `reports/segmentation/`。
細節見 [research.md](research.md)、[data-model.md](data-model.md)、[contracts/](contracts/)、[quickstart.md](quickstart.md)。

## Technical Context

**Language/Version**: Python 3.12（`pyproject.toml` 限定 `>=3.12,<3.13`）

**Primary Dependencies**: PyTorch 2.13（已安裝，CUDA 13.0）、opencv-python（NLM／CLAHE／USM、fillPoly、warpAffine）、numpy、pydantic、PyYAML、imagehash（近重複）；新增選配依賴群組 `unet = [torch, pyyaml, imagehash]`，不使用 pycocotools（多邊形自行以 cv2 光柵化）

**Storage**: 檔案系統；原始資料 `data/seg_dataset/` 唯讀；衍生資料 `data/derived/seg_<data_version>/`；權重 `models/segmentation/<run_id>/`（不入版本庫）；證據 `reports/segmentation/<run_id>/`

**Testing**: pytest；`tests/segmentation/` 以合成小樣本驗證遮罩光柵化、letterbox 還原、增強取消、指標、切分洩漏

**Target Platform**: Linux，單 GPU（RTX 5060 Laptop，8 GB VRAM）

**Project Type**: 研究管線（library + CLI 入口）

**Performance Goals**: 無最低準確度門檻（FR-012）；一次正式訓練 + 固定測試評估即達標

**Constraints**: 8 GB VRAM → 512×512 單通道輸入、batch 4、AMP；訓練資料僅 61 張（離線展開增強前），結果標示探索性單種子

**Scale/Scope**: 101 張影像、單一 U-Net、單種子；多模型、多種子、Metal 分割、牙位均不納入

**NEEDS CLARIFICATION 狀態**: 全部於 [research.md](research.md) 解決（尺寸、損失、最佳化、輪數、停止條件、增強數值、截斷判定、初始化）。仍為已揭露風險而非待釐清：無病人識別資訊、資料授權待盤點、Metal 審閱狀態未知。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] I. 架構 `research/models/unet.py`（使用者指定）；編排入口 `research/{data,train,evaluation}/`；可重用邏輯 `src/segmentation/`（不 import research）；證據 `reports/segmentation/`。各 README 同步區分既有與新增。
- [x] II. 每次 run 有 `run_id` 與 `run_manifest.json`（命令、git commit／dirty、環境、種子、超參數、資料／標註版本、切分、增強設定、初始化、權重 sha256）；失敗亦保存、不覆寫；裝置非確定性（cuDNN／AMP）記錄於 manifest；單種子標示探索性。
- [x] III. 原始資料唯讀；衍生遮罩帶多邊形來源與轉換設定；切分先查核既有目錄切分（pHash 近重複分組 + 檔名鄰近替代分群，無病人資訊明載殘餘風險）；增強僅訓練集；測試集不參與調參、閾值、選模；NLM／CLAHE／USM 為固定參數、不從資料擬合；報告不含可辨識病人資訊。
- [x] IV. tooth 標註「人工且專家確認」記錄使用者確認來源／日期／資料版本，Metal 狀態另列「未知」；指標定義見 [contracts/evaluation-protocol.md](contracts/evaluation-protocol.md)；未量測項（種子變異、穩健性、臨床能力）標「未量測」。
- [x] V. 遮罩光柵化、letterbox 還原（≤1 px）、增強取消、指標零分母、切分洩漏皆附 pytest；文件變更檢查路徑與命令；既有失效測試只記錄不宣稱通過。

**Post-design 重新檢查**: 通過，無需 Complexity Tracking。注意：`src/segmentation/` 依賴 `src/detection/` 的共用模組（前處理、letterbox、provenance、splits），屬 src 內部重用，不違反「src 不依賴 research／reports」。

## Project Structure

### Documentation (this feature)

```text
specs/002-train-unet-baseline/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   ├── train-config.md
│   └── evaluation-protocol.md
└── tasks.md             # /speckit-tasks 產出，本命令不建立
```

### Source Code (repository root)

```text
research/
├── models/unet.py                        # U-Net 架構（使用者指定位置）
├── data/prepare_tooth_seg_dataset.py     # 編排：盤點 → 洩漏檢查 → 前處理 → 遮罩 → 凍結清單
├── train/train_tooth_unet.py             # 編排：smoke／正式訓練、驗證選模
└── evaluation/evaluate_tooth_unet.py     # 編排：固定權重測試評估、疊圖、錯誤案例

src/segmentation/                         # 新增：可重用實作
├── __init__.py
├── config.py            # SegConfig（pydantic；延伸 PreprocessConfig 概念）
├── inventory.py         # COCO 多邊形盤點與合法性檢查、審閱狀態紀錄
├── masks.py             # tooth 聯集光柵化（Metal 忽略）
├── transform.py         # letterbox 512 影像／遮罩、預測還原原圖
├── augment.py           # 影像＋遮罩同步增強、新增截斷取消
├── dataset.py           # torch Dataset（離線前處理快取＋線上增強）
├── losses.py            # BCE + Dice
├── metrics.py           # 像素累計 Dice/IoU/precision/recall、逐圖、未定義處理
├── train.py             # 訓練迴圈、選模、停止條件、checkpoint
└── report.py            # 疊圖與錯誤案例分類

tests/segmentation/      # test_masks / test_transform / test_augment / test_metrics / test_splits / test_unet_smoke

configs/segmentation/unet_baseline_v1.yaml  # 全部登錄參數（訓練前凍結）
reports/segmentation/<run_id>/              # manifest、學習曲線、指標、疊圖、報告
```

**Structure Decision**: 新增 `src/segmentation/` 與 `tests/segmentation/`，與既有 `src/detection/` 平行；前處理（NLM→CLAHE→USM）、`provenance`、`splits`（pHash 分群）以 import 重用，不複製。U-Net 僅放 `research/models/unet.py`（使用者限制），`src` 訓練程式以參數接收 `nn.Module`，不 import research。

## Complexity Tracking

無違反項目。

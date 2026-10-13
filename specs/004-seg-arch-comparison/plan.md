# Implementation Plan: 牙齒分割模型架構比較（對照 baseline U-Net）

**Branch**: 沿用目前分支（`main`；setup 腳本回報 `004-seg-arch-comparison`） | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/004-seg-arch-comparison/spec.md`

## Summary

在 spec 003 的單顆牙 ROI 資料集（138／48／46）上，把 **U-Net++**（巢狀密集跳接）與 **Attention U-Net**（跳接處注意力閘門）
和既有 baseline U-Net 在「同一份凍結協議」下比較：3 模型 × 3 種子＝9 次隨機初始化訓練 → 以 val Dice 選模 → 協議凍結後 test 只評一次 →
以原圖為單位的配對 cluster bootstrap 給出差異信賴區間，並與種子間變異對照後才下結論。

技術做法：兩個候選以**專案內自行實作**（與 baseline 共用 `double_conv`、Kaiming 初始化、Conv-BN-ReLU，僅跳接結構不同），
不引入新套件，確保差異來自跳接而非編碼器；訓練沿用既有 `run_training`（以向後相容的擴充補上計時、峰值記憶體、協議 hash、
多種子與 `exploratory` 旗標處理）；新增「協議凍結」「test 一次性鎖」「比較統計」三個可重用模組放 `src/segmentation/`，
入口放 `research/`，證據放 `reports/segmentation/`。細節見 [research.md](research.md)、[data-model.md](data-model.md)、
[contracts/](contracts/)、[quickstart.md](quickstart.md)。

## Technical Context

**Language/Version**: Python 3.12（`pyproject.toml` 限定 `>=3.12,<3.13`）

**Primary Dependencies**: PyTorch 2.13（已安裝，`unet` 選配群組）、numpy、pydantic、PyYAML、opencv；**新增 scipy**（邊界指標用距離轉換，已安裝 1.18.1，需加入 `unet` 選配群組）；不新增模型庫（不用 segmentation_models_pytorch／timm）

**Storage**: 檔案系統；資料 `data/derived/roi_2a0c9441396f/`（唯讀）；協議 `configs/segmentation/compare_protocol_v1.yaml`；各模型設定 `configs/segmentation/roi_{unet,unetpp,attention_unet}_v1.yaml`；執行紀錄 `reports/segmentation/<run_id>/`；權重 `models/segmentation/<run_id>/`（不入版本庫）；比較報告 `reports/segmentation/compare_<protocol_version>/`

**Testing**: pytest（`tests/segmentation/`）：模型輸出形狀／參數量／可反傳、注意力閘門與巢狀節點結構、協議 hash 與凍結、test 鎖、邊界指標與 bootstrap 以合成資料核對、切分洩漏

**Target Platform**: Linux，單張 8 GB GPU（RTX 5060 Laptop 8151 MiB）；CPU 可跑小樣本驗證

**Project Type**: 研究實驗管線（library + CLI 入口）

**Performance Goals**: 單次訓練約數十分鐘（baseline 108 epoch 約 10–25 分鐘）；9 次合計預期數小時，可分批執行、可中斷續跑（以完成的 run 為單位）

**Constraints**: ROI 256×256、單通道；batch 16 為預設，OOM 時僅允許協議預先登錄的 batch 8 + 梯度累積 2（保持有效 batch 16，並於 manifest 標示）；AMP 開啟；test 僅 46 ROI／約 20 張原圖；不可調參

**Scale/Scope**: 9 次訓練、9 次 test 評估、1 份報告；新增程式約 6 個模組

**NEEDS CLARIFICATION 狀態**: 全部於 [research.md](research.md) 解決（實作來源、規模、等容量變體、種子、統計方法、邊界指標、test 鎖、baseline 已看過 test 的處理）。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] I. 入口放 `research/models/`（模型）、`research/train/train_tooth_seg.py`、`research/evaluation/evaluate_seg_models.py`、`research/evaluation/compare_seg_models.py`；可重用邏輯放 `src/segmentation/`（`protocol.py`、`compare_stats.py`、`boundary.py`、`run_lock.py`）；證據放 `reports/segmentation/`；`src` 不 import `research`（模型由入口注入，沿用現行 `run_training` 作法）。README 區分既有與新增。
- [x] II. 每次訓練／評估有獨立 run_id，manifest 記錄命令、git commit／dirty、環境、種子、設定與 hash、協議版本與 hash、資料與切分 sha256、初始化（random_kaiming）、權重 sha256、輸出位置；失敗保留原因且不覆寫；裝置非確定性寫入 manifest 並**修正既有「single seed only」字樣**；協議凍結後的執行不再標 `exploratory`，凍結前的標 `exploratory: true`。
- [x] III. 原始資料唯讀；全部模型共用同一份切分（同圖 ROI 同子集，pHash 查核已於 003 完成）；增強僅 train；test 由「一次性鎖」保護；缺病人資訊仍揭露殘餘洩漏風險；公開報告不含疊圖原圖（疊圖僅存本機 `reports/` 並標示不得公開）。
- [x] IV. 標註審閱狀態沿用 003（多邊形人工且專家確認、偵測框未知）；指標定義、單位、樣本數、零分母規則寫入協議；結論措辭三分（優於／未見可證實差異／較差）；未量測項（更多種子、外部資料、預訓練、等容量變體若未跑）標「未量測」；不宣稱臨床能力。
- [x] V. 模型結構、指標（邊界）、bootstrap、協議凍結、test 鎖皆附 pytest；`run_training` 擴充需通過既有 `tests/segmentation/`；改動入口同步更新 `research/README.md`、`src/README.md`；README 與 spec 003/002 之連結檢查。

**Post-design 重新檢查**: 通過，無需 Complexity Tracking。

## Project Structure

### Documentation (this feature)

```text
specs/004-seg-arch-comparison/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   └── protocol-and-outputs.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks 產出，本指令不建立
```

### Source Code (repository root)

```text
configs/segmentation/
├── compare_protocol_v1.yaml          # 新：比較協議（凍結對象）
├── roi_unet_v1.yaml                  # 新：baseline 設定（複製 unet_baseline_v1 的訓練配方，凍結）
├── roi_unetpp_v1.yaml                # 新：僅 model 區塊不同
└── roi_attention_unet_v1.yaml        # 新：僅 model 區塊不同

research/
├── train/train_tooth_unet.py         # 既有，保持可用
├── models/
│   ├── unet.py                       # 既有，不改行為
│   ├── unetpp.py                     # 新：U-Net++（無 deep supervision）
│   ├── attention_unet.py             # 新：Attention U-Net（加性注意力閘門）
│   └── registry.py                   # 新：name -> builder
├── train/train_tooth_seg.py          # 新：--model 選模型、--seed 單次訓練
├── evaluation/
│   ├── evaluate_seg_models.py        # 新：固定權重 test 一次性評估 + 鎖
│   └── compare_seg_models.py         # 新：統計與比較報告

src/segmentation/
├── protocol.py                       # 新：載入／雜湊／凍結協議，檢查 run 是否屬於協議
├── run_lock.py                       # 新：test 一次性鎖與違規記錄
├── boundary.py                       # 新：HD95 與 boundary F1（ROI 原解析度）
├── compare_stats.py                  # 新：以原圖為單位的配對 cluster bootstrap、種子變異
├── train.py                          # 小改：計時、峰值記憶體、protocol 欄位、exploratory 條件化、梯度累積
└── roi_config.py                     # 小改：Model 增加 name 的列舉驗證（unet|unetpp|attention_unet）

tests/segmentation/
├── test_models_arch.py  test_protocol.py  test_run_lock.py  test_boundary.py  test_compare_stats.py

reports/segmentation/compare_<protocol_version>/   # 新：比較報告、表格、錯誤案例
```

**Structure Decision**: 沿用 002／003 的分層（`research/` 入口 ← `src/segmentation/` 共用邏輯 → `reports/` 證據），
只新增必要檔案；`run_training` 以關鍵字參數向後相容擴充，不分叉訓練迴圈。

## Complexity Tracking

無憲章例外。兩點需使用者留意（非違規）：

| 項目 | 說明 |
|---|---|
| baseline 的 test 已被看過一次 | 既有 `roiunet-20261012T125011Z-8e067f` 已在 test 評估（探索性）。協議凍結後的 9 次正式執行是新訓練，test 仍只評一次；但設計者已知 baseline 約 0.93，報告須揭露此先驗。不影響「協議凍結後 test 只用一次」的紀錄，但確認性結論的強度受限。 |
| 容量差異 | U-Net++／注意力閘門參數與計算較多；主要比較不調整容量，等容量變體為預先登錄的次要組（見 research.md R3）。 |

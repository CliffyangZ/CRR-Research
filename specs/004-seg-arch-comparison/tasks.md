---

description: "Task list for 牙齒分割模型架構比較（U-Net++／Attention U-Net vs baseline U-Net）"
---

# Tasks: 牙齒分割模型架構比較（對照 baseline U-Net）

**Input**: [spec.md](spec.md)、[plan.md](plan.md)、[research.md](research.md)、[data-model.md](data-model.md)、[contracts/](contracts/)、[quickstart.md](quickstart.md)

**Prerequisites**: 資料 `data/derived/roi_2a0c9441396f/` 存在；單張 8 GB GPU（RTX 5060 Laptop）

**Tests**: 憲章原則 V 要求：模型結構、指標（邊界）、bootstrap、協議凍結、test 鎖皆附 pytest（`tests/segmentation/`）。

**Organization**: 依使用者故事分階段。因協議凍結須在任何正式訓練前、且凍結內容含候選設定檔並需先確認 8 GB 可行，兩個候選模型與其 smoke 驗證放在 Foundational。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可平行（不同檔案、無未完成依賴）
- **[Story]**: US1／US2／US3
- 指令一律自 repo 根目錄執行；`python` 為專案環境

## Phase 1: Setup

- [X] T001 在 `pyproject.toml` 的 `unet` 選配群組加入 `scipy>=1.14`（邊界指標用），並確認 `pip install -e ".[unet,dev]"` 可安裝
- [X] T002 [P] 建立 `configs/segmentation/roi_unet_v1.yaml`：複製 `configs/segmentation/unet_baseline_v1.yaml` 的訓練配方，`train.batch_size: 16`、`model.name: unet`（不修改原檔；與已跑 baseline manifest `reports/segmentation/roiunet-20261012T125011Z-8e067f/run_manifest.json` 的 config 逐欄比對並記錄差異）
- [X] T003 [P] 建立 `configs/segmentation/roi_unetpp_v1.yaml` 與 `configs/segmentation/roi_attention_unet_v1.yaml`：除 `model.name`（`unetpp`／`attention_unet`）外與 T002 逐欄相同
- [X] T004 [P] 在 `src/segmentation/roi_config.py` 的設定驗證加入 `model.name ∈ {unet, unetpp, attention_unet}`（`src/segmentation/config.py` 的 `Model.name` 改為受限列舉，預設仍 `unet`），確認既有 `tests/segmentation/` 不受影響

## Phase 2: Foundational（阻擋所有故事）

**⚠️ 在此階段完成前不得開始協議凍結或正式訓練**

- [X] T005 [P] 實作 `research/models/unetpp.py`：`UNetPlusPlus`（深度 4、基礎通道 32、重用 `research.models.unet.double_conv`、轉置卷積上採樣、Kaiming 初始化、**無 deep supervision**、輸出 X(0,4) 單 logit 圖）與 `build_unetpp(model_cfg)`
- [X] T006 [P] 實作 `research/models/attention_unet.py`：`AttentionUNet`（與 baseline 同編碼／解碼；每個跳接處加性注意力閘門——gating 與 skip 各經 1×1 卷積到 C/2、相加、ReLU、1×1、sigmoid，對 skip 逐像素加權後串接；Kaiming 初始化）與 `build_attention_unet(model_cfg)`
- [X] T007 建立 `research/models/registry.py`：`MODELS = {"unet": build_unet, "unetpp": ..., "attention_unet": ...}` 與 `build_model(model_cfg)`（依賴 T005、T006）
- [X] T008 [P] 撰寫 `tests/segmentation/test_models_arch.py`：三模型輸入 (2,1,256,256) → 輸出 (2,1,256,256)、可反傳且梯度非 NaN；U-Net++ 含預期的巢狀節點數、Attention U-Net 的閘門權重在 [0,1]；參數量 >0 並印出與 baseline 7,762,465 對照；同一 seed 兩次建構權重相同
- [X] T009 擴充 `src/segmentation/train.py::run_training`（僅新增關鍵字參數、預設行為不變）：記錄 `train_seconds`、`peak_gpu_mem_mb`（訓練前 `torch.cuda.reset_peak_memory_stats`）、`batch{size,accum,override_reason}`、`candidate`、`protocol_version/hash`；支援 `grad_accum`（預設 1）；僅在無協議時寫 `exploratory: true`；將 `nondeterminism_note` 改為不含「single seed only」且說明 AMP／cuDNN 非確定性；失敗路徑同樣寫入已量得的欄位
- [X] T010 撰寫 `tests/segmentation/test_run_training_ext.py`：以極小模型與合成 ROI 資料驗證新欄位存在、`grad_accum=2` 時有效 batch 不變、無協議時 `exploratory=True`、有協議時為 False、失敗時 `status=failed` 且 run 目錄不被覆寫（依賴 T009）
- [X] T011 建立 `research/train/train_tooth_seg.py`：參數依 [contracts/cli.md](contracts/cli.md)（`--model --data-version --seed --config --protocol --device --output --batch-size --grad-accum --override-reason --smoke --derived`）；以 `registry.build_model` 注入模型並呼叫 `run_training`（`run_prefix` 為模型名）；`--batch-size/--grad-accum` 僅接受協議 `batch_fallback` 組合；既有 `research/train/train_tooth_unet.py` 維持可用（依賴 T007、T009）
- [X] T012 執行三個候選的 smoke：`python -m research.train.train_tooth_seg --model {unet,unetpp,attention_unet} --data-version roi_2a0c9441396f --seed 1 --smoke --device cuda:0`；確認無 OOM、U-Net++ `peak_gpu_mem_mb` <8000；把三個 smoke run_id、參數量、峰值記憶體寫入 `specs/004-seg-arch-comparison/smoke_notes.md`；若 OOM，改用 batch 8 + `--grad-accum 2` 重測並記錄，仍失敗則停止並請使用者決定（依賴 T011）
- [ ] T013 向使用者呈現候選登錄表（名稱、設計取向、實作檔、來源與授權、實測參數量、初始化、記憶體）並**取得確認**後才進入 US1 凍結（spec FR-002）；確認結果記在 `specs/004-seg-arch-comparison/smoke_notes.md`

**Checkpoint**: 三個模型可訓練、資源可行、使用者確認候選。

## Phase 3: User Story 1 — 固定比較規則並重現 baseline (P1) 🎯 MVP

**Goal**: 凍結協議，並以 3 個種子重跑 baseline，得到種子變異與既有單次結果並列。

**Independent Test**: `src.segmentation.protocol check` 通過、凍結後修改被拒；baseline 3 個 succeeded run 具完整重現資訊，且沒有任何 test 存取。

- [X] T014 [P] [US1] 實作 `src/segmentation/protocol.py`：載入協議 YAML（pydantic）、計算 `protocol_hash`、驗證三個候選設定除 `model` 外相同（hash）、驗證 `data_version` 與 `split_sha256` 與 `data/derived/<data_version>/splits.json` 一致、`freeze` 寫 `protocol_freeze.json`（時間、git commit／dirty、hash）、`check_run(manifest)` 判斷 run 是否屬於凍結協議；CLI 子命令 `freeze`、`check`（[contracts/protocol-and-outputs.md](contracts/protocol-and-outputs.md)）；不 import `research`
- [X] T015 [P] [US1] 建立 `configs/segmentation/compare_protocol_v1.yaml`（內容依 contracts：資料版本 roi_2a0c9441396f、種子 [20261012, 20261013, 20261014]、三候選、閾值 0.5、`batch_fallback {8, 2}`、指標與零分母規則、統計方法與證據門檻）與 `configs/segmentation/PROTOCOL_CHANGELOG.md`（空白版本紀錄）
- [X] T016 [US1] 撰寫 `tests/segmentation/test_protocol.py`：hash 對鍵順序穩定；候選設定被改動一欄（非 model）→ check 失敗；split_sha256 不符 → 失敗；凍結後 `check_run` 對協議外的 config hash 拒絕；`tests/segmentation/` 既有切分洩漏測試（同原圖 ROI 同子集）涵蓋協議所用 splits（依賴 T014）
- [X] T017 [US1] 在 `research/train/train_tooth_seg.py` 加入協議閘門：帶 `--protocol` 時要求已凍結、config hash 與協議一致、seed 在清單內、否則結束碼 2 且不建立 run；run manifest 寫入 `protocol_version/hash`（依賴 T011、T014）
- [ ] T018 [US1] 執行 `python -m src.segmentation.protocol check` 後 `freeze`，把 `protocol_freeze.json` 存至 `reports/segmentation/compare_v1/`；確認 `git status` 乾淨度並記錄（若有未提交變更須在 freeze 紀錄中保留，不隱藏）（依賴 T013、T015、T016）
- [ ] T019 [US1] 訓練 baseline 3 個種子：`python -m research.train.train_tooth_seg --model unet --seed <20261012|20261013|20261014> --protocol configs/segmentation/compare_protocol_v1.yaml --data-version roi_2a0c9441396f`；任何失敗保留 run 並以新 run_id 重跑、不挑種子（依賴 T017、T018）
- [ ] T020 [US1]（`build_runset` 與其測試已完成於 T014／T016；待 T019 後執行建立 runset.json）建立 `reports/segmentation/compare_v1/runset.json`（候選×種子→採用 run_id 與被取代的失敗 run_id），並新增 `scripts` 無關的小工具函式於 `src/segmentation/protocol.py::build_runset(reports_dir, protocol)`；寫 `tests/segmentation/test_protocol.py` 對應案例（缺 run、重複 succeeded 皆報錯）（依賴 T019）
- [ ] T021 [US1] 撰寫 `reports/segmentation/compare_v1/baseline_repro.md`：三個種子的 val Dice、停止輪、訓練時間，與既有單次結果（val／test，取自 `reports/segmentation/roiunet-20261012T125011Z-8e067f/`）並列，**僅引用 val**；說明既有 test 數字為協議前探索性結果；種子變異以樣本 SD（n=3）呈現並註明估計粗略（依賴 T019）

**Checkpoint**: US1 完成＝協議凍結且 baseline 已多種子重現（MVP）。

## Phase 4: User Story 2 — 訓練並評估候選架構 (P2)

**Goal**: 兩個候選各 3 個種子在同一協議下訓練並留完整紀錄。

**Independent Test**: 6 個候選 run 皆有 succeeded manifest（或有失敗原因的紀錄），含參數量、峰值記憶體、訓練時間、權重 sha256、協議 hash。

- [ ] T022 [US2] 訓練 U-Net++ 3 個種子（指令同 T019，`--model unetpp`）；OOM 時僅用協議內的 batch 8 + `--grad-accum 2 --override-reason "OOM"`，並於 manifest 標示；失敗者保留（依賴 T018）
- [ ] T023 [US2] 訓練 Attention U-Net 3 個種子（`--model attention_unet`），規則同 T022（依賴 T018；可與 T022 在同 GPU 上依序執行，不平行）
- [ ] T024 [US2] 更新 `reports/segmentation/compare_v1/runset.json`（重跑 `build_runset`），確認 9 個計畫 run 全部 succeeded；缺者列「未完成」並請使用者決定（依賴 T019、T022、T023）
- [ ] T025 [US2] 撰寫 `reports/segmentation/compare_v1/training_summary.md`：每 run 的 run_id、epochs_run、停止原因、最佳 epoch 與 val Dice、參數量、訓練時間、峰值記憶體、batch 覆寫；僅使用 train／val 資訊（依賴 T024）

**Checkpoint**: 9 個 run 完整，可進入一次性 test 評估。

## Phase 5: User Story 3 — 公平比較與可查證結論 (P3)

**Goal**: test 只評一次，產出含 CI、種子變異、成本與錯誤案例的誠實報告。

**Independent Test**: 合成資料單元測試通過；真實資料上鎖後重跑被拒；報告 verdict 僅為三個列舉值之一。

- [X] T026 [P] [US3] 實作 `src/segmentation/boundary.py`：`hd95(pred, gt, valid)`（scipy 距離轉換，像素單位）與 `boundary_f1(pred, gt, valid, tol=2)`；任一為空回傳 `None` 與原因；不計 valid 外像素
- [X] T027 [P] [US3] 撰寫 `tests/segmentation/test_boundary.py`：完全相同遮罩 HD95=0／F1=1；平移 3 px 的方塊 HD95≈3；空 pred／空 gt → None＋原因；valid 外錯誤不影響結果（依賴 T026）
- [X] T028 [P] [US3] 實作 `src/segmentation/run_lock.py`：`check_ready(protocol, run_dirs)`（協議 frozen、9 個 run 皆 succeeded 且 hash 屬於協議）、`acquire(...)` 寫 `test_lock.json`（協議 hash、run_id＋權重 sha256、時間）、`record_violation(reason)`；鎖已存在且無違規記錄 → 拋 `TestAlreadyUsed`
- [X] T029 [P] [US3] 撰寫 `tests/segmentation/test_run_lock.py`：第二次 acquire 被拒；`record_violation` 後 report 標記確認性無效；未凍結／run 缺漏時 `check_ready` 失敗（依賴 T028）
- [X] T030 [P] [US3] 實作 `src/segmentation/compare_stats.py`：`seed_summary`（平均、樣本 SD）、`pixel_dice(counts)`、`paired_cluster_bootstrap(per_roi_by_model, baseline, candidate, B, seed)`（以 `source_image_id` 為單位重抽，同一重抽下各種子算 Dice 差再對種子平均，回傳 mean_diff、ci95、各 ROI 勝負平手）、`verdict(ci, mean_diff, sd_base, sd_cand)` → 三列舉值
- [X] T031 [P] [US3] 撰寫 `tests/segmentation/test_compare_stats.py`：合成資料中候選明顯較好→`evidence_better`；相同分布→`no_demonstrable_difference`；差異小於種子 SD 即使 CI 排除 0 亦為 `no_demonstrable_difference`；同圖 ROI 以整組重抽（驗證抽樣單位）；固定 seed 可重現（依賴 T030）
- [X] T032 [US3] 實作 `research/evaluation/evaluate_seg_models.py`（[contracts/cli.md](contracts/cli.md)）：檢查就緒＋鎖，對每個 run 載入 `best.pt` 在指定 split 推論（沿用 `predict_rois`、`GroundTruth`），輸出 `eval/per_roi.csv`（含計數、四指標、HD95、boundary F1、`source_image_id`、`truncated`、`has_neighbor`）、`eval/metrics.json`（含 `inference_ms_per_roi`：batch 1、AMP、warm-up 5 次）、`eval/eval_manifest.json`（權重 sha256、run_id）；全部完成後才寫鎖；`--split val` 為演練不寫鎖；`--record-violation` 行為依契約（依賴 T026、T028）
- [X] T033 [US3] 實作 `research/evaluation/compare_seg_models.py`：只讀 `eval/` 與鎖；輸出 `comparison.json`、`tables/{seed_table,model_summary,paired,cost}.csv`、`report.md`；report 必含 [contracts/protocol-and-outputs.md](contracts/protocol-and-outputs.md) 所列各節，並含 baseline 先前已見 test 之揭露、標註審閱狀態（多邊形人工專家確認、偵測框未知）、缺病人資訊的洩漏殘餘風險、「未量測」清單（更多種子、外部資料、預訓練、等容量變體若未跑、FLOPs）、不宣稱臨床能力（依賴 T030、T032）
- [X] T034 [US3] 在 `compare_seg_models.py` 加入錯誤案例輸出：沿用 `src/segmentation/report.classify_errors`，三類（含鄰牙、貼近邊界或截斷、牙根漏分＝漏分占 GT 比例最高的前 10 ROI 之近似定義，報告註明近似）各模型同一批 ROI 並列、零例標示；**本次不產生疊圖**（評估階段不保存預測，避免含原圖的疊圖外流）；報告僅含 roi_id 與數值（依賴 T033）
- [ ] T035 [US3] 以演練驗證流程：`evaluate_seg_models --split val`（不寫鎖）後跑 `compare_seg_models`，確認報告管線無誤、verdict 邏輯可運作（val 結果標 `dry_run`，不可作結論）（依賴 T024、T033）
- [ ] T036 [US3] **使用者確認後**執行正式一次性 test 評估：`python -m research.evaluation.evaluate_seg_models --protocol configs/segmentation/compare_protocol_v1.yaml --runs <9 個 run 目錄>`；再執行一次確認以結束碼 3 被拒（把兩次輸出存 `reports/segmentation/compare_v1/lock_check.txt`）（依賴 T024、T032、T035）
- [ ] T037 [US3] 執行 `compare_seg_models` 產生正式 `reports/segmentation/compare_v1/report.md`；檢查 SC-001～SC-006：9 個 run 具 FR-007 資訊、test 使用次數＝1、各指標有值或未定義原因、每候選有 CI＋verdict、錯誤案例至少三類；把檢查結果寫在 report 末節（依賴 T034、T036）
- [ ] T038 [US3] 在報告「後續建議」中提出至少一項有案例支持的建議（引用 error_cases 的 roi_id）；若 verdict 皆為 `no_demonstrable_difference`，建議含增加種子數與等容量變體的價值估計（依賴 T037）

## Final Phase: Polish

- [ ] T039 [P] 更新 `research/README.md`、`src/README.md`、`reports/README.md`：區分既有／新增入口與模組、列出命令與輸出位置；檢查所有連結與路徑
- [ ] T040 [P] 修正 `src/segmentation/` 與 `research/` 中受影響 import；跑 `pytest tests/segmentation -q`，既有失效測試逐一記錄於 `specs/004-seg-arch-comparison/smoke_notes.md`，不宣稱未跑的檢查通過
- [ ] T041 依 [quickstart.md](quickstart.md) 逐步核對命令參數與預期輸出是否與實作一致，必要時更新 quickstart
- [ ] T042 確認 `models/segmentation/` 與疊圖未被 git 追蹤（`.gitignore`），公開產物不含可識別病人資訊
- [ ] T043 （可選，需使用者同意）等容量變體：登錄 U-Net++／Attention U-Net 基礎通道使參數量在 baseline ±5%，建新協議版本 `compare_protocol_v2.yaml` 並於 `PROTOCOL_CHANGELOG.md` 記錄理由與新舊 hash；3 種子訓練後**另列次要表**，test 鎖需新協議的新評估並揭露為第二次使用

## Dependencies & Execution Order

- Setup(T001–T004) → Foundational(T005–T013) → US1(T014–T021) → US2(T022–T025) → US3(T026–T038) → Polish
- US3 的純函式工作（T026–T031）只依賴 Setup，可在 US1／US2 訓練（耗時）期間平行開發；T032 以後需 T024。
- T013（使用者確認候選）與 T036（使用者確認後才用 test）為人工閘門。
- 協議凍結（T018）之後不得修改協議或候選設定；需要改動時走 T043 的新版本流程。

## Parallel Opportunities

- T002、T003、T004；T005、T006、T008；T014、T015；T026–T031（五對檔案互不相干）；T039、T040、T042。
- 訓練任務共用單張 GPU，不平行。

## Implementation Strategy

- **MVP＝US1**：協議凍結＋baseline 3 種子重現，已可回答「baseline 的種子變異有多大」。
- 之後 US2 取得候選結果，最後 US3 一次性 test 與報告；在等待訓練時開發 US3 的統計與評估程式。
- 任何階段發現資源或結果異常，停下來向使用者回報，不自行改動協議。

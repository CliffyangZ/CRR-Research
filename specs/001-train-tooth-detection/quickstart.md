# Quickstart: 驗證流程

前置：Python 3.12 環境，`pip install -e ".[dev,yolo]"`，CUDA 版 PyTorch（RTX 5060 需支援 sm_120）。

1. **單元驗證**（無需 GPU／資料）
   ```bash
   python -m pytest tests/detection
   ```
   預期：letterbox 往返誤差 ≤1 px；增強三情境（界內／越界取消／原已截斷）符合 FR-005；切分無跨子集群組；指標零分母為 undefined。
2. **小樣本資料準備**
   ```bash
   python -m research.data.prepare_tooth_dataset ... --limit 12
   ```
   預期：產生 DatasetVersion／SplitManifest／疊圖；原始影像 sha256 不變。
3. **小樣本訓練**：`python -m research.train.train_tooth_yolo --model v8 --smoke ...`；預期 `run_manifest.json` 含契約欄位（見 [cli.md](contracts/cli.md)）。對 v11、v26 重複。
4. **完整流程**：全資料訓練三模型 → 評估 → 比較報告。預期每模型有權重、`metrics.json`（五項指標）、錯誤計數，種子變異標示「未量測」。
5. **驗收檢查**：對照 spec SC-001～SC-005，確認可由報告追溯設定、權重、切分與證據。

欄位定義見 [data-model.md](data-model.md)，評估規則見 [evaluation-protocol.md](contracts/evaluation-protocol.md)。

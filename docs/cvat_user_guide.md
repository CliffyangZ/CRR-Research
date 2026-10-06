# 日常啟動與停止
```
cd ~/Program/Medical-CV/cvat
```

```
CVAT_HOST=localhost docker compose up -d
```

```
docker compose down
```

```
docker compose ps
```

```
docker compose logs -f cvat_server
```

# 匯入 SAM 批次預標注（pseudo-label）

`src/core/segmentation/sam_infer.py` 會對 `data/processed/` 全部影像跑 SAM ViT-B，輸出
`reports/models/sam_pseudo_labels/coco_annotations.json`（COCO 1.0 格式）。用這個檔案可以
讓新建立的 CVAT task 一開始就帶有每張影像的牙齒 mask，標註者只需校正而不用逐張手動框選：

1. 啟動 CVAT（見上）後開啟 CVAT 網頁介面。
2. **Tasks → Create a new task**，建立名稱完全相同的 `tooth` label，Data 選擇 `data/processed/` 目錄下的全部影像（此為 CLAHE 增強後、SAM 實際使用的輸入影像，見 `preprocess.md` 第 4 節）。影像檔名與 COCO `images[].file_name` 必須一致。
3. Task 建立完成後，進入該 task → **Actions → Upload annotations**，Format 選 **COCO 1.0**，上傳
   `reports/models/sam_pseudo_labels/coco_annotations.json`。
   腳本也會輸出 annotation-only 的 `cvat_coco_annotations.zip`（內含 `annotations/instances_default.json`）。對已建立且已上傳影像的 task，優先使用前述 JSON。
   若推論時加上 `--package-cvat-dataset`，另會產生包含 `images/default/*` 與 `annotations/instances_default.json` 的完整 `cvat_coco_dataset.zip`，可直接以 COCO dataset 匯入；但目前 282 張 processed images 約 401 MB，會額外占用相近空間。
4. 匯入後每張影像會出現預先產生的 mask（多邊形），標註者逐張檢查並校正邊界（CEJ、根尖、齒尖，見
   `preprocess.md` 第 7 節）。
5. 若某張影像的預標注品質太差（可對照 `reports/models/sam_pseudo_labels/metadata.json` 裡
   `flagged_for_review: true`，即 `sam_score < 0.7` 的影像），刪掉該預標注後，改用下方的互動式
   Tooth SAM function 重新框選/加點取得新 mask——批次匯入與互動式標註互補，不是互相取代。
6. 校正完成後，**Actions → Export annotations**，Format 同樣選 **COCO 1.0**，匯出檔可作為
   `src/models/finetune_sam.py` 微調 mask decoder 的訓練資料（見 `preprocess.md` 第 9 節）。

## 互動式 Tooth SAM（個別牙齒重新框選）

`deploy/cvat/tooth-sam/nuclio/` 部署了一個互動式 SAM ViT-B 的 Nuclio serverless function
（`sam_vit_b_01ec64.pth`），CVAT 前端可對單一物件呼叫它取得即時分割結果（框選/加點後即時出 mask），
適合用來處理批次匯入結果不理想的個別牙齒，或全新加入的影像。

## Tooth YOLO + SAM：先定位再分割

`deploy/cvat/tooth-yolo-sam/nuclio/` 是 GPU Nuclio detector。它使用
`models/yolo/best.pt` 在目前影像找出牙齒框，再把每個框交給 SAM ViT-B，
回傳可在 CVAT 編輯的個別牙齒 mask。

在標註頁面按 `Ctrl+Shift+A` 開啟 **AI tools**，切到 **Detectors**，
選 **Tooth YOLO + SAM Segmentation**，將模型的 `tooth` 對應到 task 的 `tooth` label，
按 **Annotate**。可調整偵測 confidence threshold，也可指定 ROI。
執行結果是當前影格的自動標註初稿，請人工檢查並校正。

重新部署前，將 YOLO 與 SAM checkpoint 放入該 function 的建置目錄，檔名分別為
`best.pt`、`sam_vit_b_01ec64.pth`，再執行：

```bash
nuctl deploy pth-tooth-yolo-sam-segment --platform local --project-name cvat \
  --path deploy/cvat/tooth-yolo-sam/nuclio \
  --file deploy/cvat/tooth-yolo-sam/nuclio/function-gpu.yaml \
  --platform-config '{"attributes":{"network":"cvat_cvat"}}' \
  --readiness-timeout 180
```

## 畫筆人工修正（Mask / Polygon）

本機 CVAT 前端新增 **Brush edit** 入口，可快速補上或擦除分割區域：

1. 開啟 job，在右側物件的 `⋮` 選單選 **Brush edit**。
2. **Mask** 直接編輯；**Polygon** 選 **Brush edit (save as mask)**，完成時轉為 Mask，因此能保留孔洞及不相連區域。此入口適用於 2D shape，Polygon track 不提供轉換。
3. 選畫筆補上（`Shift+1`）或橡皮擦移除（`Shift+2`）；可調整筆刷大小與 Circle / Square 筆頭。按住 `Alt` 加滑鼠右鍵左右拖曳可改大小。
4. 按工具箱的 `✓` 完成，再儲存 job。按 `Esc` 取消本次編輯；取消 Polygon 編輯時不轉換、不替換原物件。
5. 完成 Polygon 編輯後，一次 Undo（`Ctrl+Z`）可還原原 Polygon；Redo 可恢復修正後的 Mask。轉換保留標籤、屬性、群組、遮擋狀態及圖層，替換後物件 ID 會改變。

畫筆操作期間可使用 CVAT 既有的 Undo / Redo。若把所有像素擦除，CVAT 會取消此次編輯並保留原物件。鎖定物件與 ground-truth 物件不能使用此入口。

原始碼位於 `../cvat`，回歸測試：

```bash
cd ~/Program/Medical-CV/cvat
node cvat-ui/tests/polygon-brush.test.cjs
BABEL_CACHE_PATH=/tmp/cvat-brush-babel-cache.json node cvat-ui/tests/polygon-brush-conversion.test.cjs
```

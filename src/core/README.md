# Core：共用核心邏輯

`core/` 提供可由研究流程與命令列工具重用的實作。
目前主要功能位於 `segmentation/`：

| 模組 | 功能 |
| --- | --- |
| `segmentation/cvat_coco.py` | COCO polygon 格式檢查、CVAT ZIP 輸出、中心牙齒審閱條件 |
| `segmentation/sam_infer.py` | SAM 提示、mask 後處理、polygon 轉換與批次推論 |
| `segmentation/yolo_sam_coco.py` | YOLO 偵測框引導 SAM，產生 COCO 偽標註及審閱資訊 |

`testing.py` 目前是空白占位模組；自動化測試放在根目錄 `tests/`。
新共用功能依用途新增模組，並保持計算邏輯與實驗編排分離。
執行範例、依賴與維護方式見 [`src/README.md`](../README.md)。

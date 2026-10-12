"""Compare evaluated runs (one per model) into a markdown report.

  python -m research.evaluation.compare_runs --runs reports/detection/runs/<id> ... --output reports/detection/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd=3):
    return "undefined" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


NEXT = {
    "missed": "漏檢最多：試驗降低輸入縮小造成的小牙損失（更大 imgsz）、檢查邊緣截斷牙標註，或增加訓練影像。",
    "false_positive": "背景誤檢最多：檢查標註是否遺漏牙齒（標註審閱）、加入難負樣本，或提高工作閾值。",
    "duplicate": "重複框最多：比較 NMS IoU 閾值（v8/v11）與 NMS-free（v26）行為，調整後處理（僅用驗證集）。",
    "localization": "定位偏差最多：框定義與標註一致性待確認（現有標註來源未驗證），並可測試更大解析度或更長訓練。",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", default="reports/detection/comparison")
    a = ap.parse_args(argv)
    rows = []
    for r in a.runs:
        r = Path(r)
        rm = json.loads((r / "run_manifest.json").read_text())
        t = json.loads((r / "eval" / "test_metrics.json").read_text())
        v = json.loads((r / "eval" / "val_metrics.json").read_text())
        rows.append((rm, t, v))
    L = ["# 牙齒偵測三模型比較（探索性單次結果）", ""]
    L += ["> 每模型僅訓練一次（seed 固定）：種子變異 **未量測**；測試集僅 20 張影像，差異不應解讀為模型排序的確認性證據。",
          f"> 標註審閱狀態：**{rows[0][1]['annotation_review_status']}**（標註來源未驗證、可能為模型產生），指標是對這些標註的一致性，不是對人工 ground truth 的準確率。",
          "> 不宣稱臨床診斷能力。", ""]
    L += ["## 實驗追溯", "", "| 模型 | run_id | data_version | 權重 sha256 | 裝置 | epochs | git dirty |", "|---|---|---|---|---|---|---|"]
    for rm, t, v in rows:
        L.append(f"| {rm['candidate']['family']} | {rm['run_id']} | {rm['data_version']} | {rm['selected_weights']['sha256'][:12]} | "
                 f"{rm['hyperparams']['device']} | {rm['hyperparams']['epochs']} | {rm['git']['dirty']} |")
    L += ["", "## 測試集指標（IoU 0.5 的 P/R/F1，τ 由驗證集選定）", "",
          f"母體：test，影像 {rows[0][1]['population']['n_images']}，牙齒標註 {rows[0][1]['population']['n_boxes']}。", "",
          "| 模型 | τ | AP@[.5:.95] | AP@.5 | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for rm, t, v in rows:
        L.append(f"| {rm['candidate']['family']} | {fmt(t['tau'])} | {fmt(t['AP'])} | {fmt(t['AP50'])} | {fmt(t['precision'])} | {fmt(t['recall'])} | {fmt(t['f1'])} |")
    L += ["", "## 驗證集指標（參考，用於選 τ／選模）", "", "| 模型 | val AP@[.5:.95] | val AP@.5 | val F1 |", "|---|---|---|---|"]
    for rm, t, v in rows:
        L.append(f"| {rm['candidate']['family']} | {fmt(v['AP'])} | {fmt(v['AP50'])} | {fmt(v['f1'])} |")
    L += ["", "## 錯誤分析（test，IoU 0.5，工作閾值 τ）", "",
          "| 模型 | 漏檢 | 背景誤檢 | 重複框 | 定位偏差 |", "|---|---|---|---|---|"]
    tot = {k: 0 for k in NEXT}
    for rm, t, v in rows:
        e = t["errors"]
        L.append(f"| {rm['candidate']['family']} | {e['missed']} | {e['false_positive']} | {e['duplicate']} | {e['localization']} |")
        for k in tot:
            tot[k] += e[k]
    L += ["", "零例的類別即為零（未發生）。代表案例疊圖（綠=標註、紅=預測）：" +
          "、".join(f"`{Path(r[0]['train_dir']).parent.name}/eval/overlays/`" for r in rows), ""]
    L += ["## 分組指標", "", "未量測：資料沒有可支持的分組資訊（病人、設備、牙位），且 test 僅 20 張，分組後樣本不足。", ""]
    top = max(tot, key=tot.get)
    L += ["## 限制與後續建議", "",
          "- 僅 101 張已標註影像；訓練 61 張；無病人 ID（以 pHash 近重複替代，病人層級洩漏**未排除**）。",
          "- 單次訓練、單一種子；v26 為 NMS-free，v8/v11 使用 NMS（後處理差異未分離）。",
          "- 標註來源未驗證（見 data report），可能影響所有指標的意義。",
          f"- 後續建議（由錯誤分析支持，三模型合計 {tot}）：{NEXT[top] if tot[top] else '無錯誤，需更嚴格的測試資料。'}",
          "- 先完成標註人工審閱，再做多種子重複與前處理對照（NLM/CLAHE/USM 消融）。", ""]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    print(out / "report.md")


if __name__ == "__main__":
    main()

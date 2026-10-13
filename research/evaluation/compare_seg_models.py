"""Statistics and report for the architecture comparison; reads only existing evaluation outputs and the test lock.

  python -m research.evaluation.compare_seg_models --protocol configs/segmentation/compare_protocol_v1.yaml
  (--dry-run reads the validation dry-run evaluation instead and labels everything as not a result)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.segmentation import compare_stats as S
from src.segmentation import run_lock
from src.segmentation.protocol import ProtocolError, build_runset, load_protocol, require_frozen

REFERENCE_RUN = "reports/segmentation/roiunet-20261012T125011Z-8e067f"
VERDICT_ZH = {S.EVIDENCE_BETTER: "有證據優於 baseline", S.NO_DIFFERENCE: "未見可證實差異", S.EVIDENCE_WORSE: "有證據較差於 baseline"}
ROOT_MISS_TOPK = 10


def fmt(v, nd=4):
    return "未定義" if v is None else f"{v:.{nd}f}"


def load_run(m: dict, sub: str) -> dict:
    ev = Path(m["output_dir"]) / sub
    metrics = json.loads((ev / "metrics.json").read_text())
    with open(ev / "per_roi.csv") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["roi_id"])
    return {"m": m, "metrics": metrics, "rows": rows}


def counts_of(rows: list[dict]) -> np.ndarray:
    return np.array([[int(r["tp"]), int(r["fp"]), int(r["fn"])] for r in rows], dtype=np.int64)


def _f(x):
    return None if x in ("", None) else float(x)


def build_comparison(p, runs: dict, dry_run: bool) -> dict:
    """runs: {candidate: {seed: loaded run}}"""
    base = p.baseline
    ids = [r["roi_id"] for r in next(iter(runs[base].values()))["rows"]]
    images = [r["source_image_id"] for r in next(iter(runs[base].values()))["rows"]]
    for c in runs.values():
        for r in c.values():
            if [x["roi_id"] for x in r["rows"]] != ids:
                raise ProtocolError(f"ROI lists differ in {r['m']['run_id']}")
    st = p.statistics
    seed_table, summary, counts = [], {}, {}
    for cand, by_seed in runs.items():
        counts[cand] = {s: counts_of(r["rows"]) for s, r in by_seed.items()}
        cols = {k: [] for k in ("dice", "iou", "precision", "recall", "per_roi_mean_dice", "hd95_mean", "boundary_f1_mean")}
        for s, r in sorted(by_seed.items()):
            a, mean = r["metrics"]["aggregate"], r["metrics"]["per_image_mean"]
            row = {"candidate": cand, "seed": s, "run_id": r["m"]["run_id"], "dice": a["dice"], "iou": a["iou"],
                   "precision": a["precision"], "recall": a["recall"], "per_roi_mean_dice": mean["dice"],
                   "hd95_mean": r["metrics"]["hd95_mean"], "boundary_f1_mean": r["metrics"]["boundary_f1_mean"]}
            seed_table.append(row)
            for k in cols:
                cols[k].append(row[k])
        summary[cand] = {k: S.seed_summary([v for v in vs if v is not None]) if any(v is not None for v in vs) else None
                         for k, vs in cols.items()}
    paired = {}
    for cand in runs:
        if cand == base:
            continue
        r = S.paired_cluster_bootstrap(counts[base], counts[cand], images, st.B, st.seed, st.ci)
        sd_b, sd_c = summary[base]["dice"]["sd"], summary[cand]["dice"]["sd"]
        r.update(seed_sd={"baseline": sd_b, "candidate": sd_c}, verdict=S.verdict(r["ci"], r["mean_diff"], sd_b, sd_c),
                 roi_wins=S.roi_wins(counts[base], counts[cand]))
        paired[cand] = r
    cost = {}
    for cand, by_seed in runs.items():
        ms = [r["m"] for r in by_seed.values()]
        cost[cand] = {"n_parameters": ms[0]["n_parameters"],
                      "train_seconds_mean": float(np.mean([m["train_seconds"] for m in ms])),
                      "peak_gpu_mem_mb_max": max(m["peak_gpu_mem_mb"] for m in ms if m["peak_gpu_mem_mb"] is not None),
                      "epochs_run": [m["stopping"]["epochs_run"] for m in ms],
                      "batch": sorted({json.dumps(m["batch"], sort_keys=True) for m in ms}),
                      "inference_ms_per_roi_mean": float(np.mean([r["metrics"]["inference"]["ms_per_roi_mean"]
                                                                   for r in by_seed.values()]))}
    return {"dry_run": dry_run, "protocol_version": p.protocol_version, "n_rois": len(ids),
            "n_source_images": len(set(images)), "seed_table": seed_table, "summary": summary, "paired": paired,
            "cost": cost, "error_cases": error_cases(runs, ids)}


def error_cases(runs: dict, ids: list[str]) -> dict:
    """Three categories on the same ROIs for every model (pooled over seeds)."""
    any_run = next(iter(next(iter(runs.values())).values()))["rows"]
    flag = {k: {r["roi_id"] for r in any_run if r[k] == "True"} for k in ("has_neighbor", "truncated")}
    fn_all = {i: [] for i in ids}
    for by_seed in runs.values():
        for r in by_seed.values():
            for row in r["rows"]:
                if _f(row["fn_frac_of_gt"]) is not None:
                    fn_all[row["roi_id"]].append(_f(row["fn_frac_of_gt"]))
    ranked = sorted((i for i in ids if fn_all[i]), key=lambda i: (-np.mean(fn_all[i]), i))[:ROOT_MISS_TOPK]
    cats = {"含鄰牙 (has_neighbor)": sorted(flag["has_neighbor"]), "貼近邊界或被截斷 (truncated)": sorted(flag["truncated"]),
            "牙根漏分（近似：漏分占 GT 比例最高的 ROI）": sorted(ranked)}
    out = {}
    for name, sel in cats.items():
        per_model = {}
        for cand, by_seed in runs.items():
            tot, fn_fracs = [], []
            for r in by_seed.values():
                rs = [row for row in r["rows"] if row["roi_id"] in set(sel)]
                if rs:
                    tot.append(S.pixel_dice(counts_of(rs)))
                    fn_fracs += [_f(x["fn_frac_of_gt"]) for x in rs if _f(x["fn_frac_of_gt"]) is not None]
            per_model[cand] = {"dice_mean_over_seeds": float(np.mean([t for t in tot if t is not None])) if tot else None,
                               "fn_frac_mean": float(np.mean(fn_fracs)) if fn_fracs else None}
        out[name] = {"n_rois": len(sel), "roi_ids": sel, "per_model": per_model}
    return out


def render(p, comp: dict, lock: dict | None, reference: dict | None) -> str:
    L = []
    dry = comp["dry_run"]
    L += ["# 牙齒分割架構比較報告" + ("（演練：驗證集，非結果）" if dry else ""), ""]
    if dry:
        L += ["> **演練輸出**：使用驗證集，僅用於檢查流程；不得作為任何結論。", ""]
    L += [f"- 協議：{p.protocol_version}；資料版本：{p.data_version}；切分 sha256：{p.split_sha256[:12]}",
          f"- 評估子集：{'val（演練）' if dry else 'test'}；樣本量：{comp['n_rois']} 個 ROI／{comp['n_source_images']} 張原圖；閾值 {p.threshold}",
          "- 標註審閱狀態：多邊形標註為人工且經專家確認（使用者 2026-10-12）；偵測框審閱狀態未知；被隔離的 41 個框不在資料內。",
          "- 指標單位 0–1（HD95 為像素）；Dice 為全體 ROI 像素累計（ROI 原圖解析度、補邊不計）；零分母標示「未定義」。",
          "- 不宣稱臨床診斷能力；結果僅對本資料集的 ROI 二元分割。", ""]
    if lock is not None:
        L += [f"- test 鎖：協議 hash {lock['protocol_hash']}，{len(lock['evaluated_runs'])} 個權重於 {lock['timestamp']} 評估一次；"
              + ("**已記錄違規，確認性結論失效**：" + "；".join(v["reason"] for v in lock["violations"]) if lock["violations"] else "無違規記錄。"), ""]
    L += ["- **先驗揭露**：協議前已有 baseline 的單次 test 結果（探索性），設計者已知該數字；此事實限制確認性結論的強度。", ""]
    if reference:
        L += [f"  參考（協議前、單種子、探索性）：baseline run `{reference['run_id']}` test Dice {fmt(reference['aggregate']['dice'])}。", ""]
    L += ["## 結論（相對 baseline U-Net）", "", "| 候選 | 平均 Dice 差 | 95% CI（以原圖重抽） | 種子 SD（baseline／候選） | ROI 勝／負／平 | 結論 |", "|---|---|---|---|---|---|"]
    for c, r in comp["paired"].items():
        w = r["roi_wins"]
        L.append(f"| {c} | {r['mean_diff']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {fmt(r['seed_sd']['baseline'])}／{fmt(r['seed_sd']['candidate'])} | "
                 f"{w['wins']}／{w['losses']}／{w['ties']} | **{VERDICT_ZH[r['verdict']]}** |")
    L += ["", f"證據門檻：CI 排除 0 且 |平均差|> 兩模型種子 SD 的較大者；否則寫「未見可證實差異」。B={p.statistics.B}，單位＝原圖"
          f"（{comp['n_source_images']} 張），3 個種子的 SD 估計很粗，檢定力有限。", ""]
    L += ["## 各模型各種子結果", "", "| 模型 | 種子 | run_id | Dice | IoU | Precision | Recall | 逐 ROI 平均 Dice | HD95 均值(px) | Boundary F1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp["seed_table"]:
        L.append(f"| {r['candidate']} | {r['seed']} | {r['run_id']} | {fmt(r['dice'])} | {fmt(r['iou'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                 f"{fmt(r['per_roi_mean_dice'])} | {fmt(r['hd95_mean'], 2)} | {fmt(r['boundary_f1_mean'])} |")
    L += ["", "## 平均 ± 標準差（跨種子，樣本 SD，n=種子數）", "", "| 模型 | Dice | IoU | Precision | Recall | HD95(px) | Boundary F1 |", "|---|---|---|---|---|---|---|"]
    for c, s in comp["summary"].items():
        def pm(k, nd=4):
            v = s[k]
            return "未定義" if v is None else f"{v['mean']:.{nd}f} ± {fmt(v['sd'], nd)}"
        L.append(f"| {c} | {pm('dice')} | {pm('iou')} | {pm('precision')} | {pm('recall')} | {pm('hd95_mean', 2)} | {pm('boundary_f1_mean')} |")
    L += ["", "## 成本（同一台機器、同一設定下才可互比）", "", "| 模型 | 參數量 | 訓練時間(s,平均) | 峰值 GPU 記憶體(MiB,最大) | 推論 ms/ROI | epochs | batch |", "|---|---|---|---|---|---|---|"]
    for c, v in comp["cost"].items():
        L.append(f"| {c} | {v['n_parameters']:,} | {v['train_seconds_mean']:.0f} | {v['peak_gpu_mem_mb_max']:.0f} | {v['inference_ms_per_roi_mean']:.2f} | {v['epochs_run']} | {'; '.join(v['batch'])} |")
    L += ["", "容量差異（參數量）不得當成架構優劣的證據；推論時間為模型前向（batch 1，AMP，暖機 5 次），不含前後處理。", "",
          "## 錯誤案例（同一批 ROI 上各模型並列）", ""]
    for name, e in comp["error_cases"].items():
        L += [f"### {name}：{e['n_rois']} 個 ROI" + ("（零例）" if e["n_rois"] == 0 else ""), ""]
        if e["n_rois"]:
            L += ["| 模型 | 該組 Dice（跨種子平均） | 平均漏分占比 |", "|---|---|---|"]
            for c, v in e["per_model"].items():
                L.append(f"| {c} | {fmt(v['dice_mean_over_seeds'])} | {fmt(v['fn_frac_mean'])} |")
            L += ["", "ROI：" + ", ".join(e["roi_ids"][:30]) + (" …" if len(e["roi_ids"]) > 30 else ""), ""]
    L += ["「牙根漏分」為近似定義：資料無牙位／上下顎資訊，故取漏分像素占 GT 比例（跨所有模型與種子平均）最高的前 "
          f"{ROOT_MISS_TOPK} 個 ROI；ROI 以 test 結果挑選，僅作展示，不回饋調參。本報告未產生疊圖。", "",
          "## 限制與未量測", "",
          "- 單一資料集、test 約 20 張原圖；同圖 ROI 不獨立，故以原圖為單位估計不確定性；差異屬探索性。",
          "- 缺病人資訊：切分以 pHash 近重複分群，無法排除病人層級洩漏。",
          "- 種子數為 3，種子變異估計粗略；更多種子：未量測。",
          "- 未量測：外部資料泛化、預訓練變體、等容量變體（除非另列）、FLOPs、臨床診斷能力、穩健性。",
          "- AMP／cuDNN 非確定性：相同種子不保證逐位元相同。", ""]
    L += ["## 追溯", "", "| 模型 | 種子 | run_id | 權重 sha256 |", "|---|---|---|---|"]
    for r in comp["seed_table"]:
        L.append(f"| {r['candidate']} | {r['seed']} | {r['run_id']} | {comp['_sha'][r['run_id']][:16]} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--dry-run", action="store_true", help="use the validation dry-run evaluation; no lock needed")
    ap.add_argument("--reference-run", default=REFERENCE_RUN, help="pre-protocol baseline run, listed as reference only")
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)
    try:
        p = load_protocol(a.protocol)
        require_frozen(p)
        lock = run_lock.load_lock(p)
        if not a.dry_run and lock is None:
            raise ProtocolError("test has not been evaluated (no test_lock.json); run evaluate_seg_models first")
        rs = build_runset(p)
        if not rs["complete"]:
            raise ProtocolError("run set incomplete")
        sub = "eval_val_dryrun" if a.dry_run else "eval"
        runs: dict = {}
        for cell in rs["runs"]:
            m = json.loads((Path(p.reports_dir) / cell["adopted"] / "run_manifest.json").read_text())
            runs.setdefault(cell["candidate"], {})[cell["seed"]] = load_run(m, sub)
        if lock is not None and not a.dry_run:
            locked = {r["run_id"]: r["weights_sha256"] for r in lock["evaluated_runs"]}
            for by in runs.values():
                for r in by.values():
                    if locked.get(r["m"]["run_id"]) != r["metrics"]["weights_sha256"]:
                        raise ProtocolError(f"{r['m']['run_id']}: evaluated weights do not match the lock")
        comp = build_comparison(p, runs, a.dry_run)
        comp["_sha"] = {r["m"]["run_id"]: r["metrics"]["weights_sha256"] for by in runs.values() for r in by.values()}
        ref = None
        if Path(a.reference_run, "eval/metrics.json").exists():
            ref = json.loads(Path(a.reference_run, "eval/metrics.json").read_text())
        out = Path(a.output) if a.output else p.compare_dir() / ("dryrun" if a.dry_run else "")
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps({k: v for k, v in comp.items() if k != "_sha"}, indent=2, ensure_ascii=False))
        with open(out / "tables" / "seed_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp["seed_table"][0]))
            w.writeheader()
            w.writerows(comp["seed_table"])
        (out / "report.md").write_text(render(p, comp, lock, ref))
        print(f"report: {out / 'report.md'}")
        for c, r in comp["paired"].items():
            print(f"{c}: diff {r['mean_diff']:+.4f} CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] -> {r['verdict']}")
    except ProtocolError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

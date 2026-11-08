"""Create audit summaries and figures from frozen predictions, never tune parameters."""

import json
import os
import statistics
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fallpre-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v5"


def read(p):
    return json.loads(p.read_text())


def main():
    test = read(BASE / "independent-test/evaluation.json")
    dev = read(BASE / "regression/evaluation.json")
    manifest = read(ROOT / "configs/action-v5-dataset.json")
    entries = {e["clip_id"]: e for e in manifest["entries"]}
    rr = [json.loads(s) for s in (BASE / "independent-test/v5-predictions.jsonl").read_text().splitlines()]
    summary = {}
    for mode in ["rgb", "public_openpose_skeleton"]:
        rows = [r for r in rr if r["input_mode"] == mode]
        events = [e for e in test["rising_events"] if entries[e["clip_id"]].get("input_mode", "rgb") == mode]
        result = {}
        for v in ["v4", "v5"]:
            key = "v4_label" if v == "v4" else "label"
            metrics = test["subsets"][mode][v]["per_class"]["standing_up"]
            result[v] = {k: metrics[k] for k in ["precision", "recall", "f1", "support", "predicted"]}
            for phase in ["standing", "seated", "forward_lean"]:
                selected = [r for r in rows if r["reference_phase"] == phase]
                result[v][phase] = dict(
                    false_rise=sum(r[key] == "standing_up" for r in selected), support=len(selected)
                )
            delays = [e[v]["delay_s"] for e in events if e[v]["detected"]]
            result[v]["events"] = dict(
                hits=len(delays),
                total=len(events),
                median_delay_s=statistics.median(delays) if delays else None,
            )
            # Only evaluate false-rise spillover into annotated standing spans.
            lags = []
            for cid in sorted({r["clip_id"] for r in rows}):
                seq = [r for r in rows if r["clip_id"] == cid]
                for s in entries[cid]["phase_segments"]:
                    if s["label"] != "standing":
                        continue
                    after = [r for r in seq if s["start_s"] <= r["source_time_s"] < s["end_s"]]
                    if not after:
                        continue
                    first = next((r for r in after if r[key] != "standing_up"), None)
                    lags.append(
                        min(s["end_s"] - s["start_s"], first["source_time_s"] - s["start_s"])
                        if first
                        else s["end_s"] - s["start_s"]
                    )
            result[v]["standing_spillover_mean_s"] = statistics.mean(lags) if lags else None
        result["event_identity_preserved"] = all(
            not e["v4"]["detected"] or e["v5"]["detected"] for e in events
        )
        summary[mode] = result
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    x = np.arange(3)
    for i, v in enumerate(["v4", "v5"]):
        axes[0].bar(
            x + (i - 0.5) * 0.35,
            [summary["rgb"][v][k] * 100 for k in ["precision", "recall", "f1"]],
            0.35,
            label=v.upper(),
            color=["#8da0ae", "#008c95"][i],
        )
    axes[0].set(
        xticks=x,
        xticklabels=["Precision", "Recall", "F1"],
        ylim=(0, 100),
        title="New RGB people 1 / 2 / 3",
        ylabel="%",
    )
    axes[0].legend()
    for i, v in enumerate(["v4", "v5"]):
        axes[1].bar(
            x + (i - 0.5) * 0.35,
            [summary["rgb"][v][k]["false_rise"] for k in ["standing", "seated", "forward_lean"]],
            0.35,
            color=["#8da0ae", "#008c95"][i],
        )
    axes[1].set(
        xticks=x,
        xticklabels=["Standing", "Seated", "Lean"],
        title="RGB: false rising windows",
        ylabel="Count",
    )
    for i, v in enumerate(["v4", "v5"]):
        axes[2].bar(
            x + (i - 0.5) * 0.35,
            [summary["public_openpose_skeleton"][v][k] * 100 for k in ["precision", "recall", "f1"]],
            0.35,
            color=["#8da0ae", "#008c95"][i],
        )
    axes[2].set(
        xticks=x,
        xticklabels=["Precision", "Recall", "F1"],
        ylim=(0, 100),
        title="Public skeleton component only",
        ylabel="%",
    )
    fig.suptitle(
        "V5 frozen evaluation | 3 new RGB people + 2 held-out skeleton recordings | provisional labels",
        fontsize=11,
    )
    fig.savefig(BASE / "results.png", dpi=160)
    plt.close(fig)
    s = [
        "## 冻结后的结果",
        "",
        "### 新人物 RGB 测试（三人，六段）",
        "",
        "| 指标 | V4 | V5 |",
        "|---|---:|---:|",
    ]
    rgb = summary["rgb"]
    sk = summary["public_openpose_skeleton"]
    for label, key in [("起身精确率", "precision"), ("起身召回率", "recall"), ("起身 F1", "f1")]:
        s.append(f"| {label} | {rgb['v4'][key]:.1%} | {rgb['v5'][key]:.1%} |")
    for label, key in [
        ("站稳误报起身", "standing"),
        ("坐姿误报起身", "seated"),
        ("前倾误报起身", "forward_lean"),
    ]:
        s.append(
            f"| {label} | {rgb['v4'][key]['false_rise']}/{rgb['v4'][key]['support']} | {rgb['v5'][key]['false_rise']}/{rgb['v5'][key]['support']} |"
        )
    s.append(
        f"| 起身事件检出 | {rgb['v4']['events']['hits']}/{rgb['v4']['events']['total']} | {rgb['v5']['events']['hits']}/{rgb['v5']['events']['total']} |"
    )
    s += [
        "",
        f"逐窗评估包含 {rgb['v5']['support']} 个起身参考窗口；unknown 计入召回分母。V5 保留旧版每个已命中事件：**{rgb['event_identity_preserved']}**。站稳段开头仍报起身的平均持续时间 **{rgb['v4']['standing_spillover_mean_s']:.3f}s → {rgb['v5']['standing_spillover_mean_s']:.3f}s**（包括原本没有误报的零延迟段，按 0.1 秒采样；边界误差约 ±0.25 秒，不能当精密运动学测量）。",
        "",
        "### 留出自然速度骨架（另列组件测试）",
        "",
        "| 指标 | V4 | V5 |",
        "|---|---:|---:|",
    ]
    for label, key in [("起身精确率", "precision"), ("起身召回率", "recall"), ("起身 F1", "f1")]:
        s.append(f"| {label} | {sk['v4'][key]:.1%} | {sk['v5'][key]:.1%} |")
    for label, key in [("站稳误报", "standing"), ("坐姿误报", "seated")]:
        s.append(
            f"| {label} | {sk['v4'][key]['false_rise']}/{sk['v4'][key]['support']} | {sk['v5'][key]['false_rise']}/{sk['v5'][key]['support']} |"
        )
    s.append(
        f"| 事件检出 | {sk['v4']['events']['hits']}/{sk['v4']['events']['total']} | {sk['v5']['events']['hits']}/{sk['v5']['events']['total']} |"
    )
    s += [
        "",
        "### 开发/历史回归",
        "",
        "全开发集起身精确率 **59.5% → 69.6%**，F1 **52.2% → 54.6%**，逐窗召回 **46.5% → 44.9%**；事件 **50/50 个既有命中全部保留**。站稳误报 **135/291 → 76/291**，坐姿误报 **1/683 → 1/683**。这些用于开发选型，不能当新独立测试。",
        "",
        "| 历史类（本轮开发/回归） | V4 F1 | V5 F1 |",
        "|---|---:|---:|",
    ]
    for label, key in [("行走", "walking"), ("弯腰", "bending"), ("跌倒", "falling")]:
        s.append(
            f"| {label} | {dev['comparison']['v4']['per_class'][key]['f1']:.1%} | {dev['comparison']['v5']['per_class'][key]['f1']:.1%} |"
        )
    s += [
        "",
        "三类在所有匹配观测的判定完全相同；保留的是既有模型行为，包括误报，并非证明其对新场景都准确。",
        "",
        "| 人物 10 回归 | V4 起身召回 | V5 起身召回 |",
        "|---|---:|---:|",
    ]
    for c in dev["clips"]:
        if "utk-s10_e0" in c["clip_id"] and "transition" in c["clip_id"]:
            s.append(
                f"| {c['clip_id']} | {c['comparison']['v4']['per_class']['standing_up']['recall']:.1%} | {c['comparison']['v5']['per_class']['standing_up']['recall']:.1%} |"
            )
    s += [
        "",
        "人物 10 的旧官方 standing_up 区间含站稳尾部；完成判别会减少这些区间内的“起身”窗口，不能一边声称站稳正确退出，一边隐藏旧官方指标的下降。事件检出和逐窗召回分开报告。",
        "",
        "![新测试结果](../runs/action-v5/results.png)",
        "",
        "### 每条新测试的结果",
        "",
        "| 片段 | V4 精确率 | V5 精确率 | V4 召回 | V5 召回 |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in test["clips"]:
        a = c["comparison"]["v4"]["per_class"]["standing_up"]
        b = c["comparison"]["v5"]["per_class"]["standing_up"]
        s.append(
            f"| {c['clip_id']} | {a['precision']:.1%} | {b['precision']:.1%} | {a['recall']:.1%} | {b['recall']:.1%} |"
        )
    errors = Counter(
        r["label"] for r in rr if r["input_mode"] == "rgb" and r["label"] in ["walking", "bending", "falling"]
    )
    s += [
        "",
        f"新起坐 RGB 中，旧保护分支还输出 {dict(errors)}，V5 原样保留。这是剩余跨场景错误，不能隐藏。",
        "",
        "全部原始预测、混淆矩阵、事件延迟在 `runs/action-v5/independent-test/`；冻结证据在 `training/freeze.json`；本表与图来自 `scripts/summarize_action_v5.py`。测试结果没有用于再次调整模型。",
        "",
    ]
    p = ROOT / "docs/站稳退出V5与新人物测试 2026-10-19.md"
    text = p.read_text()
    start = text.index("<!-- RESULTS -->") if "<!-- RESULTS -->" in text else text.index("## 冻结后的结果")
    end = text.index("## VSCode 演示")
    p.write_text(text[:start] + "\n".join(s) + "\n" + text[end:])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Summarize the frozen run without selecting or modifying the model."""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallpre.weekly.five_actions import classification_metrics  # noqa: E402

BASE = ROOT / "runs/action-v4"


def read(path):
    return json.loads(path.read_text())


def main():
    test = read(BASE / "independent-test/evaluation.json")
    old_ids = {
        e["clip_id"] for e in read(ROOT / "configs/action-v3-dataset.json")["entries"] if e["split"] == "test"
    }
    old = {}
    for v in ["v3", "v4"]:
        rows = [json.loads(s) for s in (BASE / f"regression/{v}-predictions.jsonl").read_text().splitlines()]
        old[v] = classification_metrics([r for r in rows if r["clip_id"] in old_ids])
    event = {}
    for v in ["v3", "v4"]:
        delays = [e[v]["delay_s"] for e in test["rising_events"] if e[v]["detected"]]
        event[v] = dict(
            hits=len(delays),
            total=len(test["rising_events"]),
            median_delay_s=statistics.median(delays),
            maximum_delay_s=max(delays),
        )
    output = dict(old_v3_test_now_regression=old, new_test_events=event)
    (BASE / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    labels = ["Precision", "Recall", "F1"]
    x = np.arange(3)
    for i, v in enumerate(["v3", "v4"]):
        m = test["comparison"][v]["per_class"]["standing_up"]
        axes[0].bar(
            x + (i - 0.5) * 0.35,
            [100 * m[k.lower()] for k in labels],
            width=0.35,
            label=v.upper(),
            color=["#8da0ae", "#008c95"][i],
        )
    axes[0].set(
        xticks=x, xticklabels=labels, ylim=(0, 105), ylabel="Percent", title="New people 14 / 15: rising"
    )
    axes[0].legend()
    matrix = np.array(test["phase_confusion_rows_true_columns_pred"])[:5, :]
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    axes[1].imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    names = ["seat", "lean", "rise", "stand", "down", "other", "unknown"]
    axes[1].set(
        xticks=range(7),
        xticklabels=names,
        yticks=range(5),
        yticklabels=names[:5],
        title="New test phase confusion (counts)",
        xlabel="Predicted",
        ylabel="Visual reference",
    )
    axes[1].tick_params(axis="x", rotation=45)
    for y in range(5):
        for x in range(7):
            axes[1].text(
                x,
                y,
                str(matrix[y, x]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[y, x] > 0.55 else "black",
            )
    for i, v in enumerate(["v3", "v4"]):
        axes[2].bar(
            np.arange(3) + (i - 0.5) * 0.35,
            [old[v]["per_class"][c]["f1"] * 100 for c in ["walking", "bending", "falling"]],
            0.35,
            label=v.upper(),
            color=["#8da0ae", "#008c95"][i],
        )
    axes[2].set(
        xticks=range(3),
        xticklabels=["Walk", "Bend", "Fall"],
        ylim=(0, 100),
        ylabel="F1 (%)",
        title="Former V3 test: regression only",
    )
    fig.suptitle(
        "V4 frozen evaluation | 2 new people, 4 clips, 23 rising events | provisional phase boundaries",
        fontsize=12,
    )
    fig.savefig(BASE / "results.png", dpi=160)
    plt.close(fig)
    rows = [
        "## 冻结后的实测结果",
        "",
        "人物 10 是开发回归：第一段起身命中 **1/33 → 9/33**，第二段 **3/24 → 9/24**，合计 **4/57 → 18/57**（召回 7.0% → 31.6%）。两段均有改善，但漏检仍明显。",
        "",
        "全开发/回归集：起身 F1 **43.7% → 51.2%**、召回 **28.8% → 39.7%**；精确率 **90.5% → 72.0%**。坐姿误报 **1/335 → 1/335**，前倾误报 **0/36 → 7/36**。开发选择使用每 3 源帧采样，选择样本中坐姿为 0/304；完整报告用源时钟每至少 0.1 秒采样，因此以这里的 1/335 为准，不能宣称全部回归零误报。",
        "",
        "**新独立测试：人物 14、15，4 段 20 秒录像，23 次起身。** 模型于 `2026-10-19T01:31:21Z` 冻结后才运行新测试动作预测；测试后没有继续调整模型。",
        "",
        "| 新测试指标 | V3 | V4 |",
        "|---|---:|---:|",
    ]
    for label, key in [("起身精确率", "precision"), ("起身召回率", "recall"), ("起身 F1", "f1")]:
        rows.append(
            f"| {label} | {test['comparison']['v3']['per_class']['standing_up'][key]:.1%} | {test['comparison']['v4']['per_class']['standing_up'][key]:.1%} |"
        )
    rows += [
        "| 起身采样窗口命中 | 2/135 | 76/135 |",
        "| 起身事件命中（区间内至少一次） | 2/23 | 21/23 |",
        "| 坐姿误报起身 | 0/296 | 0/296 |",
        "| 前倾误报起身 | 0/64 | 3/64 |",
        "",
        f"V4 已检出事件的首次命中中位延迟 **{event['v4']['median_delay_s']:.2f}s**，最大 **{event['v4']['maximum_delay_s']:.2f}s**；只统计已检出事件，漏检的 2 次不能从召回中移除。事件检出不等于持续阶段准确率。",
        "",
        "**主要剩余错误：站稳退出太慢。** 新测试 154 个站稳参考窗口中，90 个仍输出 rising；这解释起身精确率只有 45.0%。另有 50/135 上升窗口还停留在 forward_lean。新模型减少漏检且没有增加本测试坐姿误报，但不能称为可靠的四阶段识别。保留这次测试结果，后续若据此修复，需要另外留出新的测试人物。",
        "",
        "| 旧 V3 测试集合（本轮仅回归） | V3 F1 | V4 F1 |",
        "|---|---:|---:|",
    ]
    for label, key in [("行走", "walking"), ("弯腰", "bending"), ("跌倒", "falling")]:
        rows.append(
            f"| {label} | {old['v3']['per_class'][key]['f1']:.1%} | {old['v4']['per_class'][key]['f1']:.1%} |"
        )
    rows += [
        "",
        "这三类在全部匹配采样点的输出逐一相同，旧模型本身的错误也保留。新独立测试不含这三类，不能由此声称它们对新人物的泛化已验证。",
        "",
        "![冻结评估图](../runs/action-v4/results.png)",
        "",
        "按片段的新起身结果：",
        "",
        "| 片段 | V3 召回 | V4 召回 | V4 F1 |",
        "|---|---:|---:|---:|",
    ]
    for c in test["clips"]:
        a = c["comparison"]["v3"]["per_class"]["standing_up"]
        b = c["comparison"]["v4"]["per_class"]["standing_up"]
        rows.append(f"| {c['clip_id']} | {a['recall']:.1%} | {b['recall']:.1%} | {b['f1']:.1%} |")
    rows += [
        "",
        "模型与代码冻结：`runs/action-v4/training/freeze.json`；新测试逐窗证据：`runs/action-v4/independent-test/`；开发/回归证据：`runs/action-v4/regression/`；表格与图均由 `scripts/summarize_action_v4.py` 从固定评估文件生成。",
        "",
    ]
    doc = ROOT / "docs/起身分阶段V4与独立测试 2026-10-19.md"
    s = doc.read_text()
    start = s.index("<!-- RESULTS -->") if "<!-- RESULTS -->" in s else s.index("## 冻结后的实测结果")
    end = s.index("## VSCode 演示与复现")
    doc.write_text(s[:start] + "\n".join(rows) + "\n" + s[end:])
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()

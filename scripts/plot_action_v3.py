"""Render the frozen independent-test comparison without retraining."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
result = json.loads((ROOT / "runs/action-v3/independent-test/evaluation.json").read_text())["comparison"]
regression = json.loads((ROOT / "runs/action-v3/regression/evaluation.json").read_text())["comparison"]
labels = ["walking", "sitting_down", "standing_up", "bending", "falling"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [2.1, 1]})
colors = ["#8094ad", "#087f8c"]
x = np.arange(5)
for j, name in enumerate(["v2", "v3"]):
    y = [result[name]["per_class"][k]["f1"] for k in labels]
    bars = ax.bar(x + (j - 0.5) * 0.36, y, 0.36, label=name.upper(), color=colors[j])
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=4)
ax.set(
    xticks=x,
    xticklabels=["Walk", "Sit down", "Stand up", "Bend", "Fall"],
    ylim=(0, 1),
    ylabel="F1 (unknown counts as error)",
    title="New held-out test: 26 clips / 20 source recordings",
)
ax.legend(frameon=False, loc="upper left", ncols=2)
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
for j, name in enumerate(["v2", "v3"]):
    counts = [
        result[name]["seated_false_rise"]["false_rise"],
        regression[name]["seated_false_rise"]["false_rise"],
    ]
    bars = bx.bar(np.arange(2) + (j - 0.5) * 0.3, counts, 0.3, color=colors[j])
    bx.bar_label(bars, fmt="%d", padding=4)
bx.set(
    xticks=[0, 1],
    xticklabels=["New seated\nn=27", "Historical regression\nn=135"],
    ylim=(0, 1.6),
    yticks=[0, 1],
    ylabel="False standing-up windows",
    title="Seated-state checks",
)
bx.grid(axis="y", alpha=0.15)
bx.set_axisbelow(True)
for a in (ax, bx):
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("V3: walking and bending recover; rising remains weak", fontsize=16, x=0.07, ha="left")
fig.text(
    0.07,
    0.08,
    "New rising recall: 20.7% → 22.9%   |   Fall windows: 55/80 → 55/80; events: 8/8 → 8/8",
    fontsize=11,
)
fig.text(
    0.07,
    0.035,
    "Source: frozen evaluation JSON. Seated labels are provisional; historical regression is development data.",
    fontsize=9,
    color="#555555",
)
fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.24, wspace=0.3)
fig.savefig(ROOT / "docs/assets/action-v3-comparison.png", dpi=180)

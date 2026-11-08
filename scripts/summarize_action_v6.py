# ruff: noqa: E402
# Standalone script configures repository imports.
"""Summarize frozen V6 tests and regression, including annotation-boundary sensitivity."""

import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fallpre-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallpre.weekly.five_actions import classification_metrics

BASE = ROOT / "runs/action-v6"


def main():
    entries = {e["clip_id"]: e for e in json.load(open(ROOT / "configs/action-v6-dataset.json"))["entries"]}
    result = {}
    for split in ["regression", "independent-test"]:
        rr = [json.loads(line) for line in (BASE / split / "v6-predictions.jsonl").read_text().splitlines()]
        ev = json.load(open(BASE / split / "evaluation.json"))["rising_events"]
        for group, select in [
            ("all", lambda r: True),
            (
                "former_rgb_test",
                lambda r: r["clip_id"] in {f"telkom-{i}-{w}" for i in [1, 2, 3] for w in ["before", "after"]},
            ),
        ]:
            rows = [r for r in rr if select(r)]
            if not rows:
                continue
            ids = {r["clip_id"] for r in rows}
            events = [e for e in ev if e["clip_id"] in ids]
            comparison = {}
            for version, key in [("v5", "v5_label"), ("v6", "label")]:
                metrics = classification_metrics([dict(r, label=r[key]) for r in rows])
                d = dict(rising=metrics["per_class"]["standing_up"], all_actions=metrics["per_class"])
                for phase in ["standing", "seated", "forward_lean"]:
                    chosen = [
                        r
                        for r in rows
                        if (r["seated"] if phase == "seated" else r["reference_phase"] == phase)
                    ]
                    d[phase] = dict(
                        support=len(chosen), false_rise=sum(r[key] == "standing_up" for r in chosen)
                    )
                late = []
                interior = []
                for r in rows:
                    e = entries[r["clip_id"]]
                    s = next(
                        (
                            s
                            for s in e.get("phase_segments", [])
                            if s["start_s"] <= r["source_time_s"] < s["end_s"]
                        ),
                        None,
                    )
                    if s:
                        if s["label"] == "standing" and r["source_time_s"] - s["start_s"] > 0.25 + 1e-8:
                            late.append(r)
                        if (
                            r["source_time_s"] - s["start_s"] > 0.25 + 1e-8
                            and s["end_s"] - r["source_time_s"] > 0.25 + 1e-8
                        ):
                            interior.append(dict(r, label=r[key]))
                d["standing_after025"] = dict(
                    support=len(late), false_rise=sum(r[key] == "standing_up" for r in late)
                )
                d["interior_rising"] = classification_metrics(interior)["per_class"]["standing_up"]
                delays = [e[version]["delay_s"] for e in events if e[version]["detected"]]
                d["events"] = dict(
                    hits=len(delays),
                    total=len(events),
                    median_delay_s=statistics.median(delays) if delays else None,
                )
                comparison[version] = d
            comparison["same_detected_events"] = all(
                not e["v5"]["detected"] or e["v6"]["detected"] for e in events
            )
            comparison["protected_parity"] = all(
                (r["v5_label"] == label) == (r["label"] == label)
                for r in rows
                for label in ["walking", "bending", "falling"]
            )
            result[split + "/" + group] = comparison
        if split == "independent-test":
            result["new_test_per_subject"] = {}
            for sid in sorted({r["subject_id"] for r in rr}):
                seq = [r for r in rr if r["subject_id"] == sid]
                result["new_test_per_subject"][sid] = {
                    v: classification_metrics([dict(r, label=r[k]) for r in seq])["per_class"]["standing_up"]
                    for v, k in [("v5", "v5_label"), ("v6", "label")]
                }
    (BASE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for ax, group, title in zip(
        axes[:2],
        ["regression/former_rgb_test", "independent-test/all"],
        ["Former people 1/2/3: development", "New people 4/5/6: frozen test"],
    ):
        for i, v in enumerate(["v5", "v6"]):
            ax.bar(
                np.arange(3) + (i - 0.5) * 0.35,
                [result[group][v]["rising"][k] * 100 for k in ["precision", "recall", "f1"]],
                0.35,
                label=v.upper(),
                color=["#8da0ae", "#008c95"][i],
            )
        ax.set(
            xticks=np.arange(3),
            xticklabels=["Precision", "Recall", "F1"],
            ylim=(0, 100),
            title=title,
            ylabel="%",
        )
        ax.legend()
    for i, v in enumerate(["v5", "v6"]):
        axes[2].bar(
            np.arange(2) + (i - 0.5) * 0.35,
            [
                result["regression/former_rgb_test"][v][k]["false_rise"]
                for k in ["standing", "standing_after025"]
            ],
            0.35,
            label=v.upper(),
            color=["#8da0ae", "#008c95"][i],
        )
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["All standing", ">0.25s after boundary"],
        title="Remaining 70-window audit",
        ylabel="False rising windows",
    )
    fig.suptitle("V6 | unchanged labels | provisional boundaries ±0.25s | no new-test tuning", fontsize=11)
    fig.savefig(BASE / "results.png", dpi=160)
    print(
        json.dumps(
            {
                g: {
                    v: (
                        d[v]["rising"]["precision"],
                        d[v]["rising"]["recall"],
                        d[v]["standing"],
                        d[v]["events"],
                    )
                    for v in ["v5", "v6"]
                }
                for g, d in result.items()
                if "/" in g
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

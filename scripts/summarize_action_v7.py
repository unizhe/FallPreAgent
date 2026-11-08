# ruff: noqa: E402
"""Reproducible V7 headline metrics, subgroup preservation and source-only slow audit."""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v7"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from review_ulred_action_v7 import read_amc

from fallpre.weekly.five_actions import classification_metrics


def main():
    audit = json.loads((BASE / "audit.json").read_text())
    summary = dict(audit["summary"])
    rr = [json.loads(x) for x in (BASE / "regression/v7-predictions.jsonl").read_text().splitlines()]
    for name, ids in [("old_people123", range(1, 4)), ("old_people456", range(4, 7))]:
        cohort = {f"telkom-{i}-{w}" for i in ids for w in ["before", "after"]}
        rows = [r for r in rr if r["clip_id"] in cohort]
        summary[name] = {}
        for v in ["v6", "v7"]:
            records = [dict(r, label=r["v6_label"]) if v == "v6" else r for r in rows]
            summary[name][v] = dict(
                rise=classification_metrics(records)["per_class"]["standing_up"],
                standing_fp=sum(
                    r["reference_phase"] == "standing" and r["label"] == "standing_up" for r in records
                ),
                seated_fp=sum(r["seated"] and r["label"] == "standing_up" for r in records),
            )
    summary["old62"] = audit["old62_summary"]
    slow = []
    # Bounds chosen solely from raw hip-coordinate plots, before any action inference.
    for subject, start, end in [("S01", 11.5, 12.35), ("S02", 13.5, 14.65), ("S03", 12.0, 12.85)]:
        p = BASE / f"inputs/ulred/{subject}/marker-less/raw/SitToStandR3{subject}.amc"
        rows = read_amc(p)
        h = np.array([(r["LeftHip"][1] + r["RightHip"][1]) / 2000 for r in rows])
        tt = np.arange(len(h)) / 30
        select = (tt >= start) & (tt <= end)
        t = tt[select]
        y = h[select]
        low = float(np.min(y))
        high = float(np.max(y))
        delta = high - low
        t10 = float(t[np.where(y >= low + 0.1 * delta)[0][0]])
        t90 = float(t[np.where(y >= low + 0.9 * delta)[0][0]])
        slow.append(
            dict(
                subject=subject,
                source_path=str(p.relative_to(ROOT)),
                fps=30,
                source_only_review_interval_s=[start, end],
                hip_height_gain_m=delta,
                t10_s=t10,
                t90_s=t90,
                ascent_10_to_90_s=t90 - t10,
                qualifies_for_two_second_continuous_slow_ascent=t90 - t10 >= 2,
                used_in_training=False,
                action_model_evaluated=False,
            )
        )
    summary["slow_data"] = dict(
        ulred=slow,
        criterion="At least 2 seconds from 10% to 90% of pelvis height gain, then inspect continuity; a source-screening convention, not a clinical definition.",
        remap=dict(
            status="Metadata verified; direct directory and ZIP downloads returned timeouts/empty replies. No sample acquired, no training/test result claimed.",
            url="https://data.bris.ac.uk/data/dataset/21h9f9e30v9cl2fapjggz4q1x7",
        ),
        scope="UL-RED native 3D markerless skeleton only; no RGB and no artificial time stretch. Instructed slow repetitions do not imply sustained slow ascent.",
    )
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for j, (key, title) in enumerate(
        [
            ("old_people123", "Old people 1/2/3 (development)"),
            ("old_people456", "Old people 4/5/6 (development)"),
            ("independent-test", "New people 7/8/9 (test)"),
        ]
    ):
        values = []
        for v in ["v6", "v7"]:
            m = (
                summary[key][v]["rise"]
                if j < 2
                else summary[key]["comparison"][v]["per_class"]["standing_up"]
            )
            values.append([100 * m["precision"], 100 * m["recall"]])
        x = np.arange(2)
        for i, v in enumerate(["V6", "V7"]):
            bars = ax[j].bar(x + (i - 0.5) * 0.32, values[i], 0.32, label=v, color=["#9aa6b2", "#237c85"][i])
            ax[j].bar_label(bars, fmt="%.2f", fontsize=8)
        ax[j].set_xticks(x, ["Rise precision", "Rise recall"])
        ax[j].set_ylim(0, 100)
        ax[j].set_title(title, fontsize=10)
        ax[j].legend()
    fig.suptitle("V7: improved precision; one independent-test rise window lost")
    fig.tight_layout()
    fig.savefig(BASE / "results.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            dict(old123=summary["old_people123"], old456=summary["old_people456"], slow=slow), indent=2
        )
    )


if __name__ == "__main__":
    main()

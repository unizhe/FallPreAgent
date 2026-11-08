"""Source-only phase annotations, recorded before new-person model inference."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v7"
# (lean starts, lift starts, upright, descend starts, seated); ±0.25s provisional.
CYCLES = {
    "telkom-7-after": [
        [1.7, 1.95, 2.5, 3.15, 3.8],
        [4.8, 5.05, 5.6, 6.25, 6.95],
        [8.15, 8.4, 9, 9.65, 10.35],
        [11.15, 11.4, 12, 12.65, 13.35],
        [14.15, 14.4, 15, 15.65, 16.35],
        [17.2, 17.45, 18.05, 18.7, 19.4],
    ],
    "telkom-7-before": [
        [1.2, 1.45, 2, 2.65, 3.35],
        [4.15, 4.4, 4.95, 5.55, 6.2],
        [7.05, 7.3, 7.9, 8.55, 9.25],
        [9.9, 10.15, 10.7, 11.25, 11.95],
        [12.85, 13.1, 13.65, 14.2, 14.85],
        [15.8, 16.05, 16.6, 17.15, 17.8],
        [18.65, 18.9, 19.5, 20, 20],
    ],
    "telkom-8-after": [
        [0.7, 0.95, 1.55, 2.1, 2.8],
        [3.75, 4, 4.6, 5.15, 5.85],
        [6.8, 7.05, 7.65, 8.1, 8.75],
        [9.65, 9.9, 10.5, 11.05, 11.75],
        [13.05, 13.3, 13.9, 14.5, 15.2],
        [16.5, 16.75, 17.35, 17.85, 18.55],
    ],
    "telkom-8-before": [
        [0.45, 0.7, 1.3, 1.65, 2.35],
        [2.8, 3.05, 3.65, 4.1, 4.7],
        [5.55, 5.8, 6.35, 6.9, 7.55],
        [8.3, 8.55, 9.15, 9.65, 10.3],
        [11.5, 11.75, 12.35, 12.8, 13.5],
        [14.25, 14.5, 15.05, 15.6, 16.3],
        [17.35, 17.6, 18.2, 18.7, 19.4],
    ],
    "telkom-9-after": [
        [0.45, 0.7, 1.3, 1.65, 2.35],
        [3.2, 3.45, 4, 4.2, 4.85],
        [5.8, 6.05, 6.6, 7, 7.7],
        [8.7, 8.95, 9.5, 9.8, 10.45],
        [11.2, 11.45, 12, 12.25, 12.9],
        [13.7, 13.95, 14.5, 14.8, 15.45],
        [16.2, 16.45, 17, 17.3, 17.95],
        [19.1, 19.35, 19.9, 20, 20],
    ],
    "telkom-9-before": [
        [0.8, 1.05, 1.6, 1.85, 2.5],
        [3, 3.25, 3.8, 4.05, 4.7],
        [5.45, 5.7, 6.25, 6.65, 7.3],
        [8.1, 8.35, 8.9, 9.2, 9.85],
        [10.6, 10.85, 11.4, 11.7, 12.35],
        [13.1, 13.35, 13.9, 14.2, 14.85],
        [15.7, 15.95, 16.5, 16.75, 17.4],
        [17.9, 18.15, 18.4, 18.55, 19.25],
    ],
}


def main():
    if (BASE / "training/freeze.json").exists():
        raise ValueError("Frozen")
    old = json.load(open(ROOT / "configs/action-v6-dataset.json"))
    entries = []
    for e in old["entries"]:
        e = dict(e, legacy_v6=True, previous_v6_split=e["split"])
        if e["split"] == "test":
            e["split"] = "dev"
        entries.append(e)
    annotation = dict(
        method="Agent visual source-only 0.5s contact sheets; boundary uncertainty ±0.25s; no model output, no independent human adjudication.",
        cycles=CYCLES,
    )
    (BASE / "review/phase-boundaries.json").write_text(json.dumps(annotation, indent=2) + "\n")
    for e in json.load(open(BASE / "new-entries-unannotated.json")):
        phases = []
        last = 0
        for a, b, c, d, f in CYCLES[e["clip_id"]]:
            for lo, hi, phase in [
                (last, a, "seated"),
                (a, b, "forward_lean"),
                (b, c, "rising"),
                (c, d, "standing"),
                (d, f, "descending"),
            ]:
                if hi > lo:
                    phases.append(dict(start_s=lo, end_s=hi, label=phase))
            last = f
        if last < e["duration_s"]:
            phases.append(dict(start_s=last, end_s=e["duration_s"], label="seated"))
        mapping = dict(
            seated="background",
            forward_lean="background",
            rising="standing_up",
            standing="background",
            descending="sitting_down",
        )
        e.update(
            phase_segments=phases,
            segments=[dict(s, label=mapping[s["label"]]) for s in phases],
            state_segments=[
                dict(start_s=s["start_s"], end_s=s["end_s"], state="seated")
                for s in phases
                if s["label"] == "seated"
            ],
            annotation_basis=annotation["method"],
            annotation_evidence=f"runs/action-v7/review/{e['clip_id']}.jpg",
        )
        entries.append(e)
    (ROOT / "configs/action-v7-dataset.json").write_text(
        json.dumps(
            dict(schema="fallpre.action_v7.dataset.v1", entries=entries, independent_human_annotators=0),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

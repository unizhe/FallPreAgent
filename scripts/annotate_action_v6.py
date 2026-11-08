"""Source-only phase annotations, recorded before new-person model inference."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v6"
# (lean starts, lift starts, upright, descend starts, seated); ±0.25s provisional.
CYCLES = {
    "telkom-4-after": [
        [0.1, 0.3, 0.75, 1.15, 1.9],
        [3.05, 3.2, 3.7, 4.15, 4.8],
        [5.65, 5.95, 6.5, 7.1, 7.85],
        [8.6, 8.85, 9.5, 10.1, 10.85],
        [11.25, 11.55, 12.15, 12.75, 13.45],
        [14.35, 14.6, 15.15, 15.75, 16.45],
        [17.05, 17.3, 17.9, 18.55, 19.25],
    ],
    "telkom-4-before": [
        [0.35, 0.6, 1.15, 1.65, 2.4],
        [3.25, 3.55, 4.1, 4.65, 5.5],
        [6.55, 6.85, 7.4, 8, 8.8],
        [9.6, 9.95, 10.5, 11.1, 11.9],
        [12.8, 13.1, 13.65, 14.25, 15.15],
        [16.05, 16.3, 16.85, 17.45, 18.25],
    ],
    "telkom-5-after": [
        [0.95, 1.25, 1.9, 2.6, 3.45],
        [5.4, 5.7, 6.3, 7.05, 7.95],
        [9.65, 9.95, 10.6, 11.35, 12.2],
        [13.65, 13.95, 14.55, 15.25, 16.05],
        [17.85, 18.15, 18.8, 19.55, 20],
    ],
    "telkom-5-before": [
        [0.75, 1.05, 1.75, 2.45, 3.35],
        [5.25, 5.55, 6.2, 6.9, 7.7],
        [9.45, 9.8, 10.4, 11.15, 12],
        [13.65, 13.95, 14.55, 15.2, 16.05],
        [18.1, 18.4, 19, 19.65, 20],
    ],
    "telkom-6-after": [
        [1.85, 2.1, 2.6, 3.15, 3.95],
        [5.1, 5.4, 6, 6.55, 7.35],
        [8.65, 8.95, 9.55, 10.15, 10.95],
        [11.8, 12.05, 12.65, 13.2, 14],
        [15.05, 15.3, 15.9, 16.5, 17.3],
        [18.45, 18.75, 19.35, 19.9, 20],
    ],
    "telkom-6-before": [
        [0.35, 0.65, 1.25, 1.75, 2.55],
        [3.5, 3.8, 4.35, 4.9, 5.7],
        [6.6, 6.85, 7.4, 8, 8.8],
        [9.85, 10.1, 10.65, 11.2, 12],
        [12.9, 13.2, 13.8, 14.35, 15.2],
        [16.35, 16.6, 17.2, 17.8, 18.65],
    ],
}


def main():
    if (BASE / "training/freeze.json").exists():
        raise ValueError("Frozen")
    old = json.load(open(ROOT / "configs/action-v5-dataset.json"))
    entries = []
    for e in old["entries"]:
        e = dict(e, legacy_v5=True, previous_v5_split=e["split"])
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
            annotation_evidence=f"runs/action-v6/review/{e['clip_id']}.jpg",
        )
        entries.append(e)
    (ROOT / "configs/action-v6-dataset.json").write_text(
        json.dumps(
            dict(schema="fallpre.action_v6.dataset.v1", entries=entries, independent_human_annotators=0),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

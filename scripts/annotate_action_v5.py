"""Assemble pre-inference visual annotations; all V3 tests become regression data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v5"


def main():
    old = json.loads((ROOT / "configs/action-v4-dataset.json").read_text())
    entries = []
    for e in old["entries"]:
        e = dict(e, legacy_v4=True, previous_v4_split=e["split"])
        if e["split"] == "test":
            e["split"] = "dev"
        entries.append(e)
    annotation = json.loads((BASE / "review/phase-boundaries.json").read_text())
    for e in json.loads((BASE / "new-entries-unannotated.json").read_text()):
        phases = []
        last = 0
        for a, b, c, d, f in annotation["cycles"][e["clip_id"]]:
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
        mapping = {
            "seated": "background",
            "forward_lean": "background",
            "rising": "standing_up",
            "standing": "background",
            "descending": "sitting_down",
        }
        e["phase_segments"] = phases
        e["segments"] = [dict(s, label=mapping[s["label"]]) for s in phases]
        e["state_segments"] = [
            dict(start_s=s["start_s"], end_s=s["end_s"], state="seated")
            for s in phases
            if s["label"] == "seated"
        ]
        e["annotation_basis"] = annotation["method"]
        e["annotation_evidence"] = "runs/action-v5/review/" + e["clip_id"] + ".jpg"
        entries.append(e)
    output = ROOT / "configs/action-v5-dataset.json"
    assert not output.exists(), "Never overwrite a frozen manifest"
    output.write_text(
        json.dumps(
            dict(schema="fallpre.action_v5.dataset.v1", entries=entries, independent_human_annotators=0),
            indent=2,
        )
        + "\n"
    )
    print(len(entries))


if __name__ == "__main__":
    main()

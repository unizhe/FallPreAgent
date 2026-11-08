"""Audit all old 62 standing errors and the person4 miss against frozen V7 outputs."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v7"


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    manifest = json.loads((ROOT / "configs/action-v7-dataset.json").read_text())
    entries = {e["clip_id"]: e for e in manifest["entries"]}
    dev = read(BASE / "regression/v7-predictions.jsonl")
    old = read(ROOT / "runs/action-v6/regression/v6-predictions.jsonl")
    cohort = {f"telkom-{i}-{w}" for i in [1, 2, 3] for w in ["before", "after"]}
    old62 = [
        r
        for r in old
        if r["clip_id"] in cohort and r["reference_phase"] == "standing" and r["label"] == "standing_up"
    ]
    lookup = {(r["clip_id"], r["source_frame_id"]): r for r in dev}
    audit = []
    for r in old62:
        n = lookup[(r["clip_id"], r["source_frame_id"])]
        t = r["source_time_s"]
        start = next(
            s["start_s"]
            for s in entries[r["clip_id"]]["phase_segments"]
            if s["label"] == "standing" and s["start_s"] <= t < s["end_s"]
        )
        audit.append(
            dict(
                clip_id=r["clip_id"],
                source_time_s=t,
                source_frame_id=r["source_frame_id"],
                seconds_after_upright=t - start,
                v7_label=n["label"],
                fixed=n["label"] != "standing_up",
                up_speed=n["current_up_speed"],
                motion_veto=n["completion_motion_veto"],
                probability=n["fast_completion_probability"],
                track_epoch=n["target_epoch"],
            )
        )
    assert len(audit) == 62
    protected = ["walking", "bending", "falling", "sitting_down"]
    summary = {}
    for subset in ["regression", "independent-test"]:
        rr = read(BASE / subset / "v7-predictions.jsonl")
        ev = json.loads((BASE / subset / "evaluation.json").read_text())
        for r in rr:
            for label in protected:
                assert (r["label"] == label) == (r["v6_label"] == label)
        summary[subset] = dict(
            comparison=ev["comparison"],
            events={
                v: dict(
                    detected=sum(e[v]["detected"] for e in ev["rising_events"]),
                    total=len(ev["rising_events"]),
                )
                for v in ["v6", "v7"]
            },
            lost_tp=[
                dict(clip_id=r["clip_id"], t=r["source_time_s"])
                for r in rr
                if r["target"] == "standing_up"
                and r["v6_label"] == "standing_up"
                and r["label"] != "standing_up"
            ],
            gained_tp=[
                dict(clip_id=r["clip_id"], t=r["source_time_s"])
                for r in rr
                if r["target"] == "standing_up"
                and r["v6_label"] != "standing_up"
                and r["label"] == "standing_up"
            ],
            protected_identical=True,
        )
        for v in ["v6", "v7"]:
            late = [
                r
                for r in rr
                if r["reference_phase"] == "standing"
                and r["source_time_s"]
                - next(
                    s["start_s"]
                    for s in entries[r["clip_id"]]["phase_segments"]
                    if s["label"] == "standing" and s["start_s"] <= r["source_time_s"] < s["end_s"]
                )
                > 0.25
            ]
            summary[subset][v + "_standing_beyond_025"] = dict(
                support=len(late),
                false_rise=sum((r["label"] if v == "v7" else r["v6_label"]) == "standing_up" for r in late),
            )
    remaining = [r for r in audit if not r["fixed"]]
    result = dict(
        old62=audit,
        old62_summary=dict(
            fixed=sum(r["fixed"] for r in audit),
            remaining=len(remaining),
            remaining_within_025=sum(r["seconds_after_upright"] <= 0.25 for r in remaining),
            remaining_motion_veto=sum(bool(r["motion_veto"]) for r in remaining),
            remaining_by_clip=dict(Counter(r["clip_id"] for r in remaining)),
        ),
        person4=[r for r in dev if r["clip_id"] == "telkom-4-after" and 6.3 <= r["source_time_s"] <= 6.6],
        summary=summary,
    )
    (BASE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["old62_summary"])
    for subset, s in summary.items():
        print(
            subset, s["events"], "lost", s["lost_tp"], "gained", s["gained_tp"], s["v7_standing_beyond_025"]
        )


if __name__ == "__main__":
    main()

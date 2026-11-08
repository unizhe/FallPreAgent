"""Visual skeleton phase references, reviewed before action predictions."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v5"
# start time, phase; inspect first20 seconds or complete shorter source, retain rest unknown.
LABELS = {
    "TmJDsJCx": [
        (0, "standing"),
        (0.8, "descending"),
        (2, "seated"),
        (3.1, "forward_lean"),
        (3.35, "rising"),
        (4.15, "standing"),
        (5.15, "descending"),
        (6.3, "seated"),
        (7.55, "forward_lean"),
        (7.8, "rising"),
        (8.55, "standing"),
        (9.6, "descending"),
        (10.7, "seated"),
        (11.85, "forward_lean"),
        (12.15, "rising"),
        (12.9, "standing"),
        (14.2, "descending"),
        (15.1, "seated"),
        (16.25, "forward_lean"),
        (16.55, "rising"),
        (17.2, "standing"),
    ],
    "I0tA8b18": [
        (0, "seated"),
        (1.5, "forward_lean"),
        (1.9, "rising"),
        (2.6, "standing"),
        (4.1, "descending"),
        (4.8, "seated"),
        (6, "forward_lean"),
        (6.4, "rising"),
        (7, "standing"),
        (8.1, "descending"),
        (8.8, "seated"),
        (10.3, "forward_lean"),
        (10.6, "rising"),
        (11.2, "standing"),
        (12.8, "descending"),
        (13.5, "seated"),
        (15.4, "forward_lean"),
        (15.7, "rising"),
        (16.4, "standing"),
        (18, "descending"),
        (18.8, "seated"),
    ],
    "FWDyTLr2": [
        (0, "standing"),
        (0.6, "descending"),
        (2.4, "seated"),
        (4.5, "forward_lean"),
        (4.9, "rising"),
        (5.6, "standing"),
        (6.6, "descending"),
        (7.6, "seated"),
        (8.3, "forward_lean"),
        (8.6, "rising"),
        (9.4, "standing"),
        (10.5, "descending"),
        (11.2, "seated"),
        (12.2, "forward_lean"),
        (12.55, "rising"),
        (13.15, "standing"),
        (14.4, "descending"),
        (15.2, "seated"),
        (16.05, "forward_lean"),
        (16.35, "rising"),
        (17.1, "standing"),
        (18.3, "descending"),
        (19.2, "seated"),
    ],
    "VOHFfenU": [
        (0, "standing"),
        (3.6, "descending"),
        (4.5, "seated"),
        (5.65, "forward_lean"),
        (5.95, "rising"),
        (6.6, "standing"),
        (7.7, "descending"),
        (8.5, "seated"),
        (9.7, "forward_lean"),
        (10, "rising"),
        (10.6, "standing"),
        (11.65, "descending"),
        (12.4, "seated"),
        (13.6, "forward_lean"),
        (13.9, "rising"),
        (14.55, "standing"),
        (15.15, "descending"),
        (16, "seated"),
        (17.1, "forward_lean"),
        (17.4, "rising"),
        (18, "standing"),
        (18.7, "descending"),
        (19.5, "seated"),
    ],
    "kP1305wr": [
        (0, "seated"),
        (2.1, "forward_lean"),
        (2.4, "rising"),
        (3.15, "standing"),
        (3.65, "descending"),
        (4.8, "seated"),
        (5.95, "forward_lean"),
        (6.25, "rising"),
        (7.05, "standing"),
        (8.15, "descending"),
        (9.25, "seated"),
        (10.3, "forward_lean"),
        (10.6, "rising"),
        (11.6, "standing"),
        (12.3, "descending"),
        (13.4, "seated"),
        (14.6, "forward_lean"),
        (14.9, "rising"),
        (15.7, "standing"),
        (16.3, "descending"),
        (17.3, "seated"),
        (18.35, "forward_lean"),
        (18.6, "rising"),
        (19.4, "standing"),
    ],
    "jsr1WRji": [
        (0, "unknown"),
        (1.35, "standing"),
        (2.7, "descending"),
        (3.3, "seated"),
        (4.35, "forward_lean"),
        (4.7, "rising"),
        (5.25, "standing"),
        (6.7, "descending"),
        (7.4, "seated"),
        (8.4, "forward_lean"),
        (8.7, "rising"),
        (9.25, "standing"),
        (10.4, "descending"),
        (11, "seated"),
        (12.15, "forward_lean"),
        (12.45, "rising"),
        (13.05, "standing"),
        (14.6, "descending"),
        (15.3, "seated"),
        (16.3, "forward_lean"),
        (16.6, "rising"),
        (17.1, "standing"),
        (18.15, "descending"),
        (18.7, "seated"),
    ],
}


def main():
    if (ROOT / "runs/action-v5/training/freeze.json").exists():
        raise ValueError("V5 artifacts are frozen; create a new version for data changes")
    m = json.loads((ROOT / "configs/action-v5-dataset.json").read_text())
    m["entries"] = [e for e in m["entries"] if e.get("dataset") != "Stanford-Sit2Stand"]
    for e in json.loads((BASE / "slow-entries-unannotated.json").read_text()):
        sid = e["clip_id"][9:]
        if sid not in LABELS:
            continue
        times = LABELS[sid] + [(min(20, e["duration_s"]), "unknown")]
        ph = []
        for (lo, label), (hi, _) in zip(times, times[1:]):
            if lo < hi:
                ph.append(dict(start_s=lo, end_s=hi, label=label))
        if e["duration_s"] > 20:
            ph.append(dict(start_s=20, end_s=e["duration_s"], label="unknown"))
        e["phase_segments"] = ph
        e["segments"] = [
            dict(
                s,
                label={"rising": "standing_up", "descending": "sitting_down", "unknown": "unknown"}.get(
                    s["label"], "background"
                ),
            )
            for s in ph
        ]
        e["state_segments"] = [
            dict(start_s=s["start_s"], end_s=s["end_s"], state="seated") for s in ph if s["label"] == "seated"
        ]
        e["annotation_basis"] = (
            "Agent visual review of natural skeleton sequences at0.5s, first20s fixed, ~0.25s boundary uncertainty, no independent human rater. Author slow STS metric includes preparatory sitting time, not pure lift duration."
        )
        e["pose_cache_sha256"] = hashlib.sha256(
            (BASE / "poses" / e["clip_id"] / "observations.jsonl").read_bytes()
        ).hexdigest()
        m["entries"].append(e)
    (ROOT / "configs/action-v5-dataset.json").write_text(json.dumps(m, indent=2) + "\n")
    (BASE / "review/slow-phase-boundaries.json").write_text(json.dumps(LABELS, indent=2) + "\n")
    p = ROOT / "configs/action-v5-protocol.json"
    d = json.loads(p.read_text())
    d["real_slow_source"]["subjects"] = {
        e["clip_id"][9:]: e["split"] for e in m["entries"] if e.get("dataset") == "Stanford-Sit2Stand"
    }
    d["real_slow_source"]["caveat"] = (
        "Author STS duration includes preparatory dwell; visible lift alone is generally <1.5s here. These are real slower complete transitions, not clinical bradykinetic ascent. Extra automatic slow candidates rejected before inference: visual review found camera motion/partial poses or merged cycles. No new model predictions used."
    )
    d["real_slow_source"]["identity_limit"] = (
        "Author anonymized subject IDs used; no claim of independently verified identity across source recordings. Primary new-person test is Telkom1/2/3. Exact duplicate skeletons prohibited across splits."
    )
    p.write_text(json.dumps(d, indent=2) + "\n")
    print(len(m["entries"]))


if __name__ == "__main__":
    main()

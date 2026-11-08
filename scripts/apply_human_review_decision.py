"""Apply one explicit chat decision to a versioned annotation file.

The original experiment-one annotations, metrics and validation remain frozen.
This script records a user decision; it never infers confirmation from UI use.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/abc-v7-20261020"
REVIEW = RUN / "human-review"
BASELINE = RUN / "annotations/annotations.json"
CURRENT = RUN / "annotations/annotations.reviewed.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path)
    args = parser.parse_args()
    decision_path = args.decision.resolve()
    decision = json.loads(decision_path.read_text())
    session_path = REVIEW / "review-session.json"
    session = json.loads(session_path.read_text())
    assert decision["reviewer"] == "user" and decision["user_statement"].strip()
    assert decision["status"] in {"human_confirmed_interval", "human_confirmed_exact",
                                  "human_confirmed_proxy_observation"}
    proxy_only = decision["status"] == "human_confirmed_proxy_observation"
    baseline_hash = digest(BASELINE)
    assert baseline_hash == session["annotation_sha256_before"]
    receipt_path = REVIEW / "decisions" / f'{decision["decision_id"]}.receipt.json'
    item = next(x for x in session["queue"] if x["id"] == decision["case_id"])
    if receipt_path.exists():
        assert item["decision"]["decision_id"] == decision["decision_id"]
        assert digest(decision_path) == item["decision"]["decision_sha256"]
        print("Decision already applied; no duplicate review.")
        return
    assert item["status"] == "pending" and item["clip_id"] == decision["clip_id"]
    source_path = ROOT / session["annotation_file"]
    before_hash = digest(source_path)
    if source_path == CURRENT:
        assert before_hash == session["annotation_sha256_current"], "Current review file changed outside the recorded session"
    annotation = json.loads(source_path.read_text())
    entry = next(x for x in annotation["entries"] if x["clip_id"] == decision["clip_id"])
    landmark_key = decision["landmark_key"]
    before = copy.deepcopy(entry["event_landmarks"].get(landmark_key))
    clock = dict(zip(entry["clock"]["original_frame_ids"], entry["clock"]["source_times_s"]))
    if decision["status"] == "human_confirmed_interval":
        assert decision["frame"] is None and decision["time_s"] is None
        first, last = decision["frame_interval"]
        assert first <= last
        assert decision["time_interval_s"] == [clock[first], clock[last]]
    else:
        assert decision["frame"] in clock
        assert decision["time_s"] == clock[decision["frame"]]
        assert decision["frame_interval"] is None and decision["time_interval_s"] is None
    physical_before = copy.deepcopy(entry["event_landmarks"].get("seat_off"))
    if proxy_only:
        assert landmark_key == "rise_visible_proxy"
        assert physical_before is None or (
            not physical_before.get("human_reviewed")
            and all(physical_before.get(k) is None for k in
                    ("frame", "time_s", "frame_interval", "time_interval_s"))
        ), "A proxy-only decision cannot overwrite an established physical boundary"
    if source_path == CURRENT:
        snapshot = REVIEW / "decisions" / f"annotations-before-{before_hash}.json"
        if not snapshot.exists():
            snapshot.write_bytes(source_path.read_bytes())
    else:
        snapshot = source_path
    reviewed = copy.deepcopy(before) if before else {}
    for key in ("frame", "time_s", "frame_interval", "time_interval_s", "status"):
        reviewed[key] = decision[key]
    for key in ("stability_criterion", "proxy_definition", "clock_status", "note"):
        if key in decision:
            reviewed[key] = copy.deepcopy(decision[key])
    source_id = "human_review_" + decision["decision_id"]
    reviewed["source"] = list(dict.fromkeys(reviewed.get("source", []) + [source_id]))
    reviewed.update(human_reviewed=True, review_decision_id=decision["decision_id"],
                    review_scope=decision["scope"])
    entry["event_landmarks"][landmark_key] = reviewed
    if proxy_only:
        entry["event_landmarks"]["seat_off"] = {
            "frame": None, "time_s": None, "frame_interval": None, "time_interval_s": None,
            "status": "pending_physical_boundary", "human_reviewed": False,
            "human_review_attempted": True, "boundary_confirmed": False,
            "review_decision_id": decision["decision_id"], "source": [source_id],
            "note": "Human review confirmed only a visible-rising proxy; physical seat-off remains unresolved.",
        }
    entry["review_pending"] = [
        text for text in entry["review_pending"]
        if text not in decision.get("resolved_review_pending", [])
    ]
    entry["human_review_status"] = "partial_landmarks_only"
    annotation["sources"][source_id] = {
        "path": str(decision_path.relative_to(ROOT)), "sha256": digest(decision_path),
        "kind": "explicit_user_annotation_decision", "note": decision["scope"],
    }
    annotation["human_review_status"] = "partial"
    annotation["human_review_completed"] = False
    annotation["conventions"]["human_review"] = (
        "human_reviewed applies to each annotation item. Only items linked to explicit "
        "user decisions are confirmed. An interval confirmation does not establish an "
        "exact frame. A confirmed proxy observation does not confirm a physical event boundary. "
        "Entry/root flags remain false until all annotations are reviewed."
    )
    annotation["revision"] = {
        "baseline_file": str(BASELINE.relative_to(ROOT)), "baseline_sha256": baseline_hash,
        "purpose": "Current partially human-reviewed annotations for experiment three; experiment-one baseline remains immutable.",
        "latest_decision_id": decision["decision_id"],
    }
    write_json(CURRENT, annotation)
    after_hash = digest(CURRENT)
    receipt = {
        "schema": "fallpre.human_review.receipt.v1",
        "decision_id": decision["decision_id"],
        "decision_file": str(decision_path.relative_to(ROOT)),
        "decision_sha256": digest(decision_path),
        "annotation_before_file": str(snapshot.relative_to(ROOT)),
        "annotation_before_sha256": before_hash,
        "annotation_after_file": str(CURRENT.relative_to(ROOT)),
        "annotation_after_sha256": after_hash,
        "clip_id": decision["clip_id"], "landmark_key": landmark_key,
        "before": before, "after": reviewed,
        "baseline_unchanged": digest(BASELINE) == baseline_hash,
        "other_cases_confirmed": False,
    }
    if proxy_only:
        receipt["physical_boundary_before"] = physical_before
        receipt["physical_boundary_after"] = entry["event_landmarks"]["seat_off"]
    write_json(receipt_path, receipt)
    item_status = "reviewed_proxy_only" if proxy_only else (
        "confirmed_interval" if decision["frame"] is None else "confirmed_exact"
    )
    item.update(status=item_status,
                human_reviewed=True, decision={
                    "decision_id": decision["decision_id"],
                    "decision_file": str(decision_path.relative_to(ROOT)),
                    "decision_sha256": digest(decision_path),
                    "receipt_file": str(receipt_path.relative_to(ROOT)),
                    "display_summary": decision["display_summary"],
                    "confirmed_definition": decision.get("confirmed_definition"),
                })
    session.update(
        annotation_file=str(CURRENT.relative_to(ROOT)), annotation_sha256_current=after_hash,
        baseline_annotation_file=str(BASELINE.relative_to(ROOT)),
        status=("review_queue_complete_with_unresolved_boundaries"
                if all(x["human_reviewed"] for x in session["queue"])
                else "partially_reviewed"),
        current_case_id=decision["next_case_id"],
        completed_cases=sum(x["human_reviewed"] for x in session["queue"]),
        completed_cases_meaning="Reviewed decisions, including proxy-only outcomes; not a count of confirmed physical boundaries.",
        proxy_only_cases=[x["id"] for x in session["queue"] if x["status"] == "reviewed_proxy_only"],
    )
    write_json(session_path, session)
    assert digest(BASELINE) == baseline_hash
    print(f'Applied {decision["decision_id"]}; status={reviewed["status"]}; frame={reviewed["frame"]}; {CURRENT}')


if __name__ == "__main__":
    main()

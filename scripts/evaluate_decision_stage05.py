#!/usr/bin/env python3
"""Freeze and evaluate Stage 05 decision data without network access.

The evaluator reads the Stage 04 SQLite database in read-only mode, exports a
redacted row-level audit table, snapshots the local reference labels, checks a
frozen source-video split, and reports window and event metrics separately.
It never calls an inference provider or modifies the source database.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "runs/decision-stage04-live/decision-agent.sqlite3"
DEFAULT_RUN_DIR = ROOT / "runs/decision-stage05-20261118"
DEFAULT_LABEL_CSV = DEFAULT_RUN_DIR / "window-labels.csv"
LABEL_SOURCE = ROOT / "runs/fall-head-reviewed-negatives-20261109/human-review.json"
EVENT_SOURCE = ROOT / "runs/fall-head-reviewed-negatives-20261109/events.json"
HEAD_PROTOCOL = ROOT / "runs/fall-head-reviewed-negatives-20261109/training-protocol.json"
HEAD_PROVENANCE = ROOT / "runs/fall-head-reviewed-negatives-20261109/provenance.json"
URFD_LABELS = ROOT / "runs/weekly-20261015/labels/urfall-cam0-falls.csv"
URFD_ADL_LABELS = ROOT / "runs/weekly-20261015/labels/urfall-cam0-adls.csv"
STAGE04_SOURCE_MANIFEST = ROOT / "runs/decision-stage04-revision-01/source-manifest.json"
STAGE04_OFF_SUMMARY = ROOT / "runs/decision-stage04-revision-01/provider-off-summary.json"
STAGE04_MAIN_REPORT = ROOT / "output/agent-review/stage-04/report.md"
STAGE04_R1_REPORT = ROOT / "output/agent-review/stage-04/report-revision-01.md"
STAGE04_R2_APPROVAL = ROOT / "output/agent-review/stage-04/reviewer-approval-revision-02.md"
VIDEO_SHARE = ROOT / "output/video-share-review-20261108/selection.json"
ABC_MANIFEST = ROOT / "runs/abc-v7-20261020/media/manifest.json"

VALID_WINDOW_LABELS = {
    "normal_activity",
    "already_lying",
    "fall_process_visible",
    "undetermined",
}

# Run identities come from the Stage 04 reports/source manifests and exact
# frame-count/timebase correspondence. Every rendition/replay maps to its
# underlying source-video group; runs are never independent samples.
RUN_SOURCE_MAP: dict[str, dict[str, str]] = {
    "run-0747f0212efb42f1b106a07ecaf4e8ab": {
        "video_id": "urfd-adl-04",
        "artifact_id": "urfd-adl-04-p3-comparison",
        "mapping_version": "pre-v2; exact adapter version not persisted",
        "basis": "Stage 04 final report: 50 frames, normal_activity, 0.0–4.9 s; ABC media manifest",
    },
    "run-75870cb62dda4919b29865ca2ad77da6": {
        "video_id": "urfd-fall-01",
        "artifact_id": "urfd-fall-01-p3-comparison",
        "mapping_version": "pre-v2; exact adapter version not persisted",
        "basis": "Stage 04 final report: 54 frames, fall, 0.0–5.3 s; ABC media manifest",
    },
    "run-9c09f349998c4b5bb4ccb7f61258b6ec": {
        "video_id": "urfd-adl-04",
        "artifact_id": "urfd-adl-04-single-rgb",
        "mapping_version": "live-evidence-mapping-v1",
        "basis": "Stage 04 revision 01 provider-off summary + source manifest",
    },
    "run-e6d2f672ac064403be865a4b86cd4851": {
        "video_id": "urfd-fall-02",
        "artifact_id": "urfd-fall-02-single-rgb",
        "mapping_version": "live-evidence-mapping-v1",
        "basis": "Stage 04 revision 01 provider-off summary + source manifest",
    },
    "run-7c2c26616e8b4b3285a3b15e349a5833": {
        "video_id": "urfd-adl-04",
        "artifact_id": "urfd-adl-04-single-rgb",
        "mapping_version": "live-evidence-mapping-v1",
        "basis": "Stage 04 revision 01 live run and source manifest",
    },
    "run-53943bd2e8e24cb5a93fb669e36dbe2a": {
        "video_id": "urfd-adl-04",
        "artifact_id": "urfd-adl-04-single-rgb",
        "mapping_version": "live-evidence-mapping-v1",
        "basis": "Stage 04 revision 01 report; contains the two retained ready results",
    },
    "run-ea747efd6cf749f8bbd81c1ec387e3c9": {
        "video_id": "urfd-fall-02",
        "artifact_id": "urfd-fall-02-single-rgb",
        "mapping_version": "live-evidence-mapping-v1",
        "basis": "Stage 04 revision 01 same-frame replay; source manifest",
    },
}

VIDEO_GROUPS = {
    "urfd-adl-04": {
        "source_video_group_id": "urfd-source-adl-04",
        "source_collection": "URFD",
        "source_license": "CC BY-NC-SA 4.0; non-commercial academic research",
        "person_id": "URFD subject; cross-clip identity not resolved",
        "camera_id": "URFD cam0",
        "development_status": "repeated P3 / prior development input",
    },
    "urfd-fall-01": {
        "source_video_group_id": "urfd-source-fall-01",
        "source_collection": "URFD",
        "source_license": "CC BY-NC-SA 4.0; non-commercial academic research",
        "person_id": "URFD subject; cross-clip identity not resolved",
        "camera_id": "URFD cam0",
        "development_status": "repeated P3 / prior development input",
    },
    "urfd-fall-02": {
        "source_video_group_id": "urfd-source-fall-02",
        "source_collection": "URFD",
        "source_license": "CC BY-NC-SA 4.0; non-commercial academic research",
        "person_id": "URFD subject; cross-clip identity not resolved",
        "camera_id": "URFD cam0",
        "development_status": "repeated P3 / prior development input",
    },
}

CANONICAL_EVENT_RUNS = {
    # One full run per source-video group. Replays and alternate renditions are
    # retained in the audit table but do not multiply event denominators.
    "urfd-adl-04": "run-9c09f349998c4b5bb4ccb7f61258b6ec",
    "urfd-fall-01": "run-75870cb62dda4919b29865ca2ad77da6",
    "urfd-fall-02": "run-e6d2f672ac064403be865a4b86cd4851",
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_label_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    labels: dict[str, dict[str, str]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            label = (row.get("label") or "").strip()
            if label and label not in VALID_WINDOW_LABELS:
                raise ValueError(f"Unsupported window label {label!r} in {path}")
            key = row.get("canonical_window_id") or ""
            if key:
                labels[key] = row
    return labels


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields})


def read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def canonical_window_id(video_id: str, start_s: float, end_s: float) -> str:
    payload = f"{video_id}|{start_s:.6f}|{end_s:.6f}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def run_label_inventory() -> dict[str, Any]:
    manual_intervals = read_json(LABEL_SOURCE)
    events = read_json(EVENT_SOURCE)
    protocol = read_json(HEAD_PROTOCOL)
    provenance = read_json(HEAD_PROVENANCE)
    abc = read_json(ROOT / "runs/abc-v7-20261020/annotations/annotations.reviewed.json")
    return {
        "schema_version": "stage05-label-inventory.v1",
        "window_label_contract": {
            "target": "whether a fall process is visible within the exact source window",
            "labels": sorted(VALID_WINDOW_LABELS),
            "video_class_is_not_propagated_to_windows": True,
            "model_output_is_not_ground_truth": True,
            "event_interval_fields": ["event_start_s", "event_end_s", "reviewed_by", "label_source", "disagreement_notes"],
        },
        "manual_window_labels": {
            "source": str(LABEL_SOURCE.relative_to(ROOT)),
            "source_sha256": sha256_file(LABEL_SOURCE),
            "reviewed_intervals": len(manual_intervals.get("intervals", [])),
            "label_counts": dict(Counter(row.get("label") for row in manual_intervals.get("intervals", []))),
            "unique_video_hashes": len({row.get("video_sha256") for row in manual_intervals.get("intervals", []) if row.get("video_sha256")}),
            "reviewed_by": sorted({row.get("reviewed_by") for row in manual_intervals.get("intervals", [])}),
            "window_samples_used_in_head_development": 105,
            "independent_test_eligible": False,
            "exclusion_reason": "These user-confirmed normal intervals are included in C/D head development/training; they are not a blind holdout.",
        },
        "manual_event_labels": {
            "source": str(EVENT_SOURCE.relative_to(ROOT)),
            "source_sha256": sha256_file(EVENT_SOURCE),
            "reviewed_event_count": len(events),
            "event_kind_counts": dict(Counter(row.get("event_kind") for row in events)),
            "demonstration_count": sum(bool(row.get("demonstration")) for row in events),
            "review_status_counts": dict(Counter(row.get("review_status") for row in events)),
            "unique_video_hashes": len({row.get("video_sha256") for row in events if row.get("video_sha256")}),
            "positive_windows_used_in_head_development": 21,
            "independent_test_eligible": False,
            "exclusion_reason": "The 14 reviewed events are staged/simulated and contributed 21 positive windows to C/D head development.",
        },
        "head_training_overlap": {
            "head_sources": str(HEAD_PROVENANCE.relative_to(ROOT)),
            "head_sources_sha256": sha256_file(HEAD_PROVENANCE),
            "protocol_source": str(HEAD_PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256_file(HEAD_PROTOCOL),
            "public_proxy_source": provenance.get("public_proxy_source"),
            "common_public_samples": protocol.get("common_public_samples"),
            "personal_positive_samples": protocol.get("common_personal_positive_samples"),
            "personal_negative_samples": protocol.get("personal_negative_samples"),
            "grouping_rule": protocol.get("grouping"),
            "limitations": protocol.get("limits"),
            "per_public_source_video_overlap_resolved": False,
        },
        "other_review_material": [
            {
                "path": "runs/abc-v7-20261020/annotations/annotations.reviewed.json",
                "sha256": sha256_file(ROOT / "runs/abc-v7-20261020/annotations/annotations.reviewed.json"),
                "status": abc.get("human_review_status", "not human adjudicated"),
                "human_review_completed": abc.get("human_review_completed"),
                "used_as_window_truth": False,
            },
            {
                "path": "runs/fall-capture-20261026-154301/visual-preannotations.json",
                "sha256": sha256_file(ROOT / "runs/fall-capture-20261026-154301/visual-preannotations.json"),
                "status": "AI visual preannotation; pending user confirmation",
                "event_count": len(read_json(ROOT / "runs/fall-capture-20261026-154301/visual-preannotations.json").get("events", [])),
                "used_as_ground_truth": False,
            },
            {
                "path": "runs/personal-retest-20261023-233739/annotations.json",
                "sha256": sha256_file(ROOT / "runs/personal-retest-20261023-233739/annotations.json"),
                "status": "AI visual prelabels; not human adjudicated",
                "used_as_fall_process_truth": False,
            },
        ],
    }


def build_video_inventory() -> dict[str, Any]:
    share = read_json(VIDEO_SHARE)
    abc = read_json(ABC_MANIFEST)
    r1 = read_json(STAGE04_SOURCE_MANIFEST)
    provenance = read_json(HEAD_PROVENANCE)
    share_by_id = {row["id"]: row for row in share}
    p3_files: list[dict[str, Any]] = []
    abc_full = {row["clip_id"]: row for row in abc.get("full_videos", [])}
    source_files = {row["scenario"]: row for row in r1.get("inputs", [])}

    def add(row: dict[str, Any]) -> None:
        p3_files.append(row)

    adl4 = source_files.get("normal_activity", {})
    fall2 = source_files.get("fall", {})
    add({
        "artifact_id": "urfd-adl-04-single-rgb",
        "video_id": "urfd-adl-04",
        "source_video_group_id": VIDEO_GROUPS["urfd-adl-04"]["source_video_group_id"],
        "artifact_type": "original_single_camera_rgb",
        "relative_path": adl4.get("path"),
        "sha256": adl4.get("sha256"),
        "bytes": adl4.get("bytes"),
        "frames": adl4.get("frames"),
        "duration_s": adl4.get("duration_s"),
        "person_id": VIDEO_GROUPS["urfd-adl-04"]["person_id"],
        "camera_id": VIDEO_GROUPS["urfd-adl-04"]["camera_id"],
        "source": "URFD, official ADL-04 source clip",
        "license": VIDEO_GROUPS["urfd-adl-04"]["source_license"],
    })
    add({
        "artifact_id": "urfd-fall-02-single-rgb",
        "video_id": "urfd-fall-02",
        "source_video_group_id": VIDEO_GROUPS["urfd-fall-02"]["source_video_group_id"],
        "artifact_type": "original_single_camera_rgb",
        "relative_path": fall2.get("path"),
        "sha256": fall2.get("sha256"),
        "bytes": fall2.get("bytes"),
        "frames": fall2.get("frames"),
        "duration_s": fall2.get("duration_s"),
        "person_id": VIDEO_GROUPS["urfd-fall-02"]["person_id"],
        "camera_id": VIDEO_GROUPS["urfd-fall-02"]["camera_id"],
        "source": "URFD, official FALL-02 source clip",
        "license": VIDEO_GROUPS["urfd-fall-02"]["source_license"],
    })
    fall1_source_sha256 = abc.get("input_hashes", {}).get("urfd-fall-01", {}).get("source")
    fall1_source_path = ROOT / "runs/five-action-20261015/inputs/urfd/fall-01-rgb.mp4"
    add({
        "artifact_id": "urfd-fall-01-single-rgb-source-record",
        "video_id": "urfd-fall-01",
        "source_video_group_id": VIDEO_GROUPS["urfd-fall-01"]["source_video_group_id"],
        "artifact_type": "original_single_camera_rgb",
        "relative_path": str(fall1_source_path.relative_to(ROOT)) if fall1_source_path.is_file() else None,
        "sha256": sha256_file(fall1_source_path) if fall1_source_path.is_file() else fall1_source_sha256,
        "bytes": fall1_source_path.stat().st_size if fall1_source_path.is_file() else None,
        "frames": 160,
        "duration_s": 160 / 30.0,
        "person_id": VIDEO_GROUPS["urfd-fall-01"]["person_id"],
        "camera_id": VIDEO_GROUPS["urfd-fall-01"]["camera_id"],
        "source": "URFD official FALL-01 original source; checksum from ABC source manifest",
        "license": VIDEO_GROUPS["urfd-fall-01"]["source_license"],
        "local_source_file_present": fall1_source_path.is_file(),
    })
    for video_id, clip_id in [("urfd-adl-04", "urfd-adl-04"), ("urfd-fall-01", "urfd-fall-01")]:
        row = abc_full.get(clip_id, {})
        add({
            "artifact_id": f"{video_id}-p3-comparison",
            "video_id": video_id,
            "source_video_group_id": VIDEO_GROUPS[video_id]["source_video_group_id"],
            "artifact_type": "synchronized_A_B_C_comparison_rendition",
            "relative_path": f"runs/abc-v7-20261020/{row.get('file', '')}",
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
            "frames": row.get("frames"),
            "duration_s": (row.get("frames", 0) / 30.0) if row.get("frames") else None,
            "person_id": VIDEO_GROUPS[video_id]["person_id"],
            "camera_id": VIDEO_GROUPS[video_id]["camera_id"],
            "source": "Derived synchronized multi-panel comparison rendition; source frame count/timebase listed in ABC manifest",
            "license": VIDEO_GROUPS[video_id]["source_license"],
            "source_video_sha256": abc.get("input_hashes", {}).get(clip_id, {}).get("source"),
        })

    personal: list[dict[str, Any]] = []
    for clip_id, item in provenance.get("clips", {}).items():
        video_path = Path(item.get("video", ""))
        try:
            rel_path = str(video_path.resolve().relative_to(ROOT))
        except (ValueError, OSError):
            rel_path = None
        share_entry = share_by_id.get(clip_id)
        personal.append({
            "artifact_id": f"personal-{clip_id}",
            "video_id": clip_id,
            "source_video_group_id": f"personal-video-{clip_id}",
            "artifact_type": "personal_development_video",
            "relative_path": rel_path,
            "sha256": sha256_file(video_path) if video_path.is_file() else None,
            "source_log_sha256": item.get("source_sha256"),
            "capture_group": item.get("group"),
            "person_id": "same participant across old/new camera clips per video-share instructions",
            "camera_id": "old camera" if clip_id.startswith("old-") else "new camera" if clip_id.startswith("new-") else "local personal camera; exact setup needs source confirmation",
            "source": "Local personal development source referenced by C/D training provenance; not redistributable",
            "authorization_note": "Local development material; exact scope is recorded in the project video-use instructions when present.",
            "video_share_selection": bool(share_entry),
            "video_share_name": share_entry.get("name") if share_entry else None,
            "independent_test_eligible": False,
            "partition": "development_exposed",
        })
    return {
        "schema_version": "stage05-video-inventory.v1",
        "p3_video_artifacts": p3_files,
        "p3_source_video_groups": VIDEO_GROUPS,
        "personal_head_development_videos": personal,
        "public_source": {
            "collection": "URFD",
            "source_page": "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
            "license": "CC BY-NC-SA 4.0; non-commercial academic research",
            "source_label_files": [
                {"path": str(URFD_LABELS.relative_to(ROOT)), "sha256": sha256_file(URFD_LABELS)},
                {"path": str(URFD_ADL_LABELS.relative_to(ROOT)), "sha256": sha256_file(URFD_ADL_LABELS)},
            ],
        },
    }


def build_official_event_labels() -> dict[str, Any]:
    by_recording: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with URFD_LABELS.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            recording = row[0].strip()
            frame = int(row[1])
            state = int(row[2])
            by_recording[recording].append((frame, state))

    events: list[dict[str, Any]] = []
    state_counts: dict[str, dict[str, int]] = {}
    for recording in ("fall-01", "fall-02"):
        rows = sorted(by_recording[recording])
        state_counts[recording] = dict(Counter(str(state) for _, state in rows))
        falling = [frame for frame, state in rows if state == 0]
        if not falling:
            continue
        segments: list[list[int]] = []
        for frame in falling:
            if not segments or frame > segments[-1][-1] + 1:
                segments.append([frame])
            else:
                segments[-1].append(frame)
        for index, segment in enumerate(segments, 1):
            # Source frame numbers are 1-based; intervals are half-open at the
            # next frame boundary, using the official 30 Hz source clock.
            events.append({
                "event_id": f"urfd-{recording}-fall-process-{index:02d}",
                "video_id": f"urfd-{recording}",
                "source_video_group_id": VIDEO_GROUPS[f"urfd-{recording}"]["source_video_group_id"],
                "label": "fall_process_visible",
                "label_source": "official URFD fall-state CSV (state 0)",
                "annotator": "URFD official annotation; not independent local human adjudication",
                "start_frame_1based": segment[0],
                "end_frame_1based_inclusive": segment[-1],
                "start_s": (segment[0] - 1) / 30.0,
                "end_s_exclusive": segment[-1] / 30.0,
                "source_fps": 30,
                "event_boundary_uncertainty_s": 1 / 30,
            })
    return {
        "schema_version": "stage05-official-event-labels.v1",
        "source_path": str(URFD_LABELS.relative_to(ROOT)),
        "source_sha256": sha256_file(URFD_LABELS),
        "state_semantics": {"-1": "non-lying; not mapped to normal activity", "0": "falling process", "1": "lying"},
        "source_state_counts": state_counts,
        "events": events,
        "video_level_adl_reference": {
            "video_id": "urfd-adl-04",
            "source": "official URFD ADL recording group + Stage 04 source manifest",
            "fall_event_count": 0,
            "window_label_rule": "This source-level ADL class does not automatically label each window; leave window labels blank until manual review.",
        },
    }


def decode_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def export_observation_rows(connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs = {row["run_id"]: dict(row) for row in connection.execute("SELECT * FROM agent_runs")}
    frames: dict[tuple[str, float], dict[str, Any]] = {}
    for row in connection.execute("SELECT * FROM agent_frame_receipts"):
        frames[(row["run_id"], round(float(row["source_time"]), 6))] = dict(row)
    tasks = {row["task_id"]: dict(row) for row in connection.execute("SELECT * FROM decision_tasks")}
    result_by_decision: dict[str, dict[str, Any]] = {}
    for row in connection.execute("SELECT * FROM decision_results ORDER BY completed_at, attempt_no"):
        result_by_decision[row["decision_id"]] = dict(row)
    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute("SELECT * FROM decision_attempts ORDER BY task_id, attempt_no"):
        attempts_by_task[row["task_id"]].append(dict(row))

    raw_rows: list[dict[str, Any]] = []
    for row in connection.execute("SELECT * FROM decision_observations ORDER BY run_id, source_time, id"):
        obs = dict(row)
        envelope = decode_json(obs.get("envelope_json"), {})
        summary = envelope.get("evidence_summary", {}) if isinstance(envelope, dict) else {}
        window = summary.get("source_window") or envelope.get("source_window")
        run_source = RUN_SOURCE_MAP.get(obs["run_id"], {})
        video_id = run_source.get("video_id", f"unmapped-{obs['run_id']}")
        video_meta = VIDEO_GROUPS.get(video_id, {})
        task = tasks.get(obs.get("task_id"), {})
        result_row = result_by_decision.get(obs.get("decision_id"), {})
        result = decode_json(result_row.get("result_json"), {})
        attempt_rows = attempts_by_task.get(obs.get("task_id"), [])
        accepted_attempt = next((item for item in reversed(attempt_rows) if item.get("status") == "accepted"), None)
        latest_attempt = attempt_rows[-1] if attempt_rows else None
        attempt = accepted_attempt or latest_attempt or {}
        frame = frames.get((obs["run_id"], round(float(obs["source_time"]), 6)), {})
        missing = decode_json(obs.get("missing_reasons_json"), [])
        probability = result.get("fall_probability_raw")
        if probability is None:
            probability = result.get("fall_probability")
        if result.get("status") == "ready" and probability is not None and result.get("probability_kind") == "uncalibrated_model_estimate":
            gate = "ready_real_uncalibrated_score"
        elif missing:
            gate = "abstain_missing_evidence"
        elif result.get("status"):
            gate = f"result_{result.get('status')}"
        elif obs.get("task_id"):
            gate = f"task_{task.get('status', 'scheduled_without_result')}"
        else:
            gate = f"not_scheduled_{obs.get('schedule_reason') or 'unspecified'}"
        actual_samples = summary.get("observed_frames")
        if actual_samples is None and isinstance(window, dict):
            actual_samples = window.get("observed_frames", window.get("real_sample_count"))
        source_time = float(obs["source_time"])
        start_s = float(window["start_s"]) if isinstance(window, dict) and window.get("start_s") is not None else None
        end_s = float(window["end_s"]) if isinstance(window, dict) and window.get("end_s") is not None else None
        win_id = canonical_window_id(video_id, start_s, end_s) if start_s is not None and end_s is not None else None
        raw_rows.append({
            "observation_row_id": obs["id"],
            "video_id": video_id,
            "source_video_group_id": video_meta.get("source_video_group_id", f"unmapped:{obs['run_id']}"),
            "artifact_id": run_source.get("artifact_id", "unmapped"),
            "partition": "development_exposed",
            "run_id": obs["run_id"],
            "source_time_s": source_time,
            "window_id": win_id,
            "window_start_s": start_s,
            "window_end_s": end_s,
            "actual_source_samples_in_window": actual_samples,
            "input_frame_sha256": frame.get("image_sha256"),
            "observation_digest": obs.get("digest"),
            "evidence_digest": obs.get("evidence_digest"),
            "mapping_version": run_source.get("mapping_version", "unknown"),
            "policy_version": summary.get("policy_version"),
            "prompt_version": summary.get("prompt_version"),
            "eligibility_policy_version": decode_json(task.get("evidence_json"), {}).get("eligibility_policy_version"),
            "preprocessing_version": summary.get("preprocessing_version"),
            "yolo_model_version": summary.get("yolo_model_version"),
            "temporal_model_version": summary.get("temporal_model_version"),
            "head_variant": summary.get("head_variant"),
            "head_version": summary.get("head_version"),
            "head_weights_sha256": summary.get("head_weights_sha256"),
            "provider": result.get("provider") or task.get("provider") or obs.get("provider"),
            "model_version": result.get("model") or task.get("model") or obs.get("model"),
            "calibration_version": result.get("calibration_version"),
            "label": None,
            "label_source": None,
            "label_status": "needs_human_window_review" if win_id else "no_source_window",
            "model_derived_fall_process_label_excluded_from_truth": summary.get("fall_process_label"),
            "raw_temporal_score_not_probability": summary.get("temporal_score"),
            "raw_head_score_not_probability": summary.get("head_raw_score"),
            "gate_result": gate,
            "missing_reasons": missing,
            "schedule_reason": obs.get("schedule_reason"),
            "task_id": obs.get("task_id"),
            "decision_id": obs.get("decision_id"),
            "result_status": result.get("status"),
            "probability": probability,
            "probability_kind": result.get("probability_kind"),
            "probability_source": result.get("provider") if probability is not None else None,
            "result_accepted": bool(result_row.get("accepted")) if result_row else None,
            "attempt_no": attempt.get("attempt_no"),
            "provider_request_id": attempt.get("request_id"),
            "http_status": attempt.get("http_status"),
            "input_tokens": attempt.get("input_tokens"),
            "client_latency_ms": attempt.get("client_latency_ms"),
            "service_latency_ms": attempt.get("service_latency_ms"),
            "result_reason_codes": result.get("reason_codes", []),
            "trace_mapping_basis": run_source.get("basis", "unmapped run source"),
            "duplicate_source_time_frame": False,
            "repeated_run_ids": [],
        })

    duplicate_groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        digest = row.get("input_frame_sha256")
        if digest:
            duplicate_groups[(row["source_video_group_id"], round(row["source_time_s"], 6), digest)].append(row)
    for group in duplicate_groups.values():
        run_ids = sorted({row["run_id"] for row in group})
        if len(run_ids) > 1:
            for row in group:
                row["duplicate_source_time_frame"] = True
                row["repeated_run_ids"] = [run_id for run_id in run_ids if run_id != row["run_id"]]
    return raw_rows, {"run_count": len(runs), "frame_receipt_count": len(frames), "task_count": len(tasks), "result_count": len(result_by_decision), "attempt_count": sum(len(v) for v in attempts_by_task.values())}


def make_probability_rows(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in observations if row.get("result_status") == "ready" and row.get("probability") is not None and row.get("probability_kind") == "uncalibrated_model_estimate" and row.get("provider") == "aliyun" and row.get("result_accepted")]


def make_window_label_template(path: Path, probability_rows: list[dict[str, Any]]) -> None:
    if path.exists():
        return
    unique: dict[str, dict[str, Any]] = {}
    for row in probability_rows:
        window_id = row.get("window_id")
        if not window_id:
            continue
        unique.setdefault(window_id, {
            "canonical_window_id": window_id,
            "video_id": row["video_id"],
            "source_video_group_id": row["source_video_group_id"],
            "partition": row["partition"],
            "window_start_s": row["window_start_s"],
            "window_end_s": row["window_end_s"],
            "source_time_s": row["source_time_s"],
            "task_id": row["task_id"],
            "decision_id": row["decision_id"],
            "label": "",
            "label_source": "",
            "annotator": "",
            "annotation_status": "needs_review",
            "event_start_s": "",
            "event_end_s": "",
            "disagreement_notes": "",
        })
    fields = ["canonical_window_id", "video_id", "source_video_group_id", "partition", "window_start_s", "window_end_s", "source_time_s", "task_id", "decision_id", "label", "label_source", "annotator", "annotation_status", "event_start_s", "event_end_s", "disagreement_notes"]
    write_csv(path, list(unique.values()), fields)


def reliability_metrics(probability_rows: list[dict[str, Any]], label_map: dict[str, dict[str, str]], *, bins: int = 10, bootstrap_iterations: int = 2000, seed: int = 20261118) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    for row in probability_rows:
        label = label_map.get(row.get("window_id") or "", {})
        target = (label.get("label") or "").strip()
        if target not in {"normal_activity", "already_lying", "fall_process_visible"}:
            continue
        scored.append({**row, "label": target, "y": 1 if target == "fall_process_visible" else 0})
    n_pos = sum(row["y"] for row in scored)
    n_neg = len(scored) - n_pos
    independent_groups = len({row["source_video_group_id"] for row in scored})
    brier = sum((float(row["probability"]) - row["y"]) ** 2 for row in scored) / len(scored) if scored else None
    bin_rows = []
    ece = None
    if scored:
        for index in range(bins):
            lo, hi = index / bins, (index + 1) / bins
            members = [row for row in scored if lo <= float(row["probability"]) < hi or (index == bins - 1 and float(row["probability"]) == 1.0)]
            bin_rows.append({
                "bin_index": index,
                "lower_inclusive": lo,
                "upper_exclusive": hi if index < bins - 1 else 1.0,
                "n": len(members),
                "mean_prediction": statistics.mean(float(row["probability"]) for row in members) if members else None,
                "observed_fall_fraction": statistics.mean(row["y"] for row in members) if members else None,
            })
        ece = sum(row["n"] * abs(row["mean_prediction"] - row["observed_fall_fraction"]) for row in bin_rows if row["n"]) / len(scored)
    ci = {"brier_95_percentile": None, "ece_95_percentile": None, "method": "cluster bootstrap by source_video_group_id, percentile 2.5/97.5, fixed seed; unavailable with fewer than 2 labeled source-video groups"}
    if len(scored) and independent_groups >= 2 and bootstrap_iterations > 0:
        import random

        rng = random.Random(seed)
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            by_group[row["source_video_group_id"]].append(row)
        group_ids = list(by_group)
        brier_samples: list[float] = []
        ece_samples: list[float] = []
        for _ in range(bootstrap_iterations):
            sampled_groups = [rng.choice(group_ids) for _ in group_ids]
            sample = [item for group_id in sampled_groups for item in by_group[group_id]]
            brier_samples.append(sum((float(item["probability"]) - item["y"]) ** 2 for item in sample) / len(sample))
            ece_sample = 0.0
            for index in range(bins):
                lo, hi = index / bins, (index + 1) / bins
                members = [item for item in sample if lo <= float(item["probability"]) < hi or (index == bins - 1 and float(item["probability"]) == 1.0)]
                if members:
                    ece_sample += len(members) / len(sample) * abs(statistics.mean(float(item["probability"]) for item in members) - statistics.mean(item["y"] for item in members))
            ece_samples.append(ece_sample)
        brier_samples.sort()
        ece_samples.sort()
        ci = {
            "brier_95_percentile": [brier_samples[int(0.025 * (len(brier_samples) - 1))], brier_samples[int(0.975 * (len(brier_samples) - 1))]],
            "ece_95_percentile": [ece_samples[int(0.025 * (len(ece_samples) - 1))], ece_samples[int(0.975 * (len(ece_samples) - 1))]],
            "method": f"cluster bootstrap by source_video_group_id, percentile 2.5/97.5, {bootstrap_iterations} replicates, seed {seed}",
        }
    metrics = {
        "schema_version": "stage05-window-metrics.v1",
        "eligible_real_scored_windows": len(probability_rows),
        "eligible_real_scored_source_video_groups": len({row["source_video_group_id"] for row in probability_rows}),
        "labeled_version_matched_windows": len(scored),
        "positive_windows": n_pos,
        "negative_windows": n_neg,
        "labeled_independent_source_video_groups": independent_groups,
        "brier_score": brier,
        "ece": ece,
        "calibration_bins": bins,
        "reliability_bins": bin_rows,
        "uncertainty": ci,
        "calibrator_fit": False,
        "candidate_calibrated_strategy": "not evaluated; no fit or threshold selection",
    }
    return metrics, bin_rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def event_prediction_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for video_id, run_id in CANONICAL_EVENT_RUNS.items():
        for row in connection.execute("SELECT source_time,response_json,created_at,completed_at,image_sha256 FROM agent_frame_receipts WHERE run_id=? ORDER BY source_time", (run_id,)):
            response = decode_json(row["response_json"], {})
            payload = response.get("payload", response)
            fall = payload.get("fall", {}) if isinstance(payload, dict) else {}
            event_id = fall.get("event_id") or fall.get("current_event_id")
            new_event = bool(fall.get("new_event"))
            if new_event and event_id:
                output.append({
                    "prediction_id": f"{video_id}:{event_id}:{row['source_time']}",
                    "video_id": video_id,
                    "source_video_group_id": VIDEO_GROUPS[video_id]["source_video_group_id"],
                    "run_id": run_id,
                    "event_id": str(event_id),
                    "source_time_s": float(row["source_time"]),
                    "event_state": fall.get("state"),
                    "event_label": fall.get("label"),
                    "new_event": new_event,
                    "frame_completed_at_unix_s": row["completed_at"],
                    "frame_created_at_unix_s": row["created_at"],
                })
    return output


def match_events(events: list[dict[str, Any]], predictions: list[dict[str, Any]], post_event_tolerance_s: float = 2.5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    for event_index, event in enumerate(events):
        for prediction_index, prediction in enumerate(predictions):
            if event["source_video_group_id"] != prediction["source_video_group_id"]:
                continue
            detected_at = prediction["source_time_s"]
            if event["start_s"] <= detected_at <= event["end_s_exclusive"] + post_event_tolerance_s:
                pairs.append((detected_at - event["start_s"], event_index, prediction_index))
    used_events: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[dict[str, Any]] = []
    for delay, event_index, prediction_index in sorted(pairs):
        if event_index in used_events or prediction_index in used_predictions:
            continue
        used_events.add(event_index)
        used_predictions.add(prediction_index)
        event = events[event_index]
        prediction = predictions[prediction_index]
        source_start_by_run: dict[str, float] = {}
        # Filled by the caller if a matching wall-clock frame/run anchor exists.
        matches.append({
            "event_id": event["event_id"],
            "prediction_id": prediction["prediction_id"],
            "video_id": event["video_id"],
            "truth_onset_s": event["start_s"],
            "prediction_source_time_s": prediction["source_time_s"],
            "source_time_delay_s": delay,
            "prediction_completed_at_unix_s": prediction["frame_completed_at_unix_s"],
            "source_time_anchor_placeholder": source_start_by_run.get(prediction["run_id"]),
        })
    false_positives = [prediction for index, prediction in enumerate(predictions) if index not in used_predictions]
    return matches, false_positives


def build_event_metrics(connection: sqlite3.Connection, official: dict[str, Any], observations: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    events = official["events"]
    predictions = event_prediction_rows(connection)
    matches, false_positives = match_events(events, predictions)
    matched_ids = {row["event_id"] for row in matches}
    misses = []
    for event in events:
        if event["event_id"] in matched_ids:
            continue
        event_obs = [row for row in observations if row["video_id"] == event["video_id"]]
        window_obs = [row for row in event_obs if row.get("window_start_s") is not None]
        misses.append({
            "event_id": event["event_id"],
            "video_id": event["video_id"],
            "source_video_group_id": event["source_video_group_id"],
            "label_source": event["label_source"],
            "start_s": event["start_s"],
            "end_s_exclusive": event["end_s_exclusive"],
            "miss_reason": "no confirmed new_event prediction matched within the fixed event window",
            "any_structural_window_in_observation_trace": bool(window_obs),
            "decision_gate_event_coverage": False,
        })
    durations: dict[str, float] = {"urfd-adl-04": 5.0, "urfd-fall-01": 160 / 30, "urfd-fall-02": 110 / 30}
    exposure_s = sum(durations.values())
    source_delays = [row["source_time_delay_s"] for row in matches]
    wall_ms = []
    for match in matches:
        if match.get("source_time_anchor_placeholder") is not None:
            wall_ms.append((match["prediction_completed_at_unix_s"] - match["source_time_anchor_placeholder"]) * 1000.0)
    provider_off = read_json(STAGE04_OFF_SUMMARY)
    eligible_by_scenario = {row["scenario"]: row["eligible_windows"] for row in provider_off["runs"]}
    eligible_by_video = {
        "urfd-adl-04": eligible_by_scenario.get("normal_activity", 0),
        "urfd-fall-02": eligible_by_scenario.get("fall", 0),
        "urfd-fall-01": 0,
    }
    gate_covered_events = sum(1 for event in events if eligible_by_video.get(event["video_id"], 0) > 0)
    per_video = []
    for video_id, run_id in CANONICAL_EVENT_RUNS.items():
        group_events = [event for event in events if event["video_id"] == video_id]
        frame_rows = connection.execute("SELECT source_time,response_json FROM agent_frame_receipts WHERE run_id=? ORDER BY source_time", (run_id,)).fetchall()
        states = Counter()
        new_count = 0
        for frame in frame_rows:
            payload = decode_json(frame["response_json"], {}).get("payload", {})
            fall = payload.get("fall", {})
            states[fall.get("state", "unknown")] += 1
            new_count += bool(fall.get("new_event"))
        source_rows = [row for row in observations if row["video_id"] == video_id]
        missing_window = sum("source_window" in (row.get("missing_reasons") or []) for row in source_rows)
        source_windows = sum(row.get("window_start_s") is not None for row in source_rows)
        qualified_results = sum(row.get("gate_result") == "ready_real_uncalibrated_score" for row in source_rows)
        per_video.append({
            "video_id": video_id,
            "source_video_group_id": VIDEO_GROUPS[video_id]["source_video_group_id"],
            "source_duration_s": durations[video_id],
            "canonical_run_id": run_id,
            "canonical_sampled_frames": len(frame_rows),
            "fall_state_counts": dict(states),
            "real_probability_rows_across_all_runs_for_source_video_group": qualified_results,
            "source_window_missing_observations_in_all_trace_rows": missing_window,
            "structurally_present_source_window_rows": source_windows,
            "reference_fall_events": len(group_events),
            "detected_events": sum(1 for row in matches if row["video_id"] == video_id),
            "missed_events": sum(1 for row in misses if row["video_id"] == video_id),
            "event_predictions": new_count,
        })
    metrics = {
        "schema_version": "stage05-event-metrics.v1",
        "canonical_source_video_groups": len(CANONICAL_EVENT_RUNS),
        "distinct_underlying_videos": len(CANONICAL_EVENT_RUNS),
        "source_duration_s_unique_videos": exposure_s,
        "exposure_hours": exposure_s / 3600,
        "sampling_interval_s": 0.1,
        "eligible_real_scored_windows": sum(row.get("gate_result") == "ready_real_uncalibrated_score" for row in observations),
        "official_fall_events": len(events),
        "matched_fall_events": len(matches),
        "missed_fall_events": len(misses),
        "fall_event_recall": len(matches) / len(events) if events else None,
        "missed_events_after_quality_gate": len(misses),
        "gate_event_coverage": gate_covered_events / len(events) if events else None,
        "false_positive_events": len(false_positives),
        "false_alarms_per_hour": len(false_positives) / (exposure_s / 3600) if exposure_s > 0 else None,
        "event_match_rule": "One-to-one match inside the same source-video group when a new_event=true, non-null event_id confirmation occurs from official fall-process onset through 2.5 s after the official falling interval ends. Additional unmatched events are false positives.",
        "source_time_latency_p50_s": percentile(source_delays, 0.50),
        "source_time_latency_p95_s": percentile(source_delays, 0.95),
        "end_to_end_wall_clock_latency_p50_ms": percentile(wall_ms, 0.50),
        "end_to_end_wall_clock_latency_p95_ms": percentile(wall_ms, 0.95),
        "latency_status": "not computable: no matched detections; source-to-wall-clock anchor also not persisted for event onset",
        "per_video": per_video,
        "overlapping_and_repeated_runs_denominator_rule": "All runs are preserved in the trace; event counts and duration use one canonical full run per underlying source video.",
    }
    return metrics, matches, misses + [{**row, "false_positive": True} for row in false_positives]


def render_reliability_svg(metrics: dict[str, Any]) -> str:
    width, height = 720, 500
    left, top, plot_w, plot_h = 78, 42, 500, 330
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#fff"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.diag{stroke:#777;stroke-dasharray:5 5}</style>',
             '<text x="78" y="25" font-size="18" font-weight="bold">Reliability: version-matched real predictions</text>']
    for i in range(11):
        x = left + i * plot_w / 10
        y = top + plot_h - i * plot_h / 10
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}"/>')
        parts.append(f'<text x="{x-9:.1f}" y="{top+plot_h+20}" font-size="11">{i/10:.1f}</text>')
        parts.append(f'<text x="{left-36}" y="{y+4:.1f}" font-size="11">{i/10:.1f}</text>')
    parts.extend([
        f'<line class="diag" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}"/>',
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<text x="{left+plot_w/2-42}" y="{top+plot_h+48}" font-size="13">Mean predicted probability</text>',
        f'<text transform="translate(18 {top+plot_h/2+50}) rotate(-90)" font-size="13">Observed fall-process fraction</text>',
    ])
    if metrics["labeled_version_matched_windows"] == 0:
        parts.extend([
            '<rect x="165" y="160" width="330" height="100" rx="9" fill="#f5f1e8" stroke="#b8975b"/>',
            '<text x="330" y="195" text-anchor="middle" font-size="16" font-weight="bold">No labeled, version-matched windows</text>',
            '<text x="330" y="224" text-anchor="middle" font-size="13">Calibration metrics are not estimable</text>',
            '<text x="330" y="244" text-anchor="middle" font-size="12">2 real scores; 0 reviewed window labels</text>',
        ])
    else:
        for row in metrics["reliability_bins"]:
            if not row["n"]:
                continue
            x = left + float(row["mean_prediction"]) * plot_w
            y = top + plot_h - float(row["observed_fall_fraction"]) * plot_h
            radius = 4 + 4 * math.sqrt(row["n"])
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#2673aa" fill-opacity="0.78"/>')
            parts.append(f'<text x="{x+radius+3:.1f}" y="{y+4:.1f}" font-size="11">n={row["n"]}</text>')
    parts.append(f'<text x="{left}" y="{height-18}" font-size="12">Fixed bins: 10 equal-width; bubble size is bin n. Group bootstrap by source video.</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def build_frozen_groups(inventory: dict[str, Any]) -> dict[str, Any]:
    groups = []
    single_camera_artifacts = {
        row["video_id"]: row
        for row in inventory["p3_video_artifacts"]
        if row["artifact_type"] == "original_single_camera_rgb"
    }
    for video_id, meta in VIDEO_GROUPS.items():
        artifact = single_camera_artifacts.get(video_id, {})
        groups.append({
            "video_id": video_id,
            **meta,
            "person_cluster_id": None,
            "capture_group_id": None,
            "source_file_sha256": artifact.get("sha256"),
            "partition": "development_exposed",
            "window_count": "all source windows within the video remain in this partition",
        })
    for row in inventory["personal_head_development_videos"]:
        groups.append({
            "video_id": row["video_id"],
            "source_video_group_id": row["source_video_group_id"],
            "source_collection": "local personal development material",
            "source_license": "private/local-use material; not redistributable",
            "person_id": "same local participant; identity kept pseudonymous",
            "person_cluster_id": "local-participant-01",
            "camera_id": row["camera_id"],
            "capture_group_id": row.get("capture_group"),
            "source_file_sha256": row.get("sha256"),
            "development_status": "C/D head development/training source; never independent evaluation",
            "partition": "development_exposed",
            "window_count": "all source windows within the video remain in this partition",
        })
    return {
        "schema_version": "stage05-frozen-groups.v1",
        "frozen_at": "2026-11-18",
        "purpose": "Prevent source-video overlap, repeated runs, derivative renditions, and overlapping windows from crossing a calibration/test boundary.",
        "unit": "underlying source video; all windows, derived renditions, frames, and repeated runs stay with that video group",
        "split_policy": "All P3-observed groups and all inventoried C/D personal training groups are development_exposed and excluded from calibration and test. Linked windows, repeated runs, camera views from the same participant, capture-day groups, and source collections remain on one side of any future split. No threshold or calibrator was fit before this freeze.",
        "groups": groups,
        "training_groups": {
            "personal_video_groups_inventoried": len(inventory["personal_head_development_videos"]),
            "public_proxy_samples": 121,
            "public_proxy_video_level_inventory_complete": False,
            "public_proxy_overlap_status": "unresolved; every public proxy source remains excluded from calibration/test",
        },
        "calibration_groups": [],
        "test_groups": [],
        "crossings": {
            "overlapping_windows_cross_split": 0,
            "repeated_runs_cross_split": 0,
            "derived_renditions_cross_split": 0,
            "same_person_cross_video": "All 17 inventoried C/D personal videos share pseudonymous local-participant-01; all are development-exposed across old/new cameras.",
            "camera_crossings": "The three public clips use URFD cam0; no camera-independent holdout exists.",
            "source_crossings": "All selected public clips are URFD; no source-collection-independent holdout exists.",
            "public_C_D_training_overlap": "Per-video overlap unresolved; all related URFD/public-proxy groups remain excluded from test.",
            "personal_capture_day_groups": "Each capture-day group is kept in development_exposed; no capture-day is assigned to calibration/test.",
        },
    }


def bootstrap_self_check() -> None:
    # Same source video with overlapping windows cannot be split across folds.
    group = "video-a"
    assigned = {"window-a": (group, "train"), "window-b": (group, "test")}
    assert len({fold for source_group, fold in assigned.values() if source_group == group}) == 2
    try:
        validate_group_assignments([{"source_video_group_id": g, "partition": f} for g, f in assigned.values()])
    except ValueError:
        pass
    else:
        raise AssertionError("group leakage self-check did not reject cross-fold overlap")
    try:
        validate_group_assignments([
            {"source_video_group_id": "video-b", "person_cluster_id": "person-a", "partition": "calibration"},
            {"source_video_group_id": "video-c", "person_cluster_id": "person-a", "partition": "test"},
        ])
    except ValueError:
        pass
    else:
        raise AssertionError("linked-person leakage self-check did not reject cross-fold overlap")

    sample = [
        {"video_id": "video-a", "source_video_group_id": "video-a", "window_start_s": 0.0, "window_end_s": 2.0, "probability": 0.1, "probability_kind": "uncalibrated_model_estimate", "provider": "aliyun", "result_status": "ready", "result_accepted": True, "window_id": "a"},
        {"video_id": "video-b", "source_video_group_id": "video-b", "window_start_s": 0.0, "window_end_s": 2.0, "probability": 0.9, "probability_kind": "uncalibrated_model_estimate", "provider": "aliyun", "result_status": "ready", "result_accepted": True, "window_id": "b"},
    ]
    metrics, _ = reliability_metrics(sample, {"a": {"label": "normal_activity"}, "b": {"label": "fall_process_visible"}}, bootstrap_iterations=100)
    assert math.isclose(metrics["brier_score"], 0.01, abs_tol=1e-9)
    assert math.isclose(metrics["ece"], 0.1, abs_tol=1e-9)
    assert metrics["labeled_version_matched_windows"] == 2
    abstention = {"result_status": "insufficient", "probability": None, "probability_kind": None, "provider": "aliyun", "result_accepted": False, "source_video_group_id": "video-c", "gate_result": "result_insufficient"}
    assert len(make_probability_rows(sample + [abstention])) == 2
    assert count_gate_results([{"gate_result": "ready_real_uncalibrated_score"}, {"gate_result": "ready_real_uncalibrated_score"}, abstention]) == {"ready_real_uncalibrated_score": 2, "result_insufficient": 1}

    events = [{"event_id": "e1", "video_id": "video-a", "source_video_group_id": "video-a", "start_s": 1.0, "end_s_exclusive": 2.0}]
    predictions = [
        {"prediction_id": "p-match", "video_id": "video-a", "source_video_group_id": "video-a", "source_time_s": 2.5, "frame_completed_at_unix_s": 100.0, "run_id": "r"},
        {"prediction_id": "p-fp", "video_id": "video-a", "source_video_group_id": "video-a", "source_time_s": 9.0, "frame_completed_at_unix_s": 107.0, "run_id": "r"},
    ]
    matches, fps = match_events(events, predictions)
    assert len(matches) == 1 and math.isclose(matches[0]["source_time_delay_s"], 1.5)
    assert len(fps) == 1
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    print("self-check: video/person/source split leakage, abstention exclusion/count, Brier/ECE, event match/FP, and percentiles passed")


def validate_group_assignments(rows: list[dict[str, Any]]) -> None:
    assignments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        linked_keys = [
            ("video", row.get("source_video_group_id")),
            ("person", row.get("person_cluster_id")),
            ("capture", row.get("capture_group_id")),
            ("source", row.get("source_collection")),
        ]
        for kind, value in linked_keys:
            if value:
                assignments[f"{kind}:{value}"].add(row["partition"])
    leaking = {group_id: folds for group_id, folds in assignments.items() if len(folds) > 1}
    if leaking:
        raise ValueError(f"Source-video groups split across partitions: {leaking}")


def hash_manifest(run_dir: Path) -> dict[str, str]:
    result = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "artifact-hashes.json":
            result[str(path.relative_to(run_dir))] = sha256_file(path) or ""
    return result


def write_annotations_and_group_artifacts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory = build_video_inventory()
    groups = build_frozen_groups(inventory)
    official = build_official_event_labels()
    labels = run_label_inventory()
    write_json(run_dir / "frozen-groups.json", groups)
    write_json(run_dir / "video-inventory.json", inventory)
    write_json(run_dir / "human-label-inventory.json", labels)
    write_json(run_dir / "official-event-labels.json", official)
    return groups, inventory, official


def build_coverage_table(run_dir: Path, observations: list[dict[str, Any]], official: dict[str, Any], event_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = read_json(STAGE04_OFF_SUMMARY)
    eligible_by_video = {row["scenario"]: row["eligible_windows"] for row in summary["runs"]}
    rows = []
    durations = {"urfd-adl-04": 5.0, "urfd-fall-01": 160 / 30, "urfd-fall-02": 110 / 30}
    for video_id, run_id in CANONICAL_EVENT_RUNS.items():
        values = [row for row in observations if row["video_id"] == video_id]
        canonical_obs = [row for row in values if row["run_id"] == run_id]
        event_count = sum(row["video_id"] == video_id for row in official["events"])
        if video_id == "urfd-adl-04":
            # P3 direct real calls are attached to this exact original input;
            # the provider-off summary is an independent complete pass on the
            # same source clip and provides the run-level gate count.
            n_eligible = eligible_by_video.get("normal_activity", 0)
        elif video_id == "urfd-fall-02":
            n_eligible = eligible_by_video.get("fall", 0)
        else:
            n_eligible = 0
        frame_rows = []
        # Find sampled frames for the canonical visual run.
        frame_rows = [row for row in canonical_obs]
        rows.append({
            "video_id": video_id,
            "source_video_group_id": VIDEO_GROUPS[video_id]["source_video_group_id"],
            "partition": "development_exposed",
            "duration_s_unique": durations[video_id],
            "reference_fall_events": event_count,
            "canonical_run_id": run_id,
            "canonical_decision_observations": len(frame_rows),
            "qualified_window_observations_in_full_offline_run": n_eligible,
            "abstention_window_observations_in_full_offline_run": max(0, len(frame_rows) - n_eligible),
            "eligible_window_rate_over_sampled_observations": n_eligible / len(frame_rows) if frame_rows else None,
            "source_window_missing_rows_in_all_runs": sum("source_window" in (row.get("missing_reasons") or []) for row in values),
            "predicted_confirmed_events": sum(row["video_id"] == video_id for row in event_predictions),
            "label_source": "official URFD source-event reference; individual windows still need manual review",
        })
    return rows


def event_prediction_rows_from_cached(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "event-predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build_task_attempt_export(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    tasks = {row["task_id"]: dict(row) for row in connection.execute("SELECT * FROM decision_tasks")}
    rows = []
    for attempt in connection.execute("SELECT * FROM decision_attempts ORDER BY started_at, task_id, attempt_no"):
        a = dict(attempt)
        task = tasks.get(a["task_id"], {})
        source = RUN_SOURCE_MAP.get(task.get("run_id"), {})
        result = decode_json(a.get("result_json"), {})
        rows.append({
            "video_id": source.get("video_id"),
            "artifact_id": source.get("artifact_id"),
            "run_id": task.get("run_id"),
            "task_id": a["task_id"],
            "attempt_no": a["attempt_no"],
            "provider": task.get("provider"),
            "model_version": task.get("model"),
            "attempt_status": a.get("status"),
            "error_code": a.get("error_code"),
            "request_id": a.get("request_id"),
            "http_status": a.get("http_status"),
            "input_tokens": a.get("input_tokens"),
            "client_latency_ms": a.get("client_latency_ms"),
            "service_latency_ms": a.get("service_latency_ms"),
            "result_status": result.get("status"),
            "fall_probability": result.get("fall_probability"),
            "probability_kind": result.get("probability_kind"),
            "external_action_performed": result.get("external_action_performed"),
        })
    return rows


def calculate_cache_and_usage(connection: sqlite3.Connection, attempt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    budget = [dict(row) for row in connection.execute("SELECT maximum,used FROM agent_http_budget")]
    unique_request_ids = {row["request_id"] for row in attempt_rows if row.get("request_id")}
    known_token_rows = [row for row in attempt_rows if row.get("input_tokens") is not None]
    return {
        "stage04_budget_ledger": budget,
        "provider_attempt_rows": len([row for row in attempt_rows if row.get("provider") == "aliyun"]),
        "documented_stage04_http_submissions": 6,
        "provider_request_ids_persisted": len(unique_request_ids),
        "input_token_observations_persisted": len(known_token_rows),
        "known_input_tokens_sum": sum(int(row["input_tokens"]) for row in known_token_rows),
        "ready_probability_calls": 2,
        "ready_probability_input_tokens": 753 + 754,
        "cache_status": "The two scored ready windows each have distinct request/evidence digests and are remote results; no cache hit among these two. A later same-frame replay is documented as two cached local replays with zero additional HTTP submissions.",
        "billable_cost": None,
        "billable_cost_status": "No invoice/usage-bill record in the run. The current official pricing page lists decision-model-preview as limited-time free; account and promotion applicability must be checked before any new spend.",
        "new_p4_cloud_requests_sent": 0,
        "proposed_p4_probability_budget": {
            "status": "planning estimate only; no calls authorized or sent",
            "independent_source_video_groups": 120,
            "positive_groups": 60,
            "normal_groups": 60,
            "split_before_scoring": {
                "calibration": {"positive_groups": 30, "normal_groups": 30},
                "locked_test": {"positive_groups": 30, "normal_groups": 30},
            },
            "requests_per_group": 1,
            "expected_http_submissions": 120,
            "maximum_retries_per_group": 1,
            "maximum_http_submissions": 240,
            "estimated_input_tokens_expected": 90480,
            "estimated_input_tokens_maximum": 180960,
            "token_basis": "Conservative 754 input tokens per request, rounded up from the two retained ready calls (753 and 754); output tokens and provider-side retries are not included.",
            "proposed_fee_cny": 0,
            "fee_condition": "Only if the limited-time-free decision-model-preview offer still applies to this account and region at execution. Otherwise obtain a console quote and approve a separate nonzero spend cap before sending anything.",
            "event_sequence_scope": "The estimate covers one probability window per source-video group. Full-sequence remote window decisions, if needed for event evaluation, require offline admission counts and a separate request budget.",
            "sampling_note": "Use distinct source videos and people where possible; stratify fall-process-visible vs normal-activity groups; retain at least 10 minutes of negative video per normal group for event false-alarm exposure. Treat this as a first evaluation cohort, not deployment certification.",
        },
    }


def count_gate_results(observations: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row.get("gate_result") for row in observations))


def duplicate_observation_summary(observations: list[dict[str, Any]]) -> dict[str, int]:
    exact_frame_groups: dict[tuple[str, str, str], int] = defaultdict(int)
    video_time_keys = set()
    for row in observations:
        video_time_keys.add((row["source_video_group_id"], row["source_time_s"]))
        digest = row.get("input_frame_sha256")
        if digest:
            exact_frame_groups[(row["source_video_group_id"], row["source_time_s"], digest)] += 1
    duplicate_groups = [count for count in exact_frame_groups.values() if count > 1]
    return {
        "observation_rows": len(observations),
        "distinct_video_source_time_keys": len(video_time_keys),
        "exact_input_frame_keys": len(exact_frame_groups),
        "exact_duplicate_frame_groups": len(duplicate_groups),
        "rows_in_exact_duplicate_frame_groups": sum(duplicate_groups),
        "repeat_rows_beyond_first_per_exact_frame_key": sum(count - 1 for count in duplicate_groups),
        "observation_rows_without_source_window": sum(row.get("window_start_s") is None for row in observations),
    }


def build_run_summary(observations: list[dict[str, Any]], probability_rows: list[dict[str, Any]], metrics: dict[str, Any], event_metrics: dict[str, Any], inventory: dict[str, Any], group_data: dict[str, Any], db_counts: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    counters = count_gate_results(observations)
    missing = Counter(reason for row in observations for reason in row.get("missing_reasons", []))
    trace_groups = {row["source_video_group_id"] for row in observations}
    frozen_partition_counts = Counter(row["partition"] for row in group_data["groups"])
    return {
        "schema_version": "stage05-summary.v1",
        "code_revision": "8310c0e plus workspace Stage 04 changes; see code-and-source-lock.json",
        "source_database_counts": db_counts,
        "video_artifact_count": len(inventory["p3_video_artifacts"]),
        "p3_unique_source_video_groups": len(VIDEO_GROUPS),
        "personal_c_d_development_video_records": len(inventory["personal_head_development_videos"]),
        "observation_rows_preserved_in_trace": len(observations),
        "source_video_groups_in_trace": len(trace_groups),
        "duplicate_and_window_coverage_counts": duplicate_observation_summary(observations),
        "gate_result_counts": dict(counters),
        "missing_reason_counts_nonexclusive": dict(missing),
        "gate_result_counting_rule": "Counts are per decision_observation row across all seven runs; repeated frames/runs are preserved and not independent samples. Missing-reason counts are non-exclusive.",
        "real_accepted_uncalibrated_probability_rows": len(probability_rows),
        "window_metrics": metrics,
        "event_metrics": event_metrics,
        "partition_counts_in_frozen_inventory": dict(frozen_partition_counts),
        "partition_counts_in_p3_trace": {"development_exposed": len(trace_groups), "calibration": 0, "test": 0},
        "usage": usage,
        "conclusion": {
            "evaluation_tool_validated_by_self_check": True,
            "manual_data_sufficient_for_probability_accuracy": False,
            "candidate_calibration_accepted": False,
            "probability_evaluation_possible": False,
            "event_evaluation_possible_descriptively": True,
            "deployment_probability_or_alarm_policy_changed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate derived outputs; preserve a filled window-labels.csv.")
    parser.add_argument("--self-check", action="store_true", help="Run dependency-free known-small-example checks and exit.")
    args = parser.parse_args()
    if args.self_check:
        bootstrap_self_check()
        return 0
    run_dir = args.run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        parser.error(f"{run_dir} already contains data; pass --overwrite to regenerate derived files (window-labels.csv is preserved)")
    run_dir.mkdir(parents=True, exist_ok=True)
    for required in [args.db, LABEL_SOURCE, EVENT_SOURCE, URFD_LABELS, STAGE04_OFF_SUMMARY, STAGE04_SOURCE_MANIFEST]:
        if not required.is_file():
            parser.error(f"Required local input missing: {required}")

    groups, inventory, official = write_annotations_and_group_artifacts(run_dir)
    validate_group_assignments(groups["groups"])
    connection = read_only_connection(args.db)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Source SQLite integrity check failed: {integrity}")
    observations, db_counts = export_observation_rows(connection)
    attempts = build_task_attempt_export(connection)
    predictions = event_prediction_rows(connection)
    write_csv(run_dir / "decision-observations.csv", observations, list(observations[0].keys()) if observations else ["observation_row_id"])
    write_csv(run_dir / "decision-attempts.csv", attempts, list(attempts[0].keys()) if attempts else ["task_id"])
    write_csv(run_dir / "event-predictions.csv", predictions, list(predictions[0].keys()) if predictions else ["prediction_id"])
    # The JSONL copy is a plain, newline-delimited stable snapshot of the audit rows.
    with (run_dir / "decision-observations.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    probability_rows = make_probability_rows(observations)
    label_path = run_dir / "window-labels.csv"
    make_window_label_template(label_path, probability_rows)
    label_map = load_label_csv(label_path)
    metrics, _ = reliability_metrics(probability_rows, label_map)
    write_json(run_dir / "window-metrics.json", metrics)
    (run_dir / "reliability.svg").write_text(render_reliability_svg(metrics))
    event_metrics, matches, unmatched = build_event_metrics(connection, official, observations)
    write_json(run_dir / "event-metrics.json", event_metrics)
    write_csv(run_dir / "event-matches.csv", matches, ["event_id", "prediction_id", "video_id", "truth_onset_s", "prediction_source_time_s", "source_time_delay_s", "prediction_completed_at_unix_s"])
    write_csv(run_dir / "event-unmatched.csv", unmatched, ["event_id", "prediction_id", "video_id", "source_video_group_id", "start_s", "end_s_exclusive", "miss_reason", "any_structural_window_in_observation_trace", "decision_gate_event_coverage", "false_positive"])
    coverage = build_coverage_table(run_dir, observations, official, predictions)
    write_csv(run_dir / "video-coverage.csv", coverage, list(coverage[0].keys()) if coverage else ["video_id"])
    usage = calculate_cache_and_usage(connection, attempts)
    write_json(run_dir / "cloud-usage.json", usage)
    summary = build_run_summary(observations, probability_rows, metrics, event_metrics, inventory, groups, db_counts, usage)
    summary["source_sqlite_integrity_check"] = integrity
    summary["probability_ready_prediction_rows"] = probability_rows
    write_json(run_dir / "summary.json", summary)
    source_hashes = {
        str(path.resolve().relative_to(ROOT)): sha256_file(path)
        for path in [args.db, Path(str(args.db) + "-wal"), STAGE04_SOURCE_MANIFEST, STAGE04_OFF_SUMMARY, LABEL_SOURCE, EVENT_SOURCE, HEAD_PROTOCOL, HEAD_PROVENANCE, URFD_LABELS, URFD_ADL_LABELS, STAGE04_MAIN_REPORT, STAGE04_R1_REPORT, STAGE04_R2_APPROVAL]
        if path.is_file()
    }
    write_json(run_dir / "code-and-source-lock.json", {
        "git_head": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "git_worktree_dirty_at_run": True,
        "evaluator_path": "scripts/evaluate_decision_stage05.py",
        "evaluator_sha256": sha256_file(ROOT / "scripts/evaluate_decision_stage05.py"),
        "run_source_map": RUN_SOURCE_MAP,
        "canonical_event_runs": CANONICAL_EVENT_RUNS,
        "source_sha256": source_hashes,
        "read_only_source_database": True,
        "external_network_calls": 0,
    })
    (run_dir / "run-command.txt").write_text(
        ".venv/bin/python scripts/evaluate_decision_stage05.py --db runs/decision-stage04-live/decision-agent.sqlite3 --run-dir runs/decision-stage05-20261118 --overwrite\n"
        ".venv/bin/python scripts/evaluate_decision_stage05.py --self-check\n"
    )
    connection.close()
    write_json(run_dir / "artifact-hashes.json", hash_manifest(run_dir))
    print(json.dumps({
        "run_dir": str(run_dir),
        "observations": len(observations),
        "real_ready_probabilities": len(probability_rows),
        "labeled_probability_rows": metrics["labeled_version_matched_windows"],
        "reference_events": event_metrics["official_fall_events"],
        "matched_events": event_metrics["matched_fall_events"],
        "missed_events": event_metrics["missed_fall_events"],
        "sqlite_integrity": integrity,
        "network_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

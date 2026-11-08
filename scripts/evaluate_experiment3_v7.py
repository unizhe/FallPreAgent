"""Frozen V7 source-time audit from existing logs; no model inference or training.

The first-output rule scans each complete input from its first observation. Event
annotations are joined only after scanning and never gate or select a detection.
This is a retrospective development audit, not a preregistered prediction study.
"""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from evaluate_abc_v7 import (
    LABELS,
    ROOT,
    aggregate,
    digest,
    evaluate_clip,
    normalize,
    read_json,
    read_rows,
    reference_at,
    write_csv,
    write_json,
)

BASE = ROOT / "runs/abc-v7-20261020"
OUT = BASE / "experiment3-20261021"
MODEL_HASH = "f6281a191b3ec3a3de7f649a82b436fa2d4d69d036882fcd85fbc21d85e0169e"
REVIEW_HASH = "3663d8d2d208c116fd975127478ec6c4441bfe3e40577e5ca546e20eacf1b257"


def first_output(rows, field="label", value="falling", *, enabled=True, seconds=True):
    """Return first emitted value, including output outside reference intervals."""
    if not enabled:
        return None
    for row in rows:
        if row["prediction"].get(field) == value:
            return dict(frame=row["source_frame_id"],
                        time_s=row["source_time_s"] if seconds else None,
                        label=row["prediction"].get("label"),
                        phase=row["prediction"].get("phase"),
                        field=field, value=value)
    return None


def boundary_margin(boundary, detection):
    """Boundary minus detection: positive means detection precedes boundary."""
    if not boundary or not detection or detection.get("time_s") is None:
        return None
    bounds = boundary.get("time_interval_s")
    if bounds is None and boundary.get("time_s") is not None:
        bounds = [boundary["time_s"], boundary["time_s"]]
    if bounds is None:
        return None
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError("Invalid boundary interval")
    return [round(t - detection["time_s"], 9) for t in bounds]


def position(detection, onset_proxy, contact):
    if detection is None:
        return "no_detection"
    t = detection.get("time_s")
    onset = onset_proxy.get("time_s") if onset_proxy else None
    bounds = contact.get("time_interval_s") if contact else None
    if t is None or onset is None or bounds is None:
        return "unresolved_boundary_or_clock"
    if t < onset:
        return "before_official_onset_proxy"
    if t < bounds[0]:
        return "onset_proxy_to_contact"
    if t <= bounds[1]:
        return "within_contact_uncertainty"
    return "after_contact"


def derive_stable_confirmation(landmark, clock):
    """Availability of a persistence criterion is later than its onset label."""
    bounds = landmark.get("time_interval_s")
    duration = landmark.get("stability_criterion", {}).get("minimum_duration_s")
    if bounds is None or duration is None:
        return None
    thresholds = [round(t + duration, 9) for t in bounds]
    observations = []
    for threshold in thresholds:
        candidate = next((dict(frame=f, time_s=t) for f, t in
                          zip(clock["original_frame_ids"], clock["source_times_s"])
                          if t + 1e-12 >= threshold), None)
        observations.append(candidate)
    return dict(status="derived_not_independently_human_confirmed", human_reviewed=False,
                stability_onset_interval_s=bounds, minimum_duration_s=duration,
                continuous_threshold_interval_s=thresholds,
                available_observations_for_lower_and_upper=observations,
                exact_confirmation_frame=None, exact_confirmation_time_s=None,
                note="Derived earliest observation at/after each endpoint + 0.5 s. This is not an observed model detection or an exact human-confirmed event.")


def pending_items(annotations):
    result = []
    for entry in annotations["entries"]:
        cid = entry["clip_id"]
        for i, reason in enumerate(entry.get("review_pending", [])):
            result.append(dict(clip_id=cid, category="entry_review_requirement",
                               item=f"review_pending[{i}]", status="pending", reason=reason))
        for key, landmark in entry.get("event_landmarks", {}).items():
            if not landmark.get("human_reviewed"):
                result.append(dict(clip_id=cid, category="event_landmark", item=key,
                                   status=landmark.get("status", "pending"),
                                   reason=landmark.get("note") or "No explicit human decision confirms this landmark.",
                                   frame=landmark.get("frame"), time_s=landmark.get("time_s")))
            elif landmark.get("frame") is None:
                result.append(dict(clip_id=cid, category="remaining_precision", item=key,
                                   status="interval_confirmed_exact_unknown",
                                   reason="Human confirmed only the interval; exact frame/time remain unknown.",
                                   frame_interval=landmark.get("frame_interval"),
                                   time_interval_s=landmark.get("time_interval_s")))
        for kind in ["action_segments", "state_segments", "phase_segments", "visibility_segments", "official_state_segments"]:
            for i, segment in enumerate(entry.get(kind, [])):
                if not segment.get("human_reviewed"):
                    result.append(dict(clip_id=cid, category=kind, item=str(i),
                                       status=segment.get("status", "pending"),
                                       start_frame=segment["start_frame"], end_frame=segment["end_frame"],
                                       label=segment.get("label", segment.get("visibility")),
                                       reason=segment.get("note") or "Not independently human reviewed; official/provisional provenance retained."))
    return result


def pending_summary_zh():
    return [
        dict(clip_id="urfd-fall-01-original", item="接地、稳定倒地的剩余精度", status="区间已确认，精确时刻未知",
             detail="接地保留 101–107 帧，稳定倒地起点保留 132–138 帧；精确起点及满足 0.5 秒持续条件的精确确认时刻均未知。"),
        dict(clip_id="urfd-fall-01-original", item="真正起跌与动作区间", status="待人工复核",
             detail="83 帧仍仅为官方起跌代理；113–137 帧翻滚区间未评分，138–160 帧 background/lying 仍是暂定区间，不能由稳定倒地起点确认自动升级。"),
        dict(clip_id="urfd-fall-01", item="低清版与原始版逐帧对应", status="待人工复核",
             detail="属于同一跌倒事件，但逐帧映射未人工核准；接地、稳定倒地的人工确认没有传播到低清版。"),
        dict(clip_id="utk-s01_e01-transition", item="物理离座、阶段与时间轴", status="仅确认可见上升代理",
             detail="只确认 722 帧为首次明确可见上升观测；物理离座、真正上升开始、前倾/上升/站稳的精确边界及帧号差÷30 时间轴仍待定。"),
        dict(clip_id="utk-s10_e01-transition", item="物理离座、阶段与时间轴", status="仅确认可见上升代理",
             detail="只确认 790 帧为首次明确可见上升观测；物理离座、真正上升开始、前倾/上升/站稳的精确边界及帧号差÷30 时间轴仍待定。"),
        dict(clip_id="urfd-fall-02", item="事件边界与原始时钟", status="待人工复核与源时钟核实",
             detail="40 帧仅为官方起跌代理；接地和稳定倒地未定，70–99 帧后续翻滚未评分；原始物理时间待核实，仅比较源帧差。"),
        dict(clip_id="urfd-adl-04 / urfd-adl-07", item="日常动作、可见性及非跌倒对照", status="未完成独立人工复核",
             detail="继承动作边界、人体可见区间、静坐负例及名义 30 FPS 时间轴未全部核准；局部已标注区间内零 falling 不能推广为全视频零误报。"),
        dict(clip_id="multicam-chute01-cam8", item="时钟、身份与事件边界", status="帧级评估；动作停用",
             detail="真实帧率未核实；人体可见/空房、遮挡、多人身份、接地与稳定倒地仍待复核。源解码问题不能由派生视频可播放证明消失。"),
        dict(clip_id="all", item="独立人体、骨架与阶段真值", status="未建立完整真值",
             detail="本轮四项决定只覆盖两个事件区间和两个代理观测；其余动作、阶段、可见性、目标身份及骨架真值尚未整体完成独立人工复核。"),
    ]


def build(out=OUT):
    reviewed = BASE / "annotations/annotations.reviewed.json"
    original = BASE / "annotations/annotations.json"
    original_protocol = read_json(BASE / "evaluation-protocol.json")
    session = read_json(BASE / "human-review/review-session.json")
    assert digest(reviewed) == REVIEW_HASH == session["annotation_sha256_current"]
    assert digest(original) == original_protocol["annotations_sha256"]
    assert digest(ROOT / "runs/action-v7/training/model.json") == MODEL_HASH == original_protocol["model_sha256"]
    annotations = read_json(reviewed)
    assert session["completed_cases"] == 4 and len(session["proxy_only_cases"]) == 2
    entries = {e["clip_id"]: e for e in annotations["entries"]}
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "annotations.reviewed.snapshot.json"
    if snapshot.exists():
        assert digest(snapshot) == REVIEW_HASH
    else:
        snapshot.write_bytes(reviewed.read_bytes())

    # Lock the already established rule before this audit scans any predictions.
    protocol = dict(
        schema="fallpre.experiment3_v7.protocol.v1",
        created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        scope="Experiment 3A: retrospective source-time detection audit; future prediction 3B deferred",
        training=False, new_model_inference=False, new_visual_inference=False,
        model_sha256=MODEL_HASH, annotation_sha256=REVIEW_HASH,
        annotation_snapshot=str(snapshot.relative_to(ROOT)),
        frozen_baseline_annotation_sha256=original_protocol["annotations_sha256"],
        groups=original_protocol["groups"], source_logs=original_protocol["source_logs"],
        trigger=dict(field="prediction.label", value="falling", required_observations=1,
                     debounce_s=0, scan="complete input from first observation in source order",
                     reference_gate=False, future_smoothing=False,
                     status="existing raw-output rule restated before this audit; not retrospective preregistration",
                     note="No trigger threshold was selected or optimized against reviewed boundaries. First detections were already known in experiment 1."),
        auxiliary_triggers=["first prediction.label=standing_up", "first prediction.phase=rising", "first prediction.phase=forward_lean"],
        timing=dict(sign="boundary minus detection; positive margin means earlier source-time detection",
                    actual_alert_delivery_time_s=None, actual_end_to_end_latency_s=None,
                    actual_available_warning_s=None,
                    frame_only_event_audits=["urfd-fall-02", "urfd-fall-01"],
                    multicam="action disabled; no seconds or timing comparisons",
                    utk="source-frame delta/30 relative clock; visible-rise proxy only, physical seat-off unknown",
                    runtime="historical recovery.wall_s is recovery-only; separate benchmark may measure missing full-chain processing",
                    stable="human interval marks onset; 0.5 s persistence confirmation is a separate derived threshold"),
        independence="8 input streams, 7 event groups; fall01 original is primary and derivative secondary. Development/train overlap, small selected examples, no generalization or calibrated prediction claim.",
        source_causality_evidence=["scripts/replay_public_samples.py: frames -> current-image detection -> tracker.update -> V7 -> reference annotation",
                                   "scripts/compare_evidence_recovery.py: cached current native observations + current-image recovery -> tracker/V7",
                                   "src/fallpre/weekly/action_v7.py: current/past histories; V2 TCN uses left-only padding",
                                   "runs/abc-v7-20261020/validation.json: previous 7449 exact prediction replay validation"],
    )
    write_json(out / "protocol.json", protocol)

    all_streams, timeline_streams, primary_fall, utk, fall02, paired = [], [], [], [], [], []
    metrics, events, validations = [], [], []
    for cid, annotation in entries.items():
        ids = annotation["clock"]["original_frame_ids"]
        expected_times = annotation["clock"]["source_times_s"]
        for mode in ["A", "B", "C"]:
            record = original_protocol["source_logs"][f"{mode}/{cid}"]
            assert digest(record["file"]) == record["sha256"]
            rows = [normalize(mode, r) for r in read_rows(record["file"])]
            assert [r["source_frame_id"] for r in rows] == ids
            if expected_times is None:
                assert all(r["source_time_s"] is None for r in rows)
            else:
                assert all(abs(r["source_time_s"] - t) < 1e-9 for r, t in zip(rows, expected_times))
                assert all(a < b for a, b in zip(expected_times, expected_times[1:]))
            enabled = annotation["clock"]["unit"] != "frame"
            seconds = enabled and cid not in {"urfd-fall-01", "urfd-fall-02"}
            first = first_output(rows, enabled=enabled, seconds=seconds)
            metric, clip_events, details = evaluate_clip(mode, rows, annotation)
            metrics.append(metric)
            events.extend(clip_events)
            refs = [reference_at(annotation["action_segments"], r["source_frame_id"]) for r in rows]
            states = [reference_at(annotation["state_segments"], r["source_frame_id"], "state") for r in rows]
            falling_mask = [enabled and r["prediction"]["label"] == "falling" for r in rows]
            stream = dict(mode=mode, clip_id=cid, event_group_id=annotation["event_group_id"],
                          primary=annotation["primary_event_representation"], samples=len(rows),
                          clock_status=annotation["clock"]["status"], action_enabled=enabled,
                          event_seconds_eligible=seconds, first_falling=first,
                          falling_samples=sum(falling_mask) if enabled else None,
                          annotated_negative_falling_samples=sum(flag and ref in LABELS[:-1] and ref != "falling"
                                                                 for flag, ref in zip(falling_mask, refs)) if enabled else None,
                          annotated_negative_falling_episodes=metric["false_action_episodes"]["falling"] if enabled else None,
                          unscored_falling_samples=metric["unscored_falling_samples"],
                          unscored_falling_source_frames=[r["source_frame_id"] for r, flag, ref in zip(rows, falling_mask, refs)
                                                         if flag and ref not in LABELS[:-1]],
                          unknown_samples=metric["unknown_samples"],
                          negative_reference_status="official/provisional intervals; not all independently human reviewed",
                          recovery_wall_s_total=sum(r["recovery_wall_s"] for r in rows if r["recovery_wall_s"] is not None) if mode == "C" else None,
                          complete_processing_time_s=None, actual_delivery_time_s=None)
            all_streams.append(stream)
            timeline_streams.append(dict(clip_id=cid, mode=mode, clock=annotation["clock"]["status"],
                time_semantics="log source or nominal relative time; event eligibility is in timing.json; MC has null time",
                action_enabled=enabled, rows=[dict(frame=r["source_frame_id"], time_s=r["source_time_s"],
                    label=r["prediction"]["label"], phase=r["prediction"].get("phase"), observed=r["observed"],
                    epoch=r["epoch"], reference_action=ref, reference_state=state,
                    recovery_wall_s=r["recovery_wall_s"])
                    for r, ref, state in zip(rows, refs, states)]))

            lm = annotation["event_landmarks"]
            if cid == "urfd-fall-01-original":
                primary_fall.append(dict(mode=mode, clip_id=cid, event_group_id=annotation["event_group_id"],
                    first_falling=first, onset_proxy=lm["onset_proxy"], contact=lm["contact"], stable_lying=lm["stable_lying"],
                    delay_from_onset_proxy_s=round(first["time_s"] - lm["onset_proxy"]["time_s"], 9) if first else None,
                    contact_margin_s=boundary_margin(lm["contact"], first),
                    stable_onset_margin_s=boundary_margin(lm["stable_lying"], first),
                    classification=position(first, lm["onset_proxy"], lm["contact"]),
                    classification_note="Classification uses official onset proxy, not an independently reviewed physical onset.",
                    recovery_wall_s_through_first_detection=sum(r["recovery_wall_s"] for r in rows
                        if r["recovery_wall_s"] is not None and first and r["source_frame_id"] <= first["frame"]) if mode == "C" else None,
                    actual_warning_margin_s=None, alert_delivery_time_s=None))
            elif cid.startswith("utk-"):
                proxy = lm["rise_visible_proxy"]
                detection = first_output(rows, value="standing_up")
                rising = first_output(rows, "phase", "rising")
                lean = first_output(rows, "phase", "forward_lean")
                assert all(lm["seat_off"].get(key) is None for key in ["frame", "time_s", "frame_interval", "time_interval_s"])
                utk.append(dict(mode=mode, clip_id=cid, event_group_id=annotation["event_group_id"],
                    first_standing_up=detection, first_rising=rising, first_forward_lean=lean,
                    visible_rise_proxy=proxy, physical_seat_off=lm["seat_off"],
                    delay_to_visible_proxy_s=round(detection["time_s"] - proxy["time_s"], 9) if detection else None,
                    delay_to_visible_proxy_frames=detection["frame"] - proxy["frame"] if detection else None,
                    rising_phase_delay_to_visible_proxy_s=round(rising["time_s"] - proxy["time_s"], 9) if rising else None,
                    forward_lean_phase_delay_to_visible_proxy_s=round(lean["time_s"] - proxy["time_s"], 9) if lean else None,
                    physical_seat_off_delay_s=None,
                    seated_phase_diagnostic=dict(reference_status="ai_visual_provisional_not_human_reviewed",
                        seated_reference_samples=metric["seated_reference_samples"],
                        seated_forward_lean_samples=metric["seated_forward_lean_samples"],
                        seated_false_rise_samples=metric["seated_false_rise_samples"],
                        first_forward_lean_reference_state=reference_at(annotation["state_segments"], lean["frame"], "state") if lean else None),
                    note="Visible proxy timing is not physical seat-off delay. Early forward_lean is not a validated anticipatory detection."))
            elif cid in {"urfd-fall-01", "urfd-fall-02"}:
                item = dict(mode=mode, clip_id=cid, event_group_id=annotation["event_group_id"],
                    primary=annotation["primary_event_representation"], first_falling=first,
                    official_onset_proxy_frame=lm["onset_proxy"]["frame"],
                    delay_from_official_onset_proxy_frames=first["frame"] - lm["onset_proxy"]["frame"] if first else None,
                    contact_margin_s=None, physical_warning_margin_s=None,
                    clock_note="Frame difference only; nominal video times not assigned as physical event timing.",
                    human_contact_mapping_confirmed=False)
                (fall02 if cid == "urfd-fall-02" else paired).append(item)
            # Each first-match output must be unchanged when future observations are removed.
            for field, value in [("label", "falling"), ("label", "standing_up"), ("phase", "rising"), ("phase", "forward_lean")]:
                match = first_output(rows, field, value, enabled=enabled, seconds=seconds)
                if match:
                    cutoff = next(i for i, r in enumerate(rows) if r["source_frame_id"] == match["frame"])
                    assert first_output(rows[:cutoff + 1], field, value, enabled=enabled, seconds=seconds) == match
                    assert first_output(rows[:cutoff], field, value, enabled=enabled, seconds=seconds) is None
            validations.append(dict(mode=mode, clip_id=cid, source_hash_matches=True,
                                    source_frame_and_clock_match=True, samples=len(rows), scanner_prefix_check=True))

    totals = aggregate(metrics, events)
    assert len(all_streams) == 24 and sum(s["samples"] for s in all_streams) == 7449
    assert len({e["event_group_id"] for e in entries.values()}) == 7
    source_aggregates = {mode: {key: value for key, value in row.items() if key in {
        "unique_clocked_groups", "observations", "unscored_falling_samples", "unscored_falling_s",
        "false_action_episodes", "false_action_s", "unknown_s", "unscored_action_s", "annotated_action_s",
        "seated_false_rise_samples", "seated_forward_lean_samples", "seated_reference_samples"}}
        for mode, row in totals.items()}
    stable = derive_stable_confirmation(entries["urfd-fall-01-original"]["event_landmarks"]["stable_lying"],
                                       entries["urfd-fall-01-original"]["clock"])
    result = dict(schema="fallpre.experiment3_v7.timing.v1", primary_fall=primary_fall, utk=utk,
                  fall02_frame_only=fall02, paired_low_resolution=paired, all_streams=all_streams,
                  aggregates=source_aggregates, stable_confirmation_derived=stable,
                  limitations=["Source-time detection margin is not a measured delivered-warning margin.",
                      "Physical onset is not independently reviewed; official frame 83 is a proxy.",
                      "UTK physical seat-off remains unknown; frame 722/790 are visible observation proxies only.",
                      "Contact and stability onsets retain intervals, not fabricated exact timestamps.",
                      "All action, phase, visibility, and seated-negative intervals still require independent human review.",
                      "MultiCam action disabled and frame-only. Fall02 and low-res fall01 event timing only by source frame.",
                      "Aggregated duration diagnostics reuse experiment 1's mixed source/nominal clocks, not physical warning timing.",
                      "Selected development/train-overlap examples; no independent generalization, clinical or future-prediction validation."])
    write_json(out / "timing.json", result)
    flat = []
    for r in primary_fall:
        flat.append(dict(mode=r["mode"], clip_id=r["clip_id"], trigger="label=falling", frame=r["first_falling"]["frame"],
            time_s=r["first_falling"]["time_s"], delay_from_onset_proxy_s=r["delay_from_onset_proxy_s"],
            contact_margin_lower_s=r["contact_margin_s"][0], contact_margin_upper_s=r["contact_margin_s"][1],
            stable_onset_margin_lower_s=r["stable_onset_margin_s"][0], stable_onset_margin_upper_s=r["stable_onset_margin_s"][1],
            classification=r["classification"], actual_warning_margin_s=None))
    for r in utk:
        for key in ["first_standing_up", "first_rising", "first_forward_lean"]:
            hit = r[key]
            flat.append(dict(mode=r["mode"], clip_id=r["clip_id"], trigger=key,
                frame=hit["frame"] if hit else None, time_s=hit["time_s"] if hit else None,
                delay_to_visible_proxy_s=round(hit["time_s"] - r["visible_rise_proxy"]["time_s"], 9) if hit else None,
                physical_seat_off_delay_s=None, interpretation="proxy-only; no physical seat-off claim"))
    for r in fall02 + paired:
        flat.append(dict(mode=r["mode"], clip_id=r["clip_id"], trigger="label=falling", frame=r["first_falling"]["frame"] if r["first_falling"] else None,
            time_s=None, delay_from_onset_proxy_frames=r["delay_from_official_onset_proxy_frames"],
            interpretation="frame-only event audit; no physical seconds"))
    write_csv(out / "timing.csv", flat)
    write_json(out / "timelines.json", dict(schema="fallpre.experiment3_v7.timelines.v1", streams=timeline_streams))
    review = dict(schema="fallpre.experiment3_v7.review_summary.v1", completed_cases=4,
                  status=session["status"], all_annotations_human_reviewed=False,
                  completed_cases_meaning=session["completed_cases_meaning"],
                  proxy_only_cases=session["proxy_only_cases"], annotation_sha256=REVIEW_HASH,
                  cases=[dict(id=q["id"], clip_id=q["clip_id"], status=q["status"],
                              human_reviewed=q["human_reviewed"], decision=deepcopy(q["decision"])) for q in session["queue"]])
    write_json(out / "review-summary.json", review)
    pending = pending_items(annotations)
    write_json(out / "pending-annotations.json", dict(schema="fallpre.experiment3_v7.pending.v1",
        note="Explicit pending requirements and every unreviewed segment are retained. Confirmed interval precision remains unresolved; 4/4 decisions does not mean all annotations are verified.",
        summary_zh=pending_summary_zh(),
        items=pending))
    write_csv(out / "pending-annotations.csv", pending)
    assert digest(reviewed) == REVIEW_HASH and digest(snapshot) == REVIEW_HASH
    assert digest(original) == original_protocol["annotations_sha256"]
    assert digest(ROOT / "runs/action-v7/training/model.json") == MODEL_HASH
    for record in original_protocol["source_logs"].values():
        assert digest(record["file"]) == record["sha256"]
    write_json(out / "cache-validation.json", dict(schema="fallpre.experiment3_v7.cache_validation.v1",
        status="passed", streams_checked=24, observations_checked=7449,
        model_and_source_logs_unchanged=True, reviewed_snapshot_matches=True,
        original_annotations_unchanged=True, scanner_prefix_checks_passed=True,
        scope="Cache scanner and source integrity checks; previous exact V7 inference replay is preserved separately.",
        streams=validations))
    print(f"Wrote {out}: 24 streams, 7449 observations, 4 reviewed decisions, {len(pending)} pending entries.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    build(parser.parse_args().output)

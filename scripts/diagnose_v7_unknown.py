"""Audit frozen V7 action histories from saved selected poses; no perception run.

This is an accounting decomposition, not an independently randomized causal
ablation: wall-clock scheduling selected different source frames in each trial.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fallpre.weekly.action_v7 import CompletionModel, OnlineCompletion
from fallpre.weekly.pose_refinement import predict_with_pose_quality


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def streams(online):
    result = {}
    while online is not None:
        version = type(online).__module__.rsplit("_", 1)[-1]
        result[version] = online
        online = getattr(online, "baseline", None)
    return result


def snapshot(stream):
    history = list(stream.history)
    valid = [r for r in history if r["instances"]]
    return dict(
        epoch=stream.epoch, last_t=getattr(stream, "last_t", None),
        last_seen=getattr(stream, "last_seen", None),
        history_frames=[r.get("source_frame_id") for r in history],
        history_times=[r["t"] for r in history],
        valid_history_frames=[r.get("source_frame_id") for r in valid],
        valid_history_times=[r["t"] for r in valid],
    )


def reset_reasons(version, before, row):
    """Mirror reset predicates only; actual histories are independently replayed."""
    t = row["t"]
    reasons = []
    if row.get("target_epoch") != before["epoch"]:
        reasons.append("target_epoch_changed")
    previous = before["history_times"][-1] if before["history_times"] else None
    if version == "v3":
        if previous is not None and t - previous > .35:
            reasons.append("adjacent_processed_gap_gt_0_35")
        # OnlineMotion.reset clears last_seen before its second reset check.
        if not reasons and row["instances"] and before["last_seen"] is not None and t - before["last_seen"] > .35:
            reasons.append("last_valid_observation_gap_gt_0_35")
    else:
        last_t = before["last_t"]
        if last_t is not None and t - last_t > .35:
            reasons.append("adjacent_processed_gap_gt_0_35")
        if version in ("v6", "v7"):
            if last_t is not None and t <= last_t:
                reasons.append("nonincreasing_time")
            if not row["instances"]:
                reasons.append("missing_action_observation")
        if version == "v4" and row["instances"] and before["last_seen"] is not None and t - before["last_seen"] > .35:
            reasons.append("last_valid_observation_gap_gt_0_35")
    return reasons


def audit_trial(path, model):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    online = OnlineCompletion(model)
    chain = streams(online)
    # Count actual method invocations without changing the frozen update path.
    # These reset methods do not receive their cause; record the callsite and
    # state at invocation, then infer the predicate from that captured state.
    reset_calls = {version: [] for version in ("v3", "v4")}
    active_row = {}

    def instrument_reset(version):
        stream = chain[version]
        original = stream.reset

        def capture(*args, **kwargs):
            caller = sys._getframe(1)
            state = snapshot(stream)
            event = dict(
                caller_file=caller.f_code.co_filename,
                caller_function=caller.f_code.co_name,
                caller_line=caller.f_lineno,
                supplied_args=list(args), supplied_kwargs=kwargs,
                before=state,
                inferred_reasons=reset_reasons(version, state, active_row),
            )
            result = original(*args, **kwargs)
            event["after"] = snapshot(stream)
            reset_calls[version].append(event)
            return result

        stream.reset = capture

    for version in reset_calls:
        instrument_reset(version)
    frames = []
    counts = Counter()
    mismatch = []
    history_mismatch = []
    actual_call_mismatch = []
    warmup_origin = None
    ever_valid = False
    ever_known = False
    last_valid = None
    reset_counts = {version: Counter() for version in chain}
    for record in records:
        fid = record["source_frame_id"]
        selected = record["selected_instance"]
        row = dict(t=record["source_time_s"], source_frame_id=fid,
                   target_epoch=record["target_epoch"],
                   instances=[] if selected is None else [selected])
        withheld = bool(selected is not None and selected.get("action_eligible") is False)
        effective = dict(row, instances=[]) if withheld else row
        active_row.clear()
        active_row.update(effective)
        for calls in reset_calls.values():
            calls.clear()
        before = {version: snapshot(stream) for version, stream in chain.items()}
        reasons = {version: reset_reasons(version, before[version], effective)
                   for version in chain}
        prediction = predict_with_pose_quality(online, row)
        equal = prediction == record["prediction"]
        if not equal:
            mismatch.append(dict(source_frame_id=fid, fields={
                k: dict(replayed=prediction.get(k), logged=record["prediction"].get(k))
                for k in set(prediction) | set(record["prediction"])
                if prediction.get(k) != record["prediction"].get(k)}))
        after = {version: snapshot(stream) for version, stream in chain.items()}
        reset_audit = {}
        for version in chain:
            reset = bool(reasons[version])
            observed_calls = list(reset_calls[version]) if version in reset_calls else None
            if observed_calls is not None:
                observed_reasons = [reason for call in observed_calls for reason in call["inferred_reasons"]]
                if len(observed_calls) != int(reset) or observed_reasons != reasons[version]:
                    actual_call_mismatch.append(dict(
                        source_frame_id=fid, version=version,
                        expected_reset_calls=int(reset), observed_reset_calls=len(observed_calls),
                        expected_reasons=reasons[version], observed_reasons=observed_reasons,
                    ))
                reset = bool(observed_calls)
            old = before[version]
            nonempty = bool(old["history_frames"])
            has_valid = bool(old["valid_history_frames"])
            reset_audit[version] = dict(
                reasons=reasons[version], clear_called=reset,
                audit_method="wrapped_reset_method" if observed_calls is not None else "history_state_replay",
                observed_reset_calls=observed_calls,
                cleared_nonempty_history=reset and nonempty,
                cleared_valid_history=reset and has_valid,
                before=old, after=after[version],
            )
            if reset:
                reset_counts[version]["all_clear_calls"] += 1
                reset_counts[version]["nonempty_history_clear_calls"] += int(nonempty)
                reset_counts[version]["valid_history_clear_calls"] += int(has_valid)
                for reason in reasons[version]:
                    reset_counts[version]["reason:" + reason] += 1
            # Check both true and false branches, including normal retention
            # pruning. This verifies more than just single-row reset output.
            keep_s = {"v3": 2.05, "v4": 4.05, "v5": .7, "v6": .7, "v7": .7}[version]
            prior = [] if reset else list(zip(old["history_frames"], old["history_times"]))
            expected_history = [frame_id for frame_id, t in prior + [(fid, row["t"])]
                                if t >= row["t"] - keep_s]
            if after[version]["history_frames"] != expected_history:
                history_mismatch.append(dict(
                    source_frame_id=fid, version=version,
                    expected_history=expected_history,
                    observed_history=after[version]["history_frames"],
                    reason="history_not_equal_to_expected_reset_or_retention_branch"))
        valid_gap = None if last_valid is None else row["t"] - last_valid["time_s"]
        if reasons["v3"]:
            warmup_origin = dict(
                source_frame_id=fid, time_s=row["t"], reasons=reasons["v3"],
                previous_last_valid=last_valid, previous_valid_gap_s=valid_gap,
                initial=not ever_valid,
            )
        category = None
        budget_abandoned = bool(record.get("recovery", {}).get("budget_audit", {}).get("exhausted"))
        if prediction["label"] == "unknown":
            reason = prediction.get("reason")
            if withheld:
                category = "partial_pose_withheld"
            elif reason == "missing_observation":
                if budget_abandoned:
                    category = "recovery_abandoned_missing_observation"
                elif not ever_valid:
                    category = "initial_target_unconfirmed"
                elif record["target_id"] is None:
                    category = "target_unavailable_after_expiry_or_loss"
                else:
                    category = "other_missing_observation"
            elif reason == "warming_up":
                if warmup_origin and warmup_origin["initial"] and not ever_known:
                    category = "initial_warmup"
                elif warmup_origin and any("gap" in r for r in warmup_origin["reasons"]):
                    category = "post_gap_reset_warmup"
                elif warmup_origin and "target_epoch_changed" in warmup_origin["reasons"]:
                    category = "post_identity_reset_warmup"
                else:
                    category = "other_warmup"
            elif reason == "low_classifier_confidence":
                category = "classifier_uncertainty"
            else:
                category = "other_unknown"
            counts[category] += 1
        frame = dict(
            source_frame_id=fid, source_time_s=row["t"],
            observed=record["observed"], action_observed=bool(effective["instances"]),
            target_id=record["target_id"], target_epoch=record["target_epoch"],
            tracking_reason=record["tracking_reason"],
            label=prediction["label"], reason=prediction.get("reason"),
            unknown_category=category, prediction_exactly_equal=equal,
            previous_valid=last_valid, gap_since_last_valid_s=valid_gap,
            source_gap_s=record["source_gap_s"],
            dropped_before_this_frame=record["dropped_before_this_frame"],
            recovery_abandoned=budget_abandoned,
            recovery_reason=record["recovery"]["reason"],
            v3_warmup_origin=warmup_origin, histories=reset_audit,
        )
        frames.append(frame)
        if effective["instances"]:
            ever_valid = True
            last_valid = dict(source_frame_id=fid, time_s=row["t"])
        if prediction["label"] != "unknown":
            ever_known = True
    ids = [r["source_frame_id"] for r in records]
    drops = [fid for r in records for fid in r["dropped_before_this_frame"]]
    return dict(
        trial=path.parent.name, strategy=records[0]["strategy"],
        repeat=records[0]["repeat"], input_path=str(path), input_sha256=sha256(path),
        processed_count=len(records), processed_frames=ids,
        dropped_count=len(drops), dropped_frames=drops,
        unknown_count=sum(counts.values()), unknown_categories=dict(counts),
        unknown_frames=[{k: r[k] for k in (
            "source_frame_id", "source_time_s", "label", "reason", "unknown_category",
            "observed", "action_observed", "target_id", "tracking_reason",
            "recovery_abandoned", "recovery_reason", "previous_valid",
            "gap_since_last_valid_s", "source_gap_s", "v3_warmup_origin")}
                        for r in frames if r["unknown_category"]],
        reset_counts={k: dict(v) for k, v in reset_counts.items()},
        full_prediction_matches=len(records) - len(mismatch),
        prediction_mismatches=mismatch, history_reset_mismatches=history_mismatch,
        observed_reset_call_mismatches=actual_call_mismatch,
        observed_v3_v4_reset_calls_match_predicates=not actual_call_mismatch,
        all_predictions_exactly_equal=not mismatch,
        history_reset_predicates_verified=not history_mismatch,
        frames=frames,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/abc-v7-20261020/recovery-budget-20261021")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/abc-v7-20261020/unknown-reduction-20261021/diagnosis.json")
    args = parser.parse_args()
    source = args.input.resolve()
    paths = sorted((source / "trials").glob("*/frames.jsonl"))
    if not paths:
        raise SystemExit(f"No trials/*/frames.jsonl under {source}")
    model_path = ROOT / "runs/action-v7/training/model.json"
    code_paths = [ROOT / f"src/fallpre/weekly/action_v{n}.py" for n in range(2, 8)]
    code_paths.append(ROOT / "src/fallpre/weekly/pose_refinement.py")
    frozen = {str(p): sha256(p) for p in [model_path, *code_paths, *paths]}
    model = CompletionModel.load(model_path)
    trials = [audit_trial(path, model) for path in paths]
    grouped = {}
    for trial in trials:
        bucket = grouped.setdefault(trial["strategy"], dict(trials=0, processed=0, unknown=0, categories=Counter()))
        bucket["trials"] += 1
        bucket["processed"] += trial["processed_count"]
        bucket["unknown"] += trial["unknown_count"]
        bucket["categories"].update(trial["unknown_categories"])
    for bucket in grouped.values():
        bucket["categories"] = dict(bucket["categories"])
    result = dict(
        method="Exact frozen V7 replay of saved selected_instance and target_epoch; no detector/pose inference or training",
        input=str(source), model_sha256=sha256(model_path),
        frozen_hashes=frozen,
        frozen_hashes_unchanged=all(sha256(p) == value for p, value in frozen.items()),
        processed_prediction_count=sum(t["processed_count"] for t in trials),
        all_predictions_exactly_equal=all(t["all_predictions_exactly_equal"] for t in trials),
        history_reset_predicates_verified=all(t["history_reset_predicates_verified"] for t in trials),
        observed_v3_v4_reset_calls_match_predicates=all(t["observed_v3_v4_reset_calls_match_predicates"] for t in trials),
        audit_instrumentation=dict(
            v3_v4="Wrapped original instance reset methods to record actual invocation count, callsite and pre/post states; original method executes unchanged",
            reset_reason="Inferred from actual call prestate and source predicates; the original methods receive no cause string",
            other_layers="State replay only; no claim of capturing deque.clear calls",
            all_layers="Both reset and no-reset histories verified against append and time-window pruning",
        ),
        aggregate_by_strategy=grouped, trials=trials,
        interpretation=[
            "Unknown categories are mutually exclusive output-reason accounting, not additive independent causal effects.",
            "Latest-frame scheduling selects different processed sets per trial; dropped frames are not unknown predictions and are not correct predictions.",
            "V3/V4 last-valid-observation gaps can exceed 0.35 s even when consecutive processed-source gaps are only 0.334 s.",
            "V3 warming_up requires at least three valid poses spanning 0.16 s; V2 may preserve a falling output before this condition, so actual labels are replayed rather than inferred only from span.",
            "V6/V7 clear completion-only history on a missing action observation; this must not be counted as independently causing the inherited V3 action label to become unknown.",
            "Tracking target_id continuity and action history continuity have distinct thresholds; preserving target_id does not imply preservation of action history.",
            "History clear counts include initial epoch initialization; nonempty/valid-history counts separately identify deletion of accumulated evidence.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "processed": result["processed_prediction_count"],
                      "predictions_equal": result["all_predictions_exactly_equal"],
                      "histories_verified": result["history_reset_predicates_verified"],
                      "aggregate": grouped}, ensure_ascii=False))
    if not all(result[k] for k in ("frozen_hashes_unchanged", "all_predictions_exactly_equal", "history_reset_predicates_verified", "observed_v3_v4_reset_calls_match_predicates")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

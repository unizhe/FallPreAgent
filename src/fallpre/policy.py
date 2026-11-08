"""Deterministic prototype policy; no clinical prediction or hidden model calls."""
from __future__ import annotations

import json
from copy import deepcopy

from .config import RuleConfig
from .schemas import RiskDecision, Snapshot, Validity
from .state_store import EventRecord


def fresh(t_s: float, now_s: float, age_s: float) -> bool:
    return 0 <= now_s - t_s <= age_s + 1e-9


def bound(snapshot: Snapshot) -> bool:
    return getattr(snapshot, "binding_status", "BOUND") == "BOUND"


def quality_valid(snapshot: Snapshot, now_s: float, rules: RuleConfig) -> bool:
    q = snapshot.quality
    return (q.validity == Validity.VALID and q.camera_stable is True
            and q.t_s <= now_s < q.valid_until_s
            and fresh(q.t_s, now_s, rules.quality_ttl_s))


def spatial_valid(snapshot: Snapshot, now_s: float, rules: RuleConfig) -> bool:
    s, m, o = snapshot.spatial, snapshot.motion, snapshot.obstacle
    return bool(bound(snapshot) and quality_valid(snapshot, now_s, rules)
                and fresh(snapshot.t_s, now_s, rules.human_max_age_s)
                and s.validity == Validity.VALID and m.validity == Validity.VALID
                and o.validity == Validity.VALID and s.valid_until_s >= now_s
                and o.valid_until_s >= now_s
                and fresh(s.reference_time_s, now_s, rules.human_max_age_s)
                and fresh(m.t_s, now_s, rules.human_max_age_s)
                and snapshot.calibration_id is not None
                and s.calibration_id == m.calibration_id == snapshot.calibration_id
                and s.footprint_version == o.footprint_version
                and m.track_epoch == snapshot.track_epoch)


def approaching(snapshot: Snapshot, now_s: float, rules: RuleConfig) -> bool:
    s = snapshot.spatial
    return bool(spatial_valid(snapshot, now_s, rules)
                and (s.ttc_status == "INSIDE_RISK_REGION" or
                     (s.path_intersects is True and s.ttc_s is not None
                      and 0 <= s.ttc_s <= rules.warning_ttc_s
                      and s.closing_speed_mps is not None and s.closing_speed_mps > 0)))


def fall_confirmed(snapshot: Snapshot, now_s: float, rules: RuleConfig) -> bool:
    f = snapshot.fall
    return bool(bound(snapshot) and quality_valid(snapshot, now_s, rules)
                and fresh(snapshot.t_s, now_s, rules.human_max_age_s)
                and fresh(f.t_s, now_s, rules.human_max_age_s)
                and f.validity == Validity.VALID and f.had_upright and f.low_pose is True
                and f.vertical_drop_norm is not None and f.vertical_drop_norm >= rules.fall_drop_norm
                and ((f.tilt_deg is not None and f.tilt_deg >= rules.fall_tilt_deg)
                     or (f.aspect_ratio is not None and f.aspect_ratio >= rules.fall_aspect_ratio))
                and f.low_pose_duration_s >= rules.fall_low_duration_s)


def _wait(event: EventRecord, now_s: float, rules: RuleConfig, partition: str):
    if event.wait_deadline_s is None:
        event.wait_deadline_s = now_s + rules.wait_timeout_s
    if now_s >= event.wait_deadline_s:
        return "REVIEW_REQUIRED", "NEEDS_REVIEW", "review", ["REQUEST_CHECK"], ["Observation deadline reached; fall/risk remains unconfirmed"]
    return "WAIT", "CANDIDATE", partition, [], ["Awaiting sufficient evidence"]


def evaluate(snapshot: Snapshot, current: EventRecord, now_s: float,
             rules: RuleConfig) -> tuple[EventRecord, RiskDecision]:
    event = deepcopy(current)
    actions: list[str] = []
    reasons: list[str] = []
    if event.confirmed_alarm:
        outcome, status, partition, actions = "ALARM", "ACTIVE", "confirmed_alarm", ["FALL_ALARM"]
        reasons = ["Previously confirmed alarm remains deliverable"]
    elif event.branch == "fall":
        f = snapshot.fall
        observable = (bound(snapshot) and quality_valid(snapshot, now_s, rules)
                      and f.validity == Validity.VALID
                      and fresh(snapshot.t_s, now_s, rules.human_max_age_s)
                      and fresh(f.t_s, now_s, rules.human_max_age_s))
        if fall_confirmed(snapshot, now_s, rules):
            event.confirmed_alarm = True
            event.wait_deadline_s = None
            outcome, status, partition, actions = "ALARM", "ACTIVE", "confirmed_alarm", ["FALL_ALARM"]
            reasons = ["Reliable prior upright posture, descent, orientation and sustained low posture"]
        elif observable and f.recovered:
            outcome, status, partition = "CLEAR", "CLEARED", "recovered"
            event.wait_deadline_s = None
            reasons = ["Sustained reliable recovery evidence"]
        elif event.status == "NEEDS_REVIEW":
            outcome, status, partition, actions = "REVIEW_REQUIRED", "NEEDS_REVIEW", "review", ["REQUEST_CHECK"]
            reasons = ["Unresolved review request; no renewed automatic wait"]
        elif (observable and f.phase in {"SUSPECTED", "LOW_POSE", "RECOVERING"}) or event.status == "CANDIDATE":
            partition_name = f.phase if observable else "unobservable"
            outcome, status, partition, actions, reasons = _wait(event, now_s, rules, partition_name)
        else:
            outcome, status, partition = "NO_ACTION", "NONE", "observed_none" if observable else "unknown"
            reasons = ["No fall candidate in observed evidence" if observable else "Insufficient current fall evidence; safety is unknown"]
    else:
        valid = spatial_valid(snapshot, now_s, rules)
        is_risk = approaching(snapshot, now_s, rules)
        was_open = event.status in {"CANDIDATE", "ACTIVE", "NEEDS_REVIEW"}
        if is_risk:
            if event.risk_since_s is None:
                event.risk_since_s = snapshot.t_s
            event.clear_since_s = None
            if snapshot.t_s - event.risk_since_s >= rules.risk_enter_s - 1e-9:
                outcome, status, partition, actions = "REMIND", "ACTIVE", "risk", ["OBSTACLE_REMINDER"]
                event.wait_deadline_s = None
                reasons = ["Valid predicted entry into registered obstacle risk region"]
            else:
                outcome, status, partition, actions, reasons = _wait(event, now_s, rules, "entering")
        elif valid and was_open and snapshot.spatial.ttc_status != "INSIDE_RISK_REGION" and snapshot.spatial.clearance_m is not None and snapshot.spatial.clearance_m >= rules.clear_distance_m and snapshot.spatial.path_intersects is False:
            event.risk_since_s = None
            if event.clear_since_s is None:
                event.clear_since_s = snapshot.t_s
            if snapshot.t_s - event.clear_since_s >= rules.risk_clear_s - 1e-9:
                outcome, status, partition = "CLEAR", "CLEARED", "clear"
                event.wait_deadline_s = None
                reasons = ["Sustained sufficient clearance and non-intersecting predicted path"]
            else:
                outcome, status, partition, actions, reasons = _wait(event, now_s, rules, "clearing")
        elif was_open:
            event.risk_since_s = None
            event.clear_since_s = None
            if event.status == "NEEDS_REVIEW":
                outcome, status, partition, actions = "REVIEW_REQUIRED", "NEEDS_REVIEW", "review", ["REQUEST_CHECK"]
                reasons = ["Risk remains unresolved"]
            else:
                outcome, status, partition, actions, reasons = _wait(event, now_s, rules, "uncertain")
        else:
            outcome, status, partition = "NO_ACTION", "NONE", "no_intersection" if valid else "unknown"
            event.risk_since_s = None
            event.clear_since_s = None
            reasons = ["No obstacle trigger in supported region" if valid else "Geometry unavailable; risk is unknown"]

    # Gaps cannot accumulate dwell time as if they contained positive observations.
    # Ingest resets counters before evaluation; timer-only calls never start dwell.
    semantic = json.dumps([status, outcome, partition, snapshot.binding_id,
                          getattr(snapshot, "binding_status", "BOUND"), snapshot.track_epoch,
                          snapshot.calibration_id, snapshot.scene_version,
                          getattr(snapshot, "session_id", "default")])
    if semantic != event.semantic_key:
        event.revision += 1
        event.semantic_key = semantic
    if status == "CLEARED" and current.status != "CLEARED":
        event.action_slots.clear()
    for action in actions:
        if action not in event.action_slots:
            event.action_sequence += 1
            event.action_slots[action] = event.action_sequence
    event.outcome, event.status, event.partition = outcome, status, partition
    event.latest_snapshot_id = snapshot.snapshot_id
    expires = (event.wait_deadline_s if outcome == "WAIT" else now_s +
               (rules.wait_timeout_s if outcome == "REVIEW_REQUIRED" else rules.reminder_ttl_s))
    decision = RiskDecision(event_id=event.event_id, event_revision=event.revision,
                            snapshot_id=snapshot.snapshot_id, person_id=snapshot.person_id,
                            binding_id=snapshot.binding_id, branch=event.branch, outcome=outcome,
                            reasons=reasons, allowed_actions=actions, issued_at_s=now_s,
                            expires_at_s=expires, rule_version=rules.version,
                            calibration_id=snapshot.calibration_id, scene_version=snapshot.scene_version,
                            confirmed_alarm=event.confirmed_alarm,
                            action_sequence=event.action_slots[actions[0]] if actions else 0)
    event.latest_decision_id = decision.decision_id
    return event, decision

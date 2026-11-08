"""Observer contracts are synthetic; these tests do not establish clinical accuracy."""

from copy import deepcopy

import pytest

from fallpre.weekly.webcam_risk_points import LiveRiskTransitionPoints


def activity(label="seated", epoch=1, index=0, **changes):
    action = dict(
        label=label,
        raw_label="background",
        phase="seated",
        decision_code="accepted",
        target_index=index,
        selection=dict(target_epoch=epoch, state="locked"),
    )
    action.update(changes)
    behavior = dict(
        target_epoch=epoch,
        target_index=index,
        posture=dict(label={"standing": "upright", "standing_up": "unknown"}.get(label, label)),
        transition=dict(label="standing_up" if label == "standing_up" else "none"),
        quality=dict(observed=True, torso=True, camera_stable=True, reset_reason=None),
    )
    return action, behavior


def feed(observer, t, label="seated", fall=None, diagnostics=None, stable=True, **kwargs):
    a, b = activity(label, **kwargs)
    return observer.update(t, a, b, fall or {}, diagnostics or {}, stable)


def yellow(epoch="rgb-1"):
    return dict(
        state="suspected",
        observed=True,
        target_epoch=epoch,
        candidate_since_s=0.0,
        metrics=dict(qualifying_fallen=True),
        transitions=[],
    )


def red(t, epoch="rgb-1", event_id="rgb-fall-0001"):
    return dict(
        state="confirmed",
        observed=True,
        target_epoch=epoch,
        event_target_epoch=epoch,
        current_target_confirmed=True,
        event_id=event_id,
        confirmed_at_s=t,
        transitions=[dict(kind="confirmed", t_s=t, target_epoch=epoch, event_id=event_id)],
    )


def descent(t, epoch="rgb-1"):
    return dict(
        t_s=t,
        target_epoch=epoch,
        source="person_detector",
        reference_kind="seated",
        center_drop_norm=0.2,
        top_drop_norm=0.3,
        height_ratio=0.6,
    )


def kinds(result):
    return [event["kind"] for event in result["new_events"]]


def test_accepted_seat_to_rise_is_only_an_activity_point_not_a_danger_or_alarm():
    f = LiveRiskTransitionPoints()
    feed(f, 0.0)
    out = feed(f, 0.13, "standing_up")
    assert kinds(out) == ["sit_to_stand_started"]
    e = out["new_events"][0]
    assert e["category"] == "activity_transition" and e["pre_reminder_candidate"]
    assert not e["post_fall_evidence"] and not e["prediction_validated"]
    assert e["occurred_at_s"] == e["detected_at_s"] == 0.13
    assert out["observer_only"] and not out["reminders_enabled"]
    assert not kinds(feed(f, 0.26, "standing_up"))


def test_forward_lean_phase_alone_never_invents_rise_intent():
    f = LiveRiskTransitionPoints()
    feed(f, 0.0)
    for i in range(1, 8):
        assert not kinds(feed(f, i * 0.13, phase="forward_lean"))


@pytest.mark.parametrize(
    "boundary", ["missing", "epoch", "index", "gap", "camera", "side", "raw_fall", "auxiliary"]
)
def test_action_anchors_do_not_cross_identity_visibility_or_camera_boundaries(boundary):
    f = LiveRiskTransitionPoints()
    feed(f, 0.0)
    a, b = activity("unknown")
    t = 0.13
    stable = True
    if boundary == "missing":
        b["quality"]["observed"] = False
    if boundary == "epoch":
        b["target_epoch"] = a["selection"]["target_epoch"] = 2
    if boundary == "index":
        b["target_index"] = a["target_index"] = 1
    if boundary == "gap":
        t = 0.5
    if boundary == "camera":
        stable = False
    if boundary == "side":
        b["quality"]["reset_reason"] = "visible_side_changed"
    if boundary == "raw_fall":
        a["raw_label"] = "falling"
    if boundary == "auxiliary":
        b["quality"]["auxiliary_posture"] = True
    f.update(t, a, b, {}, {}, stable)
    assert not kinds(feed(f, t + 0.13, "standing_up"))


def test_unknown_but_continuously_visible_action_can_finish_a_known_transition():
    f = LiveRiskTransitionPoints()
    feed(f, 0.0)
    feed(f, 0.13, "unknown", decision_code="rise_pending")
    assert kinds(feed(f, 0.26, "standing_up")) == ["sit_to_stand_started"]


def test_expired_activity_and_intervening_bend_cannot_borrow_seat():
    f = LiveRiskTransitionPoints(activity_context_s=0.4)
    feed(f, 0.0)
    for i in range(1, 5):
        feed(f, i * 0.13, "unknown")
    assert not kinds(feed(f, 0.65, "standing_up"))
    feed(f, 0.78)
    feed(f, 0.91, "bending")
    assert not kinds(feed(f, 1.04, "standing_up"))


def test_squat_recovery_timestamp_is_visible_completion_not_inferred_onset():
    f = LiveRiskTransitionPoints()
    feed(f, 0.0, "squatting")
    feed(f, 0.13, "unknown")
    out = feed(f, 0.26, "standing")
    assert kinds(out) == ["squat_to_upright_completed"]
    assert out["new_events"][0]["occurred_at_s"] == 0.26
    assert out["new_events"][0]["category"] == "activity_transition"


def test_static_flicker_does_not_repeat_but_new_confirmed_sit_cycle_can():
    f = LiveRiskTransitionPoints()
    feed(f, 0.0)
    first = feed(f, 0.13, "standing_up")["new_events"][0]
    feed(f, 0.26)
    assert not kinds(feed(f, 0.39, "standing_up"))
    for t in (0.52, 0.65, 0.78):
        feed(f, t)
    second = feed(f, 0.91, "standing_up")["new_events"][0]
    assert first["event_id"] != second["event_id"]


def test_measured_descent_can_precede_yellow_without_pose_or_fallen_classification():
    f = LiveRiskTransitionPoints()
    fall = dict(state="normal", target_epoch="rgb-1", observed=True)
    out = f.update(1.0, {}, {}, fall, dict(measured_descent=descent(1.0)))
    assert kinds(out) == ["body_descent_observed"]
    e = out["new_events"][0]
    assert e["evidence"]["witness"]["reference_kind"] == "seated"
    assert not e["post_fall_evidence"] and not e["prediction_validated"]
    out = f.update(1.13, {}, {}, fall, dict(measured_descent=descent(1.13)))
    assert not kinds(out)
    out = f.update(1.26, {}, {}, yellow(), dict(handoff_ready=True))
    assert kinds(out) == ["fall_suspected"]
    assert out["new_events"][0]["occurred_at_s"] == 1.26  # Not candidate_since_s=0.
    assert out["new_events"][0]["post_fall_evidence"]


@pytest.mark.parametrize("bad", ["stale", "future", "epoch", "source", "nan", "camera"])
def test_descent_requires_current_matching_authoritative_witness(bad):
    witness = descent(1.0)
    if bad == "stale":
        witness["t_s"] = 0.9
    if bad == "future":
        witness["t_s"] = 1.1
    if bad == "epoch":
        witness["target_epoch"] = "rgb-2"
    if bad == "source":
        witness["source"] = "fallen_detector"
    if bad == "nan":
        witness["center_drop_norm"] = float("nan")
    out = LiveRiskTransitionPoints().update(
        1.0, {}, {}, dict(target_epoch="rgb-1"), dict(measured_descent=witness), bad != "camera"
    )
    assert not kinds(out)


def test_fresh_legacy_handoff_witness_is_supported_but_not_repeated():
    f = LiveRiskTransitionPoints()
    out = f.update(1.0, {}, {}, yellow(), dict(handoff_ready=True, descent_witness=descent(1.0)))
    assert kinds(out) == ["body_descent_observed", "fall_suspected"]
    assert out["new_events"][0]["post_fall_evidence"]
    assert not out["new_events"][0]["pre_reminder_candidate"]
    assert not kinds(f.update(1.13, {}, {}, yellow(), dict(handoff_ready=True, descent_witness=descent(1.0))))


@pytest.mark.parametrize("bad", ["unobserved", "not_qualifying", "no_handoff", "no_epoch"])
def test_yellow_needs_current_associated_fallen_evidence(bad):
    fall, diag = yellow(), dict(handoff_ready=True)
    if bad == "unobserved":
        fall["observed"] = False
    if bad == "not_qualifying":
        fall["metrics"]["qualifying_fallen"] = False
    if bad == "no_handoff":
        diag["handoff_ready"] = False
    if bad == "no_epoch":
        fall["target_epoch"] = None
    assert not kinds(LiveRiskTransitionPoints().update(1.0, {}, {}, fall, diag))


def test_historical_latched_red_never_becomes_new_event_after_missing_or_identity_change():
    f = LiveRiskTransitionPoints()
    out = f.update(1.0, {}, {}, red(1.0), {})
    assert kinds(out) == ["fall_confirmed"]
    f.update(1.13, {}, {}, dict(state="unobservable"), {})
    assert not kinds(f.update(1.26, {}, {}, red(1.0), {}))
    stale = red(1.39, epoch="rgb-2")
    stale["event_target_epoch"] = "rgb-1"
    assert not kinds(f.update(1.39, {}, {}, stale, {}))
    assert out["new_events"][0]["post_fall_evidence"]
    assert not out["new_events"][0]["pre_reminder_candidate"]


@pytest.mark.parametrize("bad", ["not_current", "unobserved", "no_transition", "wrong_event", "not_red"])
def test_red_requires_current_explicit_confirmation_transition(bad):
    fall = red(1.0)
    if bad == "not_current":
        fall["current_target_confirmed"] = False
    if bad == "unobserved":
        fall["observed"] = False
    if bad == "no_transition":
        fall["transitions"] = []
    if bad == "wrong_event":
        fall["transitions"][0]["event_id"] = "different"
    if bad == "not_red":
        fall["state"] = "normal"
    assert not kinds(LiveRiskTransitionPoints().update(1.0, {}, {}, fall, {}))


def test_rearm_requires_explicit_current_core_event_and_records_previous_event():
    f = LiveRiskTransitionPoints()
    f.update(1.0, {}, {}, red(1.0), {})
    normal = dict(state="normal", observed=True, target_epoch="rgb-2")
    assert not kinds(f.update(1.13, {}, {}, normal, {}))
    normal["transitions"] = [
        dict(kind="rearmed", t_s=1.26, target_epoch="rgb-2", previous_event_id="rgb-fall-0001")
    ]
    out = f.update(1.26, {}, {}, normal, {})
    assert kinds(out) == ["fall_rearmed"]
    assert out["new_events"][0]["evidence"]["previous_event_id"] == "rgb-fall-0001"
    assert not kinds(f.update(1.39, {}, {}, normal, {}))
    assert kinds(f.update(1.52, {}, {}, yellow("rgb-2"), dict(handoff_ready=True))) == ["fall_suspected"]


def test_does_not_mutate_inputs_or_expose_internal_history():
    f = LiveRiskTransitionPoints()
    a, b = activity()
    fall, diag = yellow(), dict(handoff_ready=True, measured_descent=descent(1.0))
    diag["measured_descent"]["end_box"] = [1, 2, 3, 4]
    before = deepcopy((a, b, fall, diag))
    out = f.update(1.0, a, b, fall, diag)
    assert before == (a, b, fall, diag)
    out["recent_events"][0]["evidence"]["witness"]["center_drop_norm"] = 999
    assert f.events[0]["evidence"]["witness"]["center_drop_norm"] == 0.2
    assert "end_box" not in f.events[0]["evidence"]["witness"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, True, None])
def test_invalid_or_repeated_source_clock_does_not_mutate_state(bad):
    f = LiveRiskTransitionPoints()
    feed(f, 0.0)
    before = deepcopy(f.__dict__)
    with pytest.raises(ValueError):
        feed(f, bad)
    assert f.__dict__ == before


def test_history_is_bounded_and_ids_remain_unique():
    f = LiveRiskTransitionPoints(history_limit=2)
    ids = []
    for i in range(5):
        out = f.update(i * 0.13, {}, {}, red(i * 0.13, f"rgb-{i}", f"fall-{i}"), {})
        ids.append(out["new_events"][0]["event_id"])
    assert len(out["recent_events"]) == 2 and out["event_count"] == 5
    assert len(set(ids)) == 5


def test_walking_and_turning_are_not_claimed_to_be_instability():
    a, b = activity("standing")
    b.update(locomotion=dict(label="walking"), turning=dict(label="turning"))
    out = LiveRiskTransitionPoints().update(0.0, a, b, {}, {})
    assert not kinds(out)
    assert "walking_or_turning_instability" in out["unimplemented"]

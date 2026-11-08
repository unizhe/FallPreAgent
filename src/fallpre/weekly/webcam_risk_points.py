"""Causal, observer-only transition points for later reminder research.

This module never changes an action/fall decision and never emits an alarm.
Accepted posture transitions are ordinary activities, not a diagnosis of
weakness or danger. Detector-box descent is evidence of image motion, not
proof of lost support. Yellow/red describe already observed fallen evidence;
neither their timestamps nor an earlier candidate timestamp prove prediction.
"""

from collections import deque
from copy import deepcopy
from math import isfinite


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _epoch(value):
    return value if isinstance(value, (str, int)) and not isinstance(value, bool) else None


class LiveRiskTransitionPoints:
    """Observe current pipeline outputs without connecting unobserved targets.

    ``update`` takes source-video seconds, action, behavior axes, the FINAL RGB
    fall output, RGB diagnostics, and camera stability. Do not pass pose_fall:
    that branch can contain an older latched confirmation. ``new_events`` is
    only this update; ``recent_events`` is bounded history, never current alerts.

    Action anchors require a currently observed torso, a matching pose identity
    and an upstream accepted posture. Unknown labels may occur during the same
    visible transition; missing observations, identity/visible-side changes,
    camera motion and source gaps clear its anchor. No new pose is synthesized.

    Descent and lifecycle points require a fresh authoritative core witness.
    Repeated frames are deduplicated per RGB episode. An explicit core rearm
    starts a new epoch; merely becoming normal or reacquiring a person does not.
    """

    def __init__(self, max_gap_s=0.35, activity_context_s=3.0, history_limit=128):
        if (
            not _finite(max_gap_s)
            or max_gap_s <= 0
            or not _finite(activity_context_s)
            or activity_context_s <= 0
            or not isinstance(history_limit, int)
            or isinstance(history_limit, bool)
            or history_limit < 1
        ):
            raise ValueError("Transition observer limits must be finite and positive")
        self.max_gap_s = max_gap_s
        self.activity_context_s = activity_context_s
        self.events = deque(maxlen=history_limit)
        self._seen = deque(maxlen=max(history_limit, 128))
        self._sequence = 0
        self._last_t = None
        self._pose_epoch = None
        self._pose_index = None
        self._anchor = None
        self._static_run = None
        self._consumed_posture = None

    def _clear_activity(self):
        self._anchor = self._static_run = self._consumed_posture = None

    def _emit(self, new, key, kind, text, category, t, epoch, evidence, pre_candidate=False, post_fall=False):
        if key in self._seen:
            return
        self._seen.append(key)
        self._sequence += 1
        event = dict(
            event_id=f"transition-{self._sequence:06d}",
            kind=kind,
            text=text,
            category=category,
            occurred_at_s=t,
            detected_at_s=t,
            target_epoch=epoch,
            evidence=deepcopy(evidence),
            pre_reminder_candidate=pre_candidate,
            post_fall_evidence=post_fall,
            prediction_validated=False,
        )
        self.events.append(event)
        new.append(event)

    def _activity(self, t, action, behavior, stable, source_gap, new):
        quality = _mapping(behavior.get("quality"))
        epoch = _epoch(behavior.get("target_epoch"))
        index = behavior.get("target_index")
        selection = _mapping(action.get("selection"))
        visible = bool(
            stable
            and quality.get("observed") is True
            and quality.get("torso") is True
            and quality.get("camera_stable") is True
            and epoch is not None
            and isinstance(index, int)
            and not isinstance(index, bool)
            and index >= 0
            and action.get("target_index") == index
            and selection.get("target_epoch") == epoch
            and selection.get("state") == "locked"
            and action.get("raw_label") not in {"falling", "unavailable"}
            and not quality.get("auxiliary_posture")
        )
        if (
            not visible
            or source_gap
            or epoch != self._pose_epoch
            or index != self._pose_index
            or quality.get("reset_reason")
        ):
            self._clear_activity()
        self._pose_epoch, self._pose_index = epoch, index
        if not visible:
            return False
        label = action.get("label")
        posture = _mapping(behavior.get("posture")).get("label")
        transition = _mapping(behavior.get("transition")).get("label")
        accepted = action.get("decision_code") == "accepted"
        static = posture if accepted and label == posture and posture in {"seated", "squatting"} else None
        if static:
            if not self._static_run or self._static_run["posture"] != static:
                self._static_run = dict(posture=static, since=t)
            # The first anchor uses an upstream confirmed posture immediately.
            # After an emitted transition, require a real new static interval
            # before another episode; label flicker must not repeat the point.
            if self._consumed_posture != static or t - self._static_run["since"] >= 0.25 - 1e-9:
                if not self._anchor or self._anchor["posture"] != static:
                    self._anchor = dict(posture=static, since=t, last_seen=t)
                self._anchor["last_seen"] = t
                self._consumed_posture = None
            return True
        self._static_run = None
        anchor = self._anchor
        if anchor is None:
            return True
        if t - anchor["last_seen"] > self.activity_context_s:
            self._anchor = None
            return True
        kind = None
        if anchor["posture"] == "seated" and accepted and label == transition == "standing_up":
            kind, text = "sit_to_stand_started", "已确认坐姿后开始起身（普通动作转换）"
        elif anchor["posture"] == "squatting" and posture == "upright" and label == "standing" and accepted:
            kind, text = "squat_to_upright_completed", "蹲姿后恢复直立（当前完成点）"
        if kind:
            self._emit(
                new,
                ("activity", epoch, kind, anchor["since"]),
                kind,
                text,
                "activity_transition",
                t,
                epoch,
                dict(
                    anchor_posture=anchor["posture"],
                    anchor_since_s=anchor["since"],
                    anchor_last_seen_s=anchor["last_seen"],
                    current_action=label,
                    current_posture=posture,
                    current_transition=transition,
                    decision_stage=action.get("decision_stage"),
                    interpretation="动作转换本身不代表无力、失衡或即将跌倒",
                ),
                pre_candidate=True,
            )
            self._consumed_posture = anchor["posture"]
            self._anchor = None
        elif label in {"bending", "sitting_down", "standing", "seated", "squatting"}:
            # Another accepted activity cannot borrow the previous seat/squat.
            self._anchor = None
        return True

    @staticmethod
    def _fresh(witness, t, epoch):
        return (
            witness.get("target_epoch") == epoch
            and _finite(witness.get("t_s"))
            and abs(witness["t_s"] - t) <= 1e-6
        )

    def _fall(self, t, fall, diagnostics, stable, new):
        epoch = _epoch(fall.get("target_epoch"))
        if not stable or epoch is None:
            return
        # Fresh core diagnostics can be available before a fall classification.
        # Do not require a pose skeleton or pretend the fallen detector's box
        # is measured in the person detector's reference geometry.
        measured = _mapping(diagnostics.get("measured_descent"))
        witness = measured or _mapping(diagnostics.get("descent_witness"))
        fallen_observed = fall.get("observed") is True and fall.get("state") in {"suspected", "confirmed"}
        if (
            self._fresh(witness, t, epoch)
            and witness.get("source") == "person_detector"
            and all(_finite(witness.get(k)) for k in ("center_drop_norm", "top_drop_norm", "height_ratio"))
        ):
            self._emit(
                new,
                ("descent", epoch),
                "body_descent_observed",
                "已观察到身体下降（是否失衡尚未确定）",
                "fall_evidence",
                t,
                epoch,
                dict(
                    witness={
                        key: witness.get(key)
                        for key in (
                            "t_s",
                            "target_epoch",
                            "source",
                            "reference_kind",
                            "center_drop_norm",
                            "top_drop_norm",
                            "height_ratio",
                            "bottom_clipped",
                            "person_motion_norm",
                            "fallen_motion_norm",
                            "handoff_at_s",
                            "geometry_evidence",
                        )
                        if isinstance(witness.get(key), (str, bool)) or _finite(witness.get(key))
                    },
                    interpretation="图像中的下降不能确定失足、失去支撑或发生原因",
                ),
                pre_candidate=not fallen_observed,
                post_fall=fallen_observed,
            )
        observed = fall.get("observed") is True
        metrics = _mapping(fall.get("metrics"))
        if (
            observed
            and fall.get("state") == "suspected"
            and metrics.get("qualifying_fallen") is True
            and diagnostics.get("handoff_ready") is True
        ):
            self._emit(
                new,
                ("suspected", epoch),
                "fall_suspected",
                "黄色：当前倒地证据进入疑似阶段",
                "fall_evidence",
                t,
                epoch,
                dict(
                    reason_code=fall.get("reason_code"),
                    candidate_since_s=fall.get("candidate_since_s"),
                    interpretation="已见倒地证据；候选开始时间不是提前预警时间",
                ),
                post_fall=True,
            )
        for event in fall.get("transitions") or []:
            event = _mapping(event)
            if not self._fresh(event, t, epoch):
                continue
            if (
                event.get("kind") == "confirmed"
                and observed
                and fall.get("state") == "confirmed"
                and fall.get("current_target_confirmed") is True
                and fall.get("event_target_epoch") == epoch
                and event.get("event_id")
                and event.get("event_id") == fall.get("event_id")
                and _finite(fall.get("confirmed_at_s"))
                and abs(fall["confirmed_at_s"] - t) <= 1e-6
            ):
                self._emit(
                    new,
                    ("confirmed", event["event_id"]),
                    "fall_confirmed",
                    "红色：当前跌倒事件确认",
                    "fall_confirmation",
                    t,
                    epoch,
                    dict(
                        fall_event_id=event["event_id"],
                        event_type=event.get("event_type", "fall"),
                        reason_code=fall.get("reason_code"),
                        interpretation="确认点是跌倒证据累计结果，不是跌倒发生前预测",
                    ),
                    post_fall=True,
                )
            elif event.get("kind") == "rearmed" and observed and (
                fall.get("state") == "normal" or (event.get("reference_kind") == "low_support"
                    and fall.get("lifecycle") == "low_posture_ready"
                    and fall.get("reason_code") == "low_support_ready")):
                self._emit(
                    new,
                    ("rearmed", epoch),
                    "fall_rearmed",
                    ("低位姿态参考已重建，等待新的倒卧证据" if event.get("reference_kind") == "low_support"
                     else "恢复参考已重建，跌倒分支重新待命"),
                    "lifecycle",
                    t,
                    epoch,
                    dict(
                        previous_event_id=event.get("previous_event_id"),
                        recovery_scope=event.get("recovery_scope"),
                        reference_kind=event.get("reference_kind"),
                        interpretation="核心明确完成恢复；正常标签或目标重现本身不足以证明恢复",
                    ),
                )

    def update(self, t, action, behavior, fall, rgb_diagnostics, camera_stable=True):
        """Return fresh points and bounded audit history; never mutate inputs.

        ``pre_reminder_candidate`` identifies a research candidate only, not a
        measured lead time or an enabled alert. Squat recovery is its observed
        completion point; this adapter does not backdate an unseen rising onset.
        """
        if not _finite(t) or t < 0 or self._last_t is not None and t <= self._last_t:
            raise ValueError("Transition observer source time must be finite, nonnegative and increasing")
        source_gap = self._last_t is not None and t - self._last_t > self.max_gap_s + 1e-9
        stable = camera_stable is True
        new = []
        observed = self._activity(t, _mapping(action), _mapping(behavior), stable, source_gap, new)
        self._fall(t, _mapping(fall), _mapping(rgb_diagnostics), stable, new)
        self._last_t = t
        return deepcopy(
            dict(
                schema_version="risk-transition-points-v1",
                observer_only=True,
                reminders_enabled=False,
                source_time_s=t,
                activity_observed=observed,
                new_events=new,
                recent_events=list(self.events),
                event_count=self._sequence,
                unimplemented=[
                    "forward_lean_intent",
                    "incomplete_or_repeated_rise_attempts",
                    "post_rise_instability",
                    "walking_or_turning_instability",
                    "support_loss",
                    "validated_pre_fall_prediction",
                ],
            )
        )

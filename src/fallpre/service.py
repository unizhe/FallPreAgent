"""MVP facade. Ingest trusted perception data; expose only bounded Agent tools."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from threading import RLock
from typing import Callable

from .config import RuleConfig
from .execution import ExecutionService, Handler, action_key, utc_now
from .geometry.spatial import recompute_spatial
from .policy import approaching, bound, evaluate, fresh, quality_valid
from .schemas import ExecutionResult, RiskDecision, Snapshot, Validity
from .state_store import EventRecord, ServiceError, StateStore


class MVPService:
    def __init__(self, rules: RuleConfig, db_path: str = ":memory:", dry_run: bool = True,
                 *, clock: Callable[[], float] | None = None, handler: Handler | None = None,
                 max_attempts: int = 3, retry_delay_s: float = 0.1):
        self.rules = rules
        self.store = StateStore(db_path)
        self._clock = clock
        self._time_s = 0.0
        self._lock = RLock()
        self._closed = False
        self.execution_mode = "dry_run" if dry_run else "live"
        self._observed_people: set[str] = set()
        self.execution = ExecutionService(self.store, self._validate_execution, dry_run=dry_run,
                                          handler=handler, max_attempts=max_attempts,
                                          retry_delay_s=retry_delay_s, claim=self._claim_execution)
        from .tools import ToolRegistry
        self.tools = ToolRegistry(self)
        self._recover_interrupted_observation()
        self.execution.recover_pending()

    def _recover_interrupted_observation(self):
        """A restart breaks source-clock continuity; don't invent observed dwell."""
        for current in self.store.list_events():
            if current.confirmed_alarm or current.status not in {"CANDIDATE", "ACTIVE", "NEEDS_REVIEW"}:
                continue
            if not current.latest_snapshot_id or not current.latest_decision_id:
                continue
            if self.store.decision_mode(current.latest_decision_id) != self.execution_mode:
                continue
            snapshot = self.store.get_snapshot(current.latest_snapshot_id)
            # The original immutable evidence/deadline remain in the journal.
            # No prior-session image is asserted to be a fresh observation.
            snapshot = snapshot.model_copy(update={"binding_status": "UNBOUND"})
            recovering = EventRecord(**asdict(current))
            recovering.status = "NEEDS_REVIEW"
            updated, decision = evaluate(snapshot, recovering, self.now_s, self.rules)
            decision.reasons = ["Observation process restarted; previous risk remains unconfirmed"]
            self.store.commit_decision(updated, current.revision, decision,
                                       f"restart:{decision.decision_id}", self.execution_mode)

    @property
    def now_s(self) -> float:
        value = max(self._time_s, float(self._clock())) if self._clock else self._time_s
        if not math.isfinite(value):
            raise ValueError("Source clock must be finite")
        return value

    def set_time(self, now_s: float):
        if not math.isfinite(now_s) or now_s < self._time_s:
            raise ServiceError("INVALID_TIME", "Source clock must be finite and monotonic")
        self._time_s = float(now_s)

    def _ensure_open(self):
        if self._closed:
            raise ServiceError("CLOSED", "MVP service is closed")

    def ingest_snapshot(self, snapshot: Snapshot, auto_execute: bool = True) -> list[RiskDecision]:
        self._ensure_open()
        if not isinstance(snapshot, Snapshot):
            snapshot = Snapshot.model_validate(snapshot)
        with self._lock:
            inserted = self.store.put_snapshot(snapshot)
            self._time_s = max(self._time_s, snapshot.t_s)
            self._observed_people.add(snapshot.person_id)
            existing = [] if inserted else self.store.decisions_for_snapshot(snapshot.snapshot_id)
            decisions = []
            for branch in ("fall", "prefall"):
                previous = next((d for d in reversed(existing) if d.branch == branch), None)
                if previous:
                    decisions.append(previous)
                    continue
                event = self.store.current_event(snapshot, branch)
                if (event and event.latest_decision_id
                        and self.store.decision_mode(event.latest_decision_id) != self.execution_mode):
                    event = None
                # A reliable recovery allows observing a new episode without erasing
                # or cancelling the previous event's durable confirmed alarm.
                if event is None or (branch == "fall" and event.confirmed_alarm and snapshot.fall.recovered):
                    event = self.store.create_event(snapshot, branch, self.now_s)
                decision = self._evaluate(snapshot, event.event_id, event.revision, branch, "observation")
                decisions.append(decision)
        if auto_execute:
            self._auto_execute(decisions)
        return decisions

    def _auto_execute(self, decisions: list[RiskDecision]):
        for decision in sorted(decisions, key=lambda d: 0 if d.confirmed_alarm else 1):
            for action in decision.allowed_actions:
                try:
                    self.execution.execute(decision.decision_id, action)
                except ServiceError:
                    # A newer observation can legitimately invalidate a queued reminder.
                    # This never silently bypasses confirmed-alarm execution validation.
                    if decision.confirmed_alarm:
                        raise

    def evaluate_branch(self, branch: str, snapshot_id: str, event_id: str,
                        event_revision: int) -> RiskDecision:
        self._ensure_open()
        with self._lock:
            snapshot = self.store.get_snapshot(snapshot_id)
            return self._evaluate(snapshot, event_id, event_revision, branch, "observation")

    def _evaluate(self, snapshot: Snapshot, event_id: str, expected_revision: int,
                  branch: str, trigger: str) -> RiskDecision:
        context = json.dumps(snapshot.personal_context, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(json.dumps([event_id, snapshot.snapshot_id, expected_revision,
                       self.rules.version, context, trigger]).encode()).hexdigest()
        cached = self.store.cached_decision(cache_key)
        if cached:
            return cached
        event = self.store.get_event(event_id)
        if event.branch != branch or event.person_id != snapshot.person_id or event.binding_id != snapshot.binding_id:
            raise ServiceError("EVENT_MISMATCH", "Snapshot does not belong to the event's branch/person/binding")
        if event.revision != expected_revision:
            raise ServiceError("VERSION_CONFLICT", "Event revision has advanced", latest_revision=event.revision)
        latest = self.store.latest_snapshot(snapshot.person_id)
        if trigger == "observation" and latest and latest.snapshot_id != snapshot.snapshot_id:
            raise ServiceError("STALE_SNAPSHOT", "Historical snapshots cannot advance live policy")
        if event.last_observation_s is not None and snapshot.t_s > event.last_observation_s + self.rules.max_gap_s:
            event.risk_since_s = None
            event.clear_since_s = None
        if trigger == "observation":
            event.last_observation_s = snapshot.t_s
        consumed = self._consume(snapshot)
        updated, decision = evaluate(consumed, event, self.now_s, self.rules)
        self.store.commit_decision(updated, expected_revision, decision, cache_key, self.execution_mode)
        return decision

    def _consume(self, snapshot: Snapshot) -> Snapshot:
        return snapshot.model_copy(update={"spatial": recompute_spatial(snapshot, self.now_s, self.rules)})

    def tick(self, now_s: float) -> list[RiskDecision]:
        """Host must call even when camera stalls. Uses source time, never wall time."""
        self._ensure_open()
        with self._lock:
            self.set_time(now_s)
            decisions = []
            for event in self.store.list_events():
                if event.confirmed_alarm or event.latest_snapshot_id is None:
                    continue
                if event.latest_decision_id and self.store.decision_mode(event.latest_decision_id) != self.execution_mode:
                    continue
                snapshot = self.store.get_snapshot(event.latest_snapshot_id)
                if event.outcome == "WAIT" and event.wait_deadline_s is not None and event.wait_deadline_s <= self.now_s:
                    trigger = f"deadline:{event.wait_deadline_s}"
                elif event.status == "ACTIVE" and not approaching(self._consume(snapshot), self.now_s, self.rules):
                    # A stalled stream must invalidate ACTIVE geometry and start a
                    # bounded observation-failure wait, without a fabricated frame.
                    trigger = "lost_observation"
                else:
                    continue
                decisions.append(self._evaluate(snapshot, event.event_id, event.revision,
                                                event.branch, trigger))
        self._auto_execute(decisions)
        return decisions

    def _validate_execution(self, decision: RiskDecision, action: str):
        with self._lock:
            if decision.confirmed_alarm and action == "FALL_ALARM":
                return
            event = self.store.get_event(decision.event_id)
            if event.revision != decision.event_revision or event.outcome != decision.outcome:
                raise ServiceError("SUPERSEDED_DECISION", "Event changed after the decision")
            if self.now_s > decision.expires_at_s:
                raise ServiceError("STALE_DECISION", "Decision has expired")
            if action == "REQUEST_CHECK":
                if event.status != "NEEDS_REVIEW":
                    raise ServiceError("SUPERSEDED_DECISION", "Review is no longer required")
                return  # Requests verification of a retained event, not a current obstacle.
            if decision.person_id not in self._observed_people:
                raise ServiceError("FRESH_OBSERVATION_REQUIRED", "Reacquire observations after startup")
            latest = self.store.latest_snapshot(decision.person_id)
            original = self.store.get_snapshot(decision.snapshot_id)
            if latest is None or not bound(latest) or latest.binding_id != decision.binding_id:
                raise ServiceError("BINDING_MISMATCH", "Subject binding is no longer current")
            for attr in ("track_epoch", "calibration_id", "scene_version", "session_id", "pipeline_version"):
                if getattr(latest, attr, None) != getattr(original, attr, None):
                    raise ServiceError("VERSION_MISMATCH", "Measurement or subject version changed", field=attr)
            if not fresh(latest.t_s, self.now_s, self.rules.human_max_age_s):
                raise ServiceError("STALE_SNAPSHOT", "Current observation is stale")
            if action == "OBSTACLE_REMINDER" and not approaching(self._consume(latest), self.now_s, self.rules):
                raise ServiceError("RISK_NO_LONGER_SUPPORTED", "Latest geometry does not support reminder")

    def _claim_execution(self, decision: RiskDecision, action: str, result: ExecutionResult):
        # Validation and start-claim are one serialized transition. I/O is outside
        # the lock; actions already claimed cannot generally be recalled remotely.
        with self._lock:
            self._validate_execution(decision, action)
            result.status, result.started_at = "RUNNING", result.started_at or utc_now()
            self.store.save_action(result)

    def realtime_state(self, person_id: str, max_age_ms: float) -> dict:
        with self._lock:
            snapshot = self.store.latest_snapshot(person_id)
            if snapshot is None:
                raise ServiceError("NOT_FOUND", "No observations for person")
            effective_age = min(max_age_ms / 1000, self.rules.human_max_age_s)
            consumed = self._consume(snapshot)
            result = consumed.model_dump(mode="json")
            result["validity"] = "VALID" if fresh(snapshot.t_s, self.now_s, effective_age) else "STALE"
            result["snapshot_fresh"] = result["validity"] == "VALID"
            result["supported_state_is_globally_safe"] = False
            result["effective_max_age_ms"] = effective_age * 1000
            components = {}
            def validity(name, value, t_s, ttl, valid_until=None):
                expires = min(t_s + ttl, valid_until) if valid_until is not None else t_s + ttl
                current = value if t_s <= self.now_s < expires else Validity.STALE
                components[name] = {"validity": current.value if isinstance(current, Validity) else current,
                                    "age_s": self.now_s - t_s, "valid_until_s": expires}
                if isinstance(result.get(name), dict):
                    result[name]["validity"] = components[name]["validity"]
            validity("motion", snapshot.motion.validity, snapshot.motion.t_s, self.rules.human_max_age_s)
            validity("fall", snapshot.fall.validity, snapshot.fall.t_s, self.rules.human_max_age_s)
            validity("quality", snapshot.quality.validity, snapshot.quality.t_s,
                     self.rules.quality_ttl_s, snapshot.quality.valid_until_s)
            validity("obstacle", snapshot.obstacle.validity,
                     snapshot.obstacle.last_confirmed_s if snapshot.obstacle.last_confirmed_s is not None else snapshot.obstacle.t_s,
                     self.rules.obstacle_ttl_s, snapshot.obstacle.valid_until_s)
            components["spatial"] = {"validity": consumed.spatial.validity.value,
                                     "age_s": self.now_s - snapshot.spatial.reference_time_s,
                                     "valid_until_s": consumed.spatial.valid_until_s,
                                     "computed_at_s": consumed.spatial.computed_at_s,
                                     "reasons": consumed.spatial.reasons}
            if snapshot.human:
                validity("human", snapshot.human.validity, snapshot.human.t_s, self.rules.human_max_age_s)
            else:
                components["human"] = {"validity": "UNKNOWN", "age_s": None, "valid_until_s": None}
            if not quality_valid(snapshot, self.now_s, self.rules):
                components["fall"]["validity"] = result["fall"]["validity"] = "UNKNOWN"
                components["fall"]["reason"] = "Current camera/image quality cannot support fall evidence"
            result["component_validity"] = components
            result["events"] = [asdict(e) for e in self.store.list_events(person_id)]
            result["action_keys"] = {}
            for e in self.store.list_events(person_id):
                if e.latest_decision_id:
                    d = self.store.get_decision(e.latest_decision_id)
                    result["action_keys"][d.decision_id] = {a: action_key(d, a) for a in d.allowed_actions}
            return result

    def event_evidence(self, event_id: str, event_revision: int) -> dict:
        event = self.store.get_event(event_id, event_revision)
        snapshot = self.store.get_snapshot(event.latest_snapshot_id) if event.latest_snapshot_id else None
        return {"event": asdict(event), "snapshot_id": snapshot.snapshot_id if snapshot else None,
                "fall": snapshot.fall.model_dump(mode="json") if snapshot else None,
                "spatial": snapshot.spatial.model_dump(mode="json") if snapshot else None,
                "video_evidence": None, "evidence_note": "No video recording is fabricated by the service"}

    def personal_context(self, person_id: str) -> dict:
        snapshot = self.store.latest_snapshot(person_id)
        if not snapshot:
            raise ServiceError("NOT_FOUND", "No personal observations")
        if not snapshot.personal_context:
            return {"person_id": person_id, "baseline_status": "collecting", "risk_level": None,
                    "missing_domains": ["personal_baseline"]}
        return {**snapshot.personal_context, "person_id": person_id}

    def screening_report(self, person_id: str, report_id: str) -> dict:
        # Reports are immutable snapshots supplied by the baseline worker.
        with self.store.lock:
            rows = self.store.conn.execute("SELECT body FROM snapshots WHERE person=? ORDER BY t DESC", (person_id,)).fetchall()
        for row in rows:
            context = Snapshot.model_validate_json(row["body"]).personal_context
            if context.get("report_id") == report_id:
                return {**context, "person_id": person_id}
        raise ServiceError("NOT_FOUND", "Unknown screening report")

    def close(self):
        if self._closed:
            return
        self.execution.close()
        self.store.close()
        self._closed = True

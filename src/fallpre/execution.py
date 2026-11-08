"""Persistent, deduplicated execution. No I/O occurs without an explicit handler."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from itertools import count
from queue import PriorityQueue
from threading import RLock, Thread
from typing import Any, Callable

from .schemas import ExecutionResult, RiskDecision
from .state_store import ServiceError, StateStore

Handler = Callable[[str, RiskDecision, str], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def action_key(decision: RiskDecision, action_type: str) -> str:
    return f"{decision.event_id}:{action_type}:{decision.action_sequence}"


class ExecutionService:
    def __init__(self, store: StateStore, validate: Callable[[RiskDecision, str], None],
                 *, dry_run: bool = True, handler: Handler | None = None,
                 max_attempts: int = 3, retry_delay_s: float = 0.1,
                 claim: Callable[[RiskDecision, str, ExecutionResult], None] | None = None):
        if max_attempts < 1 or retry_delay_s < 0:
            raise ValueError("Invalid retry configuration")
        self.store, self.validate, self.dry_run, self.handler = store, validate, dry_run, handler
        self.max_attempts, self.retry_delay_s = max_attempts, retry_delay_s
        self.claim = claim
        self.mode = "dry_run" if dry_run else "live"
        self._lock = RLock()
        self._queue: PriorityQueue = PriorityQueue()
        self._sequence = count()
        self._closed = False
        self._worker = None
        if not dry_run:
            self._worker = Thread(target=self._run, name="fallpre-actions", daemon=True)
            self._worker.start()

    def execute(self, decision_id: str, action_type: str, idempotency_key: str | None = None) -> ExecutionResult:
        with self._lock:
            if self._closed:
                raise ServiceError("CLOSED", "Execution service is closed")
            decision = self.store.get_decision(decision_id)
            if self.store.decision_mode(decision_id) != self.mode:
                raise ServiceError("EXECUTION_MODE_MISMATCH", "A decision from another execution mode cannot trigger delivery")
            if action_type not in decision.allowed_actions:
                raise ServiceError("ACTION_NOT_ALLOWED", "Action is not allowed by the stored decision")
            key = action_key(decision, action_type)
            if idempotency_key is not None and idempotency_key != key:
                raise ServiceError("INVALID_IDEMPOTENCY_KEY", "Use the server-derived action key", expected_key=key)
            existing = self.store.action_by_key(key)
            if existing:
                return existing
            self.validate(decision, action_type)
            result = ExecutionResult(decision_id=decision_id, event_id=decision.event_id,
                                     action_type=action_type, idempotency_key=key,
                                     channel="dry_run" if self.dry_run else "configured_handler",
                                     accepted_at=utc_now())
            self.store.save_action(result)
            if self.dry_run:
                result.status = "SIMULATED"
                result.started_at = result.completed_at = utc_now()
                result.delivery_receipt = {"simulated": True, "delivered": False}
                self.store.save_action(result)
            else:
                self._enqueue(result, 1)
            return result

    def _enqueue(self, result: ExecutionResult, attempt: int):
        priority = {"FALL_ALARM": 0, "REQUEST_CHECK": 1, "OBSTACLE_REMINDER": 2}.get(result.action_type, 3)
        self._queue.put((priority, next(self._sequence), result.action_id, attempt))

    def _run(self):
        while True:
            _, _, action_id, attempt = self._queue.get()
            try:
                if action_id is None:
                    return
                result = self.store.get_action(action_id)
                decision = self.store.get_decision(result.decision_id)
                try:
                    if self.handler is None:
                        raise ServiceError("CHANNEL_UNAVAILABLE", "No live execution handler is configured")
                    if self.claim:
                        self.claim(decision, result.action_type, result)
                    else:
                        self.validate(decision, result.action_type)
                        result.status, result.started_at = "RUNNING", result.started_at or utc_now()
                        self.store.save_action(result)
                    receipt = self.handler(result.action_type, decision, result.idempotency_key)
                    if not isinstance(receipt, dict):
                        raise ValueError("Handler must return a receipt dictionary")
                    result.delivery_receipt = {**receipt, "attempts": attempt}
                    result.status = "DELIVERED" if receipt.get("delivered") is True else "COMPLETED"
                    result.completed_at, result.error = utc_now(), None
                    self.store.save_action(result)
                except ServiceError as exc:
                    result.status, result.error = "REJECTED", f"{exc.code}: {exc}"
                    result.completed_at = utc_now()
                    self.store.save_action(result)
                except Exception as exc:
                    result.error = f"{type(exc).__name__}: {exc}"
                    if isinstance(getattr(exc, "delivery_receipt", None), dict):
                        result.delivery_receipt = exc.delivery_receipt
                    if attempt < self.max_attempts and not self._closed:
                        result.status = "RETRYING"
                        self.store.save_action(result)
                        # Only this I/O worker sleeps, never perception/policy. An alarm
                        # already queued takes priority when this attempt is requeued.
                        time.sleep(self.retry_delay_s)
                        self._enqueue(result, attempt + 1)
                    else:
                        result.status, result.completed_at = "FAILED", utc_now()
                        self.store.save_action(result)
            finally:
                self._queue.task_done()

    def recover_pending(self):
        """Replay interrupted alarms; cancel non-alarm work until fresh observations."""
        for result in self.store.list_actions():
            if result.status not in {"ACCEPTED", "RUNNING", "RETRYING"}:
                continue
            decision = self.store.get_decision(result.decision_id)
            if self.store.decision_mode(decision.decision_id) != self.mode:
                continue
            replayable = decision.confirmed_alarm and result.action_type == "FALL_ALARM"
            if result.action_type == "REQUEST_CHECK":
                event = self.store.get_event(result.event_id)
                if event.status == "NEEDS_REVIEW" and event.latest_decision_id:
                    current = self.store.get_decision(event.latest_decision_id)
                    if (self.store.decision_mode(current.decision_id) == self.mode
                            and action_key(current, "REQUEST_CHECK") == result.idempotency_key):
                        # Retain the stable action key but bind the pending check
                        # to the current restart decision, not its superseded one.
                        result.decision_id = current.decision_id
                        self.store.save_action(result)
                        replayable = True
            if replayable:
                if self.dry_run:
                    result.status, result.completed_at = "SIMULATED", utc_now()
                    result.delivery_receipt = {"simulated": True, "delivered": False, "recovered": True}
                    self.store.save_action(result)
                else:
                    self._enqueue(result, 1)
            else:
                result.status, result.completed_at = "REJECTED", utc_now()
                result.error = "RESTART_REQUIRES_FRESH_DECISION"
                self.store.save_action(result)
        # A crash may occur after committing a confirmed decision but before
        # creating its action. The decision journal itself is a durable outbox.
        for decision in self.store.list_decisions():
            if self.store.decision_mode(decision.decision_id) != self.mode:
                continue
            if decision.confirmed_alarm or decision.outcome == "REVIEW_REQUIRED":
                for action in decision.allowed_actions:
                    if self.store.action_by_key(action_key(decision, action)) is None:
                        try:
                            self.execute(decision.decision_id, action)
                        except ServiceError:
                            # Superseded non-alarm review results are historical.
                            if decision.confirmed_alarm:
                                raise

    def drain(self):
        """Testing/shutdown hook; handlers must have their own bounded I/O timeout."""
        self._queue.join()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._worker:
            # Drain accepted work; startup can resume pending alarms after a crash.
            self._queue.put((99, next(self._sequence), None, 0))
            self._worker.join()

"""Nine bounded tools. Source-clock, perception and authority are not Agent tools."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError

from .execution import action_key
from .schemas import Contract
from .state_store import ServiceError

if TYPE_CHECKING:
    from .service import MVPService


class StateArgs(Contract):
    person_id: str
    max_age_ms: float = Field(default=500, gt=0)


class EvidenceArgs(Contract):
    event_id: str
    event_revision: int = Field(ge=0)


class PersonArgs(Contract):
    person_id: str


class EvaluateArgs(EvidenceArgs):
    snapshot_id: str


class ReportArgs(PersonArgs):
    report_id: str


class ExecuteArgs(Contract):
    decision_id: str
    action_type: Literal["OBSTACLE_REMINDER", "FALL_ALARM", "REQUEST_CHECK"]
    idempotency_key: str | None = None


class StatusArgs(Contract):
    action_id: str


class NoteArgs(Contract):
    event_id: str
    note: str = Field(min_length=1, max_length=4000)
    # A model cannot attest that its own text came from a human. Human notes go
    # through a separate trusted local integration using StateStore.add_note.
    source: Literal["agent"] = "agent"


class ToolRegistry:
    def __init__(self, service: "MVPService"):
        self.service = service
        self._tools = {
            "get_realtime_state": (StateArgs, "Read a coherent state snapshot; age requirements cannot exceed server policy.", service.realtime_state),
            "get_event_evidence": (EvidenceArgs, "Read retained evidence for an exact event revision.", service.event_evidence),
            "get_personal_context": (PersonArgs, "Read baseline and risk context, preserving unknown fields.", service.personal_context),
            "evaluate_prefall": (EvaluateArgs, "Evaluate existing obstacle event with deterministic rules.", lambda **kw: self._evaluate("prefall", **kw)),
            "evaluate_fall": (EvaluateArgs, "Evaluate existing fall event; WAIT has a fixed deadline.", lambda **kw: self._evaluate("fall", **kw)),
            "get_screening_report": (ReportArgs, "Read an existing immutable screening report.", service.screening_report),
            "execute_intervention": (ExecuteArgs, "Execute only an allowed stored decision with server-validated idempotency.", self._execute),
            "get_execution_status": (StatusArgs, "Read actual execution status; simulation is never delivery.", lambda action_id: service.store.get_action(action_id).model_dump(mode="json")),
            "record_event_note": (NoteArgs, "Append an untrusted Agent note; never replace observations or human evidence.", self._note),
        }

    def _evaluate(self, branch: str, **args):
        decision = self.service.evaluate_branch(branch, **args)
        body = decision.model_dump(mode="json")
        body["action_keys"] = {action: action_key(decision, action) for action in decision.allowed_actions}
        return body

    def _execute(self, **args):
        return self.service.execution.execute(**args).model_dump(mode="json")

    def _note(self, event_id: str, note: str, source: str):
        return self.service.store.add_note(event_id, note, source, self.service.now_s)

    def list_tools(self) -> list[dict]:
        return [{"type": "function", "function": {"name": name, "description": desc,
                  "parameters": model.model_json_schema()}}
                for name, (model, desc, _) in self._tools.items()]

    def call(self, name: str, args: dict) -> dict:
        try:
            self.service._ensure_open()
            if name not in self._tools:
                raise ServiceError("UNKNOWN_TOOL", "Tool is not registered")
            model, _, handler = self._tools[name]
            parsed = model.model_validate(args)
            return handler(**parsed.model_dump())
        except ValidationError as exc:
            return {"error": {"code": "INVALID_ARGUMENTS", "message": str(exc)}}
        except ServiceError as exc:
            return {"error": {"code": exc.code, "message": str(exc), **exc.details}}

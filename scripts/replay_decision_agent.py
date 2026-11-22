"""Replay structured observations through the persistent off/mock/Aliyun loop."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fallpre.adapters.aliyun_decision import (  # noqa: E402
    DecisionProviderError,
    HTTPResponse,
    HTTPSJsonTransport,
)
from fallpre.decision_config import DecisionConfigError, load_decision_config  # noqa: E402
from fallpre.decision_contracts import DecisionEvidence  # noqa: E402
from fallpre.decision_evidence import synthetic_envelope  # noqa: E402
from fallpre.decision_loop import DecisionLoop, DecisionWorker  # noqa: E402
from fallpre.decision_state import (  # noqa: E402
    DECISION_PROMPT_VERSION,
    DECISION_STATE_SCHEMA_VERSION,
    build_decision_state,
)


class CountingTransport:
    """Counts one transport submission per adapter attempt without retaining data."""

    def __init__(self, *, max_submissions: int = 6) -> None:
        self.inner = HTTPSJsonTransport()
        self.submissions: list[dict[str, Any]] = []
        self.max_submissions = max_submissions

    def post_json(self, *, host: str, path: str, headers: dict[str, str], body: bytes,
                  deadline: float) -> HTTPResponse:
        if len(self.submissions) >= self.max_submissions:
            raise DecisionProviderError("SMOKE_HTTP_CAP_REACHED")
        entry: dict[str, Any] = {"host": host, "path": path, "http_status": None, "request_id": None}
        self.submissions.append(entry)
        response = self.inner.post_json(host=host, path=path, headers=headers, body=body, deadline=deadline)
        entry["http_status"] = response.status
        for name, value in response.headers.items():
            if name.lower() in {"x-request-id", "request-id", "x-acs-request-id", "x-dashscope-request-id"}:
                entry["request_id"] = value
                break
        return response


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _demo_inputs(session: str) -> list[dict[str, Any]]:
    full = synthetic_envelope(session_id=session, stream_id="camera-a", source_time_s=10.0,
                              suffix="normal-window", event_state="normal")
    insufficient = synthetic_envelope(session_id=session, stream_id="camera-b", source_time_s=10.0,
                                      suffix="needs-observation", sufficient=False)
    child = synthetic_envelope(session_id=session, stream_id="camera-b", source_time_s=12.0,
                               suffix="new-window-child", sufficient=True)
    return [full, insufficient, child, full]


def _set_scenario_labels(envelope: dict[str, Any], *, action: str, process: str,
                         transition: str, downward: bool) -> dict[str, Any]:
    perception = envelope["perception"]
    perception["current_action"]["label"] = action
    perception["fall_process"]["label"] = process
    geometry = perception["fall_process"]["geometry"]
    geometry["transition"] = transition
    geometry["downward_motion_observed"] = downward
    summary = envelope["evidence_summary"]
    summary.update({
        "action_label": action,
        "fall_process_label": process,
        "geometry_transition": transition,
        "geometry_downward_motion_observed": downward,
    })
    return envelope


def _synthetic_scenarios(session: str) -> list[dict[str, Any]]:
    normal = synthetic_envelope(session_id=session, stream_id="synthetic-normal", source_time_s=100.0,
                                suffix="stage03-normal", event_state="normal")
    _set_scenario_labels(normal, action="standing", process="non_fall_activity",
                         transition="upright", downward=False)
    fall = synthetic_envelope(session_id=session, stream_id="synthetic-fall", source_time_s=100.0,
                              suffix="stage03-fall", event_state="suspected", event_key="fixture-event-02")
    _set_scenario_labels(fall, action="falling", process="fall_process",
                         transition="upright_to_low", downward=True)
    ambiguous = synthetic_envelope(session_id=session, stream_id="synthetic-ambiguous", source_time_s=100.0,
                                   suffix="stage03-ambiguous", event_state="unknown")
    _set_scenario_labels(ambiguous, action="sitting", process="non_fall_activity",
                         transition="low", downward=False)
    insufficient = synthetic_envelope(session_id=session, stream_id="synthetic-insufficient", source_time_s=100.0,
                                      suffix="stage03-quality-insufficient", sufficient=False)
    duplicate_normal = json.loads(_json_line(normal))
    return [normal, fall, ambiguous, insufficient, duplicate_normal]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSONL at line {line_no}") from None
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_no} must be an object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(_json_line(row) + "\n" for row in rows), encoding="utf-8")


def _run_worker_until_terminal(loop: DecisionLoop, worker: DecisionWorker, task_id: str,
                               timeout_s: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for _ in range(2):
        task = loop.store.task(task_id)
        if not task or task["status"] != "pending":
            break
        delay = max(0.0, task["available_at"] - loop.wall_clock())
        if delay:
            remaining = max(0.0, (task.get("network_deadline_at") or loop.wall_clock()) - loop.wall_clock())
            if delay >= remaining:
                # A later worker claim records a durable unavailable result
                # without issuing another HTTP request.
                delay = 0.0
            else:
                time.sleep(delay)
        outcome = worker.run_one(timeout=timeout_s + 5.0)
        if outcome is None:
            continue
        output.append(outcome)
        if outcome.get("task_id") != task_id:
            output[-1]["unexpected_task_id"] = outcome.get("task_id")
        if outcome.get("status") != "retry_pending":
            break
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use full synthetic schema-v1 envelopes")
    source.add_argument("--input-jsonl", type=Path, help="Read schema-v1 envelopes from JSONL")
    source.add_argument("--synthetic-scenarios", action="store_true",
                        help="Run three eligible synthetic scenes, a local quality gate, and a duplicate")
    source.add_argument("--synthetic-scenario", choices=("normal", "fall", "ambiguous", "quality-insufficient"),
                        help="Run exactly one named synthetic scene")
    parser.add_argument("--provider", choices=("off", "mock", "aliyun"), default="off")
    parser.add_argument("--enable-network", action="store_true",
                        help="Explicitly allow Aliyun requests; required with --provider aliyun")
    parser.add_argument("--output", type=Path, help="New artifact directory, or export directory with --resume")
    parser.add_argument("--resume", type=Path, help="Explicitly reuse an existing decision SQLite database")
    parser.add_argument("--session", default="decision-replay-session-01", help="Synthetic producer session")
    parser.add_argument("--run-id", default="decision-replay-run-01")
    parser.add_argument("--http-request-cap", type=int, default=6,
                        help="Hard cap for this synthetic network replay (maximum 6)")
    args = parser.parse_args()

    if args.provider == "aliyun" and not args.enable_network:
        parser.error("--provider aliyun requires --enable-network")
    if args.enable_network and args.provider != "aliyun":
        parser.error("--enable-network is valid only with --provider aliyun")
    if args.enable_network and not (args.synthetic_scenarios or args.synthetic_scenario):
        parser.error("network-enabled replay is limited to explicitly named synthetic inputs")
    if type(args.http_request_cap) is not int or not 1 <= args.http_request_cap <= 6:
        parser.error("--http-request-cap must be from 1 to 6")

    try:
        config = load_decision_config(include_credentials=args.enable_network)
        config = replace(config, provider=args.provider)
        if args.enable_network:
            config.require_aliyun_configuration(provider="aliyun")
    except DecisionConfigError as error:
        print(f"configuration_error: {error}", file=sys.stderr)
        return 2

    if args.resume:
        database = args.resume.resolve()
        if not database.is_file():
            parser.error("--resume must name an existing SQLite database")
        output = args.output.resolve() if args.output else database.parent
        output.mkdir(parents=True, exist_ok=True)
    else:
        if args.output is None:
            parser.error("--output is required unless --resume is supplied")
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output = output.resolve()
        if output.exists():
            parser.error("output directory already exists; choose a new path or explicitly use --resume")
        output.mkdir(parents=True, exist_ok=False)
        database = output / "decision-agent.sqlite3"

    if args.demo:
        envelopes = _demo_inputs(args.session)
    elif args.synthetic_scenarios:
        envelopes = _synthetic_scenarios(args.session)
    elif args.synthetic_scenario:
        named = {
            "normal": 0,
            "fall": 1,
            "ambiguous": 2,
            "quality-insufficient": 3,
        }
        envelopes = [_synthetic_scenarios(args.session)[named[args.synthetic_scenario]]]
    else:
        try:
            envelopes = _read_jsonl(args.input_jsonl)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    trace: list[dict[str, Any]] = []
    transport = CountingTransport(max_submissions=args.http_request_cap) if args.enable_network else None
    service_instance_id = f"decision-service-{uuid.uuid4().hex}"
    provider_calls = 0
    provider_calls_by_stream: dict[str, int] = {}
    try:
        with DecisionLoop(database, run_id=args.run_id, config=config, provider=args.provider,
                          enable_network=args.enable_network, provider_transport=transport,
                          service_instance_id=service_instance_id) as loop:
            worker_context = (DecisionWorker(str(database), config=config, provider=args.provider,
                                             enable_network=args.enable_network, provider_transport=transport,
                                             service_instance_id=service_instance_id)
                              if args.enable_network else None)
            try:
                for index, envelope in enumerate(envelopes):
                    before = len(transport.submissions) if transport else 0
                    receipt = loop.observe(envelope)
                    outputs = []
                    if receipt.get("task_id") and receipt.get("status") != "duplicate":
                        if worker_context:
                            outputs = _run_worker_until_terminal(
                                loop, worker_context, receipt["task_id"], config.timeout_s,
                            )
                        else:
                            outputs = loop.run_pending()
                    after = len(transport.submissions) if transport else 0
                    provider_calls += after - before
                    provider_calls_by_stream[envelope["stream_id"]] = (
                        provider_calls_by_stream.get(envelope["stream_id"], 0) + after - before
                    )
                    task = loop.store.task(receipt["task_id"]) if receipt.get("task_id") else None
                    result_row = loop.store.get_result(receipt.get("decision_id")) if receipt.get("decision_id") else None
                    state = (build_decision_state(DecisionEvidence.model_validate(task["evidence"]))
                             if task and task.get("evidence") else None)
                    trace.append({
                        "event": "observation",
                        "index": index,
                        "input_origin": envelope.get("evidence_origin", "authorized_structured_input"),
                        "provider": args.provider,
                        "prompt_version": DECISION_PROMPT_VERSION if args.provider == "aliyun" else None,
                        "state_schema_version": DECISION_STATE_SCHEMA_VERSION if args.provider == "aliyun" else None,
                        "state": state,
                        "model_called": after > before,
                        "simulated": envelope.get("evidence_origin") == "synthetic_fixture",
                        "receipt": receipt,
                        "task": ({key: task.get(key) for key in (
                            "task_id", "status", "attempt", "max_attempts", "reason",
                            "network_started_at", "network_deadline_at", "network_remaining_s",
                        )} if task else None),
                        "attempt_history": [
                            {key: attempt[key] for key in (
                                "attempt_no", "status", "started_at", "finished_at", "error_code",
                                "request_id", "http_status", "client_latency_ms", "service_latency_ms",
                                "input_tokens", "retry_after_s", "diagnostic_code", "field_path",
                            )}
                            for attempt in loop.store.attempts(receipt["task_id"])
                        ] if receipt.get("task_id") else [],
                        "worker_outcomes": outputs,
                        "latest_result": result_row["provider_result"] if result_row else None,
                    })
                for stream_id in sorted({row["stream_id"] for row in loop.store.observations()}):
                    sessions = sorted({row["session_id"] for row in loop.store.observations()
                                       if row["stream_id"] == stream_id})
                    for session_id in sessions:
                        trace.append({"event": "latest_projection", "view": loop.latest(session_id, stream_id),
                                      "provider": args.provider,
                                      "simulated": any(row["envelope"].get("evidence_origin") == "synthetic_fixture"
                                                       for row in loop.store.observations(session_id=session_id)
                                                       if row["stream_id"] == stream_id),
                                      "model_called": provider_calls_by_stream.get(stream_id, 0) > 0})

                observations = loop.store.observations()
                decisions = []
                for session_id in loop.store.session_ids():
                    decisions.extend(loop.list_decisions(session_id, limit=500))
                _write_jsonl(output / "input-snapshot.jsonl", [row["envelope"] for row in observations])
                _write_jsonl(output / "trace.jsonl", trace)
                _write_jsonl(output / "decisions.jsonl", decisions)
                counts = loop.store.summary_counts()
                status_counts: dict[str, int] = {}
                for row in decisions:
                    result = row.get("provider_result")
                    if not result:
                        continue
                    name = result["status"]
                    status_counts[name] = status_counts.get(name, 0) + 1
                attempts_by_key = {}
                for row in decisions:
                    for attempt_row in row.get("attempt_history", []):
                        attempts_by_key[(attempt_row.get("task_id", row.get("task_id")),
                                         attempt_row["attempt_no"])] = attempt_row
                attempts = list(attempts_by_key.values())
                request_ids = sorted({attempt["request_id"] for attempt in attempts if attempt.get("request_id")})
                input_tokens = sum(attempt.get("input_tokens") or 0 for attempt in attempts)
                service_latency_ms = [attempt["service_latency_ms"] for attempt in attempts
                                      if attempt.get("service_latency_ms") is not None]
                client_latency_ms = [attempt["client_latency_ms"] for attempt in attempts
                                     if attempt.get("client_latency_ms") is not None]
                summary = {
                    "schema_version": "decision-replay-summary.v2",
                    "simulated": args.synthetic_scenarios or args.synthetic_scenario is not None or args.demo,
                    "simulated_means": "synthetic input/execution context; it does not mean model_called=false",
                    "model_called": provider_calls > 0,
                    "provider": args.provider,
                    "network_enabled_explicitly": args.enable_network,
                    "sessions": loop.store.session_ids(),
                    "counts": counts,
                    "decision_status_counts": status_counts,
                    "network": {
                        "transport_submissions": provider_calls,
                        "http_request_cap": args.http_request_cap if args.enable_network else 0,
                        "unique_service_request_ids": request_ids,
                        "input_tokens_total": input_tokens,
                        "service_latency_ms": service_latency_ms,
                        "client_latency_ms": client_latency_ms,
                        "attempts": attempts,
                    },
                    "artifacts": {
                        "input_snapshot": str((output / "input-snapshot.jsonl").resolve()),
                        "trace": str((output / "trace.jsonl").resolve()),
                        "decisions": str((output / "decisions.jsonl").resolve()),
                        "sqlite": str(database.resolve()),
                    },
                }
                (output / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            finally:
                if worker_context:
                    worker_context.close()
    except (ValueError, OSError, DecisionConfigError) as error:
        print(f"replay_error: {error}", file=sys.stderr)
        return 2

    print(f"output={output}")
    print(f"database={database.resolve()}")
    print(f"observations={len(envelopes)} provider={args.provider} "
          f"network_enabled={args.enable_network} transport_submissions={provider_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

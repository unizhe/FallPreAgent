"""Run deterministic SQLite/worker crash recovery scenarios; no network or key use."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fallpre.decision_config import DecisionConfig  # noqa: E402
from fallpre.decision_evidence import synthetic_envelope  # noqa: E402
from fallpre.decision_loop import DecisionLoop, InjectedCrash, InjectedStoreCrash  # noqa: E402
from fallpre.decision_provider import MockDecisionProvider  # noqa: E402


class FakeTime:
    def __init__(self) -> None:
        self.wall = 1_800_000_000.0
        self.mono = 100.0

    def wall_clock(self) -> float:
        return self.wall

    def monotonic_clock(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


class CountingMock:
    def __init__(self, calls: dict[str, int], clock: FakeTime):
        self.calls = calls
        self.inner = MockDecisionProvider(model="recovery-demo-mock", clock=clock.monotonic_clock)

    def assess(self, evidence, *, deadline):
        self.calls[evidence.stream_id] = self.calls.get(evidence.stream_id, 0) + 1
        return self.inner.assess(evidence, deadline=deadline)


def _line(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                             for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output.exists():
        parser.error("output directory already exists; recovery demo preserves prior artifacts")
    output.mkdir(parents=True, exist_ok=False)
    database = output / "decision-recovery.sqlite3"
    trace_path = output / "trace.jsonl"
    inputs: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    fake = FakeTime()
    calls: dict[str, int] = {}
    config = DecisionConfig(provider="mock", model="recovery-demo-mock", max_retries=1,
                            min_interval_s=0.1, timeout_s=1.0)

    # Crash inside the observation transaction after its row exists but before
    # task insertion. Rollback leaves neither row, and replay commits both.
    envelope = synthetic_envelope(session_id="recovery-session", stream_id="crash-before-task-register",
                                  source_time_s=5.0, suffix="crash-before-task-register")
    inputs.append(envelope)
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        monotonic_clock=fake.monotonic_clock, wall_clock=fake.wall_clock,
                        process_id="process-snapshot-rollback")
    try:
        loop.observe(envelope, fail_at="after_snapshot_insert_before_task")
    except InjectedStoreCrash:
        trace.append({"scenario": "after_snapshot_before_task", "fault_injected": True,
                      "observations_after_rollback": loop.store.summary_counts()["observations"],
                      "tasks_after_rollback": loop.store.summary_counts()["tasks"]})
    loop.close()
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        monotonic_clock=fake.monotonic_clock, wall_clock=fake.wall_clock,
                        process_id="process-snapshot-replayed")
    receipt = loop.observe(envelope)
    outcome = loop.run_one()
    trace.append({"scenario": "after_snapshot_before_task", "replayed_receipt": receipt,
                  "recovered_outcome": outcome})
    loop.close()

    # Atomic observation+task commit survives a crash before dispatch.
    envelope = synthetic_envelope(session_id="recovery-session", stream_id="accepted-before-dispatch",
                                  source_time_s=10.0, suffix="accepted-before-dispatch")
    inputs.append(envelope)
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        monotonic_clock=fake.monotonic_clock, wall_clock=fake.wall_clock,
                        process_id="process-before-dispatch")
    receipt = loop.observe(envelope)
    trace.append({"scenario": "accepted_before_dispatch", "receipt": receipt,
                  "fault": "process exits after atomic snapshot+task commit"})
    loop.close()
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        monotonic_clock=fake.monotonic_clock, wall_clock=fake.wall_clock,
                        process_id="process-after-dispatch-restart")
    duplicate = loop.observe(envelope)
    outcome = loop.run_one()
    trace.append({"scenario": "accepted_before_dispatch", "duplicate_receipt": duplicate, "recovered_outcome": outcome})
    loop.close()

    # Crash after the durable running lease but before provider invocation.
    envelope = synthetic_envelope(session_id="recovery-session", stream_id="crash-after-claim",
                                  source_time_s=20.0, suffix="after-claim")
    inputs.append(envelope)
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        provider_impl=CountingMock(calls, fake), monotonic_clock=fake.monotonic_clock,
                        wall_clock=fake.wall_clock, process_id="process-claim-crash")
    receipt = loop.observe(envelope)
    try:
        loop.run_one(fail_at="after_claim")
    except InjectedCrash:
        trace.append({"scenario": "after_claim", "fault_injected": True,
                      "task": loop.store.task(receipt["task_id"])["status"]})
    loop.close()
    fake.advance(6.0)
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        provider_impl=CountingMock(calls, fake), monotonic_clock=fake.monotonic_clock,
                        wall_clock=fake.wall_clock, process_id="process-claim-recovered")
    outcome = loop.run_one()
    trace.append({"scenario": "after_claim", "recovered_outcome": outcome,
                  "attempts": loop.store.attempts(receipt["task_id"])})
    loop.close()

    # A provider return lost before the result transaction may be called again;
    # its logical task still accepts only one immutable final result.
    envelope = synthetic_envelope(session_id="recovery-session", stream_id="provider-return-before-write",
                                  source_time_s=30.0, suffix="provider-return-before-write")
    inputs.append(envelope)
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        provider_impl=CountingMock(calls, fake), monotonic_clock=fake.monotonic_clock,
                        wall_clock=fake.wall_clock, process_id="process-result-lost")
    receipt = loop.observe(envelope)
    try:
        loop.run_one(fail_at="after_provider_before_write")
    except InjectedCrash:
        trace.append({"scenario": "after_provider_before_write", "fault_injected": True,
                      "accepted_results": loop.store.summary_counts()["accepted_results"]})
    loop.close()
    fake.advance(6.0)
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        provider_impl=CountingMock(calls, fake), monotonic_clock=fake.monotonic_clock,
                        wall_clock=fake.wall_clock, process_id="process-result-retried")
    outcome = loop.run_one()
    trace.append({"scenario": "after_provider_before_write", "recovered_outcome": outcome,
                  "provider_calls": calls["provider-return-before-write"],
                  "accepted_results": loop.store.summary_counts()["accepted_results"]})
    loop.close()

    # Commit-before-ack: replaying the observation finds the committed result.
    envelope = synthetic_envelope(session_id="recovery-session", stream_id="commit-before-receipt",
                                  source_time_s=40.0, suffix="commit-before-receipt")
    inputs.append(envelope)
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        provider_impl=CountingMock(calls, fake), monotonic_clock=fake.monotonic_clock,
                        wall_clock=fake.wall_clock, process_id="process-commit")
    receipt = loop.observe(envelope)
    try:
        loop.run_one(fail_at="after_commit_before_receipt")
    except InjectedCrash:
        trace.append({"scenario": "after_commit_before_receipt", "fault_injected": True,
                      "accepted_results": loop.store.summary_counts()["accepted_results"]})
    loop.close()
    loop = DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                        provider_impl=CountingMock(calls, fake), monotonic_clock=fake.monotonic_clock,
                        wall_clock=fake.wall_clock, process_id="process-commit-replay")
    duplicate = loop.observe(envelope)
    rerun = loop.run_pending()
    trace.append({"scenario": "after_commit_before_receipt", "duplicate_receipt": duplicate,
                  "rerun_outcomes": rerun, "provider_calls": calls["commit-before-receipt"],
                  "latest_after_restart": loop.latest("recovery-session", "commit-before-receipt")})
    _write_jsonl(trace_path, [{"simulated": True, "model_called": False, **row} for row in trace])
    _write_jsonl(output / "input-snapshot.jsonl", inputs)
    with DecisionLoop(database, run_id="recovery-demo", config=config, provider="mock",
                      monotonic_clock=fake.monotonic_clock, wall_clock=fake.wall_clock,
                      process_id="process-export") as final:
        decisions = [item for session_id in final.store.session_ids()
                     for item in final.list_decisions(session_id, limit=500)]
        _write_jsonl(output / "decisions.jsonl", decisions)
        counts = final.store.summary_counts()
        summary = {
            "schema_version": "decision-recovery-demo.v1",
            "simulated": True,
            "model_called": False,
            "network_used": False,
            "fault_windows": ["after_snapshot_before_task", "accepted_before_dispatch", "after_claim",
                              "after_provider_before_write", "after_commit_before_receipt"],
            "provider_calls": calls,
            "counts": counts,
            "accepted_results_per_task": final.store.db.execute(
                "SELECT task_id,count(*) FROM decision_results WHERE accepted=1 GROUP BY task_id").fetchall(),
            "artifacts": {"sqlite": str(database.resolve()), "trace": str(trace_path.resolve()),
                          "inputs": str((output / "input-snapshot.jsonl").resolve()),
                          "decisions": str((output / "decisions.jsonl").resolve())},
        }
        summary["accepted_results_per_task"] = {row[0]: row[1] for row in summary["accepted_results_per_task"]}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    print(f"output={output}")
    print(f"database={database.resolve()}")
    print("fault_windows=5 simulated=true model_called=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

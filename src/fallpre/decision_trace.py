"""Opt-in local JSONL tracing for the video decision workflow."""
import json
import logging
import time
from pathlib import Path

_log = logging.getLogger("fallpre.local_decision_trace")
_log.propagate = False


def enable_decision_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)


def trace_decision(stage: str, **fields: object) -> None:
    if _log.handlers:
        _log.info(json.dumps({"time_unix": time.time(), "stage": stage, **fields}, ensure_ascii=False))

"""Explicit placeholder: no fabricated Jev API, score or local model."""
from __future__ import annotations


class JevAdapter:
    def assess(self, snapshot_id: str, event_id: str, event_revision: int,
               deadline_s: float, allowed_actions: list[str]) -> dict:
        return {"status": "UNAVAILABLE", "reason": "Jev endpoint/schema is not configured",
                "snapshot_id": snapshot_id, "event_id": event_id,
                "event_revision": event_revision, "deadline_s": deadline_s,
                "suggestion": None, "model_score": None}

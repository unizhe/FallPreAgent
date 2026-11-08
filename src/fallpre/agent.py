"""Provider-neutral entrypoint for a single Agent's tool calls.

The deterministic fast loop belongs to MVPService and does not wait for an LLM.
A future model client can use describe_tools()/invoke() without importing YOLO.
"""
from __future__ import annotations


class FallPreventionAgent:
    def __init__(self, service):
        self.service = service

    def describe_tools(self) -> list[dict]:
        return self.service.tools.list_tools()

    def invoke(self, name: str, arguments: dict) -> dict:
        return self.service.tools.call(name, arguments)

    def inspect_person(self, person_id: str) -> dict:
        return self.invoke("get_realtime_state", {"person_id": person_id, "max_age_ms": 500})

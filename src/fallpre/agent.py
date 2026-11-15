"""Provider-neutral tool entrypoint for the legacy MVP and CareLoop backends."""
from __future__ import annotations


class FallPreventionAgent:
    def __init__(self, service):
        self.service = service

    def describe_tools(self) -> list[dict]:
        return self.service.tools.list_tools()

    def invoke(self, name: str, arguments: dict) -> dict:
        return self.service.tools.call(name, arguments)

    def inspect_person(self, person_id: str) -> dict:
        """Query the legacy MVPService person tool."""
        return self.invoke("get_realtime_state", {"person_id": person_id, "max_age_ms": 500})

    def observe(self, session_id: str, perception: dict) -> dict:
        """Drive the event bridge when configured with CareLoop."""
        observe = getattr(self.service, "observe", None)
        if observe is None:
            raise TypeError("Perception messages require a CareLoop backend")
        return observe(session_id, perception)

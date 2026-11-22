"""Off/mock providers plus the explicitly enabled P2 Aliyun adapter."""
from __future__ import annotations

import time
from collections.abc import Callable

from .adapters.aliyun_decision import AliyunDecisionProvider
from .decision_config import DecisionConfig
from .decision_contracts import (
    DecisionEvidence,
    DecisionStatus,
    EvidenceSufficiency,
    NextStep,
    ProbabilityKind,
    ProviderMode,
    ProviderResult,
)


class _BaseProvider:
    def __init__(self, *, model: str, clock: Callable[[], float]) -> None:
        self._model = model
        self._clock = clock

    def _result(self, evidence: DecisionEvidence, **values: object) -> ProviderResult:
        return ProviderResult(
            run_id=evidence.run_id,
            session_id=evidence.session_id,
            stream_id=evidence.stream_id,
            snapshot_id=evidence.snapshot_id,
            decision_id=evidence.decision_id,
            action_epoch=evidence.action_epoch,
            generation=evidence.generation,
            source_window=evidence.source_window,
            event_key=evidence.event_key,
            event_state=evidence.event_state,
            recovery_status=evidence.recovery_status,
            model=self._model,
            **values,
        )

    def _deadline_result(self, evidence: DecisionEvidence, provider: ProviderMode) -> ProviderResult:
        return self._result(
            evidence,
            status=DecisionStatus.UNAVAILABLE,
            provider=provider,
            fall_probability=None,
            probability_kind=None,
            evidence_sufficiency=EvidenceSufficiency.NOT_ASSESSED,
            next_step=None,
            reason_codes=["DEADLINE_EXCEEDED"],
        )


class DisabledDecisionProvider(_BaseProvider):
    """Explicitly disabled provider; never consults credentials or network."""

    def assess(self, evidence: DecisionEvidence, *, deadline: float) -> ProviderResult:
        del deadline
        return self._result(
            evidence,
            status=DecisionStatus.SKIPPED,
            provider=ProviderMode.OFF,
            evidence_sufficiency=EvidenceSufficiency.NOT_ASSESSED,
            next_step=None,
            reason_codes=["PROVIDER_DISABLED"],
        )


class MockDecisionProvider(_BaseProvider):
    """Deterministic fixture provider; values are explicitly synthetic."""

    SYNTHETIC_PROBABILITY = 0.86

    def assess(self, evidence: DecisionEvidence, *, deadline: float) -> ProviderResult:
        if self._clock() >= deadline:
            return self._deadline_result(evidence, ProviderMode.MOCK)
        if evidence.eligible:
            return self._result(
                evidence,
                status=DecisionStatus.READY,
                provider=ProviderMode.MOCK,
                fall_probability=self.SYNTHETIC_PROBABILITY,
                fall_probability_raw=None,
                fall_probability_calibrated=None,
                probability_kind=ProbabilityKind.SYNTHETIC_FIXTURE,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                next_step=NextStep.EMIT_RESULT,
                reason_codes=["SYNTHETIC_FIXTURE_ONLY", "UNCALIBRATED_SYNTHETIC_VALUE"],
            )

        conflicts = list(evidence.conflict_reason_codes)
        eligibility_reasons = list(evidence.eligibility_reasons)
        stale = "ST_RESULT_STALE_AT_TRIGGER" in eligibility_reasons
        if conflicts:
            sufficiency = EvidenceSufficiency.CONFLICTING
            next_step = NextStep.REQUEST_REVIEW
            reasons = ["EVIDENCE_CONFLICTING", *conflicts, *eligibility_reasons]
        else:
            sufficiency = EvidenceSufficiency.INSUFFICIENT
            next_step = NextStep.OBSERVE_MORE
            reasons = ["EVIDENCE_INSUFFICIENT", *eligibility_reasons]
        return self._result(
            evidence,
            status=DecisionStatus.STALE if stale else DecisionStatus.INSUFFICIENT,
            provider=ProviderMode.MOCK,
            fall_probability=None,
            fall_probability_raw=None,
            fall_probability_calibrated=None,
            probability_kind=None,
            evidence_sufficiency=sufficiency,
            next_step=next_step,
            reason_codes=list(dict.fromkeys(reasons)),
        )


def build_decision_provider(
    provider: str | ProviderMode,
    *,
    config: DecisionConfig | None = None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    enable_network: bool = False,
    transport: object | None = None,
) -> DisabledDecisionProvider | MockDecisionProvider | AliyunDecisionProvider:
    """Build one selected provider; Aliyun requires an explicit network opt-in."""
    mode = ProviderMode(provider)
    model = config.model if config is not None else "decision-model-preview"
    if mode == ProviderMode.OFF:
        return DisabledDecisionProvider(model=model, clock=clock)
    if mode == ProviderMode.MOCK:
        return MockDecisionProvider(model=model, clock=clock)
    if config is None:
        raise ValueError("aliyun provider requires a DecisionConfig")
    if not enable_network:
        raise ValueError("aliyun provider requires enable_network=True")
    config.require_aliyun_configuration(provider=mode.value)
    return AliyunDecisionProvider(config=config, clock=clock, wall_clock=wall_clock, transport=transport)


__all__ = [
    "AliyunDecisionProvider",
    "DisabledDecisionProvider",
    "MockDecisionProvider",
    "build_decision_provider",
]

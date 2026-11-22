import json

import pytest
from pydantic import ValidationError

from fallpre.decision_contracts import (
    DECISION_ELIGIBILITY_POLICY_VERSION,
    DecisionEvidence,
    DecisionStatus,
    DecisionView,
    EventState,
    ProbabilityKind,
    ProviderResult,
    RecoveryStatus,
    decision_view_from_result,
)
from fallpre.decision_fixtures import synthetic_evidence
from fallpre.decision_provider import MockDecisionProvider, build_decision_provider


def _mock_ready_result():
    return MockDecisionProvider(model="mock-fixture", clock=lambda: 10.0).assess(
        synthetic_evidence(), deadline=11.0
    )


def _model_ready_result():
    return ProviderResult.model_validate(
        {
            "run_id": "run-model",
            "session_id": "session-model",
            "stream_id": "stream-model",
            "snapshot_id": "snapshot-model",
            "decision_id": "decision-model",
            "action_epoch": 3,
            "generation": 2,
            "source_window": {"start_s": 10.0, "end_s": 12.5},
            "status": "ready",
            "provider": "aliyun",
            "model": "decision-model-preview",
            "fall_probability": 0.73,
            "fall_probability_raw": 0.73,
            "probability_kind": "uncalibrated_model_estimate",
            "evidence_sufficiency": "sufficient",
            "next_step": "emit_result",
            "event_state": "normal",
            "recovery_status": "not_established",
        }
    )


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("ready_but_null", {"fall_probability": None}),
        (
            "mock_claims_model_probability",
            {"probability_kind": "uncalibrated_model_estimate", "fall_probability_raw": 0.86},
        ),
        ("off_but_ready", {"provider": "off"}),
        ("ready_but_insufficient", {"evidence_sufficiency": "insufficient"}),
    ],
)
@pytest.mark.parametrize("contract", [ProviderResult, DecisionView])
def test_provider_and_view_reject_the_same_semantic_contradictions(name, changes, contract):
    del name
    result = _mock_ready_result()
    base = (
        result.model_dump(mode="json")
        if contract is ProviderResult
        else decision_view_from_result(result).model_dump(mode="json")
    )
    candidate = dict(base, **changes)
    with pytest.raises(ValidationError):
        contract.model_validate(candidate)
    with pytest.raises(ValidationError):
        contract.model_validate_json(json.dumps(candidate))


def test_ready_mock_and_model_results_roundtrip_through_view_and_json():
    for result, expected_kind in (
        (_mock_ready_result(), ProbabilityKind.SYNTHETIC_FIXTURE),
        (_model_ready_result(), ProbabilityKind.UNCALIBRATED_MODEL_ESTIMATE),
    ):
        restored_result = ProviderResult.model_validate_json(result.model_dump_json())
        view = decision_view_from_result(result)
        restored_view = DecisionView.model_validate_json(view.model_dump_json())
        assert restored_result == result
        assert restored_view.current_applicable is True
        assert restored_view.probability_kind == expected_kind
        assert restored_view.fall_probability == result.fall_probability


@pytest.mark.parametrize(
    ("status", "provider", "sufficiency", "next_step"),
    [
        ("pending", "aliyun", "not_assessed", None),
        ("insufficient", "mock", "insufficient", "observe_more"),
        ("unavailable", "aliyun", "not_assessed", None),
        ("stale", "aliyun", "not_assessed", None),
        ("skipped", "off", "not_assessed", None),
        ("cancelled", "aliyun", "not_assessed", None),
    ],
)
def test_all_non_ready_statuses_roundtrip_with_null_current_probability(status, provider, sufficiency, next_step):
    result = ProviderResult.model_validate(
        {
            "run_id": "run-non-ready",
            "session_id": "session-non-ready",
            "stream_id": "stream-non-ready",
            "snapshot_id": "snapshot-non-ready",
            "decision_id": "decision-non-ready",
            "action_epoch": 1,
            "generation": 1,
            "source_window": {"start_s": 4.0, "end_s": 5.5},
            "status": status,
            "provider": provider,
            "model": "decision-model-preview",
            "fall_probability": None,
            "probability_kind": None,
            "evidence_sufficiency": sufficiency,
            "next_step": next_step,
            "reason_codes": ["P0_FIXTURE_STATUS"],
        }
    )
    restored = ProviderResult.model_validate_json(result.model_dump_json())
    view = decision_view_from_result(restored)
    restored_view = DecisionView.model_validate_json(view.model_dump_json())
    assert restored.status == getattr(DecisionStatus, status.upper())
    assert restored.fall_probability is None
    assert restored_view.status == restored.status
    assert restored_view.fall_probability is None
    assert restored_view.current_applicable is False


def _evidence_with(**changes):
    values = synthetic_evidence().model_dump(mode="json")
    values.update(changes)
    return DecisionEvidence.model_validate(values)


@pytest.mark.parametrize(("observations", "eligible"), [(7, False), (8, True)])
def test_st_gate_counts_real_observations_not_resampled_positions(observations, eligible):
    evidence = _evidence_with(observed_frames=observations)
    assert evidence.eligibility_policy_version == DECISION_ELIGIBILITY_POLICY_VERSION
    assert evidence.eligible is eligible
    if not eligible:
        assert "ST_REAL_OBSERVATIONS_BELOW_MINIMUM" in evidence.eligibility_reasons


@pytest.mark.parametrize(
    ("span_s", "eligible"),
    [
        (1.5 - 2e-6, False),
        (1.5 - 0.5e-6, True),
        (1.5, True),
        (2.5, True),
        (2.5 + 0.5e-6, True),
        (2.5 + 2e-6, False),
    ],
)
def test_st_window_span_boundaries_and_float_tolerance(span_s, eligible):
    evidence = _evidence_with(source_window={"start_s": 12.5 - span_s, "end_s": 12.5})
    assert evidence.eligible is eligible
    if not eligible:
        assert any(
            reason in evidence.eligibility_reasons
            for reason in ("ST_WINDOW_SPAN_BELOW_MINIMUM", "ST_WINDOW_SPAN_ABOVE_MAXIMUM")
        )


@pytest.mark.parametrize(
    ("age_s", "eligible"),
    [
        (0.6 - 5e-9, True),
        (0.6, True),
        (0.6 + 0.5e-9, True),
        (0.6 + 2e-9, False),
    ],
)
def test_st_trigger_freshness_uses_source_time_and_float_tolerance(age_s, eligible):
    evidence = _evidence_with(source_time_s=12.5 + age_s)
    assert evidence.eligible is eligible
    if not eligible:
        assert "ST_RESULT_STALE_AT_TRIGGER" in evidence.eligibility_reasons


def test_stale_st_evidence_returns_stale_with_null_current_probability():
    evidence = _evidence_with(source_time_s=13.5)
    assert not evidence.eligible
    assert "ST_RESULT_STALE_AT_TRIGGER" in evidence.eligibility_reasons
    result = build_decision_provider("mock", clock=lambda: 10.0).assess(evidence, deadline=11.0)
    assert result.status == DecisionStatus.STALE
    assert result.fall_probability is None
    assert "ST_RESULT_STALE_AT_TRIGGER" in result.reason_codes


def test_future_window_is_rejected_as_invalid_source_clock_evidence():
    with pytest.raises(ValidationError, match="cannot end after source_time_s"):
        _evidence_with(source_window={"start_s": 12.0, "end_s": 12.5001})


@pytest.mark.parametrize("event_state", list(EventState))
@pytest.mark.parametrize("recovery_status", list(RecoveryStatus))
def test_upstream_event_and_recovery_domains_roundtrip_independently(event_state, recovery_status):
    changes = {"event_state": event_state.value, "recovery_status": recovery_status.value}
    if event_state == EventState.UNOBSERVABLE:
        changes.update(observable=False, event_decidable=False)
    evidence = _evidence_with(**changes)
    result = build_decision_provider("mock", clock=lambda: 10.0).assess(evidence, deadline=11.0)
    view = decision_view_from_result(result)
    restored = DecisionView.model_validate_json(view.model_dump_json())
    assert evidence.event_state == event_state
    assert result.event_state == view.event_state == restored.event_state == event_state
    assert result.recovery_status == view.recovery_status == restored.recovery_status == recovery_status


def test_recovered_is_not_an_event_or_recovery_status_alias():
    with pytest.raises(ValidationError):
        _evidence_with(event_state="recovered")
    with pytest.raises(ValidationError):
        _evidence_with(recovery_status="recovered")
    low_support = _evidence_with(event_state="confirmed", recovery_status="low_support_only")
    result = build_decision_provider("mock", clock=lambda: 10.0).assess(low_support, deadline=11.0)
    assert result.event_state == EventState.CONFIRMED
    assert result.recovery_status == RecoveryStatus.LOW_SUPPORT_ONLY

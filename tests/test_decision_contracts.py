import pytest
from pydantic import ValidationError

from fallpre.decision_contracts import (
    DecisionEvidence,
    DecisionStatus,
    DecisionView,
    EvidenceStatus,
    EvidenceSufficiency,
    IdentityLink,
    NextStep,
    ProbabilityKind,
    ProviderMode,
    ProviderResult,
    decision_view_from_result,
)
from fallpre.decision_fixtures import synthetic_evidence
from fallpre.decision_provider import DisabledDecisionProvider, MockDecisionProvider


def _ready_model_result(probability=0.73, **overrides):
    values = {
        "run_id": "run-1",
        "session_id": "session-1",
        "stream_id": "stream-1",
        "snapshot_id": "snapshot-1",
        "decision_id": "decision-1",
        "action_epoch": 4,
        "generation": 2,
        "source_window": {"start_s": 10.0, "end_s": 12.5},
        "status": "ready",
        "provider": "aliyun",
        "model": "decision-model-preview",
        "fall_probability": probability,
        "fall_probability_raw": probability,
        "fall_probability_calibrated": None,
        "probability_kind": "uncalibrated_model_estimate",
        "evidence_sufficiency": "sufficient",
        "next_step": "emit_result",
    }
    values.update(overrides)
    return ProviderResult.model_validate(values)


def test_evidence_contract_requires_identity_window_quality_and_versions():
    evidence = synthetic_evidence()
    assert evidence.schema_version == "decision-evidence.v1"
    assert evidence.eligible
    assert evidence.source_time_s >= evidence.source_window.end_s
    assert evidence.action_epoch == 3
    assert evidence.generation == 2
    assert evidence.temporal_status == EvidenceStatus.READY
    assert evidence.identity_link == IdentityLink.SAME_TARGET

    values = evidence.model_dump()
    values.pop("stream_id")
    with pytest.raises(ValidationError):
        DecisionEvidence.model_validate(values)


@pytest.mark.parametrize(
    "window",
    [
        {"start_s": 12.5, "end_s": 12.5},
        {"start_s": 13.0, "end_s": 12.5},
    ],
)
def test_inverted_or_empty_source_windows_are_rejected(window):
    values = synthetic_evidence().model_dump()
    values["source_window"] = window
    with pytest.raises(ValidationError):
        DecisionEvidence.model_validate(values)


def test_source_window_cannot_extend_beyond_evidence_source_time():
    values = synthetic_evidence().model_dump()
    values["source_time_s"] = 12.4
    with pytest.raises(ValidationError, match="cannot end after source_time_s"):
        DecisionEvidence.model_validate(values)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_probability_endpoints_are_valid_and_raw_semantics_are_explicit(probability):
    result = _ready_model_result(probability)
    assert result.fall_probability == probability
    assert result.fall_probability_raw == probability
    assert result.fall_probability_calibrated is None
    assert result.probability_kind == ProbabilityKind.UNCALIBRATED_MODEL_ESTIMATE
    assert result.probability_target == "observed_fall_in_window"
    assert result.changes_alarm is False
    assert result.external_action_performed is False


@pytest.mark.parametrize("probability", [True, -0.01, 1.01, float("nan"), float("inf")])
def test_invalid_or_boolean_probabilities_are_rejected(probability):
    with pytest.raises(ValidationError):
        _ready_model_result(probability)


def test_calibrated_probability_requires_version_and_matches_current_value():
    with pytest.raises(ValidationError, match="calibration_version"):
        _ready_model_result(
            0.68,
            probability_kind="calibrated_model_estimate",
            fall_probability_raw=0.73,
            fall_probability_calibrated=0.68,
        )
    calibrated = _ready_model_result(
        0.68,
        probability_kind="calibrated_model_estimate",
        fall_probability_raw=0.73,
        fall_probability_calibrated=0.68,
        calibration_version="cal-v1",
    )
    assert calibrated.fall_probability == calibrated.fall_probability_calibrated


def test_non_ready_results_cannot_publish_probability_or_mock_as_model():
    with pytest.raises(ValidationError, match="non-ready result"):
        _ready_model_result(status="insufficient")
    with pytest.raises(ValidationError, match="requires provider=mock"):
        _ready_model_result(probability_kind="synthetic_fixture")


def test_mock_full_fixture_is_explicitly_synthetic_and_bound_to_input():
    evidence = synthetic_evidence()
    provider = MockDecisionProvider(model="decision-model-preview", clock=lambda: 10.0)
    result = provider.assess(evidence, deadline=11.0)

    assert result.status == DecisionStatus.READY
    assert result.provider == ProviderMode.MOCK
    assert result.probability_kind == ProbabilityKind.SYNTHETIC_FIXTURE
    assert result.fall_probability == 0.86
    assert result.fall_probability_raw is None
    assert result.run_id == evidence.run_id
    assert result.snapshot_id == evidence.snapshot_id
    assert result.source_window == evidence.source_window
    assert "SYNTHETIC_FIXTURE_ONLY" in result.reason_codes


def test_mock_insufficient_evidence_returns_null_and_observe_more():
    evidence = synthetic_evidence(sufficient=False)
    result = MockDecisionProvider(model="mock-fixture", clock=lambda: 10.0).assess(evidence, deadline=11.0)
    assert result.status == DecisionStatus.INSUFFICIENT
    assert result.fall_probability is None
    assert result.next_step == NextStep.OBSERVE_MORE
    assert result.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT


def test_off_result_is_skipped_and_view_has_no_current_probability():
    evidence = synthetic_evidence()
    result = DisabledDecisionProvider(model="decision-model-preview", clock=lambda: 10.0).assess(
        evidence, deadline=11.0
    )
    view = decision_view_from_result(result)
    assert result.status == DecisionStatus.SKIPPED
    assert result.fall_probability is None
    assert result.next_step is None
    assert view.status == DecisionStatus.SKIPPED
    assert view.current_applicable is False
    assert view.fall_probability is None
    assert view.schema_version == "agent-decision.v1"
    assert view.changes_alarm is False
    assert view.external_action_performed is False


def test_deadline_uses_injected_monotonic_clock():
    result = MockDecisionProvider(model="mock-fixture", clock=lambda: 9.0).assess(
        synthetic_evidence(), deadline=9.0
    )
    assert result.status == DecisionStatus.UNAVAILABLE
    assert result.reason_codes == ["DEADLINE_EXCEEDED"]
    assert result.fall_probability is None


def test_decision_view_rejects_non_ready_current_probability():
    values = decision_view_from_result(
        DisabledDecisionProvider(model="mock", clock=lambda: 1.0).assess(
            synthetic_evidence(), deadline=2.0
        )
    ).model_dump()
    values["fall_probability"] = 0.0
    with pytest.raises(ValidationError, match="non-ready result"):
        DecisionView.model_validate(values)

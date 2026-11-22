"""Versioned evidence, provider and view contracts for decision P0.

These contracts wrap perception evidence. They intentionally do not modify
the existing perception schema v1 or grant authority over its event lifecycle.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
SourceSeconds = Annotated[StrictFloat, Field(ge=0.0)]

# Frozen P0 gate copied from the existing TemporalConfig and
# FallEvidenceConfig/PerceptionChain boundaries. These are source-clock checks;
# they are separate from future wall-clock result-age budgets.
DECISION_ELIGIBILITY_POLICY_VERSION = "decision-eligibility-p0-v1"
MIN_ST_REAL_OBSERVATIONS = 8
MIN_ST_WINDOW_SPAN_S = 1.5
MAX_ST_WINDOW_SPAN_S = 2.5
MAX_ST_RESULT_AGE_AT_TRIGGER_S = 0.6
ST_WINDOW_SPAN_TOLERANCE_S = 1e-6
ST_RESULT_AGE_TOLERANCE_S = 1e-9


class DecisionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class ProviderMode(str, Enum):
    OFF = "off"
    MOCK = "mock"
    ALIYUN = "aliyun"


class DecisionStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class EvidenceStatus(str, Enum):
    READY = "ready"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    UNKNOWN = "unknown"


class IdentityLink(str, Enum):
    SAME_TARGET = "same_target"
    DIFFERENT_TARGET = "different_target"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class ProbabilityKind(str, Enum):
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    UNCALIBRATED_MODEL_ESTIMATE = "uncalibrated_model_estimate"
    CALIBRATED_MODEL_ESTIMATE = "calibrated_model_estimate"


class EvidenceSufficiency(str, Enum):
    NOT_ASSESSED = "not_assessed"
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


class NextStep(str, Enum):
    EMIT_RESULT = "emit_result"
    OBSERVE_MORE = "observe_more"
    REQUEST_REVIEW = "request_review"


class EventState(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    UNOBSERVABLE = "unobservable"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"


class RecoveryStatus(str, Enum):
    UNKNOWN = "unknown"
    CONFIRMED = "confirmed"
    RECOVERING = "recovering"
    LOW_SUPPORT_ONLY = "low_support_only"
    NOT_ESTABLISHED = "not_established"


class GeometryTransition(str, Enum):
    UPRIGHT_TO_LOW = "upright_to_low"
    LOW = "low"
    UPRIGHT = "upright"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class NoulAnswer(DecisionContract):
    type: Literal["noul"]
    noul: Probability


class ChoiceAnswer(DecisionContract):
    type: Literal["choice"]
    choice: NonEmpty
    confidence: Probability
    probabilities: dict[NonEmpty, Probability]


class DecisionAnswers(DecisionContract):
    fall_occurred: NoulAnswer
    evidence_sufficiency: ChoiceAnswer
    next_step: ChoiceAnswer


ProviderDiagnosticCode = Literal[
    "ANSWER_SET_INVALID", "FIELD_MISSING", "FIELD_TYPE", "ANSWER_TYPE",
    "ENUM_VALUE_INVALID", "NUMBER_TYPE", "NUMBER_RANGE",
    "DISTRIBUTION_KEYS_INVALID", "DISTRIBUTION_SUM_INVALID",
    "USAGE_TOKENS_INVALID", "LATENCY_TYPE_INVALID", "LATENCY_RANGE_INVALID",
    "MODEL_ECHO_MISMATCH", "REQUEST_ID_MISSING_OR_INVALID", "MODEL_VALIDATION_FAILED",
]
ProviderFieldPath = Literal[
    "model", "request_id", "answers", "answers.fall_occurred",
    "answers.fall_occurred.type", "answers.fall_occurred.noul",
    "answers.evidence_sufficiency", "answers.evidence_sufficiency.type",
    "answers.evidence_sufficiency.choice", "answers.evidence_sufficiency.probabilities",
    "answers.evidence_sufficiency.confidence", "answers.next_step",
    "answers.next_step.type", "answers.next_step.choice", "answers.next_step.probabilities",
    "answers.next_step.confidence", "usage", "usage.input_tokens", "latency_ms",
]


class ProviderMetadata(DecisionContract):
    request_id: NonEmpty | None = None
    http_status: Annotated[StrictInt, Field(ge=100, le=599)] | None = None
    input_tokens: Annotated[StrictInt, Field(ge=0)] | None = None
    service_latency_ms: Annotated[StrictFloat, Field(ge=0)] | None = None
    client_latency_ms: Annotated[StrictFloat, Field(ge=0)] | None = None
    diagnostic_code: ProviderDiagnosticCode | None = None
    field_path: ProviderFieldPath | None = None
    answers: DecisionAnswers | None = None


class SourceWindow(DecisionContract):
    start_s: SourceSeconds
    end_s: SourceSeconds

    @model_validator(mode="after")
    def ordered(self) -> SourceWindow:
        if self.end_s <= self.start_s:
            raise ValueError("source window end must be after start")
        return self


class DecisionEvidence(DecisionContract):
    """Immutable-by-convention evidence snapshot for one source-time window."""

    schema_version: Literal["decision-evidence.v1"] = "decision-evidence.v1"
    run_id: NonEmpty
    session_id: NonEmpty
    stream_id: NonEmpty
    snapshot_id: NonEmpty
    decision_id: NonEmpty
    request_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    action_epoch: Annotated[StrictInt, Field(ge=0)]
    generation: Annotated[StrictInt, Field(ge=0)]
    fall_target_epoch: NonEmpty | None = None
    identity_link: IdentityLink

    source_time_s: SourceSeconds
    source_window: SourceWindow
    observed_frames: Annotated[
        StrictInt,
        Field(ge=1, description="Real source observations contributing to ST; excludes resampled model positions."),
    ]
    eligibility_policy_version: Literal[DECISION_ELIGIBILITY_POLICY_VERSION] = DECISION_ELIGIBILITY_POLICY_VERSION

    observable: StrictBool
    event_decidable: StrictBool
    camera_stable: StrictBool
    person_count: Annotated[StrictInt, Field(ge=0)]
    pose_valid_fraction: Probability
    continuous: StrictBool
    missing_reason_codes: list[NonEmpty] = Field(default_factory=list)

    temporal_status: EvidenceStatus
    temporal_model_version: NonEmpty | None = None
    temporal_score: Probability | None = None
    head_status: EvidenceStatus
    head_variant: Literal["tail_joint", "cd"]
    head_version: NonEmpty | None = None
    head_weights_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    head_raw_score: Probability | None = None
    conflict_reason_codes: list[NonEmpty] = Field(default_factory=list)

    event_key: NonEmpty | None = None
    event_state: EventState = EventState.UNKNOWN
    recovery_status: RecoveryStatus = RecoveryStatus.UNKNOWN
    alarm_authority: Literal["upstream_only"] = "upstream_only"

    yolo_model_version: NonEmpty
    preprocessing_version: NonEmpty
    prompt_version: NonEmpty
    policy_version: NonEmpty
    shared_pose_evidence: StrictBool
    action_label: Literal[
        "standing", "sitting", "lying", "walking", "falling", "sit_to_stand",
        "standing_to_sitting", "unknown", "other",
    ] | None = None
    fall_process_label: Literal["fall_process", "non_fall_activity", "unknown", "other"] | None = None
    geometry_status: EvidenceStatus | None = None
    geometry_transition: GeometryTransition | None = None
    geometry_downward_motion_observed: StrictBool | None = None
    geometry_score: Probability | None = None
    geometry_conflict_reason_codes: list[NonEmpty] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window_and_ready_versions(self) -> DecisionEvidence:
        if self.source_time_s < self.source_window.end_s:
            raise ValueError("source window cannot end after source_time_s")
        if self.temporal_status == EvidenceStatus.READY and not self.temporal_model_version:
            raise ValueError("ready temporal evidence requires temporal_model_version")
        if self.head_status == EvidenceStatus.READY and not self.head_version:
            raise ValueError("ready head evidence requires head_version")
        return self

    @property
    def eligibility_reasons(self) -> tuple[str, ...]:
        reasons = list(self.missing_reason_codes)
        if not self.observable:
            reasons.append("OBSERVATION_UNAVAILABLE")
        if not self.event_decidable:
            reasons.append("EVENT_NOT_DECIDABLE")
        if not self.camera_stable:
            reasons.append("CAMERA_UNSTABLE")
        if self.person_count != 1:
            reasons.append("PERSON_COUNT_NOT_UNIQUE")
        if self.pose_valid_fraction < 0.5:
            reasons.append("POSE_QUALITY_INSUFFICIENT")
        if not self.continuous:
            reasons.append("OBSERVATION_NOT_CONTINUOUS")
        if self.identity_link != IdentityLink.SAME_TARGET:
            reasons.append("IDENTITY_NOT_VERIFIED")
        if self.observed_frames < MIN_ST_REAL_OBSERVATIONS:
            reasons.append("ST_REAL_OBSERVATIONS_BELOW_MINIMUM")
        span_s = self.source_window.end_s - self.source_window.start_s
        if span_s < MIN_ST_WINDOW_SPAN_S - ST_WINDOW_SPAN_TOLERANCE_S:
            reasons.append("ST_WINDOW_SPAN_BELOW_MINIMUM")
        if span_s > MAX_ST_WINDOW_SPAN_S + ST_WINDOW_SPAN_TOLERANCE_S:
            reasons.append("ST_WINDOW_SPAN_ABOVE_MAXIMUM")
        if self.temporal_result_age_at_trigger_s > (
            MAX_ST_RESULT_AGE_AT_TRIGGER_S + ST_RESULT_AGE_TOLERANCE_S
        ):
            reasons.append("ST_RESULT_STALE_AT_TRIGGER")
        if self.temporal_status != EvidenceStatus.READY or self.temporal_score is None:
            reasons.append("TEMPORAL_EVIDENCE_NOT_READY")
        if self.head_status != EvidenceStatus.READY or self.head_raw_score is None:
            reasons.append("FALL_HEAD_EVIDENCE_NOT_READY")
        return tuple(dict.fromkeys(reasons))

    @property
    def temporal_result_age_at_trigger_s(self) -> float:
        """Age between source-clock trigger and ST window end; never wall time."""
        return self.source_time_s - self.source_window.end_s

    @property
    def eligible(self) -> bool:
        return not self.eligibility_reasons and not self.conflict_reason_codes and not self.geometry_conflict_reason_codes


class DecisionOutcome(DecisionContract):
    """Fields and semantic validation shared by provider and public view."""

    run_id: NonEmpty
    session_id: NonEmpty
    stream_id: NonEmpty
    snapshot_id: NonEmpty
    decision_id: NonEmpty
    action_epoch: Annotated[StrictInt, Field(ge=0)]
    generation: Annotated[StrictInt, Field(ge=0)]
    source_window: SourceWindow

    status: DecisionStatus
    provider: ProviderMode
    model: NonEmpty
    fall_probability: Probability | None = None
    fall_probability_raw: Probability | None = None
    fall_probability_calibrated: Probability | None = None
    probability_kind: ProbabilityKind | None = None
    probability_target: Literal["observed_fall_in_window"] = "observed_fall_in_window"
    calibration_version: NonEmpty | None = None
    event_key: NonEmpty | None = None
    event_state: EventState = EventState.UNKNOWN
    recovery_status: RecoveryStatus = RecoveryStatus.UNKNOWN
    evidence_sufficiency: EvidenceSufficiency
    next_step: NextStep | None
    reason_codes: list[NonEmpty] = Field(default_factory=list)
    provider_metadata: ProviderMetadata | None = None
    changes_alarm: Literal[False] = False
    external_action_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_probability_semantics(self) -> DecisionOutcome:
        if self.status == DecisionStatus.READY:
            if self.fall_probability is None or self.probability_kind is None:
                raise ValueError("ready result requires a current probability and probability_kind")
            if self.evidence_sufficiency != EvidenceSufficiency.SUFFICIENT:
                raise ValueError("ready result requires sufficient evidence")
            if self.next_step != NextStep.EMIT_RESULT:
                raise ValueError("ready result requires next_step=emit_result")
        elif self.fall_probability is not None:
            raise ValueError("non-ready result must not expose a current probability")

        if self.status == DecisionStatus.SKIPPED and self.provider != ProviderMode.OFF:
            raise ValueError("skipped status requires provider=off")
        if self.status == DecisionStatus.INSUFFICIENT:
            if self.evidence_sufficiency not in {
                EvidenceSufficiency.INSUFFICIENT,
                EvidenceSufficiency.CONFLICTING,
            }:
                raise ValueError("insufficient status requires insufficient or conflicting evidence")
            expected_step = (
                NextStep.REQUEST_REVIEW
                if self.evidence_sufficiency == EvidenceSufficiency.CONFLICTING
                else NextStep.OBSERVE_MORE
            )
            if self.next_step != expected_step:
                raise ValueError("insufficient status requires a matching next_step")

        if self.provider == ProviderMode.OFF:
            if self.status != DecisionStatus.SKIPPED:
                raise ValueError("provider=off requires status=skipped")
            if any(
                value is not None
                for value in (
                    self.fall_probability,
                    self.fall_probability_raw,
                    self.fall_probability_calibrated,
                    self.probability_kind,
                    self.calibration_version,
                )
            ):
                raise ValueError("provider=off cannot carry probabilities or probability source")
            if self.evidence_sufficiency != EvidenceSufficiency.NOT_ASSESSED or self.next_step is not None:
                raise ValueError("provider=off cannot claim evidence sufficiency or recommend a next step")

        if self.probability_kind == ProbabilityKind.SYNTHETIC_FIXTURE:
            if self.provider != ProviderMode.MOCK:
                raise ValueError("synthetic_fixture probability requires provider=mock")
            if (
                self.fall_probability_raw is not None
                or self.fall_probability_calibrated is not None
                or self.calibration_version is not None
            ):
                raise ValueError("synthetic_fixture cannot carry model scores or calibration metadata")
        elif self.probability_kind == ProbabilityKind.UNCALIBRATED_MODEL_ESTIMATE:
            if self.provider != ProviderMode.ALIYUN:
                raise ValueError("uncalibrated model estimate requires provider=aliyun")
            if self.calibration_version is not None:
                raise ValueError("uncalibrated result cannot carry calibration_version")
            if self.status == DecisionStatus.READY and self.fall_probability_raw is None:
                raise ValueError("uncalibrated model estimate requires fall_probability_raw")
            if self.fall_probability_calibrated is not None:
                raise ValueError("uncalibrated result cannot carry a calibrated probability")
            if self.status == DecisionStatus.READY and self.fall_probability != self.fall_probability_raw:
                raise ValueError("current probability must equal raw probability before calibration")
        elif self.probability_kind == ProbabilityKind.CALIBRATED_MODEL_ESTIMATE:
            if self.provider != ProviderMode.ALIYUN:
                raise ValueError("calibrated model estimate requires provider=aliyun")
            if (self.calibration_version is None) != (self.fall_probability_calibrated is None):
                raise ValueError("calibrated probability and calibration_version must be supplied together")
            if self.status == DecisionStatus.READY:
                if self.calibration_version is None or self.fall_probability_calibrated is None:
                    raise ValueError("calibrated result requires calibration_version and calibrated probability")
                if self.fall_probability != self.fall_probability_calibrated:
                    raise ValueError("current probability must equal calibrated probability")
        elif self.status == DecisionStatus.READY:
            raise ValueError("ready result requires an explicit probability source")

        if self.fall_probability_raw is not None and self.provider != ProviderMode.ALIYUN:
            raise ValueError("raw model probability requires provider=aliyun")
        if self.fall_probability_calibrated is not None and self.probability_kind != ProbabilityKind.CALIBRATED_MODEL_ESTIMATE:
            raise ValueError("calibrated probability requires calibrated_model_estimate source")
        if self.calibration_version is not None and self.probability_kind != ProbabilityKind.CALIBRATED_MODEL_ESTIMATE:
            raise ValueError("calibration_version requires calibrated_model_estimate source")
        return self


class ProviderResult(DecisionOutcome):
    """Provider answer tied to the exact identity and source window assessed."""

    schema_version: Literal["provider-result.v1"] = "provider-result.v1"


class DecisionView(DecisionOutcome):
    """UI/API projection; shares all provider-result semantic validation."""

    schema_version: Literal["agent-decision.v1"] = "agent-decision.v1"
    current_applicable: StrictBool

    @model_validator(mode="after")
    def current_probability_is_applicable(self) -> DecisionView:
        expected = self.status == DecisionStatus.READY
        if self.current_applicable != expected:
            raise ValueError("current_applicable must match ready status")
        if not self.current_applicable and self.fall_probability is not None:
            raise ValueError("non-applicable decision view must have null fall_probability")
        return self


def decision_view_from_result(result: ProviderResult) -> DecisionView:
    """Create a non-persistent view projection from one provider result."""
    values = result.model_dump(exclude={"schema_version"})
    values["current_applicable"] = result.status == DecisionStatus.READY
    return DecisionView.model_validate(values)


@runtime_checkable
class DecisionProvider(Protocol):
    """Decision provider. ``deadline`` uses the injected monotonic clock domain."""

    def assess(self, evidence: DecisionEvidence, *, deadline: float) -> ProviderResult:
        """Assess one fixed evidence window before the monotonic deadline."""


__all__ = [
    "DecisionEvidence",
    "DecisionAnswers",
    "DecisionProvider",
    "DecisionStatus",
    "DecisionView",
    "EvidenceSufficiency",
    "EvidenceStatus",
    "GeometryTransition",
    "IdentityLink",
    "NextStep",
    "ProbabilityKind",
    "ProviderMode",
    "ProviderMetadata",
    "ProviderResult",
    "SourceWindow",
    "decision_view_from_result",
]

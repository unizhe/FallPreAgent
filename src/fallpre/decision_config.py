"""Configuration for the P0 decision contract.

The project's .env file is parsed as data. It is never executed as shell code,
and the secret value is only exposed through an explicitly named accessor for a
future provider implementation.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_REGIONS = frozenset({"cn-beijing"})
_PROVIDERS = frozenset({"off", "mock", "aliyun"})
_HEADS = frozenset({"tail_joint", "cd"})
_ENV_NAMES = frozenset(
    {
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_WORKSPACE_ID",
        "DASHSCOPE_REGION",
        "FALLPRE_DECISION_MODEL",
        "FALLPRE_DECISION_PROVIDER",
        "FALLPRE_FALL_HEAD",
        "FALLPRE_DECISION_TIMEOUT_S",
        "FALLPRE_DECISION_MAX_RETRIES",
        "FALLPRE_DECISION_MIN_INTERVAL_S",
        "FALLPRE_DECISION_MAX_SOURCE_AGE_S",
        "FALLPRE_DECISION_MAX_WALL_AGE_S",
        "FALLPRE_DECISION_MAX_REOBSERVATIONS",
        "FALLPRE_AGENT_DB",
    }
)
_WORKSPACE_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_ALLOWED_MODEL = "decision-model-preview"


class DecisionConfigError(ValueError):
    """A safe configuration error that contains names, never values."""


class SecretValue:
    """Opaque secret wrapper with safe string and repr representations."""

    __slots__ = ("__value",)

    def __init__(self, value: str | None = None) -> None:
        self.__value = value

    def get_secret_value(self) -> str | None:
        """Return the secret for an explicitly authorized provider boundary."""
        return self.__value

    def __repr__(self) -> str:
        return "SecretValue(<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"

    def __bool__(self) -> bool:
        return bool(self.__value)


@dataclass(frozen=True, repr=False)
class DecisionConfig:
    api_key: SecretValue = field(default_factory=SecretValue, repr=False)
    workspace_id: str | None = None
    region: str = "cn-beijing"
    model: str = _ALLOWED_MODEL
    provider: str = "off"
    fall_head: str = "tail_joint"
    timeout_s: float = 3.0
    max_retries: int = 1
    min_interval_s: float = 2.0
    max_source_age_s: float = 2.5
    max_wall_age_s: float = 5.0
    max_reobservations: int = 1
    db_path: str = "runs/agent/agent.sqlite3"

    def __repr__(self) -> str:
        return (
            "DecisionConfig(api_key=<redacted>, "
            f"workspace_id={_safe_repr(self.workspace_id)}, region={self.region!r}, "
            f"model={self.model!r}, provider={self.provider!r}, fall_head={self.fall_head!r}, "
            f"timeout_s={self.timeout_s!r}, max_retries={self.max_retries!r}, "
            f"min_interval_s={self.min_interval_s!r}, max_source_age_s={self.max_source_age_s!r}, "
            f"max_wall_age_s={self.max_wall_age_s!r}, "
            f"max_reobservations={self.max_reobservations!r}, db_path={self.db_path!r})"
        )

    @property
    def missing_items(self) -> tuple[str, ...]:
        if self.provider != "aliyun":
            return ()
        missing: list[str] = []
        if not self.api_key:
            missing.append("DASHSCOPE_API_KEY")
        if not self.workspace_id:
            missing.append("DASHSCOPE_WORKSPACE_ID")
        return tuple(missing)

    @property
    def configured(self) -> bool:
        return not self.missing_items

    def public_status(self) -> dict[str, object]:
        """Return the only supported public configuration representation."""
        return {
            "configured": self.configured,
            "provider": self.provider,
            "model": self.model,
            "region": self.region,
            "missing_items": list(self.missing_items),
        }

    def require_aliyun_configuration(self, *, provider: str | None = None) -> None:
        selected_provider = self.provider if provider is None else provider
        if selected_provider != "aliyun":
            raise DecisionConfigError("provider must be aliyun for this configuration check")
        if self.region not in _ALLOWED_REGIONS:
            raise DecisionConfigError("DASHSCOPE_REGION is not in the P2 allowlist")
        if self.model != _ALLOWED_MODEL:
            raise DecisionConfigError("FALLPRE_DECISION_MODEL must be decision-model-preview")
        if self.workspace_id and not _WORKSPACE_LABEL.fullmatch(self.workspace_id):
            raise DecisionConfigError("DASHSCOPE_WORKSPACE_ID must be one valid domain label")
        missing_items = []
        if not self.api_key:
            missing_items.append("DASHSCOPE_API_KEY")
        if not self.workspace_id:
            missing_items.append("DASHSCOPE_WORKSPACE_ID")
        if missing_items:
            missing = ", ".join(missing_items)
            raise DecisionConfigError(f"missing required configuration: {missing}")


def _safe_repr(value: str | None) -> str:
    if value is None:
        return "None"
    return repr(value)


def _unquote_dotenv_value(raw: str, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        closing = value.find(quote, 1)
        if closing < 0 or value[closing + 1 :].strip() not in {"", "#"} and not value[closing + 1 :].lstrip().startswith("#"):
            raise DecisionConfigError(f"invalid .env syntax at line {line_number}")
        return value[1:closing]
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def _read_dotenv(path: Path, *, include_credentials: bool = False) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        raise DecisionConfigError("unable to read .env configuration") from None
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, raw_value = stripped.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise DecisionConfigError(f"invalid .env syntax at line {line_number}")
        if key == "DASHSCOPE_API_KEY" and not include_credentials:
            continue
        if key in _ENV_NAMES:
            values[key] = _unquote_dotenv_value(raw_value, line_number)
    return values


def _value(values: Mapping[str, str], key: str, default: str) -> str:
    return values.get(key, default).strip()


def _parse_float(values: Mapping[str, str], name: str, default: float, *, maximum: float) -> float:
    raw = _value(values, name, str(default))
    try:
        number = float(raw)
    except (TypeError, ValueError):
        raise DecisionConfigError(f"{name} must be a finite number") from None
    if number != number or number in {float("inf"), float("-inf")} or not 0 < number <= maximum:
        raise DecisionConfigError(f"{name} must be positive and at most {maximum:g}")
    return number


def _parse_int(values: Mapping[str, str], name: str, default: int, *, maximum: int) -> int:
    raw = _value(values, name, str(default))
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", raw):
        raise DecisionConfigError(f"{name} must be a non-negative integer")
    number = int(raw)
    if number > maximum:
        raise DecisionConfigError(f"{name} must be at most {maximum}")
    return number


def load_decision_config(
    *,
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    include_credentials: bool = False,
) -> DecisionConfig:
    """Load settings using process environment > .env > safe defaults.

    Supplying ``environ`` replaces (rather than augments) the process mapping,
    making tests independent from the developer's real credentials.
    """
    if env_file is None:
        root = Path(project_root) if project_root is not None else _ROOT
        dotenv_path = root / ".env"
    else:
        dotenv_path = Path(env_file)
    values = _read_dotenv(dotenv_path, include_credentials=include_credentials)
    process_values = os.environ if environ is None else environ
    process_names = _ENV_NAMES if include_credentials else _ENV_NAMES - {"DASHSCOPE_API_KEY"}
    values.update({name: process_values[name] for name in process_names if name in process_values})

    provider = _value(values, "FALLPRE_DECISION_PROVIDER", "off").lower()
    if provider not in _PROVIDERS:
        raise DecisionConfigError("FALLPRE_DECISION_PROVIDER must be off, mock, or aliyun")
    region = _value(values, "DASHSCOPE_REGION", "cn-beijing")
    if region not in _ALLOWED_REGIONS:
        raise DecisionConfigError("DASHSCOPE_REGION is not in the P0 allowlist")
    model = _value(values, "FALLPRE_DECISION_MODEL", "decision-model-preview")
    if model != _ALLOWED_MODEL:
        raise DecisionConfigError("FALLPRE_DECISION_MODEL must be decision-model-preview")
    head = _value(values, "FALLPRE_FALL_HEAD", "tail_joint")
    if head not in _HEADS:
        raise DecisionConfigError("FALLPRE_FALL_HEAD must be tail_joint or cd")
    workspace = _value(values, "DASHSCOPE_WORKSPACE_ID", "") or None
    if workspace is not None and not _WORKSPACE_LABEL.fullmatch(workspace):
        raise DecisionConfigError("DASHSCOPE_WORKSPACE_ID must be one valid domain label")
    secret = _value(values, "DASHSCOPE_API_KEY", "") or None
    db_path = _value(values, "FALLPRE_AGENT_DB", "runs/agent/agent.sqlite3")
    if not db_path:
        raise DecisionConfigError("FALLPRE_AGENT_DB must not be empty")

    config = DecisionConfig(
        api_key=SecretValue(secret),
        workspace_id=workspace,
        region=region,
        model=model,
        provider=provider,
        fall_head=head,
        timeout_s=_parse_float(values, "FALLPRE_DECISION_TIMEOUT_S", 3.0, maximum=300.0),
        max_retries=_parse_int(values, "FALLPRE_DECISION_MAX_RETRIES", 1, maximum=1),
        min_interval_s=_parse_float(values, "FALLPRE_DECISION_MIN_INTERVAL_S", 2.0, maximum=3600.0),
        max_source_age_s=_parse_float(values, "FALLPRE_DECISION_MAX_SOURCE_AGE_S", 2.5, maximum=3600.0),
        max_wall_age_s=_parse_float(values, "FALLPRE_DECISION_MAX_WALL_AGE_S", 5.0, maximum=3600.0),
        max_reobservations=_parse_int(values, "FALLPRE_DECISION_MAX_REOBSERVATIONS", 1, maximum=3),
        db_path=db_path,
    )
    return config

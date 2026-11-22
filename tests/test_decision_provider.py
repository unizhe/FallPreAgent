import http.client
import socket
import urllib.request

import pytest

from fallpre.adapters.aliyun_decision import AliyunDecisionProvider
from fallpre.decision_config import DecisionConfigError, load_decision_config
from fallpre.decision_contracts import DecisionStatus, ProbabilityKind, ProviderMode
from fallpre.decision_fixtures import synthetic_evidence
from fallpre.decision_provider import build_decision_provider


def _block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden in decision P0 tests")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)


def _config(tmp_path, provider, *, credentials=True):
    values = {"FALLPRE_DECISION_PROVIDER": provider}
    if credentials:
        values.update(
            {
                "DASHSCOPE_API_KEY": "synthetic-test-secret-only",
                "DASHSCOPE_WORKSPACE_ID": "synthetic-test-space",
            }
        )
    return load_decision_config(env_file=tmp_path / "no-real-env", environ=values,
                                include_credentials=provider == "aliyun")


@pytest.mark.parametrize("mode", ["off", "mock"])
def test_off_and_mock_ignore_fake_credentials_and_never_network(mode, tmp_path, monkeypatch):
    _block_network(monkeypatch)
    config = _config(tmp_path, mode)
    monkeypatch.setattr(
        type(config.api_key),
        "get_secret_value",
        lambda self: pytest.fail("off/mock must not read the API key"),
    )
    provider = build_decision_provider(mode, config=config, clock=lambda: 10.0)
    result = provider.assess(synthetic_evidence(), deadline=11.0)
    assert result.provider == ProviderMode(mode)
    if mode == "off":
        assert result.status == DecisionStatus.SKIPPED
        assert result.fall_probability is None
    else:
        assert result.status == DecisionStatus.READY
        assert result.probability_kind == ProbabilityKind.SYNTHETIC_FIXTURE


def test_aliyun_mode_requires_credentials_and_explicit_network_opt_in(tmp_path, monkeypatch):
    _block_network(monkeypatch)
    missing = _config(tmp_path, "aliyun", credentials=False)
    with pytest.raises(DecisionConfigError, match="DASHSCOPE_API_KEY, DASHSCOPE_WORKSPACE_ID"):
        build_decision_provider("aliyun", config=missing, clock=lambda: 10.0, enable_network=True)

    configured = _config(tmp_path, "aliyun")
    with pytest.raises(ValueError, match="enable_network=True"):
        build_decision_provider("aliyun", config=configured, clock=lambda: 10.0)
    provider = build_decision_provider("aliyun", config=configured, clock=lambda: 10.0,
                                       enable_network=True, transport=object())
    assert isinstance(provider, AliyunDecisionProvider)

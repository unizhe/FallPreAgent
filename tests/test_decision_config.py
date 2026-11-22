import json

import pytest

from fallpre.decision_config import DecisionConfigError, load_decision_config


def test_process_environment_overrides_dotenv_and_defaults_without_mutating_environment(tmp_path):
    env_file = tmp_path / ".env.fixture"
    env_file.write_text(
        "FALLPRE_DECISION_PROVIDER=mock\n"
        "FALLPRE_DECISION_MODEL=decision-model-preview\n"
        "FALLPRE_DECISION_TIMEOUT_S=2.25\n"
        "DASHSCOPE_API_KEY=dotenv-secret-fixture\n"
        "DASHSCOPE_WORKSPACE_ID=dotenv-space\n",
        encoding="utf-8",
    )

    config = load_decision_config(
        env_file=env_file,
        environ={
            "FALLPRE_DECISION_MODEL": "decision-model-preview",
            "FALLPRE_DECISION_TIMEOUT_S": "4.5",
            "DASHSCOPE_API_KEY": "process-secret-fixture",
        },
        include_credentials=True,
    )

    assert config.provider == "mock"  # .env beats the safe default.
    assert config.model == "decision-model-preview"  # Process environment wins.
    assert config.timeout_s == 4.5
    assert config.workspace_id == "dotenv-space"
    assert config.api_key.get_secret_value() == "process-secret-fixture"
    assert config.region == "cn-beijing"
    assert config.max_retries == 1
    assert config.db_path == "runs/agent/agent.sqlite3"


def test_project_root_loads_its_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text("FALLPRE_DECISION_PROVIDER=mock\n", encoding="utf-8")
    config = load_decision_config(project_root=tmp_path, environ={})
    assert config.provider == "mock"


@pytest.mark.parametrize("provider", ["off", "mock"])
def test_off_and_mock_need_no_credentials(provider, tmp_path):
    config = load_decision_config(
        env_file=tmp_path / "missing.env",
        environ={"FALLPRE_DECISION_PROVIDER": provider},
    )
    assert config.configured
    assert config.missing_items == ()
    assert config.api_key.get_secret_value() is None


def test_credentials_are_not_loaded_without_explicit_network_authorization(tmp_path):
    env_file = tmp_path / ".env.fixture"
    env_file.write_text(
        "FALLPRE_DECISION_PROVIDER=aliyun\n"
        "DASHSCOPE_API_KEY=synthetic-secret-never-loaded\n"
        "DASHSCOPE_WORKSPACE_ID=synthetic-space\n",
        encoding="utf-8",
    )
    config = load_decision_config(env_file=env_file, environ={})
    assert config.provider == "aliyun"
    assert config.api_key.get_secret_value() is None


def test_aliyun_missing_credentials_are_reported_by_name_only(tmp_path):
    secret_marker = "secret-marker-must-never-appear"
    config = load_decision_config(
        env_file=tmp_path / "missing.env",
        environ={"FALLPRE_DECISION_PROVIDER": "aliyun", "DASHSCOPE_API_KEY": secret_marker},
        include_credentials=True,
    )
    assert not config.configured
    assert config.missing_items == ("DASHSCOPE_WORKSPACE_ID",)
    with pytest.raises(DecisionConfigError) as captured:
        config.require_aliyun_configuration()
    assert "DASHSCOPE_WORKSPACE_ID" in str(captured.value)
    assert secret_marker not in str(captured.value)


def test_secret_is_redacted_from_repr_and_public_json(tmp_path):
    secret_marker = "fixture-secret-7f5a"
    env_file = tmp_path / ".env.fixture"
    env_file.write_text(f"DASHSCOPE_API_KEY='{secret_marker}'\n", encoding="utf-8")
    config = load_decision_config(env_file=env_file, environ={}, include_credentials=True)

    assert secret_marker not in repr(config)
    assert secret_marker not in repr(config.api_key)
    assert secret_marker not in str(config.api_key)
    public_json = json.dumps(config.public_status(), sort_keys=True)
    assert secret_marker not in public_json
    assert set(config.public_status()) == {"configured", "provider", "model", "region", "missing_items"}


def test_env_file_is_parsed_as_data_not_executed_as_shell(tmp_path):
    marker = tmp_path / "shell-was-not-run"
    env_file = tmp_path / ".env.fixture"
    env_file.write_text(f"FALLPRE_AGENT_DB=$(touch {marker})\n", encoding="utf-8")
    config = load_decision_config(env_file=env_file, environ={})
    assert config.db_path == f"$(touch {marker})"
    assert not marker.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FALLPRE_DECISION_PROVIDER", "live"),
        ("DASHSCOPE_REGION", "cn-unknown"),
        ("DASHSCOPE_WORKSPACE_ID", "bad.workspace"),
        ("DASHSCOPE_WORKSPACE_ID", "-bad"),
        ("FALLPRE_DECISION_MODEL", "another-model"),
        ("FALLPRE_FALL_HEAD", "unknown"),
        ("FALLPRE_DECISION_MODEL", "   "),
        ("FALLPRE_AGENT_DB", "  "),
        ("FALLPRE_DECISION_TIMEOUT_S", "0"),
        ("FALLPRE_DECISION_TIMEOUT_S", "-1"),
        ("FALLPRE_DECISION_TIMEOUT_S", "nan"),
        ("FALLPRE_DECISION_TIMEOUT_S", "inf"),
        ("FALLPRE_DECISION_TIMEOUT_S", "true"),
        ("FALLPRE_DECISION_MAX_RETRIES", "-1"),
        ("FALLPRE_DECISION_MAX_RETRIES", "1.0"),
        ("FALLPRE_DECISION_MAX_RETRIES", "true"),
        ("FALLPRE_DECISION_MAX_RETRIES", "6"),
        ("FALLPRE_DECISION_MAX_REOBSERVATIONS", "4"),
        ("FALLPRE_DECISION_MIN_INTERVAL_S", "3601"),
    ],
)
def test_invalid_config_values_are_rejected_without_echoing_values(name, value, tmp_path):
    with pytest.raises(DecisionConfigError) as captured:
        load_decision_config(env_file=tmp_path / "missing.env", environ={name: value})
    assert name in str(captured.value)
    assert f"{name}={value}" not in str(captured.value)


def test_malformed_env_error_does_not_include_line_contents(tmp_path):
    env_file = tmp_path / ".env.fixture"
    env_file.write_text("this line contains secret-marker\n", encoding="utf-8")
    with pytest.raises(DecisionConfigError) as captured:
        load_decision_config(env_file=env_file, environ={})
    assert "secret-marker" not in str(captured.value)

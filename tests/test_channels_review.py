"""Channel isolation tests never contact a webhook or play audio."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fallpre.adapters.channels import ChannelDeliveryError, ConfiguredChannels


def test_webhook_failure_does_not_suppress_local_fall_alarm(monkeypatch, tmp_path):
    monkeypatch.setattr("fallpre.adapters.channels.platform.system", lambda: "Darwin")
    network = Mock(side_effect=TimeoutError("endpoint unavailable"))
    speech = Mock(return_value={"status": "PLAYBACK_COMPLETED"})
    monkeypatch.setattr("fallpre.adapters.channels.urllib.request.urlopen", network)
    channels = ConfiguredChannels(speech=True, webhook_url="https://example.invalid/alert", cache_dir=tmp_path)
    monkeypatch.setattr(channels, "_play_speech", speech)
    decision = SimpleNamespace(event_id="evt-1", person_id="p-1", confirmed_alarm=True)
    with pytest.raises(ChannelDeliveryError) as error:
        channels("FALL_ALARM", decision, "key-1")
    assert network.call_count == 1
    assert speech.call_count == 1
    assert error.value.delivery_receipt["receipts"]["speech"]["status"] == "PLAYBACK_COMPLETED"


@pytest.mark.parametrize("failing_channel", ["webhook", "speech"])
def test_retry_only_repeats_failed_channel(monkeypatch, tmp_path, failing_channel):
    monkeypatch.setattr("fallpre.adapters.channels.platform.system", lambda: "Darwin")
    channels = ConfiguredChannels(speech=True, webhook_url="https://example.invalid/alert", cache_dir=tmp_path)
    successes = {"speech": {"status": "PLAYBACK_COMPLETED"}, "webhook": {"http_status": 200}}
    mocks = {}
    for name, method in (("speech", "_play_speech"), ("webhook", "_send_webhook")):
        mock = Mock(side_effect=[TimeoutError("unavailable"), successes[name]]) if name == failing_channel else (
            Mock(return_value=successes[name])
        )
        monkeypatch.setattr(channels, method, mock)
        mocks[name] = mock
    decision = SimpleNamespace(event_id="evt-2", person_id="p-1", confirmed_alarm=True)
    with pytest.raises(ChannelDeliveryError):
        channels("FALL_ALARM", decision, "key-2")
    assert all(mock.call_count == 1 for mock in mocks.values())
    result = channels("FALL_ALARM", decision, "key-2")
    assert result["receipts"] == successes
    assert mocks[failing_channel].call_count == 2
    other = "speech" if failing_channel == "webhook" else "webhook"
    assert mocks[other].call_count == 1
    assert channels("FALL_ALARM", decision, "key-2") == result
    assert mocks[failing_channel].call_count == 2
    assert mocks[other].call_count == 1

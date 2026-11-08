import json

import pytest

from fallpre.showcase import run_showcase


def test_real_tools_and_direct_alarm_recorded_without_duplicate_actions(tmp_path):
    summary = run_showcase(tmp_path)
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert summary["dry_run"] is True
    expected = [("OBSTACLE_REMINDER", "agent_tool", 3.3),
                ("FALL_ALARM", "fast_loop", 2.5), ("REQUEST_CHECK", "agent_tool", 5.0)]
    for scenario, (action_type, route, source_s) in zip(trace["scenarios"], expected, strict=True):
        assert len(scenario["actions"]) == 1
        action = scenario["actions"][0]
        assert (action["action_type"], action["route"], action["t_s"]) == (action_type, route, source_s)
        assert action["status"] == "SIMULATED"
        calls = scenario["calls"]
        assert all(not call["result"].get("error") for call in calls)
        assert any(call["name"] == "get_execution_status" for call in calls)
        assert all(call["frame_id"] == frame["frame_id"]
                   for frame in scenario["frames"] for call in (calls[i] for i in frame["calls"]))
        if route == "fast_loop":
            at_alarm = [call for call in calls if call["t_s"] == source_s]
            assert at_alarm[0]["actor"] == "deterministic_fast_loop"
            assert not any(call["name"] == "execute_intervention" for call in calls)
        else:
            assert sum(call["name"] == "execute_intervention" for call in calls) == 1
    assert any(f["outcome"] == "CLEAR" for f in trace["scenarios"][0]["frames"])
    assert trace["scenarios"][0]["frames"][-1]["outcome"] == "NO_ACTION"
    assert trace["scenarios"][2]["frames"][-1]["outcome"] == "REVIEW_REQUIRED"
    with pytest.raises(FileExistsError):
        run_showcase(tmp_path)

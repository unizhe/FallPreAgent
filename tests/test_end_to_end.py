import json
import sqlite3

from fallpre.demo import run_demo


def actions(path):
    with sqlite3.connect(path) as connection:
        return [json.loads(body) for (body,) in connection.execute("SELECT body FROM actions")]


def test_composed_geometry_temporal_policy_and_execution(tmp_path):
    report = run_demo(tmp_path)
    prefall = report["scenarios"]["approach_stop_retreat"]
    assert {"REMIND", "CLEAR"} <= set(prefall["outcomes"])
    assert "INSIDE_RISK_REGION" in prefall["observed_states"]["spatial"]
    prefall_actions = actions(prefall["database"])
    assert [a["action_type"] for a in prefall_actions] == ["OBSTACLE_REMINDER"]
    fall = report["scenarios"]["fall_after_upright"]
    assert "ALARM" in fall["outcomes"]
    assert [a["action_type"] for a in actions(fall["database"])] == ["FALL_ALARM"]
    initial_low = report["scenarios"]["initial_low_pose_review"]
    assert "REVIEW_REQUIRED" in initial_low["outcomes"] and "ALARM" not in initial_low["outcomes"]
    assert [a["action_type"] for a in actions(initial_low["database"])] == ["REQUEST_CHECK"]
    assert all(a["status"] == "SIMULATED" for s in report["scenarios"].values() for a in actions(s["database"]))

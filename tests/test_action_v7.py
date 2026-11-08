from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from test_action_v2 import person, row

from fallpre.weekly.action_v7 import OnlineCompletion


def stream(speed=0.0, knee=0.99, label="standing_up"):
    s = OnlineCompletion.__new__(OnlineCompletion)
    s.model = SimpleNamespace(config=dict(threshold=0.6, max_up_speed=0.1, minimum_knee=0.96, window_s=0.067))
    s.baseline = SimpleNamespace(
        update=lambda r: dict(
            label="background" if label == "standing_up" else label,
            phase="standing",
            v4_phase="rising",
            v5_label=label,
            v5_phase="rising",
            fast_completion_probability=0.9,
        )
    )
    s.history = deque()
    s.epoch = None
    s.last_t = None
    s.rise_start = None
    s.completed = False
    s.votes = deque()
    x = np.zeros(54)
    x[1] = knee
    x[10] = -speed
    x[11] = -speed
    s.feature_provider = lambda h, p, t: x
    return s


def test_current_ascent_restores_v6_early_termination_without_timeout():
    s = stream(speed=0.234, knee=0.918)
    for i in range(500):
        p = s.update(row(i, [person()]))
        assert p["label"] == "standing_up" and p["v6_label"] == "background"
        assert p["completion_motion_veto"] and not p["motion_completion_latched"]


def test_stationary_flexed_pose_cannot_end_rise():
    s = stream(knee=0.918)
    assert all(s.update(row(i, [person()]))["label"] == "standing_up" for i in range(80))


def test_completion_requires_supported_still_upright_pose_and_latches():
    s = stream()
    assert s.update(row(0, [person()]))["label"] == "standing_up"
    for i in range(1, 20):
        p = s.update(row(i, [person()]))
    assert p["label"] == "background" and p["motion_completion_latched"]
    s.feature_provider = lambda h, p, t: np.zeros(54)
    assert s.update(row(20, [person()]))["motion_completion_latched"]


@pytest.mark.parametrize("change", ["identity", "gap", "clock", "missing", "phase"])
def test_completion_state_resets(change):
    s = stream()
    for i in range(20):
        s.update(row(i, [person()]))
    r = row(20, [person()])
    if change == "identity":
        r["target_epoch"] = 9
    if change == "gap":
        r["t"] += 2
    if change == "clock":
        r["t"] = 0
    if change == "missing":
        r["instances"] = []
    if change == "phase":
        s.baseline.update = lambda r: dict(label="background", phase="seated", v4_phase="seated")
    assert not s.update(r)["motion_completion_latched"]


@pytest.mark.parametrize("label", ["walking", "bending", "falling", "sitting_down", "background", "unknown"])
def test_only_v5_rising_output_can_change(label):
    s = stream(label=label)
    assert all(s.update(row(i, [person()]))["label"] == label for i in range(50))

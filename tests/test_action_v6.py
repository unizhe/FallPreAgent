from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from test_action_v2 import person, row

from fallpre.weekly.action_v6 import OnlineCompletion


def stream(prob=0.9, label="standing_up"):
    s = OnlineCompletion.__new__(OnlineCompletion)
    s.model = SimpleNamespace(
        config=dict(threshold=0.7, window_s=0.067, vote_fraction=0.65, minimum_rise_s=0.15),
        probability=lambda x: prob,
    )
    s.baseline = SimpleNamespace(update=lambda r: dict(label=label, phase="rising", v4_phase="rising"))
    s.history = deque()
    s.epoch = None
    s.last_t = None
    s.rise_start = None
    s.completed = False
    s.votes = deque()
    s.feature_provider = lambda h, p, t: np.zeros(54)
    return s


def test_completion_latches_and_resets_on_new_cycle():
    s = stream()
    for i in range(20):
        p = s.update(row(i, [person()]))
    assert p["label"] == "background" and p["v5_label"] == "standing_up"
    s.model.probability = lambda x: 0
    assert s.update(row(20, [person()]))["fast_completion_latched"]
    s.baseline.update = lambda r: dict(label="background", phase="seated", v4_phase="seated")
    assert not s.update(row(21, [person()]))["fast_completion_latched"]
    s.baseline.update = lambda r: dict(label="standing_up", phase="rising", v4_phase="rising")
    assert s.update(row(22, [person()]))["label"] == "standing_up"


@pytest.mark.parametrize("change", ["identity", "gap", "missing", "clock"])
def test_stale_votes_do_not_cross_discontinuities(change):
    s = stream()
    for i in range(20):
        s.update(row(i, [person()]))
    r = row(20, [person()])
    if change == "identity":
        r["target_epoch"] = 8
    if change == "gap":
        r["t"] += 2
    if change == "missing":
        r["instances"] = []
    if change == "clock":
        r["t"] = 0
    assert not s.update(r)["fast_completion_latched"]


@pytest.mark.parametrize("label", ["walking", "bending", "falling", "sitting_down", "background", "unknown"])
def test_other_labels_exactly_preserved(label):
    s = stream(label=label)
    assert all(s.update(row(i, [person()]))["label"] == label for i in range(30))


def test_continuous_slow_rise_has_no_timeout():
    s = stream(prob=0.2)
    assert all(s.update(row(i, [person()]))["label"] == "standing_up" for i in range(500))

import numpy as np
import pytest
import torch

from fallpre.weekly.action_v2 import (
    CausalTCN,
    Config,
    OnlineTemporalAction,
    PersonTracker,
    TemporalModel,
    sequence_features,
)


def person(x=100, y=100, score=0.9):
    points = np.zeros((17, 3))
    points[:] = [x, y, 0.9]
    for idx, xx, yy in [
        (5, -15, 0),
        (6, 15, 0),
        (11, -12, 50),
        (12, 12, 50),
        (13, -12, 85),
        (14, 12, 85),
        (15, -12, 120),
        (16, 12, 120),
    ]:
        points[idx] = [x + xx, y + yy, 0.9]
    return {"box": [x - 30, y - 25, x + 30, y + 130], "det": score, "kpts": points.tolist()}


def row(i, instances):
    return {"t": i / 30, "instances": instances, "source_frame_id": i + 1, "target_epoch": 1}


def rise_model():
    from dataclasses import asdict

    network = CausalTCN()
    with torch.no_grad():
        for p in network.parameters():
            p.zero_()
        network.head.bias[2] = 8.0
    return TemporalModel(
        {
            "schema": "fallpre.action_tcn.v2",
            "config": asdict(Config()),
            "mean": [0.0] * 28,
            "scale": [1.0] * 28,
            "state_dict": {k: v.tolist() for k, v in network.state_dict().items()},
        }
    )


def test_extra_detection_does_not_reset_target():
    tracker = PersonTracker()
    first = tracker.update(row(0, [person()]))
    second = tracker.update(row(1, [person(400), person(101)]))
    assert second["target_epoch"] == first["target_epoch"]
    assert second["target_index"] == 1


def test_low_score_pass_preserves_original_detection_index():
    tracker = PersonTracker()
    tracker.update(row(0, [person()]))
    result = tracker.update(row(1, [person(400, score=0.8), person(101, score=0.2)]))
    assert result["target_index"] == 1
    assert result["instances"][0]["det"] == 0.2


def test_target_gap_does_not_borrow_other_person():
    tracker = PersonTracker()
    tracker.update(row(0, [person()]))
    result = tracker.update(row(1, [person(400)]))
    assert result["instances"] == []
    assert result["tracking_reason"] == "temporarily_unobserved"


def test_short_missing_observation_is_not_filled_and_history_recovers():
    stream = OnlineTemporalAction(rise_model())
    for i in range(20):
        stream.update(row(i, [person()]))
    missing = stream.update(row(20, []))
    assert missing["label"] == "unknown"
    recovered = stream.update(row(21, [person()]))
    assert recovered["model_inference_executed"]
    assert recovered["support_frames"] >= 20


def test_static_seated_pose_cannot_be_reported_as_transition():
    model = rise_model()
    history = [row(i, [person()]) for i in range(40)]
    assert model.predict(history, guard=False)["label"] == "standing_up"
    assert model.predict(history)["label"] == "background"
    assert model.predict(history)["reason"] == "rise_rejected_no_upward_motion"


def test_real_upward_motion_can_pass_transition_guard():
    history = [row(i, [person(y=100 - i)]) for i in range(40)]
    result = rise_model().predict(history)
    assert result["upward_torso_units"] > 0.12
    assert result["label"] == "standing_up"


def test_missing_ankles_does_not_discard_entire_body_sequence():
    p = person()
    p["kpts"][15][2] = p["kpts"][16][2] = 0
    features, info = sequence_features([row(i, [p]) for i in range(30)])
    assert features is not None and info["reason"] is None
    assert np.isfinite(features).all()
    assert features[13, -1] == 0  # knee reliability stays explicit


def test_identity_switch_restarts_temporal_support():
    stream = OnlineTemporalAction(rise_model())
    for i in range(20):
        stream.update(row(i, [person()]))
    switched = dict(row(20, [person(400)]), target_epoch=2)
    assert stream.update(switched)["label"] == "unknown"


def test_long_gap_restarts_temporal_support():
    stream = OnlineTemporalAction(rise_model())
    for i in range(20):
        stream.update(row(i, [person()]))
    for i in range(20, 40):
        stream.update(row(i, []))
    assert stream.update(row(40, [person()]))["label"] == "unknown"


def test_nonincreasing_time_rejected():
    tracker = PersonTracker()
    tracker.update(row(0, [person()]))
    with pytest.raises(ValueError):
        tracker.update(row(0, [person()]))


def test_causal_convolution_has_no_future_dependency():
    net = CausalTCN().eval()
    x = torch.randn(1, 28, 21)
    altered = x.clone()
    altered[:, :, 16:] += 50

    def hidden(a):
        for i, conv in enumerate(net.blocks):
            a = torch.relu(conv(torch.nn.functional.pad(a, (2 * 2**i, 0))))
        return a

    torch.testing.assert_close(hidden(x)[:, :, :16], hidden(altered)[:, :, :16])

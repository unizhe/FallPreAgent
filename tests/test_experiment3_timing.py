"""Semantic safeguards for retrospective timing, without inventing boundaries."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_experiment3_v7 import boundary_margin, derive_stable_confirmation, first_output, position


def row(frame, time, label="background", phase="unknown"):
    return dict(source_frame_id=frame, source_time_s=time, prediction=dict(label=label, phase=phase))


def test_interval_margin_preserves_bounds_and_sign():
    detection = {"time_s": 2.869}
    assert boundary_margin({"time_interval_s": [3.336, 3.537]}, detection) == [0.467, 0.668]
    assert boundary_margin({"time_s": 2.736}, detection) == [-0.133, -0.133]


def test_unknown_boundary_or_clock_is_null_not_zero():
    assert boundary_margin({"time_s": None, "time_interval_s": None}, {"time_s": 8.6}) is None
    assert boundary_margin({"time_interval_s": [3.336, 3.537]}, {"time_s": None}) is None
    assert boundary_margin({"time_s": 3.336}, None) is None


def test_interval_order_rejected():
    with pytest.raises(ValueError):
        boundary_margin({"time_interval_s": [3.537, 3.336]}, {"time_s": 2.869})


def test_first_trigger_is_prefix_invariant_and_does_not_select_best_later_output():
    prefix = [row(1, 0), row(2, 0.03, "falling")]
    future = [row(3, 0.07), row(4, 0.1, "falling")]
    assert first_output(prefix + future) == first_output(prefix)
    assert first_output(prefix[:1]) is None


def test_reference_and_future_labels_cannot_gate_first_detection():
    rows = [row(1, 0, "falling"), row(2, 0.03)]
    altered = deepcopy(rows)
    for r in altered:
        r.update(reference_action="background", contact_time_s=999, future_ground_truth="not_falling")
    assert first_output(rows) == first_output(altered)
    assert first_output(rows)["frame"] == 1


def test_multicam_disabled_and_frame_only_has_no_seconds():
    rows = [row(1, 0, "falling")]
    assert first_output(rows, enabled=False) is None
    assert first_output(rows, seconds=False)["time_s"] is None


def test_proxy_classification_keeps_contact_uncertainty():
    onset, contact = {"time_s": 2.736}, {"time_interval_s": [3.336, 3.537]}
    assert position({"time_s": 2.869}, onset, contact) == "onset_proxy_to_contact"
    assert position({"time_s": 3.4}, onset, contact) == "within_contact_uncertainty"
    assert position({"time_s": 3.6}, onset, contact) == "after_contact"
    assert position({"time_s": 2.5}, onset, contact) == "before_official_onset_proxy"
    assert position({"time_s": 2.869}, {}, contact) == "unresolved_boundary_or_clock"


def test_stability_confirmation_uses_future_threshold_not_onset_as_availability():
    landmark = {"time_interval_s": [4.371, 4.571], "stability_criterion": {"minimum_duration_s": 0.5}}
    clock = {"original_frame_ids": [132, 138, 147, 152, 153], "source_times_s": [4.371, 4.571, 4.871, 5.038, 5.072]}
    result = derive_stable_confirmation(landmark, clock)
    assert result["continuous_threshold_interval_s"] == [4.871, 5.071]
    assert result["available_observations_for_lower_and_upper"] == [{"frame": 147, "time_s": 4.871}, {"frame": 153, "time_s": 5.072}]
    assert result["human_reviewed"] is False
    assert result["exact_confirmation_frame"] is None

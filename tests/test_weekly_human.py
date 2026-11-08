"""Source-clock, continuity and label-boundary checks for the real-model demo."""
import copy
import json
import zipfile

import pytest

from fallpre.weekly.human import (
    LABEL_SUPPORT,
    PastWindow,
    ProjectedTransitions,
    load_urfd_frames,
    projected_signals,
    validate_instances,
)


def row(i, *, t=None, count=1):
    points = [[100., 100., .95] for _ in range(17)]
    for indices, coords in (((5, 6), ((80, 50), (120, 50))),
                            ((11, 12), ((80, 100), (120, 100))),
                            ((13, 14), ((80, 150), (120, 150))),
                            ((15, 16), ((80, 200), (120, 200)))):
        for idx, (x, y) in zip(indices, coords):
            points[idx] = [float(x), float(y), .95]
    return {"source_frame_id": i, "frame_index": i - 1, "t": i / 30 if t is None else t,
            "instances": [{"box": [50., 20., 150., 220.], "det": .9, "kpts": points}
                          for _ in range(count)]}


def test_window_is_strictly_past_and_does_not_short_wrap():
    stream = PastWindow(100, 30)
    for i in range(1, 100):
        assert stream.add(row(i)) is None
    first = stream.add(row(100))
    assert [r["source_frame_id"] for r in first] == list(range(1, 101))
    for i in range(101, 130):
        assert stream.add(row(i)) is None
    second = stream.add(row(130))
    assert [r["source_frame_id"] for r in second] == list(range(31, 131))
    assert first[-1]["t"] == pytest.approx(100/30)


@pytest.mark.parametrize("count", [0, 2])
def test_no_stitching_across_missing_or_multiple_people(count):
    stream = PastWindow()
    for i in range(1, 100):
        stream.add(row(i))
    assert stream.add(row(100, count=count)) is None
    previous_run = stream.run_id
    for i in range(101, 200):
        assert stream.add(row(i)) is None
    result = stream.add(row(200))
    assert result[0]["source_frame_id"] == 101
    assert stream.run_id == previous_run


@pytest.mark.parametrize("boundary", ["frame_gap", "time_gap", "backward", "jump"])
def test_discontinuities_never_inherit_history(boundary):
    stream = PastWindow()
    for i in range(1, 100):
        stream.add(row(i))
    observation = row(100)
    if boundary == "frame_gap":
        observation["source_frame_id"] = 101
    elif boundary == "time_gap":
        observation["t"] += 1
    elif boundary == "backward":
        observation["t"] = 0
    else:
        observation["instances"][0]["box"] = [500., 20., 600., 220.]
    assert stream.add(observation) is None
    assert len(stream.history) <= 1
    assert stream.run_id == 1


def test_rejects_fake_2d_joint_format_and_nonfinite_scores():
    observation = row(1)
    observation["instances"][0]["kpts"][4][2] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_instances(observation["instances"])
    observation = row(1)
    observation["instances"][0]["kpts"].pop()
    with pytest.raises(ValueError, match="COCO17"):
        validate_instances(observation["instances"])


def test_urfd_retains_irregular_official_clock_and_exact_frame_join(tmp_path):
    archive = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for i in (1, 2, 3):
            zipped.writestr(f"fall-01-cam0-rgb/fall-01-cam0-rgb-{i:03}.png", b"fixture")
    sync = tmp_path / "sync.csv"
    sync.write_text("1,0,1.2\n2,37,1.1\n3,91,1.3\n")
    rows = load_urfd_frames(archive, sync, "fall-01")
    assert [r["t"] for r in rows] == [0., .037, .091]
    assert "label" not in rows[0]  # CSV third column is not an action annotation.
    sync.write_text("1,0,1.2\n2,37,1.1\n")
    with pytest.raises(ValueError, match="frame IDs differ"):
        load_urfd_frames(archive, sync, "fall-01")


def test_projected_geometry_quality_and_identity_reset():
    observation = row(1)
    signals = projected_signals(observation["instances"][0])
    assert signals["knee_mean_deg"] == 180
    assert signals["torso_tilt_deg"] == 0
    observer = ProjectedTransitions()
    observer.update(observation, 1)
    next_row = row(20)
    result = observer.update(next_row, 2)
    assert result["evidence_window_s"] == [next_row["t"], next_row["t"]]
    missing = copy.deepcopy(next_row)
    missing["instances"][0]["kpts"][13][2] = .1
    assert observer.update(missing, 2)["signals"]["quality"] == "unavailable"
    assert not observer.history


def test_requested_actions_do_not_mislabel_ntu_pickup_or_two_person_walk():
    assert LABEL_SUPPORT["bending"]["label"] is None
    assert LABEL_SUPPORT["walking"]["label"] is None
    assert LABEL_SUPPORT["sitting_down"]["label"] == "sitting down"
    assert LABEL_SUPPORT["standing_up"]["label"] == "standing up (from sitting position)"
    assert "pickup" not in json.dumps(LABEL_SUPPORT)

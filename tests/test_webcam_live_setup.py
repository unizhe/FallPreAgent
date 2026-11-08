import copy

import pytest

from fallpre.weekly.webcam_live_setup import LiveSetup


def lying_frame():
    p = [[0.0, 0.0, 0.0] for _ in range(17)]
    for s, h, k, a, dy in [(5, 11, 13, 15, -7), (6, 12, 14, 16, 7)]:
        for j, x, y in [(s, 130, 200), (h, 220, 210), (k, 290, 220), (a, 360, 225)]:
            p[j] = [x, y + dy, 0.95]
    person = dict(box=[100, 170, 400, 260], keypoints=p, confidence=0.95)
    result = dict(
        persons=[person],
        width=640,
        height=480,
        quality=dict(camera_stable=True, code="stable"),
        fall={"reason_code": "waiting_reference"},
    )
    rgb = {
        "raw_candidates": {
            "persons": [dict(box=person["box"][:], confidence=0.95)],
            "fallen": [dict(box=person["box"][:], confidence=0.90)],
        }
    }
    return result, rgb


def folded_lying_frame():
    result, rgb = lying_frame()
    person = result["persons"][0]
    person["box"] = [100, 160, 340, 300]
    for s, h, k, a, dy in [(5, 11, 13, 15, -7), (6, 12, 14, 16, 7)]:
        for j, x, y in [(s, 130, 200), (h, 230, 210), (k, 240, 260), (a, 300, 270)]:
            person["keypoints"][j] = [x, y + dy, 0.95]
    for group in rgb["raw_candidates"].values():
        group[0]["box"] = person["box"][:]
    return result, rgb


def test_cold_lying_has_separate_posture_without_red_or_event():
    s = LiveSetup()
    r, rgb = lying_frame()
    original = copy.deepcopy(r)
    for i in range(8):
        checks, lying = s.update(r, i * 0.13, rgb)
        assert lying["observed"] is (i >= 5)
        assert lying["new_event"] is False and lying["event_id"] is None
    assert r == original
    assert checks["complete_sides"] == 2
    assert checks["weights_finetuned_on_personal_videos"] is False


def test_folded_side_lying_observes_posture_without_reusing_a_red_event():
    s = LiveSetup()
    result, rgb = folded_lying_frame()
    result["fall"].update(state="unobservable", last_confirmed_event_id="rgb-fall-0001")
    original = copy.deepcopy(result)
    for i in range(8):
        _, out = s.update(result, i * 0.13, rgb)
        assert out["observed"] is (i >= 5)
        assert out["new_event"] is False and out["event_id"] is None
    assert out["evidence_type"] == "bent_knee_side_lying"
    assert result == original


@pytest.mark.parametrize("reason", ["one_leg", "straight_knees", "high_hips", "vertical_torso", "weak_torso", "feet_toward_head"])
def test_folded_side_lying_requires_low_torso_and_real_folded_legs(reason):
    s = LiveSetup()
    result, rgb = folded_lying_frame()
    p = result["persons"][0]["keypoints"]
    if reason == "one_leg":
        p[15][2] = 0.2
    if reason == "straight_knees":
        for h, k, a in [(11, 13, 15), (12, 14, 16)]:
            p[k][:2] = [(p[h][j] + p[a][j]) / 2 for j in (0, 1)]
    if reason == "high_hips":
        for a in (15, 16):
            p[a][1] = 400
    if reason == "vertical_torso":
        for shoulder in (5, 6):
            p[shoulder][:2] = [230, 100]
    if reason == "weak_torso":
        p[5][2] = 0.60
    if reason == "feet_toward_head":
        for a in (15, 16):
            p[a][0] = 170
    for i in range(8):
        _, out = s.update(result, i * 0.13, rgb)
        assert not out["observed"]


@pytest.mark.parametrize("frame", [lying_frame, folded_lying_frame])
@pytest.mark.parametrize(
    "reason",
    [
        "camera",
        "unknown_camera",
        "missing_pose",
        "multi_person",
        "weak_fallen",
        "vertical_legs",
        "clipped",
        "gap",
        "competing_person",
        "competing_fallen",
    ],
)
def test_lying_requires_current_unambiguous_geometry_and_temporal_evidence(reason, frame):
    s = LiveSetup()
    r, rgb = frame()
    for i in range(8):
        s.update(r, i * 0.13, rgb)
    if reason == "camera":
        r["quality"]["camera_stable"] = False
    if reason == "unknown_camera":
        r["quality"]["camera_stable"] = None
    if reason == "missing_pose":
        r["persons"] = []
    if reason == "multi_person":
        r["persons"].append(copy.deepcopy(r["persons"][0]))
    if reason == "weak_fallen":
        rgb["raw_candidates"]["fallen"][0]["confidence"] = 0.65
    if reason == "vertical_legs":
        for a in [15, 16]:
            r["persons"][0]["keypoints"][a] = [220, 420, 0.95]
    if reason == "clipped":
        r["persons"][0]["box"][0] = 0
    if reason == "competing_person":
        rgb["raw_candidates"]["persons"].append(copy.deepcopy(rgb["raw_candidates"]["persons"][0]))
    if reason == "competing_fallen":
        rgb["raw_candidates"]["fallen"].append(copy.deepcopy(rgb["raw_candidates"]["fallen"][0]))
    _, out = s.update(r, 2.0 if reason == "gap" else 1.04, rgb)
    assert not out["observed"]


def test_view_check_distinguishes_unknown_background_and_slow_sampling():
    s = LiveSetup()
    r, rgb = lying_frame()
    r["quality"] = dict(camera_stable=None, code="background_features_missing", reason="纹理不足")
    for i in range(7):
        out, _ = s.update(r, i * 0.45, rgb)
    assert out["camera_stable"] is None and out["camera_reason"] == "纹理不足"
    assert out["sampling_slow"] and out["median_gap_s"] == pytest.approx(0.45)

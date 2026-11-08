import copy

import numpy as np
import pytest
from test_webcam_daily_activity import straight_pose

from fallpre.weekly.webcam_daily_activity import DailyActivity
from fallpre.weekly.webcam_frontal_bend import FrontalBend


def standing_reference(branch):
    for i in range(7):
        info = branch.update(i * 0.1, 1, straight_pose(), {"label": "standing", "raw_label": "standing"})
    assert info["ready"]


@pytest.mark.parametrize("roll", [-15, 0, 15])
def test_frontal_compression_with_stationary_extended_legs_confirms(roll):
    b = FrontalBend()

    def rotate(p):
        p = copy.deepcopy(p)
        angle = np.deg2rad(roll)
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for q in p["kpts"]:
            q[:2] = (matrix @ (np.array(q[:2]) - 100) + 100).tolist()
        return p

    for i in range(7):
        info = b.update(i * 0.1, 1, rotate(straight_pose()), {"label": "standing"})
    assert info["ready"]
    p = rotate(straight_pose(shoulder_drop=30))
    for i in range(12):
        info = b.update(0.7 + i * 0.13, 1, p, {"label": "standing"})
        if i >= 2:
            assert info["confirmed"]
            assert info["compression"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize(
    "reason",
    [
        "knees",
        "feet",
        "hip",
        "missing",
        "camera",
        "epoch",
        "gap",
        "fall_vote",
        "no_reference",
        "expired",
        "sit",
        "rise",
    ],
)
def test_frontal_branch_rejects_unsafe_or_ambiguous_shortcuts(reason):
    b = FrontalBend()
    if reason != "no_reference":
        standing_reference(b)
    p = straight_pose(shoulder_drop=30)
    r = {"label": "unknown", "raw_label": "background"}
    if reason == "knees":
        for j in [13, 14]:
            p["kpts"][j][0] += 30
    if reason == "feet":
        for j in [15, 16]:
            p["kpts"][j][0] += 20
    if reason == "hip":
        for j in [11, 12]:
            p["kpts"][j][1] += 15
    if reason == "missing":
        p["kpts"][15][2] = 0.2
    if reason == "fall_vote":
        r["raw_label"] = "falling"
    if reason == "sit":
        r["label"] = "seated"
    if reason == "rise":
        r["label"] = "standing_up"
    for i in range(5):
        t = (10 if reason == "expired" else 1.2 if reason == "gap" else 0.7) + i * 0.13
        info = b.update(t, 2 if reason == "epoch" else 1, p, r, camera_stable=reason != "camera")
        assert not info["confirmed"]


def test_frontal_bend_recovers_after_dynamic_evidence_has_aged_out():
    d = DailyActivity()
    for i in range(7):
        d.update(i * 0.1, 1, straight_pose(), {"label": "standing", "raw_label": "standing"})
    p = straight_pose(shoulder_drop=30)
    outputs = []
    for i in range(15):
        out = d.update(0.7 + i * 0.13, 1, p, {"label": "unknown", "raw_label": "background"})
        outputs.append(out)
    assert all(r["label"] == "bending" for r in outputs[3:])
    assert outputs[-1]["decision_stage"] == "frontal_bend"

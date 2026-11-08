import pytest

from fallpre.weekly.action_v7 import CompletionModel
from fallpre.weekly.human import ROOT
from fallpre.weekly.webcam_actions import SinglePersonActions as LiveActions, SeatEvidence, display_decision
from test_action_v2 import person


def info(**overrides):
    return dict(dict(reliable=True, hip_speed=0, knee=.98, hip_angle=.98, tilt=.02), **overrides)


@pytest.mark.parametrize('label,phase,features,expected', [
    ('background', 'standing', info(), 'standing'),
    ('background', 'seated', info(knee=.6, hip_angle=.6), 'seated'),
    ('background', 'forward_lean', info(knee=.6, hip_angle=.5), 'unknown'),
    ('standing_up', 'rising', info(hip_speed=.2), 'standing_up'),
    ('standing_up', 'rising', info(), 'unknown'),
    ('sitting_down', 'descending', info(hip_speed=-.2), 'sitting_down'),
    ('sitting_down', 'seated', info(), 'unknown'),
    ('bending', 'seated', info(knee=.6, hip_angle=.6), 'unknown'),
    ('unknown', 'standing', info(), 'unknown'),
    ('background', 'seated', info(), 'unknown'),
    ('background', 'standing', None, 'unknown'),
])
def test_evidence_mapping(label, phase, features, expected):
    assert display_decision(dict(label=label, phase=phase), features)[0] == expected


@pytest.fixture(scope='module')
def model():
    return CompletionModel.load(ROOT / 'runs/action-v7/training/model.json')


def detection(x=100):
    p = person(x=x)
    return dict(box=p['box'], confidence=p['det'], keypoints=p['kpts'])


def test_missing_partial_and_multiple_people_clear_history(model):
    for kind in ['missing', 'partial', 'multiple']:
        live = LiveActions(model)
        for i in range(20):
            live.update([detection()], i / 30, 640, 480)
        assert live.history
        points = [detection()]
        if kind == 'missing':
            points = []
        elif kind == 'partial':
            for j in (15, 16):
                points[0]['keypoints'][j][2] = 0
        else:
            points.append(detection(400))
        out = live.update(points, 20 / 30, 640, 480)
        assert out['label'] == 'unknown' and not live.history


def test_clock_gaps_restart_and_isolation(model):
    live = LiveActions(model)
    for i in range(20):
        live.update([detection()], i / 30, 640, 480)
    with pytest.raises(ValueError):
        live.update([detection()], .5, 640, 480)
    assert live.update([detection()], 2, 640, 480)['label'] == 'unknown'
    assert len(live.history) <= 1
    independent = LiveActions(model)
    assert independent.update([detection()], 0, 640, 480)['evidence_s'] == 0
    assert independent.online is not live.online


def test_out_of_frame_joints_cannot_qualify(model):
    p = detection()
    for i in (15, 16):
        p['keypoints'][i][1] = 800
    out = LiveActions(model).update([p], 0, 640, 480)
    assert out['label'] == 'unknown' and '关节不完整' in out['reason']


def seat_info(y=200, **overrides):
    return dict(info(knee=.65, hip_angle=.65), hip_y=y, scale=60, **overrides)


def test_jitter_no_longer_restarts_seat_confirmation():
    evidence = SeatEvidence()
    for i in range(25):
        # The old instantaneous-speed gate would be repeatedly reset.
        value = seat_info(200 + (1 if i % 2 else -1))
        value['hip_speed'] = .2 if i % 2 else -.2
        out = evidence.update(i / 30, dict(label='background', phase='seated'), value)
    assert out['accepted'] and out['stable']


def test_forward_lean_alone_never_establishes_seat():
    evidence = SeatEvidence()
    for i in range(90):
        out = evidence.update(i / 30, dict(label='background', phase='forward_lean'), seat_info())
        assert not out['accepted'] and not out.get('anchor_available', False)


def anchored_evidence():
    evidence = SeatEvidence()
    for i in range(20):
        evidence.update(i / 30, dict(label='background', phase='seated'), seat_info())
    return evidence


def test_anchored_forward_lean_requires_current_bent_stable_pose():
    evidence = anchored_evidence()
    for i in range(20, 40):
        out = evidence.update(i / 30, dict(label='background', phase='forward_lean'), seat_info())
    assert out['accepted']
    changed = seat_info()
    changed['knee'] = .75  # still bent, but actively opening from the seat anchor
    assert not evidence.update(40 / 30, dict(label='background', phase='forward_lean'), changed)['accepted']


def test_ascent_and_stale_anchor_cannot_preserve_seated():
    evidence = anchored_evidence()
    for i in range(20, 31):
        out = evidence.update(i / 30, dict(label='background', phase='forward_lean'),
                              seat_info(200 - (i - 19) * 1.2))
    assert not out['accepted']
    evidence = anchored_evidence()
    for i in range(20, 340):
        out = evidence.update(i / 30, dict(label='background', phase='forward_lean'), seat_info())
    assert not out['accepted'] and not out['anchor_available']


@pytest.mark.parametrize('label,phase', [('standing_up', 'rising'), ('sitting_down', 'descending'),
                                       ('bending', 'other'), ('unknown', 'unknown')])
def test_actions_clear_seated_anchor(label, phase):
    evidence = anchored_evidence()
    out = evidence.update(.7, dict(label=label, phase=phase), seat_info())
    assert not out['accepted'] and evidence.anchor is None and not evidence.samples


def test_missing_quality_resets_seated_anchor():
    evidence = anchored_evidence()
    assert not evidence.update(.7, dict(label='background', phase='seated'), None)['accepted']
    assert evidence.anchor is None and not evidence.samples

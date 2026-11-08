"""Earlier bend confirmation must use distinct, current motion observations."""
import copy

import numpy as np
import pytest

from fallpre.weekly.webcam_daily_activity import DailyActivity, sequential_hinge
from test_webcam_daily_activity import straight_pose


def observations(times=(0, .13, .26)):
    return [(t, np.asarray(straight_pose(drop, hip)['kpts']), 60., 1.)
            for t, drop, hip in zip(times, (0, 8, 16), (0, 2, 4))]


def result(raw='background'):
    return dict(label='unknown', raw_label=raw, decision_code='history_warmup')


def test_two_distinct_head_first_steps_support_the_first_geometric_candidate():
    activity = DailyActivity()
    outputs = []
    for t, p, _, _ in observations():
        outputs.append(activity.update(t, 1, dict(kpts=p.tolist(), box=[50,50,200,300], det=.95), result()))
    assert [out['label'] for out in outputs] == ['unknown', 'unknown', 'bending']
    assert len(activity.history) == 1  # real motion predates the candidate timer


@pytest.mark.parametrize('mode', ['single_jump', 'whole_body_translation', 'one_knee_bent',
                                  'one_foot_steps', 'missing_joint', 'low_confidence'])
def test_shortcut_rejects_motion_or_support_that_does_not_prove_a_continuous_hinge(mode):
    rows = observations()
    if mode == 'single_jump':
        rows[1] = (rows[1][0], rows[0][1].copy(), rows[1][2], rows[1][3])
    elif mode == 'whole_body_translation':
        for i, (_, p, _, _) in enumerate(rows):
            p[:, :2] = rows[0][1][:, :2] + [0, i*8]
    elif mode == 'one_knee_bent':
        rows[-1][1][13, 0] += 30
    elif mode == 'one_foot_steps':
        rows[-1][1][15, 0] += 12
    elif mode == 'missing_joint':
        rows[1][1][15, 2] = 0
    else:
        rows[1][1][11, 2] = .7
    snapshot = copy.deepcopy(rows)
    assert not sequential_hinge(rows)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(rows, snapshot))


@pytest.mark.parametrize('times', [(0, .02, .04), (0, .2, .4), (0, .13, .13), (0, .13, .10)])
def test_shortcut_needs_real_increasing_observations_over_a_short_motion_interval(times):
    assert not sequential_hinge(observations(times))


def test_one_motion_difference_is_not_two_observations_of_motion():
    assert not sequential_hinge(observations()[-2:])


@pytest.mark.parametrize('mode', ['gap', 'epoch'])
def test_discontinuity_cannot_borrow_motion_before_the_reset(mode):
    activity = DailyActivity()
    for t, p, _, _ in observations()[:2]:
        activity.update(t, 1, dict(kpts=p.tolist(), box=[50,50,200,300], det=.95), result())
    _, p, _, _ = observations()[-1]
    output = activity.update(.8 if mode == 'gap' else .26, 2 if mode == 'epoch' else 1,
                             dict(kpts=p.tolist(), box=[50,50,200,300], det=.95), result())
    assert output['label'] == 'unknown'


def test_sequential_hinge_never_overrides_current_raw_falling_vote():
    activity = DailyActivity()
    for t, p, _, _ in observations():
        output = activity.update(t, 1, dict(kpts=p.tolist(), box=[50,50,200,300], det=.95),
                                 result('falling' if t > .2 else 'background'))
    assert output['label'] == 'unknown'
    assert output['raw_label'] == 'falling'

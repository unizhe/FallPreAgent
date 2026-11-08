import copy

import numpy as np
import pytest

from fallpre.weekly.budgeted_recovery import BudgetedRefinedPoseRecovery
from fallpre.weekly.cost_aware_recovery import CostAwareRefinedPoseRecovery
from test_action_v2 import person
from test_budgeted_recovery import Clock, IMAGE, weak_complete_pose
from test_pose_recovery import initialized
from test_verified_tracking import observation


def test_second_call_estimated_not_to_fit_is_never_started():
    clock = Clock()
    calls = []

    def backend(_):
        calls.append(True)
        clock.advance(.12)
        return []

    row = observation(2, [])
    out, audit = CostAwareRefinedPoseRecovery(
        backend=backend, budget_s=.2, clock=clock,
    ).augment(IMAGE, row, initialized())
    b = audit['budget_audit']
    assert out is row and not audit['accepted'] and audit['candidates'] == []
    assert len(calls) == b['total_calls'] == 1
    assert b['exhaustion_stage'] == b['exhaustion_cause'] == 'predicted_call_overrun'
    assert b['estimated_next_call_s'] == pytest.approx(.12)
    assert b['remaining_budget_s'] == pytest.approx(.08)
    assert b['elapsed_s'] == pytest.approx(.12) and b['overrun_s'] == 0
    assert b['blocked_call']['view'] == 'full_1'
    assert b['discarded_incomplete_recovery']


def test_estimate_uses_maximum_not_last_duration_and_keeps_view_order():
    clock = Clock()
    durations = iter([.03, .08, .04])

    def backend(_):
        clock.advance(next(durations))
        return []

    _, audit = CostAwareRefinedPoseRecovery(
        backend=backend, budget_s=.2, clock=clock,
    ).augment(IMAGE, observation(2, []), initialized())
    b = audit['budget_audit']
    assert [c['view'] for c in b['calls']] == ['full_0', 'full_1', 'full_2']
    assert b['blocked_call']['view'] == 'full_3'
    assert b['estimated_next_call_s'] == pytest.approx(.08)
    assert b['remaining_budget_s'] == pytest.approx(.05)


def test_prediction_at_exact_remaining_budget_is_rejected():
    clock = Clock()

    def backend(_):
        clock.advance(.1)
        return []

    _, audit = CostAwareRefinedPoseRecovery(
        backend=backend, budget_s=.2, clock=clock,
    ).augment(IMAGE, observation(2, []), initialized())
    assert audit['budget_audit']['total_calls'] == 1
    assert audit['budget_audit']['exhaustion_stage'] == 'predicted_call_overrun'


def test_first_call_can_overrun_and_is_marked_as_actual_exhaustion():
    clock = Clock()

    def backend(_):
        clock.advance(.3)
        return [person()]

    row = observation(2, [])
    out, audit = CostAwareRefinedPoseRecovery(
        backend=backend, budget_s=.2, clock=clock,
    ).augment(IMAGE, row, initialized())
    b = audit['budget_audit']
    assert out is row and not audit['accepted']
    assert b['exhaustion_stage'] == 'after_call'
    assert b['exhaustion_cause'] == 'elapsed_budget_exhausted'
    assert b['overrun_s'] == pytest.approx(.1)
    assert b['estimated_next_call_s'] is None and b['remaining_budget_s'] is None


def test_estimate_is_not_a_hard_deadline_for_later_calls():
    clock = Clock()
    durations = iter([.04, .18])

    def backend(_):
        clock.advance(next(durations))
        return []

    _, audit = CostAwareRefinedPoseRecovery(
        backend=backend, budget_s=.2, clock=clock,
    ).augment(IMAGE, observation(2, []), initialized())
    b = audit['budget_audit']
    assert b['total_calls'] == 2
    assert b['exhaustion_stage'] == 'after_call'
    assert b['overrun_s'] == pytest.approx(.02)


def test_actual_elapsed_check_takes_precedence_before_call():
    ticks = iter([0., 0., .1, .2, .2])
    _, audit = CostAwareRefinedPoseRecovery(
        backend=lambda _: [], budget_s=.2, clock=lambda: next(ticks),
    ).augment(IMAGE, observation(2, []), initialized())
    b = audit['budget_audit']
    assert b['total_calls'] == 1 and b['exhaustion_stage'] == 'before_call'
    assert b['exhaustion_cause'] == 'elapsed_budget_exhausted'
    assert b['estimated_next_call_s'] is None


def test_backend_kinds_do_not_share_cost_estimates():
    clock = Clock()

    def backend(_):
        clock.advance(.12)
        return [weak_complete_pose()]

    def refiner(image, box, turn, expansion):
        clock.advance(.03)
        return np.asarray(person()['kpts'])

    row = observation(2, [])
    out, audit = CostAwareRefinedPoseRecovery(
        backend=backend, refiner=refiner, budget_s=.2, clock=clock,
    ).augment(IMAGE, row, initialized())
    b = audit['budget_audit']
    # The RTMO .12 estimate must not block the first RTMPose call with .08 left.
    assert b['rtmo_calls'] == 1 and b['rtmpose_calls'] == 2
    assert b['estimated_next_call_s'] == pytest.approx(.03)
    assert b['remaining_budget_s'] == pytest.approx(.02)
    assert [(c['rotation'], c['expansion']) for c in b['calls'][1:]] == [(0, 1.), (0, 1.15)]
    assert b['blocked_call']['kind'] == 'rtmpose'
    # An earlier accepted RTMO candidate must not escape an unfinished refinement.
    assert out is row and not audit['accepted'] and not out['instances']


def test_each_attempt_starts_without_prior_frame_costs():
    clock = Clock()
    times = iter([.12, .09, .09])

    def backend(_):
        clock.advance(next(times))
        return []

    recovery = CostAwareRefinedPoseRecovery(backend=backend, budget_s=.2, clock=clock)
    _, first = recovery.augment(IMAGE, observation(2, []), initialized())
    _, second = recovery.augment(IMAGE, observation(2, []), initialized())
    assert first['budget_audit']['total_calls'] == 1
    assert first['budget_audit']['exhaustion_stage'] == 'predicted_call_overrun'
    assert second['budget_audit']['total_calls'] == 2
    assert second['budget_audit']['exhaustion_stage'] == 'predicted_call_overrun'
    assert second['budget_audit']['estimated_next_call_s'] == pytest.approx(.09)
    assert second['budget_audit']['elapsed_s'] == pytest.approx(.18)
    assert recovery._active is None


@pytest.mark.parametrize('empty', [False, True])
def test_default_unlimited_keeps_original_result_and_audit(empty):
    backend = lambda _: [] if empty else [person()]
    row = observation(2, [])
    expected, expected_audit = BudgetedRefinedPoseRecovery(backend=backend, clock=Clock()).augment(
        IMAGE, row, initialized())
    actual, audit = CostAwareRefinedPoseRecovery(backend=backend, clock=Clock()).augment(
        IMAGE, row, initialized())
    b = audit['budget_audit']
    assert b.pop('exhaustion_cause') is None
    assert b.pop('estimated_next_call_s') is None
    assert b.pop('remaining_budget_s') is None
    assert actual == expected and audit == expected_audit


def tracker_snapshot(tracker):
    return (
        copy.deepcopy({k: v for k, v in vars(tracker).items() if k != 'raw'}),
        copy.deepcopy({k: v for k, v in vars(tracker.raw).items() if k != 'tracker'}),
        tracker.raw.tracker.frame_id,
    )


@pytest.mark.parametrize('scenario', ['success', 'predicted_abort', 'exception'])
def test_complete_abort_and_exception_preserve_tracker_and_input(scenario):
    clock = Clock()

    def backend(_):
        clock.advance(.12)
        if scenario == 'exception':
            raise ValueError('backend failure')
        return [person()] if scenario == 'success' else []

    recovery = CostAwareRefinedPoseRecovery(backend=backend, budget_s=.2, clock=clock)
    tracker = initialized()
    row = observation(2, [])
    original_row, original_tracker = copy.deepcopy(row), tracker_snapshot(tracker)
    if scenario == 'exception':
        with pytest.raises(ValueError, match='backend failure'):
            recovery.augment(IMAGE, row, tracker)
    else:
        out, audit = recovery.augment(IMAGE, row, tracker)
        if scenario == 'success':
            assert audit['accepted'] and out['instances']
        else:
            assert not audit['accepted'] and out is row
    assert tracker_snapshot(tracker) == original_tracker
    assert row == original_row and recovery._active is None


def test_native_gate_never_invokes_backends():
    def forbidden(*_):
        raise AssertionError('Native observations bypass recovery')

    row = observation(2, [person()], ())
    out, audit = CostAwareRefinedPoseRecovery(
        backend=forbidden, refiner=forbidden, budget_s=.2, clock=Clock(),
    ).augment(IMAGE, row, initialized())
    assert out is row and not audit['executed']
    assert audit['reason'] == 'native_pose_available'
    assert audit['budget_audit']['calls'] == []


def test_configuration_changes_only_call_admission():
    options = dict(allow_detector_proposals=True, resolve_native_ambiguity=True)
    expected = BudgetedRefinedPoseRecovery(budget_s=.2, **options).configuration()
    actual = CostAwareRefinedPoseRecovery(budget_s=.2, **options).configuration()
    admission = actual['recovery_budget'].pop('call_admission')
    assert actual == expected
    assert admission['history_scope'] == 'current_attempt_only'
    assert not admission['hard_deadline']

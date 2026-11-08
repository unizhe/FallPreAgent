import copy

import numpy as np
import pytest

from fallpre.weekly.budgeted_recovery import BudgetedRefinedPoseRecovery
from fallpre.weekly.pose_refinement import RefinedPoseRecovery
from test_action_v2 import person
from test_pose_recovery import initialized
from test_verified_tracking import observation


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


IMAGE = np.zeros((300, 600, 3), np.uint8)


@pytest.mark.parametrize('budget', [0, -1, float('nan'), float('inf'), -float('inf'), True])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        BudgetedRefinedPoseRecovery(budget_s=budget)


@pytest.mark.parametrize('empty', [False, True])
def test_unlimited_output_and_audit_equal_original_except_budget(empty):
    backend = lambda _: [] if empty else [person()]
    row = observation(2, [])
    expected, expected_audit = RefinedPoseRecovery(backend=backend).augment(IMAGE, row, initialized())
    actual, audit = BudgetedRefinedPoseRecovery(backend=backend, clock=Clock()).augment(IMAGE, row, initialized())
    budget = audit.pop('budget_audit')
    assert actual == expected and audit == expected_audit
    assert budget['rtmo_calls'] == (8 if empty else 1)
    assert not budget['exhausted']
    if empty:
        assert [c['view'] for c in budget['calls']] == [
            'full_0', 'full_1', 'full_2', 'full_3', 'crop_0', 'crop_1', 'crop_2', 'crop_3']


def test_native_pose_never_calls_backend_or_changes_row():
    def forbidden(*_):
        raise AssertionError('Native observations must bypass recovery')
    row = observation(2, [person()], ())
    out, audit = BudgetedRefinedPoseRecovery(
        backend=forbidden, refiner=forbidden, budget_s=.2, clock=Clock(),
    ).augment(IMAGE, row, initialized())
    assert out is row and not audit['executed']
    assert audit['reason'] == 'native_pose_available'
    assert audit['budget_audit']['calls'] == []


def test_inflight_overrun_discarded_and_actual_duration_visible():
    clock = Clock()
    def slow(_):
        clock.advance(.3)
        return [person()]
    row = observation(2, [])
    out, audit = BudgetedRefinedPoseRecovery(backend=slow, budget_s=.2, clock=clock).augment(
        IMAGE, row, initialized())
    b = audit['budget_audit']
    assert out is row and not out['instances'] and not audit['accepted']
    assert audit['candidates'] == [] and audit['reason'] == 'recovery_budget_exhausted'
    assert b['exhausted'] and b['exhaustion_stage'] == 'after_call'
    assert b['rtmo_calls'] == 1 and b['elapsed_s'] == pytest.approx(.3)
    assert b['overrun_s'] == pytest.approx(.1)
    assert b['calls'][0]['duration_s'] == pytest.approx(.3)


def test_exact_budget_boundary_is_exhausted():
    clock = Clock()
    def backend(_):
        clock.advance(.2)
        return [person()]
    out, audit = BudgetedRefinedPoseRecovery(backend=backend, budget_s=.2, clock=clock).augment(
        IMAGE, observation(2, []), initialized())
    assert audit['budget_audit']['exhausted'] and not audit['accepted']


def test_second_call_can_overrun_and_stops_later_views():
    clock = Clock()
    def backend(_):
        clock.advance(.12)
        return []
    _, audit = BudgetedRefinedPoseRecovery(backend=backend, budget_s=.2, clock=clock).augment(
        IMAGE, observation(2, []), initialized())
    assert audit['inference_passes'] == 2
    assert [c['view'] for c in audit['budget_audit']['calls']] == ['full_0', 'full_1']
    assert audit['budget_audit']['elapsed_s'] == pytest.approx(.24)


def test_pre_call_check_does_not_start_a_call_after_budget_is_used():
    ticks = iter([0., 0., .1, .2, .2])
    calls = []
    def backend(_):
        calls.append(True)
        return []
    _, audit = BudgetedRefinedPoseRecovery(
        backend=backend, budget_s=.2, clock=lambda: next(ticks),
    ).augment(IMAGE, observation(2, []), initialized())
    b = audit['budget_audit']
    assert len(calls) == b['total_calls'] == 1
    assert b['exhaustion_stage'] == 'before_call'
    assert b['blocked_call']['view'] == 'full_1'
    assert audit['inference_passes'] == 1


def test_postprocessing_crossing_budget_discards_completed_candidate():
    ticks = iter([0., 0., .1, .25, .25])
    row = observation(2, [])
    out, audit = BudgetedRefinedPoseRecovery(
        backend=lambda _: [person()], budget_s=.2, clock=lambda: next(ticks),
    ).augment(IMAGE, row, initialized())
    assert out is row and not audit['accepted']
    assert audit['budget_audit']['exhaustion_stage'] == 'after_augment'
    assert audit['budget_audit']['overrun_s'] == pytest.approx(.05)


def weak_complete_pose():
    p = person()
    # Valid RTMO body but not a complete pose: refinement is still required.
    for j in range(5):
        p['kpts'][j][2] = 0
    for j in (15, 16):
        p['kpts'][j][2] = 0
    return p


def test_refiner_exhaustion_discards_already_accepted_rtmo_candidate():
    clock = Clock()
    def backend(_):
        clock.advance(.05)
        return [weak_complete_pose()]
    def refiner(image, box, turn, expansion):
        clock.advance(.1)
        return np.asarray(person()['kpts'])
    row = observation(2, [])
    out, audit = BudgetedRefinedPoseRecovery(
        backend=backend, refiner=refiner, budget_s=.2, clock=clock,
    ).augment(IMAGE, row, initialized())
    assert out is row and not out['instances']
    assert not audit['accepted'] and audit['candidates'] == []
    assert audit['refinement'] == dict(executed=True, accepted=False, inference_passes=2)
    b = audit['budget_audit']
    assert b['rtmo_calls'] == 1 and b['rtmpose_calls'] == 2
    assert b['elapsed_s'] == pytest.approx(.25)
    assert [(c['rotation'], c['expansion']) for c in b['calls'][1:]] == [(0, 1.), (0, 1.15)]


def test_unlimited_refinement_keeps_original_decisions_and_view_order():
    backend = lambda _: [weak_complete_pose()]
    refiner = lambda image, box, turn, expansion: np.asarray(person()['kpts'])
    row = observation(2, [])
    expected, expected_audit = RefinedPoseRecovery(backend=backend, refiner=refiner).augment(
        IMAGE, row, initialized())
    actual, audit = BudgetedRefinedPoseRecovery(backend=backend, refiner=refiner, clock=Clock()).augment(
        IMAGE, row, initialized())
    b = audit.pop('budget_audit')
    assert actual == expected and audit == expected_audit
    assert b['rtmpose_calls'] == 8 and audit['refinement']['accepted']
    assert [(c['rotation'], c['expansion']) for c in b['calls'][1:]] == [
        (turn, scale) for turn in range(4) for scale in (1., 1.15)]


def test_exhaustion_does_not_mutate_tracker_or_observation():
    clock = Clock()
    def backend(_):
        clock.advance(.3)
        return [person()]
    tracker = initialized()
    row = observation(2, [person(500)], ())
    state = copy.deepcopy({k: v for k, v in vars(tracker).items() if k != 'raw'})
    raw_state = copy.deepcopy({k: v for k, v in vars(tracker.raw).items() if k != 'tracker'})
    raw_frame = tracker.raw.tracker.frame_id
    original_row = copy.deepcopy(row)
    out, _ = BudgetedRefinedPoseRecovery(backend=backend, budget_s=.2, clock=clock).augment(
        IMAGE, row, tracker)
    assert {k: v for k, v in vars(tracker).items() if k != 'raw'} == state
    assert {k: v for k, v in vars(tracker.raw).items() if k != 'tracker'} == raw_state
    assert tracker.raw.tracker.frame_id == raw_frame
    assert row == original_row and out is row
    # The wrapper does not force missing input: the original tracker still decides.
    assert len(out['instances']) == 1


def test_unrelated_backend_error_is_not_swallowed_and_state_resets():
    clock = Clock()
    def broken(_):
        raise ValueError('backend failure')
    recovery = BudgetedRefinedPoseRecovery(backend=broken, budget_s=.2, clock=clock)
    for _ in range(2):
        with pytest.raises(ValueError, match='backend failure'):
            recovery.augment(IMAGE, observation(2, []), initialized())
        assert recovery._active is None


def test_each_frame_has_fresh_budget():
    clock = Clock()
    def backend(_):
        clock.advance(.1)
        return [person()]
    recovery = BudgetedRefinedPoseRecovery(backend=backend, budget_s=.2, clock=clock)
    for _ in range(3):
        _, audit = recovery.augment(IMAGE, observation(2, []), initialized())
        assert audit['accepted'] and not audit['budget_audit']['exhausted']
        assert audit['budget_audit']['elapsed_s'] == pytest.approx(.1)


def test_configuration_keeps_original_thresholds_and_adds_budget():
    options = dict(allow_detector_proposals=True, resolve_native_ambiguity=True)
    cfg = BudgetedRefinedPoseRecovery(budget_s=.2, **options).configuration()
    budget = cfg.pop('recovery_budget')
    assert cfg == RefinedPoseRecovery(**options).configuration()
    assert budget['seconds'] == .2 and budget['unchanged_view_order']

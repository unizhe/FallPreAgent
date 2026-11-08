"""Handoff contracts; synthetic streams do not measure model accuracy."""
from copy import deepcopy
import pytest
from test_webcam_rgb_fall import pose
from fallpre.weekly.webcam_fall_handoff import LiveFallHandoff

UP = [200., 40., 300., 400.]
STEPS = [[180., 100., 310., 430.], [160., 180., 340., 460.], [140., 280., 380., 480.]]
LOW = STEPS[-1]


def d(box, score=.9):
    return dict(box=list(box), confidence=score)


def feed(f, t, people=(), fallen=(), **kwargs):
    return f.update(dict(persons=list(people), fallen=list(fallen)), t, 640, 480, **kwargs)


def admitted(f):
    for t in (0., .2, .4, .6):
        o = feed(f, t, [d(UP)], pose_person=pose(UP), pose_epoch=7)
    assert o['fall']['target_epoch'] is not None


def handoff(f):
    admitted(f)
    for t, box in zip((.8, 1., 1.2), STEPS):
        o = feed(f, t, [d(box)], [d(box)])
    assert o['diagnostics']['handoff_ready']
    return o


def test_current_fallen_track_survives_pose_and_person_loss_without_fake_geometry():
    f = LiveFallHandoff(); handoff(f)
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        o = feed(f, t, fallen=[d(LOW)])
    assert o['fall']['state'] == 'confirmed'
    assert o['fall']['candidate_duration_s'] == 1.
    assert o['fall']['metrics']['height_ratio'] is None
    assert o['fall']['metrics']['center_drop_norm'] is None
    assert o['diagnostics']['descent_witness']['source'] == 'person_detector'
    assert o['observation']['person']['keypoints'] == []
    assert o['observation']['selection']['raw_index'] is None
    assert o['observation']['selection']['detector'] == 'fallen'


def test_initial_fallen_without_witnessed_person_cannot_adopt_or_alarm():
    f = LiveFallHandoff()
    for i in range(20):
        o = feed(f, i * .2, fallen=[d(LOW)])
    assert o['fall']['state'] == 'unobservable' and not f.confirmed_events


def test_static_overlapping_furniture_cannot_hand_over():
    f = LiveFallHandoff(); admitted(f)
    # Same stationary fall proposal; despite overlap, it never moves downward.
    for i, box in enumerate(STEPS):
        o = feed(f, .8 + i * .2, [d(box)], [d([140, 140, 380, 480])])
    assert not o['diagnostics']['handoff_ready']
    assert feed(f, 1.4, fallen=[d([140, 140, 380, 480])])['fall']['state'] == 'unobservable'


def test_no_descent_no_handoff_even_with_repeated_strong_labels():
    f = LiveFallHandoff(); admitted(f)
    for i in range(10):
        o = feed(f, .8 + i * .2, [d(UP)], [d(UP)])
    assert not o['diagnostics']['handoff_ready'] and not f.confirmed_events


@pytest.mark.parametrize('case', ['two_people', 'near_weak_person', 'two_falls', 'near_weak_fall', 'camera', 'size', 'gap', 'jump'])
def test_ambiguity_and_discontinuity_clear_witnessed_handoff(case):
    f = LiveFallHandoff(); handoff(f)
    raw = dict(persons=[], fallen=[d(LOW)])
    t, width, stable = 1.4, 640, True
    if case == 'two_people': raw['persons'] = [d(LOW), d([450, 30, 550, 430])]
    if case == 'near_weak_person': raw['persons'] = [d(LOW), d([150, 285, 390, 475], .3)]
    if case == 'two_falls': raw['fallen'].append(d(LOW))
    if case == 'near_weak_fall': raw['fallen'].append(d([150, 290, 390, 475], .3))
    if case == 'camera': stable = False
    if case == 'size': width = 800
    if case == 'gap': t = 1.8
    if case == 'jump': raw['fallen'] = [d([430, 300, 620, 470])]
    o = f.update(raw, t, width, 480, camera_stable=stable)
    assert o['fall']['state'] == 'unobservable'
    if case != 'jump': assert not o['diagnostics']['handoff_ready']
    assert not f.confirmed_events and o['observation']['person'] is None


def test_missing_intervals_do_not_count_or_draw_old_box():
    f = LiveFallHandoff(); handoff(f)
    o = feed(f, 1.4)
    assert not o['fall']['observed'] and o['observation']['person'] is None
    o = feed(f, 1.6, fallen=[d(LOW)])
    assert o['fall']['candidate_duration_s'] == 0
    assert o['fall']['state'] == 'suspected'


def test_weak_fallen_does_not_extend_observation_ttl():
    f = LiveFallHandoff(); handoff(f)
    for t in (1.4, 1.6, 1.8): o = feed(f, t, fallen=[d(LOW, .59)])
    assert not o['diagnostics']['handoff_ready'] and not f.confirmed_events


def test_historical_red_cannot_color_reacquired_upright_person():
    f = LiveFallHandoff(); handoff(f)
    for t in (1.4, 1.6, 1.8, 2., 2.2): feed(f, t, fallen=[d(LOW)])
    old = f.confirmed_events[0]['target_epoch']
    for t in (3., 3.2, 3.4, 3.6):
        o = feed(f, t, [d(UP)], pose_person=pose(UP), pose_epoch=8)
    assert o['fall']['target_epoch'] != old and o['fall']['state'] != 'confirmed'
    assert len(f.confirmed_events) == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., 0.])
def test_invalid_source_clock_rejected_without_state_mutation(bad):
    f = LiveFallHandoff(); feed(f, 0.)
    before = deepcopy(f.export())
    with pytest.raises(ValueError): feed(f, bad)
    assert f.export() == before


def test_weak_classification_after_handoff_is_unknown_not_green():
    f = LiveFallHandoff(); handoff(f)
    o = feed(f, 1.4, [d(LOW)], [d(LOW, .59)])
    assert o['fall']['state'] == 'unobservable'
    assert not o['fall']['observed'] and o['fall']['candidate_duration_s'] == 0


def test_static_false_fall_classification_is_gray_not_yellow():
    f = LiveFallHandoff(); admitted(f)
    for i in range(10):
        o = feed(f, .8 + i * .2, [d(UP)], [d(UP)])
        assert o['fall']['state'] == 'unobservable'
        assert o['fall']['reason_code'] == 'classification_geometry_conflict'
    assert not f.confirmed_events


def test_short_fallen_overlap_retains_upright_reference_without_accepting_candidates():
    f = LiveFallHandoff(); admitted(f)
    epoch = f._epoch
    o = feed(f, .7, [d(UP)], [d(UP), d(UP, .3)], pose_epoch=7)
    assert o['fall']['state'] == 'unobservable'
    assert o['observation']['person'] is None
    assert o['diagnostics']['ambiguity_retained'] and f._epoch == epoch
    assert not f._pairs and f._low_duration == 0
    for t, box in zip((.8, 1., 1.2), STEPS):
        o = feed(f, t, [d(box)], [d(box)], pose_epoch=7)
    assert o['diagnostics']['handoff_ready']
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        o = feed(f, t, fallen=[d(LOW)])
    assert o['fall']['state'] == 'confirmed'


@pytest.mark.parametrize('boundary', ['prolonged', 'identity', 'weak_person', 'camera'])
def test_overlap_retention_does_not_cross_unproven_boundaries(boundary):
    f = LiveFallHandoff(); admitted(f)
    feed(f, .7, [d(UP)], [d(UP), d(UP, .3)], pose_epoch=7)
    t = 1.1 if boundary == 'prolonged' else .8
    o = feed(f, t, [d(UP, .5 if boundary == 'weak_person' else .9)],
             [d(UP), d(UP, .3)], pose_epoch=8 if boundary == 'identity' else 7,
             camera_stable=boundary != 'camera')
    assert f._reference is None and not o['diagnostics']['handoff_ready']
    assert o['fall']['state'] == 'unobservable'


def test_unique_return_after_ambiguity_must_still_match_admitted_identity():
    f = LiveFallHandoff(); admitted(f)
    feed(f, .7, [d(UP)], [d(UP), d(UP, .3)], pose_epoch=7)
    o = feed(f, .8, [d(STEPS[0])], [d(STEPS[0])], pose_epoch=8)
    assert f._reference is None and not o['diagnostics']['handoff_ready']


def test_ambiguity_never_adds_time_to_confirmation_even_when_reference_is_retained():
    f = LiveFallHandoff(); handoff(f)
    feed(f, 1.4, [d(LOW)], [d(LOW)], pose_epoch=7)
    previous = f._low_duration
    o = feed(f, 1.5, [d(LOW)], [d(LOW), d(LOW, .3)], pose_epoch=7)
    assert o['fall']['state'] == 'unobservable' and f._low_duration == previous
    o = feed(f, 1.6, [d(LOW)], [d(LOW)], pose_epoch=7)
    assert f._low_duration == previous and o['fall']['state'] == 'suspected'


def test_projected_support_conflict_is_diagnostic_not_a_fall_or_admission_veto():
    f = LiveFallHandoff()
    for i in range(4):
        result = feed(f, i * .2, [d(UP)], [d(UP)], pose_person=pose(UP), pose_epoch=7)
        assert result['fall']['state'] == 'unobservable'
        evidence = result['diagnostics']['conflict_evidence']
        assert evidence['current_torso_observed']
        assert evidence['interpretation'] == 'fall_process_unresolved_not_nonfall'
    assert f._reference is not None and not f.confirmed_events


def test_rgb_conflict_does_not_erase_previously_witnessed_reference():
    f = LiveFallHandoff()
    admitted(f)
    for i in range(8):
        result = feed(f, .8 + i * .2, [d(UP)], [d(UP)], pose_person=pose(UP), pose_epoch=7)
        assert f._reference is not None
        assert not result['diagnostics']['initial_support_conflict']
    assert not f.confirmed_events

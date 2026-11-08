"""LiveSession wiring contracts for the observer-only transition recorder."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_webcam_fall_v17_boundaries import DESCENT, LOW, UP, detection, standing_pose
from test_webcam_risk_points import activity, red
from test_webcam_service import ROOT, frame, observations, service, stable_quality

from fallpre.weekly.action_v7 import CompletionModel


@pytest.fixture
def model():
    return CompletionModel.load(ROOT / 'runs/action-v7/training/model.json')


def session(model):
    result = service.LiveSession(model, 'yolo26n-pose', 'video', 'rgb')
    result.quality = stable_quality()
    return result


def rgb(box, low=False, present=True):
    return dict(detections=[], model=dict(id='synthetic'), inference_ms=0.,
                raw_candidates=dict(persons=[detection(box)] if present else [],
                                    fallen=[detection(box)] if low else []))


def test_actual_observer_leaves_session_action_behavior_and_fall_outputs_identical(model):
    with_observer = session(model)
    without_observer = session(model)
    without_observer.risk_points = SimpleNamespace(update=lambda *args, **kwargs: {})
    sequence = [(i * .2, UP, False, True) for i in range(9)]
    sequence += [(1.8 + i * .2, box, True, True) for i, box in enumerate(DESCENT)]
    sequence += [(2.4 + i * .2, LOW, True, False) for i in range(8)]
    sequence += [(4. + i * .2, UP, False, True) for i in range(8)]
    image = frame()
    for t, box, low, present in sequence:
        persons = [standing_pose(box)] if present else []
        first, second = observations(deepcopy(persons)), observations(deepcopy(persons))
        before = deepcopy(persons)
        with_observer.update(image, first, t, rgb_result=rgb(box, low, present))
        without_observer.update(image, second, t, rgb_result=rgb(box, low, present))
        points = first.pop('risk_points')
        second.pop('risk_points')
        assert first == second
        assert first['persons'] == before
        assert points['observer_only'] is True
        assert points['reminders_enabled'] is False
    events = list(with_observer.risk_points.events)
    assert any(e['kind'] == 'fall_confirmed' for e in events)
    assert all(e['prediction_validated'] is False for e in events)


def test_live_session_passes_final_rgb_not_historical_pose_red_to_observer(model):
    live = session(model)
    stale_red = red(0., epoch='old-pose', event_id='old-pose-fall')
    live.fall = SimpleNamespace(update=lambda *args, **kwargs: deepcopy(stale_red), export=lambda: {})
    image = frame()
    for i in range(10):
        out = observations([standing_pose(UP)])
        live.update(image, out, .2 + i * .2, rgb_result=rgb(UP))
        assert out['pose_fall']['state'] == 'confirmed'
        assert out['fall']['state'] not in ('suspected', 'confirmed')
        assert not [e for e in out['risk_points']['new_events'] if e['kind'] == 'fall_confirmed']
    assert not [e for e in live.risk_points.events if e['kind'] == 'fall_confirmed']


def test_normal_rise_is_activity_record_and_never_changes_rgb_alarm(model):
    live = session(model)
    current = {}
    live.actions = SimpleNamespace(update=lambda *args, **kwargs: deepcopy(current['action']))
    live.behavior = SimpleNamespace(update=lambda *args, **kwargs: deepcopy(current['behavior']))
    image = frame()
    for i, label in enumerate(('seated', 'standing_up', 'standing_up')):
        action, behavior = activity(label)
        current.update(action=action, behavior=behavior)
        out = observations([standing_pose(UP)])
        live.update(image, out, i * .13, rgb_result=rgb(UP))
        assert out['action'] == action
        assert out['fall']['state'] not in ('suspected', 'confirmed')
        assert out['risk_points']['reminders_enabled'] is False
    events = list(live.risk_points.events)
    assert len(events) == 1
    event = events[0]
    assert event['kind'] == 'sit_to_stand_started'
    assert event['category'] == 'activity_transition'
    assert event['post_fall_evidence'] is False
    assert event['prediction_validated'] is False

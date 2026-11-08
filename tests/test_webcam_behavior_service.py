"""The independent observer must not borrow another branch's person or history."""
import copy

from test_webcam_service import engine, frame, observations, service, stable_quality
from test_webcam_actions import detection
from fallpre.weekly.webcam_actions import LiveActions


def test_behavior_is_additive_and_bound_to_current_action_target(engine):
    session = service.LiveSession(engine.action_model, 'yolo26n-pose', 'video')
    session.quality = stable_quality()
    legacy = LiveActions(engine.action_model)
    image = frame()
    for i in range(15):
        people = [copy.deepcopy(detection())]
        result = observations(copy.deepcopy(people))
        session.update(image, result, i / 10)
        assert result['action'] == legacy.update(people, i / 10, 640, 480)
        axes = result['behavior']
        assert axes['target_index'] == result['action']['target_index']
        assert axes['target_epoch'] == result['action']['selection']['target_epoch']
        assert axes['schema_version'] == 'behavior-axes-v1'
    assert axes['posture']['label'] == 'upright'
    assert axes['locomotion']['label'] == 'stationary'
    # No current action subject: never continue its labels on a missing frame.
    result = observations([])
    session.update(image, result, 1.5)
    assert result['behavior']['target_index'] is None
    assert all(result['behavior'][k]['label'] == 'unknown'
               for k in ['posture', 'locomotion', 'turning'])


def test_behavior_history_is_per_session_and_camera_motion_clears_it(engine):
    from types import SimpleNamespace
    one = service.LiveSession(engine.action_model, 'yolo26n-pose', 'video')
    two = service.LiveSession(engine.action_model, 'yolo26n-pose', 'video')
    assert one.behavior is not two.behavior
    one.quality = stable_quality()
    image = frame()
    for i in range(15):
        result = observations([copy.deepcopy(detection())])
        one.update(image, result, i / 10)
    assert one.behavior.history and not two.behavior.history
    one.quality = SimpleNamespace(update=lambda *args: dict(camera_stable=False, reason='camera moved'))
    result = observations([copy.deepcopy(detection())])
    one.update(image, result, 1.5)
    assert not one.behavior.history
    assert result['behavior']['quality']['reset_reason'] == 'camera_motion'
    assert result['behavior']['locomotion']['label'] == 'unknown'

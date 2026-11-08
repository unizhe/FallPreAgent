import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_webcam_actions import detection
from test_webcam_fall import person
from test_webcam_fall_target import moved
from test_webcam_pose_recovery import forward

from fallpre.weekly.action_v7 import CompletionModel
from fallpre.weekly.webcam_actions import LiveActions

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('webcam_service', ROOT / 'scripts/webcam_pose_demo.py')
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


@pytest.fixture
def engine():
    instance = service.PoseEngine.__new__(service.PoseEngine)
    instance.default_model = 'yolo26n-pose'
    instance.action_model = CompletionModel.load(ROOT / 'runs/action-v7/training/model.json')
    instance.streams = {}

    def infer(image, model_id):
        return dict(persons=[copy.deepcopy(detection())], width=640, height=480,
                    model={'id': model_id}, inference_ms=0, device='cpu')

    instance.infer = infer
    return instance


def frame():
    return np.random.default_rng(19).integers(35, 225, (480, 640, 3), dtype=np.uint8)


def test_fall_branch_preserves_existing_action_results(engine):
    original = LiveActions(engine.action_model)
    image = frame()
    for index in range(15):
        actual = engine.infer_live(image, 'session1', index / 10)
        expected = original.update([detection()], index / 10, 640, 480)
        assert actual['action'] == expected
        assert actual['model']['id'] == 'yolo26n-pose'
        assert actual['fall']['state'] in {'normal', 'unobservable'}
        json.dumps(actual, allow_nan=False)
    assert actual['quality']['camera_stable'] is True


def test_model_source_clock_and_event_state_are_isolated_per_session(engine):
    image = frame()
    engine.infer_live(image, 'first', 1, 'yolo26n-pose', 'video')
    for time in [1, .5, float('nan'), -1]:
        with pytest.raises(ValueError):
            engine.infer_live(image, 'first', time, 'yolo26n-pose', 'video')
    with pytest.raises(ValueError, match='切换模型'):
        engine.infer_live(image, 'first', 2, 'yolo11n-pose', 'video')
    with pytest.raises(ValueError, match='切换模型'):
        engine.infer_live(image, 'first', 2, 'yolo26n-pose', 'camera')
    output = engine.infer_live(image, 'second', 0, 'yolo11n-pose', 'video')
    assert output['model']['id'] == 'yolo11n-pose'
    assert output['action']['diagnostics']['summary']['total_frames'] == 1
    one, two = engine.streams['first'][0], engine.streams['second'][0]
    assert one.fall is not two.fall and one.quality is not two.quality
    del engine.streams['first']
    assert engine.infer_live(image, 'first', 0)['source_time'] == 0


def test_invalid_model_does_not_create_session(engine):
    with pytest.raises(ValueError):
        engine.infer_live(frame(), 'bad', 0, 'untrusted-file-path')
    assert not engine.streams


def stable_quality():
    return SimpleNamespace(update=lambda image, persons, t=None: dict(camera_stable=True, reason='受控固定相机'))


def observations(persons):
    return dict(persons=persons, width=640, height=480)


def test_session_fall_torso_continues_after_knee_loss_without_changing_actions(engine):
    session = service.LiveSession(engine.action_model, 'yolo26n-pose', 'video')
    session.quality = stable_quality()
    original = LiveActions(engine.action_model)
    image = frame()
    established_epoch = None
    for i in range(20):
        p = person('standing' if i < 9 else 'low')
        if i >= 9:
            for joint in (13, 14, 15, 16):
                p['keypoints'][joint] = [0, 0, 0]
        raw = [p]
        before = copy.deepcopy(raw)
        expected = original.update(copy.deepcopy(raw), i / 10, 640, 480)
        output = observations(raw)
        session.update(image, output, i / 10)
        assert raw == before and output['action'] == expected
        if i == 8:
            established_epoch = output['fall']['target_epoch']
            assert output['fall']['metrics']['had_upright']
        if i >= 9:
            assert output['action']['target_index'] is None
            assert output['fall_observation']['source'] == 'raw'
            assert output['fall']['target_epoch'] == established_epoch
            assert output['fall']['observed']
    assert output['fall']['state'] == 'confirmed'


def test_session_current_image_recovery_reaches_only_fall_branch(engine):
    session = service.LiveSession(engine.action_model, 'yolo26n-pose', 'video')
    session.quality = stable_quality()
    original = LiveActions(engine.action_model)
    for i in range(20):
        raw = [person()] if i < 9 else []
        output = observations(raw)
        expected = original.update(copy.deepcopy(raw), i / 10, 640, 480)
        calls = []
        pixel = 40 + i
        image = np.full((480, 640, 3), pixel, np.uint8)

        def infer_current(view):
            # Every accepted pose must be inferred from this frame's pixels.
            assert np.all(view == pixel)
            turn = session.pose_recovery.config.rotations[len(calls)]
            calls.append(turn)
            crop_height, crop_width = view.shape[:2] if turn % 2 == 0 else view.shape[1::-1]
            return [forward(person('low'), crop_width, crop_height, turn)]

        session.update(image, output, i / 10, recover_infer=infer_current)
        assert output['persons'] == raw and output['action'] == expected
        if i < 9:
            assert calls == []
        else:
            assert raw == [] and output['action']['target_index'] is None
            recovered_pose = output['fall_observation']['person']
            assert recovered_pose['recovery_verified'] is True
            assert recovered_pose['keypoints'] == person('low')['keypoints']
            assert output['fall_observation']['source'] == 'recovered'
            assert output['fall_observation']['recovery']['accepted']
            assert len(set(calls)) >= 2 and output['fall']['observed']
    assert output['fall']['state'] == 'confirmed'
    assert output['fall']['candidate_duration_s'] == 1.


@pytest.mark.parametrize('target_case,expected_support', [('same', True), ('different', False), ('recovered', False)])
def test_session_action_falling_support_cannot_be_attributed_to_another_target(engine, target_case, expected_support):
    session = service.LiveSession(engine.action_model, 'yolo26n-pose', 'video')
    session.quality = stable_quality()
    real_actions_update = session.actions.update

    def classified_falling(*args):
        # Control only the model label. Real target selection and action history
        # still run, so this checks provenance rather than a copied condition.
        result = real_actions_update(*args)
        result['raw_label'] = 'falling'
        return result

    session.actions.update = classified_falling
    image = frame()
    for i in range(10):
        if target_case == 'different' and i < 6:
            session.quality = SimpleNamespace(update=lambda image, persons, t=None: dict(camera_stable=None, reason='参考未就绪'))
        else:
            session.quality = stable_quality()
        raw = [person()]
        if target_case == 'different' and i >= 6:
            raw[0]['confidence'] = .35  # Existing action identity remains observable but cannot be newly admitted.
            raw.append(moved(person(), 300))
        output = observations(raw)
        session.update(image, output, i / 10)
    if target_case == 'different':
        assert output['action']['target_index'] == 0
        assert output['fall_observation']['selection']['raw_index'] == 1
    elif target_case == 'recovered':
        calls = []

        def infer_current(view):
            turn = session.pose_recovery.config.rotations[len(calls)]
            calls.append(turn)
            crop_height, crop_width = view.shape[:2] if turn % 2 == 0 else view.shape[1::-1]
            return [forward(person('low'), crop_width, crop_height, turn)]

        output = observations([])
        session.update(image, output, 1., recover_infer=infer_current)
        assert output['fall_observation']['source'] == 'recovered'
    else:
        assert output['action']['target_index'] == output['fall_observation']['selection']['raw_index'] == 0
    assert output['action']['raw_label'] == 'falling'
    assert output['fall']['metrics']['action_falling_support'] is expected_support


def test_rgb_handoff_preserves_actions_and_rejects_missing_inputs_atomically(engine):
    from test_webcam_fall_handoff import UP, STEPS, LOW, d
    from test_webcam_rgb_fall import pose
    session = service.LiveSession(engine.action_model, 'yolo26n-pose', 'video', 'rgb')
    session.quality = stable_quality()
    original = LiveActions(engine.action_model)
    sequence = [(i * .2, UP, True, False) for i in range(9)]
    sequence += [(t, b, True, True) for t, b in zip((1.8, 2., 2.2), STEPS)]
    sequence += [(t, LOW, False, True) for t in (2.4, 2.6, 2.8, 3., 3.2)]
    for t, box, present, low in sequence:
        poses = [pose(box)] if present else []
        output = observations(copy.deepcopy(poses))
        raw = dict(persons=[d(box)] if present else [], fallen=[d(box)] if low else [])
        session.update(frame(), output, t, rgb_result=dict(detections=[], model={'id':'test'}, inference_ms=1, raw_candidates=raw))
        assert output['action'] == original.update(poses, t, 640, 480)
        assert output['persons'] == poses
    assert output['fall']['state'] == 'confirmed'
    assert output['pose_fall']['state'] == 'unobservable'
    before = session.last_t, copy.deepcopy(session.rgb_fall.export())
    with pytest.raises(ValueError):
        session.update(frame(), observations([]), 3.4, rgb_result=dict(detections=[], model={}, inference_ms=1))
    assert (session.last_t, session.rgb_fall.export()) == before

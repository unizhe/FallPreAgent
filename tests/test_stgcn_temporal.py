"""Temporal input integrity matters even when the closed-set model is uncertain."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from fallpre.weekly.stgcn_temporal import (
    CHECKPOINT_SHA256,
    PreparedWindow,
    STGCNRuntime,
    TemporalConfig,
    TemporalPoseBuffer,
    prepare_tensor,
    prepare_tensor_hold,
    prepare_tensor_linear,
)


def person(offset=0):
    points = np.column_stack((np.linspace(300, 500, 17),
                              np.linspace(100, 600, 17), np.full(17, .9)))
    points[:, 0] += offset
    return {'keypoints': points.tolist()}


def populated(config=None, epoch=1):
    buf = TemporalPoseBuffer(config or TemporalConfig(resampling='past_hold'))
    for t in np.arange(0, 2.501, .125):
        buf.observe(person(float(t)*10), float(t), 1280, 720, epoch)
    return buf


@pytest.mark.parametrize('values', [dict(window_s=float('inf')), dict(min_span_s=3.),
                                    dict(max_gap_s=0), dict(sample_count=30), dict(resampling='cyclic')])
def test_configuration_preserves_bounded_context_and_checkpoint_shape(values):
    with pytest.raises(ValueError):
        TemporalConfig(**values)


def test_causal_window_uses_elapsed_time_and_reports_real_count():
    buf = populated()
    window = buf.prepare()
    assert window.metadata['span_s'] == 2.5
    assert window.metadata['observations'] == 21
    assert window.source_times[-1] == buf.last_t
    assert np.all(np.diff(window.source_times) > 0)
    assert not window.keypoints.flags.writeable
    buf.observe(person(30), 2.625, 1280, 720, 1)
    assert window.source_times[-1] == 2.5  # Snapshot cannot acquire future input.
    assert buf.prepare() is None  # Cadence uses source time, not invocation speed.
    window2 = buf.prepare(force=True)
    assert window2.source_times[0] == .125
    assert window2.source_times[-1] == 2.625


def test_no_motion_wrap_when_low_fps_window_is_expanded_to_100_samples():
    window = populated().prepare()
    tensor, metadata = prepare_tensor(window)
    assert tensor.shape == (1, 1, 2, 100, 17, 3)
    # Input x coordinate monotonically moves right, no end-to-start cycle.
    x = tensor[0, 0, 0, :, 0, 0]
    assert np.all(np.diff(x) >= 0)
    assert metadata['unique_sampled_observations'] == 21
    assert metadata['repeated_samples'] == 79
    assert metadata['temporal_wrap'] is False
    assert np.count_nonzero(tensor[0, 0, 1]) == 0
    assert tensor[0, 0, 0, 0, 0, 0] == pytest.approx((300-640)/640)
    assert tensor[0, 0, 0, 0, 0, 1] == pytest.approx((100-360)/360)
    assert tensor[0, 0, 0, 0, 0, 2] == pytest.approx(.9)


def test_short_input_never_claims_ready_by_repeating_it():
    buf = TemporalPoseBuffer()
    for t in np.arange(0, .3, .01):
        buf.observe(person(), float(t), 1280, 720, 1)
    assert buf.status['observations'] == 30
    assert buf.status['status'] == 'warming_up'
    assert buf.prepare() is None


def test_sparse_input_needs_real_observations_and_continuity():
    buf = TemporalPoseBuffer()
    for t in (0, .4, .8, 1.2, 1.6):
        buf.observe(person(), t, 1280, 720, 1)
    assert buf.status['span_s'] == pytest.approx(1.6)
    assert buf.status['observations'] == 5
    assert buf.prepare() is None


def test_missing_current_target_suppresses_inference_without_borrowing():
    buf = populated()
    old_window = buf.prepare()
    old_result = dict(generation=old_window.generation, target_epoch=1,
                      window=old_window.metadata)
    assert buf.is_result_current(old_result)
    buf.observe(None, 2.625, 1280, 720, None)
    assert buf.status['status'] == 'target_unavailable'
    assert buf.prepare(force=True) is None
    assert not buf.is_result_current(old_result)
    buf.observe(person(), 2.75, 1280, 720, 1)
    new_window = buf.prepare(force=True)
    assert new_window.generation == old_window.generation
    assert 2.625 not in new_window.source_times
    assert new_window.metadata['max_observation_gap_s'] == .25


def test_track_change_and_long_gap_invalidate_previous_results():
    buf = populated()
    window = buf.prepare()
    old = dict(generation=window.generation, target_epoch=1, window=window.metadata)
    buf.observe(person(), 2.625, 1280, 720, 2)
    assert buf.generation > window.generation
    assert buf.status['observations'] == 1
    assert not buf.is_result_current(old)
    assert buf.prepare() is None
    generation = buf.generation
    buf.observe(person(), 3.2, 1280, 720, 2)
    assert buf.generation > generation
    assert buf.status['observations'] == 1


@pytest.mark.parametrize('change', ['camera', 'dimensions'])
def test_camera_changes_cannot_join_a_window(change):
    buf = populated()
    old_generation = buf.generation
    if change == 'camera':
        status = buf.observe(person(), 2.625, 1280, 720, 1, camera_stable=False)
        assert status['status'] == 'camera_unstable'
        assert status['observations'] == 0
    else:
        status = buf.observe(person(), 2.625, 1920, 1080, 1)
        assert status['observations'] == 1
    assert buf.generation > old_generation
    assert buf.prepare() is None


@pytest.mark.parametrize('bad', ['nan', 'shape', 'scores', 'hidden'])
def test_bad_skeleton_does_not_enter_or_drive_window(bad):
    buf = populated()
    p = person()
    points = np.asarray(p['keypoints'])
    if bad == 'nan':
        points[0, 0] = np.nan
    elif bad == 'shape':
        points = points[:16]
    elif bad == 'scores':
        points[0, 2] = 1.1
    else:
        points[5:, 2] = .1
    status = buf.observe({'keypoints': points}, 2.625, 1280, 720, 1)
    assert status['status'] == 'pose_unreliable'
    assert status['observations'] == 21
    assert buf.prepare(force=True) is None


def test_weak_joints_are_preserved_without_synthetic_skeleton():
    buf = TemporalPoseBuffer()
    p = person()
    p['keypoints'][0] = [0., 0., 0.]
    for t in np.arange(0, 1.501, .125):
        buf.observe(p, float(t), 1280, 720, 1)
    window = buf.prepare()
    tensor, _ = prepare_tensor(window)
    assert (window.keypoints[:, 0] == 0).all()
    # Official PreNormalize2D keeps the missing coordinate at (-1,-1),
    # accompanied by zero confidence; no invented nose or depth channel.
    np.testing.assert_array_equal(tensor[0, 0, 0, :, 0],
                                  np.tile([-1., -1., 0.], (100, 1)))


def test_backward_duplicate_and_nonfinite_time_are_rejected():
    for bad in (2.5, 1., float('nan'), float('inf'), -1):
        buf = populated()
        with pytest.raises(ValueError):
            buf.observe(person(), bad, 1280, 720, 1)
        assert buf.status['observations'] == 0
        assert buf.prepare(force=True) is None


def test_freshness_checks_reject_future_and_old_async_windows():
    buf = populated()
    window = buf.prepare()
    result = dict(generation=window.generation, target_epoch=window.target_epoch,
                  window=window.metadata)
    assert buf.is_result_current(result)
    assert not buf.is_result_current(result, t=2.49)
    assert not buf.is_result_current(result, t=3.51)
    assert not buf.is_result_current(dict(result, generation=99))


def linear_window(gap=.2, right_score=.4, right_x=340):
    points=np.asarray([person()['keypoints'],person()['keypoints']],dtype=np.float32)
    points[1,:,0]=right_x
    points[1,:,2]=right_score
    return PreparedWindow(points,np.array([0.,gap]),1280,720,1,1,
                          dict(start_s=0.,end_s=gap,observations=2,resampling='linear_supported'))


def test_linear_supported_is_declared_default_and_preserves_real_observation_count():
    buf=populated(TemporalConfig())
    window=buf.prepare()
    tensor,meta=prepare_tensor(window)
    assert window.metadata['resampling']=='linear_supported'
    assert meta['resampling']=='linear_supported'
    assert 'linear-supported100' in meta['preprocessing']
    assert meta['real_observations']==21
    assert meta['interpolated_sample_count']>0
    assert meta['synthetic_samples_are_not_observations'] is True
    assert 'repeated_samples' not in meta
    hold,hold_meta=prepare_tensor_hold(window)
    assert hold_meta['resampling']=='past_hold'
    assert not np.array_equal(tensor,hold)


def test_linear_trajectory_and_conservative_confidence_without_extrapolation():
    tensor,meta=prepare_tensor(linear_window())
    actual=tensor[0,0,0,:,0]
    np.testing.assert_allclose(actual[:,0]*640+640,np.linspace(300,340,100),atol=5e-5)
    assert actual[0,2]==pytest.approx(.9)
    np.testing.assert_allclose(actual[1:,2],.4)
    assert meta['real_observations']==2
    assert meta['interpolated_sample_count']==98
    assert meta['interpolated_joint_count']==98*17


@pytest.mark.parametrize('values',[dict(right_score=.1),dict(right_x=-10),dict(gap=.6)])
def test_linear_never_repairs_unreliable_or_out_of_frame_joints_or_bridges_long_gap(values):
    window=linear_window(**values)
    tensor,meta=prepare_tensor(window)
    hold,_=prepare_tensor_hold(window)
    np.testing.assert_array_equal(tensor,hold)
    assert meta['interpolated_sample_count']==0
    assert meta['real_observations']==2


def test_linear_missing_joint_is_held_while_other_supported_joints_interpolate():
    window=linear_window()
    window.keypoints[1,0,2]=0
    tensor,meta=prepare_tensor_linear(window)
    hold,_=prepare_tensor_hold(window)
    np.testing.assert_array_equal(tensor[0,0,0,:,0],hold[0,0,0,:,0])
    assert not np.array_equal(tensor[0,0,0,:,1],hold[0,0,0,:,1])
    assert meta['interpolated_joint_count']==98*16


def test_promoted_linear_matches_frozen_evaluated_candidate():
    prototype=Path(__file__).resolve().parents[1]/'runs/stgcn-integration-20261029/dense-probe/linear_probe.py'
    if not prototype.exists():
        pytest.skip('Development experiment source not distributed with minimal package')
    spec=importlib.util.spec_from_file_location('stgcn_evaluated_candidate',prototype)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for window in [linear_window(),linear_window(gap=.6),linear_window(right_score=.1),
                   populated(TemporalConfig()).prepare()]:
        production,_=prepare_tensor_linear(window)
        candidate,_=module.prepare_tensor_linear(window)
        np.testing.assert_array_equal(production,candidate)


@pytest.mark.skipif(os.environ.get('FALLPRE_TEST_STGCN_MODEL') != '1',
                    reason='Opt in to actual 5.9 MB pretrained model smoke test')
def test_actual_pretrained_model_matches_official_transform_and_is_deterministic():
    runtime = STGCNRuntime()
    window = populated().prepare()
    tensor, _ = prepare_tensor(window)
    from mmaction.datasets.transforms import FormatGCNInput, GenSkeFeat, PreNormalize2D
    times = window.source_times
    indices = np.searchsorted(times, np.linspace(times[0], times[-1], 100), side='right')-1
    chosen = window.keypoints[indices]
    raw = dict(keypoint=chosen[None, :, :, :2].copy(),
               keypoint_score=chosen[None, :, :, 2].copy(), img_shape=(720, 1280))
    official = FormatGCNInput(num_person=2)(GenSkeFeat(dataset='coco', feats=['j'])(PreNormalize2D()(raw)))
    np.testing.assert_array_equal(tensor[0], official['keypoint'])
    a, b = runtime.predict(window), runtime.predict(window)
    np.testing.assert_allclose(a['all_scores'], b['all_scores'], rtol=0, atol=0)
    assert sum(a['all_scores']) == pytest.approx(1, abs=1e-6)
    assert len(a['all_scores']) == 60
    assert runtime.metadata['weights_sha256'] == CHECKPOINT_SHA256
    assert runtime.metadata['parameters'] == 1393016
    assert a['relevant_scores']['falling'] == a['all_scores'][42]
    assert a['window_start_s'] == 0 and a['window_end_s'] == 2.5

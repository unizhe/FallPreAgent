"""Async results must not cross subjects, outages, stop or source-time expiry."""
from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace

from fallpre.weekly.webcam_temporal import LiveTemporalBranch


class ControlledExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, function, window):
        future = Future()
        self.jobs.append((future, window))
        return future

    def finish(self, index=-1, error=None):
        future, window = self.jobs[index]
        if error:
            future.set_exception(error)
        else:
            future.set_result(dict(status='ready', window={'end_s': window.source_times[-1]},
                                   window_end_s=window.source_times[-1],
                                   generation=window.generation, target_epoch=window.target_epoch,
                                   top3=[dict(label='falling', label_zh='跌倒动作', score=.99)]))


def setup():
    executor = ControlledExecutor()
    branch = LiveTemporalBranch(SimpleNamespace(executor=executor,
                               runtime=SimpleNamespace(metadata={'id': 'test'}, predict=lambda w: w)))
    points = [[200., 100.+j*10, .95] for j in range(17)]
    result = dict(persons=[dict(keypoints=points)], width=640, height=480,
                  action=dict(target_index=0, selection={'target_epoch': 1}, label='standing'),
                  fall=dict(state='normal', new_event=False), quality={'camera_stable': True})
    for i in range(17):
        output = branch.update(result, i*.1)
    assert output['status'] == 'pending'
    return branch, executor, result


def test_temporal_score_never_overwrites_primary_or_alarm_and_queue_is_bounded():
    branch, executor, result = setup()
    original = deepcopy(result)
    for i in range(17, 24):
        branch.update(result, i*.1)
    assert len(executor.jobs) == 1
    executor.finish()
    output = branch.update(result, 2.4)
    assert output['status'] == 'ready' and output['top3'][0]['score'] == .99
    assert output['changes_primary_decision'] is False
    assert result == original


def test_completed_result_is_rejected_after_subject_switch():
    branch, executor, result = setup()
    executor.finish()
    result['action']['selection']['target_epoch'] = 2
    output = branch.update(result, 1.7)
    assert output['status'] == 'warming_up' and not output['top3']


def test_current_missing_pose_suppresses_recent_prediction():
    branch, executor, result = setup()
    executor.finish()
    assert branch.update(result, 1.7)['status'] == 'ready'
    result['persons'] = []
    output = branch.update(result, 1.8)
    assert output['status'] == 'unobservable' and not output['top3']


def test_too_slow_result_is_not_published_as_current():
    branch, executor, result = setup()
    for i in range(17, 29):
        branch.update(result, i*.1)
    executor.finish()
    output = branch.update(result, 2.9)
    assert output['status'] == 'pending' and not output['top3']


def test_worker_error_does_not_interrupt_base_result():
    branch, executor, result = setup()
    original = deepcopy(result)
    executor.finish(error=RuntimeError('test'))
    output = branch.update(result, 1.7)
    assert output['status'] == 'error' and result == original


def test_stop_cancels_waiting_work_and_clears_result():
    branch, executor, result = setup()
    branch.close()
    assert executor.jobs[0][0].cancelled()
    output = branch.update(result, 1.7)
    assert output['status'] == 'unobservable' and not output['top3']

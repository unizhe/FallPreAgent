"""Bounded asynchronous temporal evidence alongside the established live rules.

ST-GCN++ describes its past window, never overwrites action/fall decisions, and
never turns a closed-set class score into an alarm probability.
"""
from concurrent.futures import ThreadPoolExecutor

from .stgcn_temporal import TemporalPoseBuffer


class TemporalService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stgcn')

    def session(self):
        return LiveTemporalBranch(self)

    def status(self):
        return dict(enabled=True, ready=True, mode='shadow', model=dict(self.runtime.metadata),
                    integration='parallel evidence; existing action and fall decisions retained',
                    validation_report_url='http://127.0.0.1:8805/')

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


class LiveTemporalBranch:
    """At most one outstanding task per stream; no source-time lookahead."""

    def __init__(self, service):
        self.service = service
        self.buffer = TemporalPoseBuffer()
        self.future = None
        self.last_result = None
        self.error = None
        self.closed = False

    def close(self):
        self.closed = True
        if self.future is not None:
            self.future.cancel()
        self.buffer.reset('会话结束')
        self.last_result = None

    def update(self, result, t):
        if self.closed:
            return dict(status='unobservable', reason='会话已结束', mode='shadow', top3=[])
        action = result.get('action') or {}
        index = action.get('target_index')
        epoch = (action.get('selection') or {}).get('target_epoch')
        persons = result.get('persons') or []
        person = persons[index] if isinstance(index, int) and 0 <= index < len(persons) else None
        status = self.buffer.observe(person, t, result['width'], result['height'], epoch,
                                     camera_stable=result.get('quality', {}).get('camera_stable') is True)
        if self.future is not None and self.future.done():
            try:
                predicted = self.future.result()
                if self.buffer.is_result_current(predicted):
                    self.last_result = predicted
                self.error = None
            except Exception as exc:
                # Only the optional branch degrades; no pixels or frame content
                # are logged and the established inference path stays available.
                self.error = type(exc).__name__
            self.future = None
        base = dict(status, mode='shadow', schema_version=1,
                    model=dict(self.service.runtime.metadata), top3=[],
                    observed_frames=status['observations'],
                    current_source_time_s=t, changes_primary_decision=False)
        if status['status'] != 'ready':
            self.last_result = None
            base['status'] = 'warming_up' if status['status'] == 'warming_up' else 'unobservable'
            return base
        if self.future is None:
            window = self.buffer.prepare()
            if window is not None:
                self.future = self.service.executor.submit(self.service.runtime.predict, window)
        if self.last_result and self.buffer.is_result_current(self.last_result):
            output = dict(base, **self.last_result)
            output.update(reason='最近历史片段的模型候选；未改写当前动作或跌倒判断',
                          result_age_s=round(t - self.last_result['window_end_s'], 6),
                          inference_pending=self.future is not None)
            return output
        if self.error:
            return dict(base, status='error', reason=f'时序分支暂不可用（{self.error}）；基础识别继续运行')
        if self.last_result:
            self.last_result = None
            return dict(base, status='stale', reason='上一片段结果已过期，等待最新连续证据')
        return dict(base, status='pending', reason='正在分析最近的连续动作片段')

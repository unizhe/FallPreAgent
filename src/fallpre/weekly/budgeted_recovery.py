"""Optional soft compute budget around the unchanged pose-recovery pipeline.

An in-flight inference is never interrupted. Its real duration remains visible,
and a result that crosses the budget is discarded together with the incomplete
recovery attempt. The original current-frame observations are returned intact.
"""
import math
import time

from .pose_recovery import RTMOBackend
from .pose_refinement import RTMPoseBackend, RefinedPoseRecovery


class _RecoveryBudgetExhausted(Exception):
    """Internal control flow; never suppress unrelated backend exceptions."""


class BudgetedRefinedPoseRecovery:
    """Same recovery gates, models and view order, with an optional time budget.

    ``budget_s=None`` executes the original algorithm with timing instrumentation.
    A finite positive budget is soft: it is checked between synchronous calls,
    after each call, and after final postprocessing. No hard deadline is promised.
    Instances are intended for a single sequential worker, like the wrapped
    recovery implementation. ``clock`` supports deterministic no-inference tests.
    """

    def __init__(self, backend=None, config=None, refiner=None, *, budget_s=None,
                 clock=None, allow_partial_observation=True,
                 allow_detector_proposals=False, resolve_native_ambiguity=False):
        if budget_s is not None:
            if isinstance(budget_s, bool) or not math.isfinite(budget_s) or budget_s <= 0:
                raise ValueError('Recovery budget must be finite and positive, or None')
            budget_s = float(budget_s)
        self.budget_s = budget_s
        self.clock = clock or time.perf_counter
        self._backend = backend
        self._refiner = refiner
        self._active = None
        self.recovery = RefinedPoseRecovery(
            backend=self._rtmo, config=config, refiner=self._rtmpose,
            allow_partial_observation=allow_partial_observation,
            allow_detector_proposals=allow_detector_proposals,
            resolve_native_ambiguity=resolve_native_ambiguity,
        )

    def configuration(self):
        return dict(self.recovery.configuration(), recovery_budget=dict(
            seconds=self.budget_s, clock='monotonic_perf_counter',
            enforcement='soft_non_preemptive_before_and_after_each_call_and_after_augment',
            on_exhaustion='discard_incomplete_recovery_return_original_observation',
            unchanged_view_order=True,
        ))

    def _check(self, elapsed, stage, call=None):
        if self.budget_s is not None and elapsed >= self.budget_s:
            self._active.update(exhausted=True, exhaustion_stage=stage,
                                exhaustion_elapsed_s=elapsed, blocked_call=call)
            raise _RecoveryBudgetExhausted

    def _call(self, kind, invoke, **metadata):
        state = self._active
        if state is None:
            raise RuntimeError('Recovery backends must be called through augment')
        index = sum(c['kind'] == kind for c in state['calls']) + 1
        call = dict(kind=kind, index=index, **metadata)
        start = self.clock() - state['origin']
        self._check(start, 'before_call', call)
        call['start_s'] = start
        state['calls'].append(call)
        try:
            result = invoke()
        finally:
            end = self.clock() - state['origin']
            call.update(end_s=end, duration_s=end-start)
        self._check(end, 'after_call', dict(kind=kind, index=index, **metadata))
        return result

    def _rtmo(self, image):
        # PoseRecovery's frozen ordering: full_0, full_1..3, crop_0..3.
        index = sum(c['kind'] == 'rtmo' for c in self._active['calls'])
        views = ['full_0', 'full_1', 'full_2', 'full_3',
                 'crop_0', 'crop_1', 'crop_2', 'crop_3']
        if index >= len(views):
            raise RuntimeError('Unexpected RTMO call order; update the audit adapter')

        def invoke():
            if self._backend is None:
                self._backend = RTMOBackend()
            return self._backend(image)

        return self._call('rtmo', invoke, view=views[index],
                          rotation=index if index < 4 else index-4,
                          expansion=None)

    def _rtmpose(self, image, box, turn=0, expansion=1.0):
        def invoke():
            if self._refiner is None:
                self._refiner = RTMPoseBackend()
            return self._refiner(image, box, turn, expansion)

        return self._call('rtmpose', invoke, rotation=turn, expansion=expansion)

    def augment(self, image, row, tracker):
        if self._active is not None:
            raise RuntimeError('Concurrent or nested augment calls are not supported')
        state = dict(origin=self.clock(), calls=[], exhausted=False,
                     exhaustion_stage=None, exhaustion_elapsed_s=None, blocked_call=None)
        self._active = state
        try:
            try:
                augmented, audit = self.recovery.augment(image, row, tracker)
                # Include postprocessing time, while leaving no-request gates alone.
                if state['calls']:
                    self._check(self.clock()-state['origin'], 'after_augment')
            except _RecoveryBudgetExhausted:
                # Recovery reads the tracker but never updates it. Reconstruct only
                # its causal request context; do not claim unreturned candidates.
                context = tracker.recovery_context(
                    row['t'], row['instances'],
                    allow_ambiguous=self.recovery.resolve_native_ambiguity,
                )
                rtmo_count = sum(c['kind'] == 'rtmo' for c in state['calls'])
                refiner_count = sum(c['kind'] == 'rtmpose' for c in state['calls'])
                augmented = row
                audit = dict(context, trigger=context['reason'],
                             executed=bool(state['calls']), accepted=False,
                             reason='recovery_budget_exhausted', candidates=[],
                             inference_passes=rtmo_count,
                             refinement=dict(executed=bool(refiner_count), accepted=False,
                                             inference_passes=refiner_count),
                             incomplete_recovery_discarded=True)
            elapsed = self.clock()-state['origin']
            audit = dict(audit, budget_audit=dict(
                budget_s=self.budget_s, enforcement='soft_non_preemptive',
                elapsed_s=elapsed, exhausted=state['exhausted'],
                exhaustion_stage=state['exhaustion_stage'],
                exhaustion_elapsed_s=state['exhaustion_elapsed_s'],
                overrun_s=max(0., elapsed-self.budget_s) if self.budget_s is not None else 0.,
                blocked_call=state['blocked_call'], calls=state['calls'],
                rtmo_calls=sum(c['kind'] == 'rtmo' for c in state['calls']),
                rtmpose_calls=sum(c['kind'] == 'rtmpose' for c in state['calls']),
                total_calls=len(state['calls']),
                max_call_s=max((c['duration_s'] for c in state['calls']), default=0.),
                discarded_incomplete_recovery=state['exhausted'],
            ))
            return augmented, audit
        finally:
            self._active = None

"""Opt-in, within-attempt call admission for the existing soft recovery budget.

The first call of each backend kind is always admitted if time remains. Later
calls use the longest completed call of that kind in this attempt as a cost
estimate. This is not a hard deadline: the first call, or any later call whose
cost exceeds the estimate, can still overrun the budget.
"""
from .budgeted_recovery import BudgetedRefinedPoseRecovery, _RecoveryBudgetExhausted


class CostAwareRefinedPoseRecovery(BudgetedRefinedPoseRecovery):
    """Keep recovery unchanged, but avoid calls estimated not to fit the budget.

    Constructor arguments and defaults match ``BudgetedRefinedPoseRecovery``.
    ``budget_s=None`` disables both budgeting and cost-based admission. With a
    finite budget, estimates use only completed calls in the current augment;
    no timing information carries over between frames or backend kinds.
    """

    def configuration(self):
        config = super().configuration()
        config['recovery_budget']['call_admission'] = dict(
            estimator='max_completed_duration_same_backend_kind_current_attempt',
            reject_when='elapsed_plus_estimate_greater_than_or_equal_to_budget',
            no_sample='admit_if_elapsed_budget_remains',
            history_scope='current_attempt_only',
            hard_deadline=False,
        )
        return config

    def _check(self, elapsed, stage, call=None):
        # Actual elapsed-time exhaustion takes precedence over an estimate.
        super()._check(elapsed, stage, call)
        if self.budget_s is None or stage != 'before_call':
            return
        completed = [c['duration_s'] for c in self._active['calls']
                     if c['kind'] == call['kind'] and 'duration_s' in c]
        if not completed:
            return
        estimate = max(completed)
        if elapsed + estimate >= self.budget_s:
            blocked = dict(call, estimated_duration_s=estimate,
                           remaining_budget_s=self.budget_s-elapsed)
            self._active.update(exhausted=True,
                                exhaustion_stage='predicted_call_overrun',
                                exhaustion_elapsed_s=elapsed, blocked_call=blocked)
            raise _RecoveryBudgetExhausted

    def augment(self, image, row, tracker):
        augmented, audit = super().augment(image, row, tracker)
        budget = audit['budget_audit']
        predicted = budget['exhaustion_stage'] == 'predicted_call_overrun'
        blocked = budget['blocked_call'] or {}
        budget.update(
            exhaustion_cause=('predicted_call_overrun' if predicted else
                              'elapsed_budget_exhausted' if budget['exhausted'] else None),
            estimated_next_call_s=blocked.get('estimated_duration_s'),
            remaining_budget_s=blocked.get('remaining_budget_s'),
        )
        return augmented, audit

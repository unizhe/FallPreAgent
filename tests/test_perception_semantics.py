from copy import deepcopy

import pytest
from test_perception_chain import source

from fallpre.weekly.perception_semantics import decision_context


def observation(code='classification_geometry_conflict', state='unobservable'):
    result = source(3)
    result['fall'] = dict(state=state, reason_code=code, observed=True, target_epoch=7,
                          event_id='historical-1', new_event=False)
    result['fall_observation'] = dict(person=result['persons'][0],
                                      selection=dict(state='locked', target_epoch=7))
    return result


def test_visible_conflict_is_unresolved_not_unobservable_or_normal():
    result = observation()
    original = deepcopy(result)
    context = decision_context(result, 3)
    assert context['pose_observation_status'] == 'observable'
    assert context['event_identity_status'] == 'current'
    assert context['fall_evidence_status'] == 'unresolved'
    assert context['event_state'] == 'unobservable'
    assert not context['changes_alarm']
    assert not context['establishes_safety']
    assert result == original


@pytest.mark.parametrize('condition', ['missing', 'camera', 'weak_pose', 'unlocked'])
def test_current_quality_required_independently_of_conflict_code(condition):
    result = observation()
    if condition == 'missing':
        result['persons'] = []
    elif condition == 'camera':
        result['quality']['camera_stable'] = False
    elif condition == 'weak_pose':
        for point in result['persons'][0]['keypoints']:
            point[2] = 0
    else:
        result['action']['selection']['state'] = 'searching'
    context = decision_context(result, 3)
    assert context['pose_observation_status'] == 'unavailable'
    assert context['fall_evidence_status'] == 'unresolved'
    assert not context['establishes_recovery']


def test_visible_action_person_does_not_resolve_multiple_event_candidates():
    context = decision_context(observation('multiple_person_candidates'), 3)
    assert context['pose_observation_status'] == 'observable'
    assert context['event_identity_status'] == 'ambiguous'
    assert context['fall_evidence_status'] == 'unavailable'


def test_wrong_event_epoch_cannot_become_current_via_action_person():
    result = observation()
    result['fall_observation']['selection']['target_epoch'] = 9
    assert decision_context(result, 3)['event_identity_status'] == 'not_established'


def test_old_confirmed_event_cannot_claim_current_identity():
    result = observation('verified_handoff_fall', 'confirmed')
    result['fall']['event_target_epoch'] = 99
    context = decision_context(result, 3)
    assert context['event_identity_status'] == 'not_established'
    assert result['fall']['event_target_epoch'] == 99


@pytest.mark.parametrize(('code', 'status'), [
    ('low_support_ready', 'low_posture_reference'),
    ('recovery_pending', 'recovery_pending'),
    ('recovery_transition', 'recovery_pending'),
    ('weak_fallen_support', 'unresolved'),
])
def test_pending_evidence_never_establishes_recovery(code, status):
    result = observation(code)
    context = decision_context(result, 3)
    assert context['fall_evidence_status'] == status
    assert not context['establishes_recovery']
    assert result['fall']['event_id'] == 'historical-1'


def test_experimental_score_has_no_presentation_authority():
    result = observation('no_supported_fall', 'normal')
    context = decision_context(result, 3)
    assert context['fall_evidence_status'] == 'not_supported'
    result['temporal']['fall_heads']['st_geometry']['fall_process_score'] = 0
    assert decision_context(result, 3) == context


def test_confirmed_event_is_preserved_when_pose_is_now_missing():
    result = observation('verified_handoff_fall', 'confirmed')
    result['persons'] = []
    context = decision_context(result, 3)
    assert context['pose_observation_status'] == 'unavailable'
    assert context['fall_evidence_status'] == 'confirmed'
    assert not context['establishes_recovery']

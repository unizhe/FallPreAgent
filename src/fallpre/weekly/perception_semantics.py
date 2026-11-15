"""Read-only presentation of current pose quality and unresolved fall evidence.

This does not reinterpret the legacy state machine or supply alarm/recovery
evidence. A visible ACTION skeleton does not establish the RGB event identity.
"""
from .stgcn_temporal import TemporalPoseBuffer

_AMBIGUOUS_CODES = {
    'multiple_person_candidates', 'expanded_person_ambiguity', 'ambiguous_person_candidate',
    'ambiguous_fallen_candidates', 'overlapping_fallen_competitor', 'multiple_strong_candidates',
    'nearby_candidates_compete',
}


def decision_context(result, t, *, fall=None, observation=None):
    """Describe this frame only, without modifying any input or rule state.

    A fresh buffer reuses the production single-frame quality checks. Its history
    readiness is deliberately ignored: visible now and sufficient history are
    different questions. Event identity comes from the event branch, not this
    action skeleton. Keep legacy ``fall.state``/``event_decidable`` for consumers.
    """
    fall = result.get('fall', {}) if fall is None else fall
    observation = result.get('fall_observation', {}) if observation is None else observation
    action = result.get('action') or {}
    selection = action.get('selection') or {}
    index = action.get('target_index')
    people = result.get('persons') or []
    person = people[index] if type(index) is int and 0 <= index < len(people) else None
    pose_visible = False
    buffer = TemporalPoseBuffer()
    try:
        buffer.observe(person, t, result.get('width'), result.get('height'),
                       selection.get('target_epoch'),
                       (result.get('quality') or {}).get('camera_stable') is True)
        pose_visible = (buffer.current_valid and selection.get('state') == 'locked'
                        and selection.get('target_index', index) == index)
    except (TypeError, ValueError, KeyError, IndexError):
        pass
    code = fall.get('reason_code', 'unspecified')
    event_selection = observation.get('selection') or {}
    if code in _AMBIGUOUS_CODES or event_selection.get('state') == 'ambiguous':
        identity = 'ambiguous'
    elif (observation.get('person') is not None and fall.get('observed') is True
          and event_selection.get('state') == 'locked'
          and fall.get('target_epoch') is not None
          and fall.get('target_epoch') == event_selection.get('target_epoch')
          and (fall.get('state') != 'confirmed'
               or fall.get('event_target_epoch') == fall.get('target_epoch'))):
        identity = 'current'
    else:
        identity = 'not_established'
    state = fall.get('state', 'unobservable')
    if state == 'confirmed':
        evidence, text = 'confirmed', '跌倒事件已确认'
    elif state == 'suspected':
        evidence, text = 'suspected', '跌倒证据正在累计'
    elif code in {'recovery_pending', 'recovery_transition'}:
        evidence, text = 'recovery_pending', '当前事件证据待核对，恢复尚未确认'
    elif code == 'weak_fallen_support':
        evidence, text = 'unresolved', '历史事件保留，当前倒地证据不足；恢复未确认'
    elif code == 'low_support_ready':
        evidence, text = 'low_posture_reference', '已建立低位参考，继续观察'
    elif code == 'classification_geometry_conflict':
        evidence, text = 'unresolved', '跌倒证据待确认'
    elif state == 'normal':
        evidence, text = 'not_supported', '当前未见足够跌倒证据'
    else:
        evidence, text = 'unavailable', '跌倒证据暂不可用'
    if state == 'unobservable':
        if identity == 'ambiguous':
            text = '目标身份有歧义，跌倒证据待核对'
        elif not pose_visible:
            text = '当前骨架观测不足；' + text
        elif evidence == 'unavailable':
            text = '当前骨架可见，跌倒判断所需证据不足'
        else:
            text = '当前骨架可见，' + text
    return dict(schema_version=1, source_time_s=t,
                pose_observation_status='observable' if pose_visible else 'unavailable',
                pose_observation_source='current_action_target',
                event_identity_status=identity, fall_evidence_status=evidence,
                event_state=state, reason_code=code, text=text,
                changes_alarm=False, establishes_safety=False, establishes_recovery=False)

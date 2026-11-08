"""The candidate API is opt-in and cannot change established action or alarm."""
import copy

from test_webcam_service import engine, frame  # noqa: F401


def test_candidate_is_opt_in_and_preserves_primary_outputs(engine):  # noqa: F811
    image = frame()
    engine.fusion_enabled = False
    a = []
    for i in range(25):
        a.append(copy.deepcopy(engine.infer_live(image, 'baseline', i / 10)))
    assert all('fused_action' not in r for r in a)
    engine.fusion_enabled = True
    for i, original in enumerate(a):
        b = engine.infer_live(image, 'candidate', i / 10)
        assert b['fused_action']['changes_primary_decision'] is False
        assert b['fused_action']['changes_alarm'] is False
        assert b['action'] == original['action']
        assert b['fall'] == original['fall']
    assert engine.streams['baseline'][0].fusion is None
    assert engine.streams['candidate'][0].fusion is not None
    assert engine.fusion_status()['mode'] == 'shadow'

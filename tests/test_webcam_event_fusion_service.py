from types import SimpleNamespace

from test_webcam_service import engine, frame  # noqa: F401


def test_event_fusion_service_is_opt_in_and_exports_baseline_and_tool_contract(engine):  # noqa: F811
    engine.default_fall_mode = 'rgb'
    engine.rgb_backend = SimpleNamespace(infer=lambda image: dict(
        detections=[], raw_candidates=dict(persons=[], fallen=[]), model=dict(id='test'), inference_ms=0.))
    original = engine.infer_live(frame(), 'original', 0.)
    assert 'baseline_fall' not in original and 'tool_event' not in original
    engine.fall_event_fusion_enabled = True
    engine.fall_temporal_enabled = True
    result = engine.infer_live(frame(), 'fused', 0.)
    assert result['fall'] == result['fused_fall']['fall']
    assert result['baseline_fall'] == original['fall']
    assert result['tool_event']['state'] == 'unobservable'
    assert not result['tool_event']['new_event']
    assert engine.fall_event_fusion_status()['mode'] == 'limited_primary'
    assert engine.streams['original'][0].fall_event_fusion is None
    assert engine.streams['fused'][0].fall_event_fusion is not None

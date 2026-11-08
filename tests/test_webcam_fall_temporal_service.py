"""Fall-centric evidence is opt-in, isolated per stream and cannot mutate alarms."""
from test_webcam_service import engine, frame  # noqa: F401


def test_fall_evidence_does_not_change_any_primary_field(engine):  # noqa: F811
    image = frame()
    original = [engine.infer_live(image, 'primary', i/10) for i in range(30)]
    assert all('fall_temporal_evidence' not in r for r in original)
    engine.fall_temporal_enabled = True
    for i, ref in enumerate(original):
        actual = engine.infer_live(image, 'fall-evidence', i/10)
        evidence = actual['fall_temporal_evidence']
        assert not evidence['changes_alarm'] and not evidence['changes_event']
        for key in ['action','fall','quality','behavior','risk_points','fall_observation']:
            assert actual[key] == ref[key]
    assert engine.streams['primary'][0].fall_temporal is None
    assert engine.streams['fall-evidence'][0].fall_temporal is not None
    assert engine.fall_temporal_status()['mode'] == 'shadow'

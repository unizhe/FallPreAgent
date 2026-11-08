"""Unknown partial people must remain protected during camera-check repair."""
import pytest

from fallpre.weekly.webcam_fall_handoff import LiveFallHandoff


@pytest.mark.parametrize('box', [
    [590, 350, 640, 480], [0, 350, 50, 480],
    [590, 0, 640, 130], [0, 0, 50, 130], [450, 60, 550, 430],
])
def test_static_partial_second_person_without_pose_is_not_discarded(box):
    tracker = LiveFallHandoff()
    raw = dict(persons=[dict(box=[150, 50, 300, 430], confidence=.95),
                        dict(box=box, confidence=.4)], fallen=[])
    for index in range(30):
        result = tracker.update(raw, index * .1, 640, 480, camera_stable=True)
        assert result['fall']['state'] == 'unobservable'
        assert result['fall']['reason_code'] == 'multiple_person_candidates'
        assert result['observation']['person'] is None
    assert not tracker.confirmed_events

"""Appearance evidence must not become a position or missing-pose exemption."""
import cv2
import numpy as np
import pytest

from fallpre.weekly.nonhuman_appearance import NonhumanAppearance


def reference(seed=31):
    image = np.random.default_rng(seed).integers(0, 255, (224, 128, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


@pytest.mark.parametrize('position', [(0, 0), (200, 0), (0, 240), (200, 240)])
def test_matching_is_position_independent(position):
    ref = reference()
    verifier = NonhumanAppearance(ref)
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    x, y = position
    image[y:y+224, x:x+128] = ref
    box = [x, y, x+128, y+224]
    removed, audit = verifier.verify(image, [dict(box=box, confidence=.9)], [])
    assert removed == [0] and audit[0]['verified']


def test_pose_veto_overrides_even_exact_object_appearance():
    ref = reference()
    verifier = NonhumanAppearance(ref)
    box = [0, 0, 128, 224]
    # Partial person: two visible joints are enough to retain protection.
    points = [[40., 30., .3], [60., 40., .3]] + [[0., 0., 0.]] * 15
    removed, audit = verifier.verify(ref, [dict(box=box, confidence=.4)],
                                      [dict(box=box, keypoints=points, confidence=.25)])
    assert removed == [] and audit[0]['reason'] == 'current_pose_protected'


def test_unknown_appearance_and_occlusion_do_not_get_exemption():
    ref = reference()
    verifier = NonhumanAppearance(ref)
    assert not verifier.compare(reference(99))['verified']
    assert not verifier.compare(np.zeros_like(ref))['verified']
    changed = ref.copy()
    changed[:112] = 0
    assert not verifier.compare(changed)['verified']
    assert not verifier.compare(ref[:4])['verified']


def test_untextured_reference_is_rejected():
    with pytest.raises(ValueError):
        NonhumanAppearance(np.zeros((224, 128, 3), dtype=np.uint8))

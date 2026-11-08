import hashlib

import pytest
from fallpre.weekly import webcam_rgb_model as rgb


def test_unmatched_furniture_never_supplies_person_geometry_or_confidence():
    people = [dict(box=[100, 20, 200, 400], confidence=.4)]
    chairs = [dict(box=[400, 200, 600, 400], confidence=.99)]
    out = rgb.match_current_persons(people, chairs)
    assert out[0]['label'] == 'person' and out[0]['confidence'] == .4
    assert out[0]['box'] == people[0]['box'] and out[0]['fall_index'] is None


def test_current_overlap_labels_fallen_without_boosting_person_score():
    out = rgb.match_current_persons([dict(box=[100, 200, 400, 450], confidence=.4)],
                                    [dict(box=[90, 190, 410, 450], confidence=.8)])
    assert out[0]['label'] == 'fallen' and out[0]['confidence'] == .4
    assert out[0]['fall_confidence'] == .8


def test_local_weight_requires_exact_checksum_and_models_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(rgb, 'ROOT', tmp_path)
    (tmp_path / 'models').mkdir()
    weight = tmp_path / 'models/test.pt'; weight.write_bytes(b'test-weight')
    spec = dict(path='models/test.pt', sha256=hashlib.sha256(b'test-weight').hexdigest())
    assert rgb.verified_path(spec) == weight
    with pytest.raises(ValueError, match='校验失败'): rgb.verified_path({**spec, 'sha256':'incorrect'})
    outside = tmp_path / 'other.pt'; outside.write_bytes(b'test-weight')
    with pytest.raises(ValueError): rgb.verified_path({**spec, 'path':'other.pt'})

import pytest
from test_stgcn_fall_evidence import result, temporal, warmed


def replay(score, mismatch=None):
    branch = warmed()
    for step in range(1, 8):
        t = 1.9 + step / 10
        # NTU60 votes sitting; a process head remains a distinct model source.
        r, prediction = result(step), temporal(t, index=7)
        head = dict(status='ready', fall_process_score=score, window_end_s=t,
                    target_epoch=1, generation=0, weights_sha256='test-frozen-head')
        if mismatch:
            head[mismatch] = {'window_end_s': t + .1, 'target_epoch': 99,
                              'generation': 99, 'fall_process_score': float('nan')}[mismatch]
        output = branch.update(r, t, prediction, process_model=head)
    return output


def test_binary_head_is_distinct_from_ntu60_class_and_still_requires_geometry():
    output = replay(.95)
    assert output['support']
    assert output['model_evidence']['top_class_index'] == 7
    assert output['model_evidence']['source'] == 'binary_process_head'
    assert output['model_evidence']['falling_score'] == .95
    assert output['geometry']['descending'] and output['geometry']['low_pose']
    assert not output['changes_alarm']


def test_negative_head_abstains_instead_of_cancelling_event():
    output = replay(.05)
    assert not output['support']
    assert output['code'] == 'model_fall_support_insufficient'
    assert not output['changes_event']


@pytest.mark.parametrize('mismatch', ['window_end_s', 'target_epoch', 'generation', 'fall_process_score'])
def test_mismatched_head_cannot_supply_support(mismatch):
    output = replay(.99, mismatch)
    assert not output['support']
    assert output['code'] == 'process_head_unavailable_or_mismatched'

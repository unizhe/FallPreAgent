import copy
import json

import pytest

from fallpre.weekly.webcam_target import StableSubject, candidate
from fallpre.weekly.webcam_actions import LiveActions, SinglePersonActions
from fallpre.weekly.action_v7 import CompletionModel
from fallpre.weekly.human import ROOT
from test_action_v2 import person


def detection(x=100, score=.9):
    p = person(x=x, score=score)
    return dict(box=p['box'], confidence=p['det'], keypoints=p['kpts'])


def chair():
    # Bigger than the person and a high detector score, but no leg evidence.
    p = detection(450, .99)
    p['box'] = [280, 20, 630, 470]
    for j in [11,12,13,14,15,16]:
        p['keypoints'][j][2] = .1
    return p


def lock(s, items=None):
    for i in range(12):
        out = s.update(items or [detection(), chair()], i/30, 640, 480)
    assert out['state']=='locked'
    return out


def test_large_high_score_background_never_wins_and_raw_input_is_unchanged():
    s=StableSubject();items=[chair(),detection()];saved=copy.deepcopy(items)
    assert lock(s,items)['target_index']==1
    assert items==saved


def test_raw_order_and_confidence_changes_do_not_switch_identity():
    s=StableSubject();epoch=lock(s)['target_epoch']
    for i in range(12,30):
        p=detection(100+(i-12), .3)  # continuation can use lower detector scores
        items=[chair(),p] if i%2 else [p,chair()]
        r=s.update(items,i/30,640,480)
        assert r['target_index']==(1 if i%2 else 0) and r['target_epoch']==epoch


def test_two_plausible_people_at_start_do_not_select_by_size_or_score():
    s=StableSubject()
    for i in range(30):
        assert s.update([detection(),detection(400,.99)],i/30,640,480)['target_index'] is None


def test_distinct_new_person_does_not_displace_existing_target():
    s=StableSubject();lock(s)
    r=s.update([detection(400,.99),detection(101,.6)],.4,640,480)
    assert r['target_index']==1


def test_overlapping_competitors_abstain_instead_of_switching():
    s=StableSubject();lock(s)
    r=s.update([detection(99),detection(101)],.4,640,480)
    assert r['state']=='ambiguous' and r['target_index'] is None


def test_missing_target_cannot_borrow_a_distant_person_and_expires():
    s=StableSubject();epoch=lock(s)['target_epoch']
    for i in range(12,26):
        assert s.update([detection(400)],i/30,640,480)['target_index'] is None
    results=[s.update([detection(400)],i/30,640,480) for i in range(26,43)]
    assert results[-1]['target_epoch']>epoch and results[-1]['target_index']==0


def test_gap_requires_reconfirmation():
    s=StableSubject();lock(s)
    assert s.update([detection()],2,640,480)['target_index'] is None


def test_single_bad_frame_never_creates_a_target():
    s=StableSubject()
    s.update([detection()],0,640,480)
    for i in range(1,10):assert s.update([chair()],i/30,640,480)['target_index'] is None


def test_partial_knee_can_continue_without_imputation():
    s=StableSubject();lock(s)
    p=detection()
    for j in [13,15,16]:p['keypoints'][j][2]=0
    original=copy.deepcopy(p)
    assert s.update([chair(),p],.4,640,480)['target_index']==1
    assert original==p


def test_out_of_frame_or_zero_length_body_cannot_initialize():
    for mode in ['outside','collapsed']:
        p=detection()
        for j in [5,6,11,12,13,14]:
            p['keypoints'][j][:2]=[100,800] if mode=='outside' else [100,100]
        assert candidate(p,640,480) is None


def test_lying_orientation_is_not_rejected_by_vertical_order():
    p=detection()
    for k in p['keypoints']:k[0],k[1]=k[1],k[0]
    p['box']=[75,70,230,130]
    assert candidate(p,640,480)['strong']


def test_parallel_before_after_uses_same_raw_detections_and_preserves_current_index():
    model=CompletionModel.load(ROOT/'runs/action-v7/training/model.json')
    new=LiveActions(model);old=SinglePersonActions(model)
    for i in range(40):
        items=[chair(),detection()];snapshot=copy.deepcopy(items)
        a=old.update(items,i/30,640,480);b=new.update(items,i/30,640,480)
        assert items==snapshot
        assert (b['previous_label'],b['previous_raw_label'],b['previous_phase'])==(a['label'],a['raw_label'],a['phase'])
        json.dumps(b,allow_nan=False)
    assert b['target_index']==1 and b['previous_label']=='unknown'
    assert b['raw_label']!='unavailable' and b['evidence_s']>.5


def test_brief_ambiguity_preserves_history_without_inventing_a_pose():
    model=CompletionModel.load(ROOT/'runs/action-v7/training/model.json')
    live=LiveActions(model)
    for i in range(15):live.update([detection()],i*.1,640,480)
    history=list(live.selected.history)
    out=live.update([detection(99),detection(101)],1.5,640,480)
    assert out['label']=='unknown' and out['target_index'] is None
    assert out['target_continuity']['history_retained']
    assert list(live.selected.history)==history
    out=live.update([detection()],1.6,640,480)
    assert out['raw_label']!='unavailable'
    assert out['evidence_s']>1.
    assert 'target_missing' not in out['reset_reasons']


def test_long_missing_interval_does_not_reuse_stale_action_history():
    model=CompletionModel.load(ROOT/'runs/action-v7/training/model.json')
    live=LiveActions(model)
    for i in range(15):live.update([detection()],i*.1,640,480)
    for t in [1.5,1.6,1.7,1.8]:out=live.update([],t,640,480)
    assert out['label']=='unknown' and not live.selected.history
    assert not out['target_continuity']['history_retained']


def folded_detection():
    p=detection()
    for s,h,k in [(5,11,13),(6,12,14)]:
        p['keypoints'][s][:2]=[p['keypoints'][h][0]+9,p['keypoints'][h][1]+4]
    return p


def test_foreshortened_torso_can_only_continue_a_locked_geometry_subject():
    p=folded_detection()
    assert candidate(p,640,480) is None
    s=StableSubject(folded_continuation=True);epoch=lock(s)['target_epoch']
    for t in [.4,.5,.6,.7,.8]:
        r=s.update([p],t,640,480)
        assert r['target_index']==0 and r['target_epoch']==epoch
    assert StableSubject(folded_continuation=True).update([p],0,640,480)['target_index'] is None


@pytest.mark.parametrize('mode',['missing_legs','translation','competition','gap'])
def test_folded_continuation_needs_real_stationary_unambiguous_legs(mode):
    s=StableSubject(folded_continuation=True);lock(s);p=folded_detection()
    if mode=='missing_legs':p['keypoints'][15][2]=.1
    if mode=='translation':
        for j in range(17):p['keypoints'][j][0]+=100
    items=[p,copy.deepcopy(p)] if mode=='competition' else [p]
    assert s.update(items,1. if mode=='gap' else .4,640,480)['target_index'] is None

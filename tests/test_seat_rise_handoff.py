import pytest
from fallpre.weekly.seat_rise_handoff import SeatRiseHandoff
from test_observed_rise import pose


def result(label='seated',raw='background'):
    return dict(label=label,raw_label=raw)


def anchored():
    d=SeatRiseHandoff()
    for i in range(6):d.update(i*.13,1,pose(),result())
    assert d.anchor is not None
    return d


def test_verified_visible_seat_can_handoff_without_original_phase_anchor():
    d=anchored()
    outs=[d.update(.78+i*.13,1,pose(up=up),result('unknown')) for i,up in enumerate([3,9,18,28])]
    assert any(o['label']=='standing_up' for o in outs)


@pytest.mark.parametrize('mode',['lean','translation','missing','falling','walking','gap','epoch','single_jump','no_verified_seat','squat'])
def test_fallback_rejects_false_ascent_and_broken_identity(mode):
    d=anchored() if mode!='no_verified_seat' else SeatRiseHandoff()
    outputs=[]
    for i,up in enumerate([3,9,18,28]):
        p=pose(up=up,mode=mode if mode in ['lean','translation'] else 'lift',missing=(15,16) if mode=='missing' else ())
        if mode=='single_jump':p=pose(up=28)
        label='squatting' if mode=='squat' else 'unknown'
        raw=mode if mode in ['walking','falling'] else 'background'
        t=(2 if mode=='gap' else .78)+i*.13
        outputs.append(d.update(t,2 if mode=='epoch' else 1,p,result(label,raw)))
    assert all(o['label']!='standing_up' for o in outputs)


def test_fallback_never_vetoes_existing_seat_or_standing():
    d=anchored()
    for i,up in enumerate([3,9,18,28]):
        assert d.update(.78+i*.13,1,pose(up=up),result())['label']=='seated'
    assert d.update(1.4,1,pose(up=45),result('standing'))['label']=='standing'
    assert d.anchor is None


def test_stable_lower_seat_refreshes_reference_but_partial_rise_does_not_raise_it():
    d=anchored();original=d.anchor['points'][11,1]
    for i in range(8):d.update(.78+i*.13,1,pose(up=-6),result())
    assert d.anchor['points'][11,1]>original
    lower=d.anchor['points'].copy()
    for i in range(6):d.update(1.82+i*.13,1,pose(up=3),result())
    assert d.anchor is not None
    assert d.anchor['points'][11,1]==lower[11,1]


def test_large_pelvis_ascent_can_lead_shoulder_recovery_from_forward_lean():
    d=anchored();outs=[]
    for i,(up,shoulder_down) in enumerate([(12,2),(18,3),(24,2)]):
        p=pose(up=up)
        for j in [5,6]:p['kpts'][j][1]=100+shoulder_down
        outs.append(d.update(.78+i*.13,1,p,result('unknown')))
    assert outs[-1]['label']=='standing_up'
    assert outs[-1]['seat_rise_handoff']['lean_ascent']


@pytest.mark.parametrize('raw,moving_feet,expected',[('standing_up',False,'standing_up'),('background',False,'unknown'),('standing_up',True,'unknown')])
def test_terminal_torso_extension_requires_original_rise_and_planted_feet(raw,moving_feet,expected):
    d=anchored()
    for i,y in enumerate([90,84,78]):
        p=pose()
        for s,h,k,a in [(5,11,13,15),(6,12,14,16)]:
            p['kpts'][s][:2]=[p['kpts'][k][0],y]
            p['kpts'][h][:2]=[p['kpts'][k][0],120]
            if moving_feet:p['kpts'][a][0]+=i*20
        out=d.update(.78+i*.13,1,p,dict(label='unknown',raw_label=raw,decision_code='rise_motion_stopped'))
    assert out['label']==expected
    if expected=='standing_up':assert out['seat_rise_handoff']['terminal_extension']

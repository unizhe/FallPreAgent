"""Frozen RTMO recovery of current-frame poses for recently confirmed targets."""
from pathlib import Path
import sys
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .human import ROOT, sha256
from .verified_tracking import overlap

RTMO_PATH = ROOT / 'runs/meeting-20261014/models/rtmo-s.onnx'
RTMO_SHA256 = 'd0703d40d19f3921da51ae725402d5fdae4d2478c7442072d3101bd396f370d8'


class RTMOBackend:
    """Official locally archived RTMO-s ONNX, COCO17, BGR top-left letterbox.

    Pre/postprocessing follows the already archived RTMLib RTMO implementation;
    unlike its empty-output placeholder, no detection returns an empty list.
    No downloads or training are performed by this adapter.
    """
    def __init__(self, weights=RTMO_PATH):
        weights = Path(weights)
        if sha256(weights) != RTMO_SHA256:
            raise ValueError('Unexpected RTMO weights')
        runtime = ROOT / 'tmp/group-meeting/rtmo-deps'
        if runtime.exists() and str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(weights), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image):
        h, w = image.shape[:2]
        ratio = min(640/h, 640/w)
        resized = cv2.resize(image, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((640,640,3),114,np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        tensor = np.ascontiguousarray(padded.transpose(2,0,1)[None],dtype=np.float32)
        det, pose = self.session.run(None,{self.input_name:tensor})
        boxes = det[0,:,:4]/ratio
        scores = det[0,:,4]
        order = [int(i) for i in np.argsort(-scores) if scores[i] >= .35]
        keep=[]
        for i in order:
            if all(overlap(boxes[i],boxes[j]) <= .45 for j in keep):keep.append(i)
        result=[]
        for i in keep:
            points=pose[0,i].copy();points[:,:2]/=ratio
            result.append(dict(box=boxes[i].astype(float).tolist(),det=float(scores[i]),
                               kpts=points.astype(float).tolist(),pose_backend='rtmo-s'))
        return result


@dataclass(frozen=True)
class RecoveryConfig:
    primary_confidence: float = .50
    joint_confidence: float = .30
    minimum_joints: int = 8
    crop_scale: float = 2.0
    consensus_iou: float = .50
    torso_agreement: float = .20
    ambiguity_margin: float = .15


def restore_points(points, width, height, quarter_turns):
    """Inverse of np.rot90: preserve keypoint identity, transform coordinates."""
    p=np.asarray(points,float).copy();x,y=p[...,0].copy(),p[...,1].copy()
    if quarter_turns==1:p[...,0],p[...,1]=width-1-y,x
    elif quarter_turns==2:p[...,0],p[...,1]=width-1-x,height-1-y
    elif quarter_turns==3:p[...,0],p[...,1]=y,height-1-x
    elif quarter_turns!=0:raise ValueError('Expected 0..3 quarter turns')
    return p


def restore_instance(person, width, height, turns, origin=(0,0)):
    b=person['box'];corners=np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])
    corners=restore_points(corners,width,height,turns)+np.asarray(origin)
    k=restore_points(person['kpts'],width,height,turns);k[:,:2]+=np.asarray(origin)
    return dict(person,box=np.r_[corners.min(0),corners.max(0)].tolist(),kpts=k.tolist())


class PoseRecovery:
    """Fallback only; no native pose replacement, identity creation or gap fill.

    A confident full-frame RTMO result can recover a recent target. Rotated or
    zoomed proposals require agreement from two distinct views. Returned poses
    always come from inference on the CURRENT image, never historical copying.
    """
    def __init__(self, backend=None, config=None, *, resolve_native_ambiguity=False):
        self.backend=backend
        self.config=config or RecoveryConfig()
        self.resolve_native_ambiguity=resolve_native_ambiguity

    def _valid(self, p, frame_shape):
        h,w=frame_shape[:2]
        k=np.asarray(p['kpts'],float)
        if k.shape!=(17,3) or not np.isfinite(k).all():return False
        visible=(k[:,0]>=0)&(k[:,0]<w)&(k[:,1]>=0)&(k[:,1]<h)
        good=(k[:,2]>=self.config.joint_confidence)&visible
        return bool(good.sum()>=self.config.minimum_joints and good[5:7].any() and good[11:13].any())

    def _agreement(self, a, b):
        if overlap(a['box'],b['box']) < self.config.consensus_iou:return False
        x,y=np.asarray(a['kpts']),np.asarray(b['kpts'])
        ids=np.array([5,6,11,12]);good=(x[ids,2]>=self.config.joint_confidence)&(y[ids,2]>=self.config.joint_confidence)
        if good.sum()<2:return False
        diag=np.linalg.norm(np.array(a['box'])[2:]-np.array(a['box'])[:2])
        return bool(np.median(np.linalg.norm(x[ids[good],:2]-y[ids[good],:2],axis=1))/max(diag,1) <= self.config.torso_agreement)

    def _choose(self, candidates):
        if not candidates:return None
        ranked=sorted(candidates,key=lambda p:p['continuity_score'],reverse=True)
        if len(ranked)>1 and ranked[0]['continuity_score']-ranked[1]['continuity_score']<self.config.ambiguity_margin:
            if overlap(ranked[0]['box'],ranked[1]['box']) < self.config.consensus_iou:return None
        return ranked[0]

    def augment(self, image, row, tracker):
        context=(tracker.recovery_context(row['t'],row['instances'],allow_ambiguous=True)
                 if self.resolve_native_ambiguity else tracker.recovery_context(row['t'],row['instances']))
        audit=dict(context,executed=False,accepted=False,candidates=[],inference_passes=0)
        audit['trigger']=context['reason']
        if not context['allowed']:return row,audit
        self.backend=self.backend or RTMOBackend()
        audit['executed']=True
        h,w=image.shape[:2]
        candidates=[]
        def infer(view,origin,turn,name):
            vh,vw=view.shape[:2]
            result=self.backend(np.ascontiguousarray(np.rot90(view,turn)))
            audit['inference_passes']+=1
            for raw in result:
                p=restore_instance(raw,vw,vh,turn,origin)
                score=tracker._continuity(p['box'],row['t'],tracker.step+1)
                valid=self._valid(p,image.shape)
                p.update(recovery_view=name,continuity_score=score,quality_pass=valid)
                audit['candidates'].append(p)
                if valid and score is not None:candidates.append(p)
        infer(image,(0,0),0,'full_0')
        chosen=self._choose([p for p in candidates if p['det']>=self.config.primary_confidence])
        if chosen is None:
            b=np.array(context['anchor']);c=(b[:2]+b[2:])/2
            side=max(b[2:]-b[:2])*self.config.crop_scale
            lo=np.maximum([0,0],np.floor(c-side/2)).astype(int)
            hi=np.minimum([w,h],np.ceil(c+side/2)).astype(int)
            crop=image[lo[1]:hi[1],lo[0]:hi[0]]
            for turn in [1,2,3]:infer(image,(0,0),turn,f'full_{turn}')
            if crop.size and min(crop.shape[:2])>=16:
                for turn in [0,1,2,3]:infer(crop,tuple(lo),turn,f'crop_{turn}')
            consistent=[p for p in candidates if any(q['recovery_view']!=p['recovery_view']
                        and self._agreement(p,q) for q in candidates)]
            chosen=self._choose(consistent)
        if chosen is None:
            audit['reason']='no_unambiguous_quality_consensus'
            return row,audit
        # The wrapper is only allowed to append a current-frame observation.
        # Existing admission and continuity logic still makes the final decision.
        recovered=dict(chosen,pose_backend='rtmo-s',recovered=True)
        audit.update(accepted=True,reason='current_image_recovery',selected_view=chosen['recovery_view'],
                     appended_index=len(row['instances']))
        return dict(row,instances=[*row['instances'],recovered]),audit

    def configuration(self):
        return dict(asdict(self.config),resolve_native_ambiguity=self.resolve_native_ambiguity)

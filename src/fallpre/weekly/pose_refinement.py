"""Current-image top-down pose verification, without historical keypoint input."""
import sys
from pathlib import Path

import cv2
import numpy as np

from .human import ROOT, sha256
from .pose_recovery import PoseRecovery, restore_points
from .verified_tracking import overlap

RTMPOSE_PATH = ROOT / 'runs/pose-refinement-20261020/models/rtmpose-m.onnx'
RTMPOSE_SHA256 = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'


def rotate_points(points, width, height, turn):
    p = np.asarray(points, float).copy()
    x, y = p[..., 0].copy(), p[..., 1].copy()
    if turn == 1: p[..., 0], p[..., 1] = y, width - 1 - x
    elif turn == 2: p[..., 0], p[..., 1] = width - 1 - x, height - 1 - y
    elif turn == 3: p[..., 0], p[..., 1] = height - 1 - y, x
    elif turn != 0: raise ValueError('Expected quarter turn 0..3')
    return p


class RTMPoseBackend:
    """Archived official ONNX SDK: RGB normalize, 1.25 padding, SimCC /2.

    The SDK pipeline.json is authoritative for RGB conversion; the upstream
    lightweight demo and some RTMLib versions omit that conversion.
    """
    def __init__(self, weights=RTMPOSE_PATH):
        if sha256(Path(weights)) != RTMPOSE_SHA256:
            raise ValueError('Unexpected RTMPose weights')
        runtime = str(ROOT / 'tmp/group-meeting/rtmo-deps')
        if runtime not in sys.path: sys.path.insert(0, runtime)
        import onnxruntime as ort
        opt = ort.SessionOptions(); opt.intra_op_num_threads = 2; opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(weights), sess_options=opt, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image, box, turn=0, expansion=1.0):
        h, w = image.shape[:2]
        b = np.asarray(box, float)
        corners = rotate_points([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],w,h,turn)
        lo, hi = corners.min(0), corners.max(0)
        center = (lo+hi)/2
        scale = (hi-lo)*1.25*expansion
        if scale[0] > scale[1]*.75: scale[1] = scale[0]/.75
        else: scale[0] = scale[1]*.75
        factor = np.array([192,256])/scale
        matrix = np.array([[factor[0],0,96-center[0]*factor[0]],
                           [0,factor[1],128-center[1]*factor[1]]],np.float32)
        crop = cv2.warpAffine(np.ascontiguousarray(np.rot90(image,turn)),matrix,(192,256))
        tensor = (crop[:,:,::-1].astype(np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]
        x,y = self.session.run(None,{self.input_name:np.ascontiguousarray(tensor.transpose(2,0,1)[None],dtype=np.float32)})
        coords = np.stack([x[0].argmax(-1),y[0].argmax(-1)],axis=-1)/2
        # Conservative per-axis confidence; never boost weak axes by averaging.
        score = np.minimum(x[0].max(-1),y[0].max(-1))
        coords = coords/[192,256]*scale+center-scale/2
        pts = restore_points(np.c_[coords,score],w,h,turn)
        return pts


def visible_joints(points, shape, confidence=.30):
    p = np.asarray(points)
    h,w = shape[:2]
    return (p[:,2] >= confidence) & np.isfinite(p).all(1) & (p[:,0] >= 0) & (p[:,0] < w) & (p[:,1] >= 0) & (p[:,1] < h)


def consensus_joints(views, shape, diagonal, tolerance=.08):
    """Current-image agreement only; unsupported joints have zero confidence.

    Coordinates are robust medians from supporting current-image predictions.
    A similarly supported competing location makes that joint unknown.
    """
    values = np.asarray(views,float)
    valid = np.array([visible_joints(p,shape) for p in values])
    result = np.zeros((17,3)); support = []
    radius = max(1,diagonal*tolerance)
    for j in range(17):
        ids = np.flatnonzero(valid[:,j])
        groups = []
        for i in ids:
            members = [int(q) for q in ids if np.linalg.norm(values[i,j,:2]-values[q,j,:2])<=radius]
            if len(members)>=2: groups.append(members)
        groups.sort(key=lambda g:(len(g),sum(values[g,j,2])),reverse=True)
        chosen = groups[0] if groups else []
        if chosen:
            center = np.median(values[chosen,j,:2],axis=0)
            conflict = any(len(g)>=max(2,len(chosen)-1) and
                           np.linalg.norm(np.median(values[g,j,:2],axis=0)-center)>2*radius for g in groups[1:])
            if conflict: chosen=[]
            else:
                result[j,:2]=center
                result[j,2]=np.min(values[chosen,j,2])
        support.append(chosen)
    return result,support


def partial_body_evidence(points, reliable, agreed, diagonal):
    """Two disjoint, cross-model-supported limb segments can locate a person.

    This is presence evidence only, not a complete pose or an action signal.
    Confidence, current-image agreement and identity gates are not relaxed.
    """
    limbs=[(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]
    segments=[(a,b) for a,b in limbs if reliable[a] and reliable[b] and agreed[a] and agreed[b]
              and .02 <= np.linalg.norm(points[a,:2]-points[b,:2])/max(diagonal,1) <= .60]
    pairs=[(a,b) for i,a in enumerate(segments) for b in segments[i+1:] if not set(a)&set(b)]
    return dict(eligible=bool(reliable.sum()>=6 and agreed.sum()>=4 and pairs),
                supported_segments=segments,disjoint_pairs=pairs)


def predict_with_pose_quality(online, row):
    """Keep target presence separate from action-model input sufficiency."""
    if online is None:return dict(label='disabled_frame_only',phase='disabled')
    withheld=bool(row['instances'] and row['instances'][0].get('action_eligible') is False)
    result=online.update(dict(row,instances=[]) if withheld else row)
    if withheld:
        result=dict(result,label='unknown',phase='unknown',pose_input_withheld=True,
                    reason='partial_person_without_action_pose')
    return result


class RefinedPoseRecovery(PoseRecovery):
    """Verify weak current RTMO observations with frozen top-down pose views.

    The original detector-confirmation gate and native-pose priority remain.
    No historical joint coordinates are passed to the top-down model. Its crop
    is from CURRENT RTMO detections, not the previous frame's box.
    """
    def __init__(self, backend=None, config=None, refiner=None, *, allow_partial_observation=True,
                 allow_detector_proposals=False, resolve_native_ambiguity=False):
        super().__init__(backend,config,resolve_native_ambiguity=resolve_native_ambiguity)
        self.refiner=refiner
        self.allow_partial_observation=allow_partial_observation
        self.allow_detector_proposals=allow_detector_proposals

    def configuration(self):
        return dict(super().configuration(), refinement=dict(joint_confidence=.30,
            agreement_distance=.08, minimum_supported_joints=6, rtmo_body_agreement=.15,
            expansions=[1.0,1.15], rotations=[0,1,2,3], minimum_added_joints=2,
            single_proposal_confidence=.60, single_rotation_requires_rtmo_joint=True,
            detector_proposal=dict(enabled=self.allow_detector_proposals,confidence=.15,
                iou=.70,minimum_cross_model_body_joints=4,role='proposal_only_not_admission'),
            partial_observation=dict(enabled=self.allow_partial_observation,minimum_joints=6,
                minimum_cross_model_body_joints=4,disjoint_limb_segments=2,
                normalized_segment_length=[.02,.60],action_input='withheld')))

    def _detector_support(self, proposal, row):
        """Weak person boxes only trigger current-image verification, never admission."""
        return bool(self.allow_detector_proposals and any(
            d['class_name']=='person' and d['conf']>=.15 and overlap(proposal['box'],d['box'])>=.70
            for d in row.get('detector_instances',[])))

    def augment(self,image,row,tracker):
        augmented,audit=super().augment(image,row,tracker)
        audit['refinement']=dict(executed=False,accepted=False,inference_passes=0)
        if not audit['executed']: return augmented,audit
        previous = augmented['instances'][-1] if audit['accepted'] else None
        detector_proposal=False
        if previous is not None:
            good=visible_joints(previous['kpts'],image.shape)
            if good.sum()>=12 and good[11:13].all() and good[15:17].any():
                return augmented,audit
            proposal=previous
        else:
            # Two current object proposals agree spatially. At least one carries
            # four visible body joints; top-down must independently corroborate.
            candidates=[p for p in audit['candidates'] if p['continuity_score'] is not None]
            supported=[p for p in candidates if visible_joints(p['kpts'],image.shape)[5:].sum()>=4
                       and (p['det']>=.60 or self._detector_support(p,row) or any(q['recovery_view']!=p['recovery_view'] and overlap(p['box'],q['box'])>=.50
                               for q in candidates))]
            proposal=self._choose(supported)
            if proposal is None:return augmented,audit
            detector_proposal=bool(proposal['det']<.60 and not any(
                q['recovery_view']!=proposal['recovery_view'] and overlap(proposal['box'],q['box'])>=.50
                for q in candidates))
        self.refiner=self.refiner or RTMPoseBackend()
        b=np.asarray(proposal['box']);diagonal=np.linalg.norm(b[2:]-b[:2])
        views=[self.refiner(image,b,turn,expansion) for turn in range(4) for expansion in [1.0,1.15]]
        k,support=consensus_joints(views,image.shape,diagonal)
        old=np.asarray(proposal['kpts'])
        old_visible=visible_joints(old,image.shape)
        # Resizing the same orientation is insufficient independent evidence
        # for a new joint. Require another rotation OR the current RTMO joint.
        for j,g in enumerate(support):
            if len({i//2 for i in g})<2 and not (old_visible[j] and
                    np.linalg.norm(k[j,:2]-old[j,:2])<=.08*diagonal):
                k[j]=0;support[j]=[]
        reliable=visible_joints(k,image.shape)
        shared=reliable & old_visible
        shared[:5]=False
        agreed=shared & (np.linalg.norm(k[:,:2]-old[:,:2],axis=1)<=.15*diagonal)
        chains=[(5,7,9),(6,8,10),(11,13,15),(12,14,16),(5,11,13),(6,12,14)]
        enough=bool(reliable.sum()>=6 and agreed.sum()>=3 and any(reliable[list(c)].all() for c in chains))
        if detector_proposal:enough=bool(enough and agreed.sum()>=4)
        partial=partial_body_evidence(k,reliable,agreed,diagonal)
        presence_only=bool(self.allow_partial_observation and previous is None and not enough and partial['eligible'])
        improved=previous is None or reliable.sum()>=visible_joints(previous['kpts'],image.shape).sum()+2
        details=dict(executed=True,accepted=bool((enough and improved) or presence_only),inference_passes=len(views),
                     proposal=proposal,views=[v.tolist() for v in views],joint_support=support,
                     supported_joints=int(reliable.sum()),rtmo_agreed_body_joints=int(agreed.sum()),
                     enough_evidence=enough,improved_coverage=bool(improved),
                     partial_body_evidence=partial,presence_only=presence_only,
                     detector_supported_proposal=detector_proposal)
        audit['refinement']=details
        if not details['accepted']:return augmented,audit
        recovered=dict(proposal,kpts=k.tolist(),pose_backend='rtmpose-m-consensus',recovered=True,
                       refined=True,pose_coverage='partial' if not reliable.all() else 'all_joints_supported',
                       joint_supported=reliable.tolist(),joint_support=support)
        if presence_only:
            recovered.update(observation_quality='partial_person',action_eligible=False)
        audit.update(accepted=True,reason='current_image_refinement',selected_view='RTMPose current-view consensus',
                     appended_index=len(row['instances']))
        return dict(row,instances=[*row['instances'],recovered]),audit

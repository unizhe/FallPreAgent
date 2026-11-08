"""Bounded current-frame YOLO crop verification for an already tracked person.

No old joints are supplied to inference. Cropping proposes where to look; only
agreement between different orientations of the CURRENT image yields a pose.
The independent fall tracker must still accept that pose's identity continuity.
"""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from .pose_recovery import restore_points


@dataclass(frozen=True)
class LiveRecoveryConfig:
    crop_scale: float = 2.0
    rotations: tuple = (0, 1, 3, 2)
    joint_confidence: float = .30
    minimum_body_joints: int = 6
    minimum_mean_confidence: float = .50
    agreement_iou: float = .50
    agreement_joint_distance: float = .12


def restore_pose(person, width, height, turn, origin):
    """Restore keypoint centers and continuous box edges to full-image space."""
    restored = deepcopy(person)
    points = restore_points(person['keypoints'], width, height, turn)
    points[:, :2] += origin
    x1, y1, x2, y2 = person['box']
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
    x, y = corners[:, 0].copy(), corners[:, 1].copy()
    if turn == 1:
        corners[:, 0], corners[:, 1] = width-y, x
    elif turn == 2:
        corners[:, 0], corners[:, 1] = width-x, height-y
    elif turn == 3:
        corners[:, 0], corners[:, 1] = y, height-x
    elif turn != 0:
        raise ValueError('Expected quarter turns 0..3')
    corners += origin
    restored.update(box=np.r_[corners.min(0), corners.max(0)].tolist(),
                    keypoints=points.tolist(), recovery_rotation=turn)
    return restored


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(intersection/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection, 1))


class LivePoseRecovery:
    def __init__(self, config=None):
        self.config = config or LiveRecoveryConfig()

    def configuration(self):
        return asdict(self.config)

    def valid(self, person, shape):
        points = np.asarray(person['keypoints'], float)
        box = np.asarray(person['box'], float)
        if (points.shape != (17, 3) or box.shape != (4,) or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])):
            return False
        height, width = shape[:2]
        good = (np.isfinite(points).all(1) & (points[:, 2] >= self.config.joint_confidence)
                & (points[:, 0] >= 0) & (points[:, 0] < width)
                & (points[:, 1] >= 0) & (points[:, 1] < height))
        margin = .1*(box[2:]-box[:2])
        good &= ((points[:, :2] >= box[:2]-margin) & (points[:, :2] <= box[2:]+margin)).all(1)
        good[:5] = False
        return bool(good[[5, 6, 11, 12]].all() and good.sum() >= self.config.minimum_body_joints
                    and np.mean(points[good, 2]) >= self.config.minimum_mean_confidence)

    def same_geometry(self, a, b):
        if overlap(a['box'], b['box']) < self.config.agreement_iou:
            return False
        ids = [5, 6, 11, 12]
        first, second = np.asarray(a['keypoints'])[ids, :2], np.asarray(b['keypoints'])[ids, :2]
        diagonal = max(np.linalg.norm(np.asarray(a['box'])[2:]-a['box'][:2]),
                       np.linalg.norm(np.asarray(b['box'])[2:]-b['box'][:2]), 1)
        return bool(np.max(np.linalg.norm(first-second, axis=1)) <= self.config.agreement_joint_distance*diagonal)

    def agree(self, a, b):
        return a['recovery_rotation'] != b['recovery_rotation'] and self.same_geometry(a, b)

    def consensus(self, proposals):
        verified = []
        for p in proposals:
            supporters = [q for q in proposals if self.agree(p, q)]
            if not supporters:
                continue
            candidate = deepcopy(p)
            candidate.update(recovery_verified=True,
                             recovery_views=sorted({p['recovery_rotation'], *[q['recovery_rotation'] for q in supporters]}))
            # Keep all instances. Cross-view agreement is looser than identity
            # deduplication; merging here could erase a nearby second person.
            verified.append(candidate)
        return verified

    def recover(self, image, anchor_box, infer):
        """infer(view) returns raw poses from the same selected YOLO model."""
        start = time.perf_counter()
        audit = dict(executed=False, accepted=False, inference_passes=0, proposals=0,
                     verified_candidates=0, reason='invalid_recovery_region', elapsed_ms=0.)
        height, width = image.shape[:2]
        anchor = np.asarray(anchor_box, float)
        if anchor.shape != (4,) or not np.isfinite(anchor).all() or np.any(anchor[2:] <= anchor[:2]):
            return [], audit
        center = (anchor[:2]+anchor[2:])/2
        side = float(max(anchor[2:]-anchor[:2])*self.config.crop_scale)
        lo = np.maximum(0, np.floor(center-side/2)).astype(int)
        hi = np.minimum([width, height], np.ceil(center+side/2)).astype(int)
        if min(hi-lo) < 16:
            return [], audit
        crop = image[lo[1]:hi[1], lo[0]:hi[0]]
        proposals, verified = [], []
        audit['executed'] = True
        for turn in self.config.rotations:
            current = infer(np.ascontiguousarray(np.rot90(crop, turn)))
            audit['inference_passes'] += 1
            for raw in current:
                pose = restore_pose(raw, crop.shape[1], crop.shape[0], turn, lo)
                if self.valid(pose, image.shape):
                    proposals.append(pose)
            verified = self.consensus(proposals)
            if verified:
                break
        # A single-view person is not a recovered observation, but remains
        # evidence of identity ambiguity. Never hide it from the final tracker.
        competitors = [dict(deepcopy(p), recovery_verified=False, recovery_identity_competitor=True)
                       for p in proposals if not any(self.agree(p, q) for q in proposals)] if verified else []
        audit.update(proposals=len(proposals), verified_candidates=len(verified),
                     identity_competitors=len(competitors),
                     reason='current_view_consensus' if verified else 'no_current_view_consensus',
                     elapsed_ms=(time.perf_counter()-start)*1000)
        return verified + competitors, audit

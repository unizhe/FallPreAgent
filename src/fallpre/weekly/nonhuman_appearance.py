"""Offline candidate verification against a human-confirmed nonhuman exemplar.

No coordinates, camera IDs or detector-score exemptions are stored. A match
requires current texture, geometry and dense appearance agreement. Missing pose
alone is never evidence for rejecting a person. This is not enabled live.
"""
from __future__ import annotations

import cv2
import numpy as np

from .webcam_rgb_fall import _overlap


class NonhumanAppearance:
    def __init__(self, reference):
        if reference is None or min(reference.shape[:2]) < 8:
            raise ValueError('Nonhuman reference image is missing or too small')
        self.orb = cv2.ORB_create(nfeatures=500, edgeThreshold=5, patchSize=15, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.gray, self.keys, self.descriptors = self.features(reference)
        if self.descriptors is None or len(self.descriptors) < 12:
            raise ValueError('Nonhuman reference lacks distinctive texture')

    def features(self, image):
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (128, 224))
        keys, descriptors = self.orb.detectAndCompute(gray, None)
        return gray, keys, descriptors

    def compare(self, image):
        if min(image.shape[:2]) < 8:
            return dict(verified=False, reason='crop_too_small')
        gray, keys, descriptors = self.features(image)
        if descriptors is None:
            return dict(verified=False, reason='texture_missing')
        forward = [p[0] for p in self.matcher.knnMatch(self.descriptors, descriptors, k=2)
                   if len(p) == 2 and p[0].distance < .7 * p[1].distance]
        reverse = {(p[0].trainIdx, p[0].queryIdx)
                   for p in self.matcher.knnMatch(descriptors, self.descriptors, k=2)
                   if len(p) == 2 and p[0].distance < .7 * p[1].distance}
        # Repeated texture can map several reference points onto one point.
        # Require reciprocal, unambiguous correspondence before measuring
        # geometric consensus; keep all admission thresholds unchanged.
        matches = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse]
        if len(matches) < 12:
            return dict(verified=False, reason='texture_mismatch')
        origin = np.float32([self.keys[m.queryIdx].pt for m in matches])
        current = np.float32([keys[m.trainIdx].pt for m in matches])
        transform, mask = cv2.estimateAffinePartial2D(origin, current, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.)
        if transform is None or mask is None:
            return dict(verified=False, reason='geometry_unknown')
        inliers = mask.ravel().astype(bool)
        coverage = float(np.prod(np.ptp(origin[inliers], axis=0)) / self.gray.size)
        fraction = float(inliers.mean())
        warped = cv2.warpAffine(gray, transform, (128, 224),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        a, b = self.gray[12:-12, 12:-12].ravel(), warped[12:-12, 12:-12].ravel()
        correlation = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1 else 0.
        corners = np.array([[0, 0, 1], [128, 0, 1], [0, 224, 1], [128, 224, 1]], np.float32)
        shift = float(np.max(np.linalg.norm(corners @ transform.T - corners[:, :2], axis=1)))
        verified = bool(inliers.sum() >= 12 and fraction >= .85 and coverage >= .35
                        and correlation >= .95 and shift <= 22)
        return dict(verified=verified, reason='appearance_match' if verified else 'appearance_uncertain',
                    inliers=int(inliers.sum()), inlier_fraction=fraction, coverage=coverage,
                    correlation=correlation, normalized_corner_shift=shift)

    def verify(self, image, candidates, pose_people):
        height, width = image.shape[:2]
        verified, audit = [], []
        for index, candidate in enumerate(candidates):
            box = candidate['box']
            # Any meaningful current skeleton overlapping the candidate vetoes
            # the exemplar exemption, including incomplete and low-score poses.
            protected = False
            for person in pose_people:
                if _overlap(person['box'], box) < .1:
                    continue
                points = person.get('keypoints', [])
                visible = [len(p) >= 3 and p[2] >= .25 for p in points]
                if sum(visible) >= 2:
                    protected = True
                    break
            if protected:
                audit.append(dict(index=index, verified=False, reason='current_pose_protected'))
                continue
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            result = self.compare(image[y1:y2, x1:x2])
            audit.append(dict(index=index, **result))
            if result['verified']:
                verified.append(index)
        return verified, audit

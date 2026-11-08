"""Current-pixel witnesses for fall transitions, never synthesized joints."""
from __future__ import annotations

import math

from .webcam_rgb_fall import _angle, _box, _overlap


def pose_geometry(person, box, width, height):
    if not isinstance(person, dict) or _box(person.get('box')) is None or _overlap(person['box'], box) < .5:
        return None
    try:
        p = person['keypoints']
        if len(p) != 17:
            return None
        valid = [len(q) == 3 and all(math.isfinite(float(v)) for v in q)
                 and q[2] >= .5 and 0 <= q[0] < width and 0 <= q[1] < height for q in p]
        if not all(valid[j] for j in (5, 6, 11, 12)):
            return None
        shoulder = tuple((p[5][i] + p[6][i]) / 2 for i in (0, 1))
        hip = tuple((p[11][i] + p[12][i]) / 2 for i in (0, 1))
        length = math.dist(shoulder, hip)
        if length < 20:
            return None
        tilt = math.degrees(math.atan2(abs(hip[0]-shoulder[0]), abs(hip[1]-shoulder[1])))
        sides = [s for s in ((5, 11, 13, 15), (6, 12, 14, 16)) if all(valid[j] for j in s)]
        seated = bool(sides and hip[1] > shoulder[1] and tilt <= 25 and all(
            45 <= _angle(p[s], p[h], p[k]) <= 140
            and 45 <= _angle(p[h], p[k], p[a]) <= 145
            and -.12 <= (p[k][1]-p[h][1])/length <= .65
            and (p[a][1]-p[h][1])/length >= .75
            and (p[a][1]-p[k][1])/length >= .35 for s,h,k,a in sides))
        # Low sitting / squatting is an episode boundary, never a safety claim.
        # A vertical torso alone is insufficient: visible folded legs must put
        # the pelvis near ankle height. Chair sitting and extended standing are
        # still handled by the existing higher support references.
        low_support = bool(sides and hip[1] > shoulder[1] and tilt <= 28
            and all(-.15 <= (p[a][1]-p[h][1])/length <= .90 for _,h,_,a in sides)
            and any((p[a][1]-p[h][1])/length <= .65
                    and _angle(p[h],p[k],p[a]) <= 145
                    and -.70 <= (p[k][1]-p[h][1])/length <= .80 for _,h,k,a in sides))
        return dict(hip=hip, shoulder=shoulder, length=length, tilt=tilt, seated=seated,
                    low_support=low_support,
                    feet=[tuple(p[s[3]][:2]) for s in sides], torso_confidence=min(p[j][2] for j in (5,6,11,12)))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def contained_fraction(small, large):
    area = (small[2]-small[0])*(small[3]-small[1])
    intersection = max(0., min(small[2],large[2])-max(small[0],large[0])) * max(0.,min(small[3],large[3])-max(small[1],large[1]))
    return intersection / area


def deduplicate_parts(people, anchor, pose_person, width, height):
    """Only a current full torso plus a continuing outer box may explain a part.

    Similar-sized/independent competitors remain ambiguous. Suppressed proposals
    are returned for audit, not silently discarded from exported raw detections.
    """
    if anchor is None or len(people) < 2:
        return people, []
    eligible = [(i,p) for i,p in people if _overlap(anchor,p['box']) >= .65
                and pose_geometry(pose_person,p['box'],width,height) is not None]
    if len(eligible) != 1:
        return people, []
    outer_i, outer = eligible[0]
    b = outer['box']
    area = (b[2]-b[0])*(b[3]-b[1])
    ignored = []
    for i,p in people:
        q = p['box']
        ratio = (q[2]-q[0])*(q[3]-q[1])/area
        if (i != outer_i and .10 <= ratio <= .55 and contained_fraction(q,b) >= .97
                and p['confidence'] <= outer['confidence']+.15):
            ignored.append(dict(index=i, outer_index=outer_i, containment=contained_fraction(q,b), area_ratio=ratio))
    return [(i,p) for i,p in people if i not in {x['index'] for x in ignored}], ignored


def containing_competitor(people, anchor, pose_person, width, height):
    """A short ambiguity may be retained, never accepted as an observation.

    A low-score image-edge box can expand over the person and furniture. Only
    one current torso must still closely match the recent real person box.
    This does NOT discard the competitor or resolve a genuine second person.
    """
    if anchor is None or len(people) != 2:
        return False
    current = [(i,p) for i,p in people if _overlap(anchor,p['box']) >= .65
               and pose_geometry(pose_person,p['box'],width,height) is not None]
    if len(current) != 1:
        return False
    i,p = current[0]
    q = next(q for j,q in people if j != i)
    a,b = p['box'],q['box']
    ratio = ((b[2]-b[0])*(b[3]-b[1])) / ((a[2]-a[0])*(a[3]-a[1]))
    return bool(p['confidence'] >= .50 and q['confidence'] <= p['confidence']-.10
                and ratio >= 2.5 and contained_fraction(a,b) >= .97
                and _overlap(pose_person['box'],b) < .35
                and (b[0] <= 2 or b[1] <= 2 or b[2] >= width-2))

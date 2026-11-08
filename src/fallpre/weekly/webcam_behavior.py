"""Independent causal descriptions of posture, translation and body rotation.

These observations neither change the frozen action vote nor feed the fall
pipeline. Distances are image/torso ratios, not metres, yaw angles or clinical
gait measures. A body turn requires agreeing shoulder AND pelvis evidence;
neither head motion nor a change of travel direction alone is a body turn.
"""

from collections import deque

import numpy as np

from .action_v2 import angle

TEXT = {
    "posture": {
        "upright": "直立",
        "seated": "坐姿",
        "bending": "弯腰",
        "squatting": "蹲姿",
        "unknown": "未知",
    },
    "transition": {"standing_up": "起身", "sitting_down": "坐下", "none": "无姿态转换", "unknown": "未知"},
    "locomotion": {
        "stationary": "静止",
        "starting": "开始行走",
        "walking": "行走",
        "stopping": "停步中",
        "moving": "移动（步态未确认）",
        "unknown": "未知",
    },
    "turning": {"turning": "转身", "not_turning": "未见转身", "unknown": "不可判断"},
    "path_change": {"changing": "路径改变", "straight": "未见路径改变", "unknown": "不可判断"},
}
SIDES = ((5, 11, 13, 15), (6, 12, 14, 16))


def axis(kind, label="unknown", reason="等待当前有效观测"):
    return {"label": label, "text": TEXT[kind][label], "reason": reason}


def observe(person, t, width, height):
    if not isinstance(person, dict):
        return None
    try:
        p = np.asarray(person.get("keypoints", []), dtype=float)
    except (ValueError, TypeError):
        return None
    if p.shape != (17, 3) or not np.isfinite(p).all():
        return None
    valid = (p[:, 2] >= 0.5) & (p[:, 0] > 0) & (p[:, 0] < width) & (p[:, 1] > 0) & (p[:, 1] < height)
    torso = [s for s in SIDES if valid[s[0]] and valid[s[1]]]
    if not torso:
        return None
    scale = float(np.median([np.linalg.norm(p[s, :2] - p[h, :2]) for s, h, _, _ in torso]))
    if scale < 12:
        return None
    shoulders = p[[s[0] for s in torso], :2].mean(0)
    hips = p[[s[1] for s in torso], :2].mean(0)
    legs = [s for s in torso if all(valid[list(s)])]
    knee = min((angle(p[h, :2], p[k, :2], p[a, :2]) for _, h, k, a in legs), default=0.0)
    hip = min((angle(p[s, :2], p[h, :2], p[k, :2]) for s, h, k, _ in legs), default=0.0)
    leg_height = min(((p[a, 1] - p[h, 1]) / scale for _, h, _, a in legs), default=0.0)
    upright = bool(
        legs and knee > 0.62 and hip > 0.84 and leg_height > 0.90 and (hips[1] - shoulders[1]) / scale > 0.78
    )
    bilateral = bool(all(valid[[5, 6, 11, 12]]))
    widths = np.array([p[6, 0] - p[5, 0], p[12, 0] - p[11, 0]]) / scale if bilateral else None
    return dict(
        t=t,
        p=p.copy(),
        valid=valid,
        sides=tuple(s[0] for s in torso),
        hips=hips,
        shoulders=shoulders,
        scale=scale,
        upright=upright,
        widths=widths,
        feet=bool(all(valid[[13, 14, 15, 16]])),
    )


def apply_auxiliary_bend(behavior, action, persons, t, width, height, camera_stable=True):
    """Expose a verified CURRENT auxiliary bend without borrowing its gait history.

    The main action identity and independent fall route remain unchanged. An
    explicitly different main target, stale evidence or conflicting vote vetoes
    this display handoff. No missing keypoint or bounding box is synthesized.
    """
    evidence = action.get('auxiliary_posture') or {}
    index = evidence.get('target_index')
    if (not camera_stable or action.get('label') != 'bending'
            or action.get('raw_label') == 'falling' or evidence.get('label') != 'bending'
            or evidence.get('source_time') != t or evidence.get('target_epoch') is None
            or not isinstance(index, int) or not 0 <= index < len(persons)
            or action.get('target_index') not in (None, index)):
        return behavior
    row = observe(persons[index], t, width, height)
    if row is None:
        return behavior
    behavior.update(target_index=index, target_epoch=f"bend-{evidence['target_epoch']}")
    behavior['posture'] = axis('posture', 'bending', '当前辅助目标的连续关节已确认弯腰；未沿用主目标历史')
    behavior['transition'] = axis('transition', 'none', '当前为已确认弯腰姿态')
    # Only posture is transferred. Never infer walking/turning from another track.
    for kind in ('locomotion', 'turning', 'path_change'):
        behavior[kind] = axis(kind, reason='辅助弯腰目标仅提供当前姿态，不提供移动或转向历史')
    behavior['quality'].update(observed=True, torso=True, feet=row['feet'],
                               bilateral=row['widths'] is not None, auxiliary_posture=True)
    return behavior


class BehaviorAxes:
    """Describe only actual observations; gaps clear temporal evidence."""

    def __init__(self):
        self.history = deque()
        self.last_t = None
        self.epoch = None
        self.walk_since = None
        self.last_walk_t = None
        self.walk_candidate_since = None
        self.last_strong_walk_t = None

    def clear(self):
        self.history.clear()
        self.walk_since = None
        self.last_walk_t = None
        self.walk_candidate_since = None
        self.last_strong_walk_t = None

    def update(self, person, t, epoch, action, width, height, camera_stable=True):
        result = dict(
            schema_version="behavior-axes-v1",
            **{k: axis(k) for k in TEXT},
            metrics={},
            quality=dict(
                observed=False,
                torso=False,
                feet=False,
                bilateral=False,
                camera_stable=bool(camera_stable),
                reset_reason=None,
            ),
            target_epoch=epoch,
        )
        reset_reason = None
        if not np.isfinite(t) or t < 0:
            self.clear()
            self.last_t = None
            result["quality"]["reset_reason"] = "invalid_timestamp"
            return result
        if self.last_t is not None and not 0 < t - self.last_t <= 0.35:
            reset_reason = "source_gap"
        if self.last_t is not None and epoch != self.epoch:
            reset_reason = "target_changed"
        self.last_t, self.epoch = t, epoch
        if reset_reason:
            self.clear()
        row = observe(person, t, width, height)
        if row is None or epoch is None or not camera_stable:
            self.clear()
            reason = "画面移动，暂停运动和转向判断" if not camera_stable else "当前目标或躯干不可观察"
            result.update({k: axis(k, reason=reason) for k in TEXT})
            result["quality"]["reset_reason"] = reset_reason or (
                "camera_motion" if not camera_stable else "missing_target_or_torso"
            )
            return result
        result["quality"].update(
            observed=True,
            torso=True,
            feet=row["feet"],
            bilateral=row["widths"] is not None,
            reset_reason=reset_reason,
        )
        label = action.get("label", "unknown")
        if action.get("raw_label") == "falling":
            self.clear()
            result.update({k: axis(k, reason="原始跌倒投票冲突，独立行为分支暂不确认") for k in TEXT})
            result["quality"]["reset_reason"] = "fall_vote_conflict"
            return result
        mapping = {"standing": "upright", "seated": "seated", "bending": "bending", "squatting": "squatting"}
        if label in mapping:
            result["posture"] = axis("posture", mapping[label], "保留当前已确认姿态")
        elif row["upright"] and label not in {"standing_up", "sitting_down"}:
            result["posture"] = axis("posture", "upright", "当前可见躯干直立、髋高于脚且腿部展开；不要求静止")
        if label in {"standing_up", "sitting_down"}:
            result["transition"] = axis("transition", label, "保留当前起坐结果")
        elif result["posture"]["label"] != "unknown":
            result["transition"] = axis("transition", "none", "当前未确认起坐转换")
        # Retain only contiguous observations of the same visible body sides.
        if self.history and row["sides"] != self.history[-1]["sides"]:
            self.clear()
            result["quality"]["reset_reason"] = "visible_side_changed"
        row["eligible"] = result["posture"]["label"] == "upright" and label not in {
            "standing_up",
            "sitting_down",
        }
        self.history.append(row)
        while self.history and t - self.history[0]["t"] > 1.2:
            self.history.popleft()
        recent = [r for r in self.history if t - r["t"] <= 0.55]
        eligible = row["eligible"]
        if not eligible:
            self.walk_since = self.last_walk_t = self.walk_candidate_since = None
            # Body translation for non-upright tasks is not walking.
            reason = "当前处于起坐、弯腰或蹲姿活动，未确认直立移动"
            result["locomotion"] = axis("locomotion", reason=reason)
            result["turning"] = axis("turning", reason="当前非可确认直立姿态，不判断站立或行走转身")
            if (
                len(recent) >= 3
                and t - recent[0]["t"] >= 0.24
                and label in {"seated", "bending", "squatting"}
            ):
                q = recent[0]
                displacement = (
                    max(
                        np.linalg.norm(row["hips"] - q["hips"]),
                        np.linalg.norm(row["shoulders"] - q["shoulders"]),
                    )
                    / row["scale"]
                )
                if displacement < 0.045:
                    result["locomotion"] = axis(
                        "locomotion", "stationary", "当前姿态下躯干位置稳定；不推断脚部支撑状态"
                    )
            return result
        # Movement performed during rising, descending or bending cannot be
        # borrowed to declare walking on the first upright observation.
        for i in range(len(recent) - 1, -1, -1):
            if not recent[i]["eligible"]:
                recent = recent[i + 1 :]
                break
        if len(recent) < 3 or t - recent[0]["t"] < 0.24:
            return result
        first = recent[0]
        scale = float(np.median([r["scale"] for r in recent]))
        dt = t - first["t"]
        hip_delta = (row["hips"] - first["hips"]) / scale
        shoulder_delta = (row["shoulders"] - first["shoulders"]) / scale
        body_distance = float(np.linalg.norm(hip_delta))
        coherent = float(np.linalg.norm(hip_delta - shoulder_delta)) < 0.15
        depth_change = abs(float(np.log(row["scale"] / first["scale"])))
        travel = max(body_distance, depth_change)
        feet_rows = all(r["feet"] for r in recent)
        foot_motion = articulation = 0.0
        fresh_motion = False
        latest_foot_speed = 0.0
        if feet_rows:
            foot_motion = float(
                np.max(np.linalg.norm((row["p"][[15, 16], :2] - first["p"][[15, 16], :2]) / scale, axis=1))
            )
            # Inter-ankle changes remove shared image translation. Normalizing
            # each observation also removes simple approach/retreat scaling.
            spread = np.array([(r["p"][15, :2] - r["p"][16, :2]) / r["scale"] for r in recent])
            articulation = float(np.max(np.linalg.norm(spread - spread[0], axis=1)))
            fresh = [r for r in recent if t - r["t"] <= 0.27]
            q = fresh[0]
            fresh_travel = max(
                float(np.linalg.norm(row["hips"] - q["hips"]) / scale),
                abs(float(np.log(row["scale"] / q["scale"]))),
            )
            fresh_feet = float(
                np.max(np.linalg.norm((row["p"][[15, 16], :2] - q["p"][[15, 16], :2]) / scale, axis=1))
            )
            fresh_motion = bool(t - q["t"] >= 0.10 and fresh_travel > 0.018 and fresh_feet > 0.025)
            latest_foot_speed = float(
                np.max(
                    np.linalg.norm((row["p"][[15, 16], :2] - recent[-2]["p"][[15, 16], :2]) / scale, axis=1)
                )
                / (t - recent[-2]["t"])
            )
        # Low image displacement alone cannot prove a stop during approach /
        # retreat: ongoing apparent-scale change remains movement evidence.
        latest_hip_speed = float(
            np.linalg.norm(row["hips"] - recent[-2]["hips"]) / scale / (t - recent[-2]["t"])
        )
        fresh_hip = float(np.linalg.norm(row["hips"] - q["hips"]) / scale) if feet_rows else 0.0
        settling = bool(feet_rows and latest_hip_speed < 0.08 and fresh_hip < 0.04 and depth_change < 0.05)
        m = dict(
            latest_hip_speed=latest_hip_speed,
            fresh_hip=fresh_hip,
            settling=settling,
            window_s=dt,
            body_displacement=body_distance,
            depth_scale_change=depth_change,
            foot_displacement=foot_motion,
            inter_ankle_change=articulation,
            coherent_translation=coherent,
            fresh_step_motion=fresh_motion,
            latest_foot_speed=latest_foot_speed,
        )
        # Paired projected shoulder/pelvis changes discriminate body rotation
        # from translation, head turns and isolated shoulder landmark noise.
        turn_rows = [r for r in recent if r["widths"] is not None]
        turning = False
        if len(turn_rows) == len(recent):
            delta = row["widths"] - first["widths"]
            increments = np.diff(np.array([r["widths"] for r in turn_rows]), axis=0)
            agreement = float(np.mean(increments[:, 0] * increments[:, 1] >= 0))
            recent_delta = row["widths"] - recent[-2]["widths"]
            raw_delta = (
                np.array(
                    [
                        (row["p"][6, 0] - row["p"][5, 0]) - (first["p"][6, 0] - first["p"][5, 0]),
                        (row["p"][12, 0] - row["p"][11, 0]) - (first["p"][12, 0] - first["p"][11, 0]),
                    ]
                )
                / scale
            )
            progressive = sum(d[0] * delta[0] > 0.001 and d[1] * delta[1] > 0.0004 for d in increments)
            turning = bool(
                abs(delta[0]) > 0.14
                and abs(delta[1]) > 0.075
                and delta[0] * delta[1] > 0
                and agreement >= 0.66
                and abs(raw_delta[0]) > 0.09
                and abs(raw_delta[1]) > 0.045
                and raw_delta[0] * delta[0] > 0
                and raw_delta[1] * delta[1] > 0
                and progressive >= 2
                and abs(recent_delta[0]) > 0.015
                and abs(recent_delta[1]) > 0.008
            )
            m.update(
                shoulder_projection_change=float(delta[0]),
                pelvis_projection_change=float(delta[1]),
                rotation_agreement=agreement,
            )
            result["turning"] = axis(
                "turning",
                "turning" if turning else "not_turning",
                "肩线和骨盆投影连续同向改变，支持身体转身；不估计真实转角"
                if turning
                else "当前窗口未见肩线与骨盆一致的转身证据",
            )
        else:
            result["turning"] = axis("turning", reason="双侧肩髋连续证据不足，不能用头部或单侧变化代替转身")
        walking = bool(
            feet_rows
            and coherent
            and fresh_motion
            and travel > 0.09
            and foot_motion > 0.10
            and articulation > 0.10
        )
        strong_walk = walking
        # An established gait may briefly take smaller projected steps. A fixed
        # last-strong-observation deadline prevents weak continuation renewing
        # itself; genuine current feet motion and pelvis travel remain required.
        continued_walk = bool(
            self.walk_since is not None
            and self.last_strong_walk_t is not None
            and t - self.last_strong_walk_t <= 0.65
            and t - self.walk_since >= 0.40
            and feet_rows
            and coherent
            and fresh_motion
            and body_distance > 0.045
            and foot_motion > 0.12
            and articulation > 0.12
        )
        # Do not carry a previous half-second of footsteps into the instant
        # after the feet stopped. Normalize by elapsed source time, not FPS.
        walking = bool(
            (walking or continued_walk)
            and not settling
            and (self.walk_since is None or latest_foot_speed > 0.10)
        )
        m["bounded_walk_continuation"] = bool(continued_walk and not strong_walk and walking)
        # Keep only recent established evidence through one brief projection
        # mismatch. This frame still abstains; a later real strong frame can
        # resume without another onset timer. Missing joints/targets clear it.
        transient_shape_change = bool(
            not coherent
            and self.last_walk_t is not None
            and t - self.last_walk_t <= 0.30
            and feet_rows
            and fresh_motion
            and body_distance > 0.075
            and foot_motion > 0.25
            and articulation > 0.25
        )
        m["transient_shape_change"] = transient_shape_change
        if walking:
            if strong_walk:
                self.last_strong_walk_t = t
            if self.walk_candidate_since is None:
                self.walk_candidate_since = t
            if t - self.walk_candidate_since < 0.10:
                result["locomotion"] = axis(
                    "locomotion", "moving", "当前有步态变化，等待下一次真实观测确认行走"
                )
            else:
                if self.walk_since is None:
                    self.walk_since = t
                self.last_walk_t = t
                state = "starting" if t - self.walk_since < 0.26 else "walking"
                result["locomotion"] = axis(
                    "locomotion",
                    state,
                    "已确认行走；当前小步位移和双脚步态持续，短时衔接"
                    if continued_walk and not strong_walk
                    else "躯干连续移动且双脚位置关系发生步态变化",
                )
        elif (
            settling
            and self.last_walk_t is not None
            and t - self.last_walk_t <= 0.65
            and (travel >= 0.045 or foot_motion >= 0.075)
        ):
            self.walk_candidate_since = self.walk_since = None
            result["locomotion"] = axis(
                "locomotion", "stopping", "身体行进已减弱，当前脚位调整按停步观察，不沿用旧窗口行走"
            )
        elif coherent and travel > 0.075:
            self.walk_candidate_since = None
            if not (self.last_walk_t is not None and t - self.last_walk_t <= 0.25 and fresh_motion):
                self.walk_since = None
            result["locomotion"] = axis("locomotion", "moving", "观察到躯干位移，但连续双脚步态证据不足")
        elif travel < 0.045 and (not feet_rows or foot_motion < 0.075):
            self.walk_candidate_since = None
            self.walk_since = None
            stopping = self.last_walk_t is not None and t - self.last_walk_t <= 0.45
            if not feet_rows:
                result["locomotion"] = axis(
                    "locomotion", "unknown", "躯干位置稳定，但脚步不可完整观察，不能确认静止或停步"
                )
            else:
                result["locomotion"] = axis(
                    "locomotion",
                    "stopping" if stopping else "stationary",
                    "行走后位移减小，正在确认停步" if stopping else "当前短窗口躯干位移较小，脚部位置稳定",
                )
        else:
            if not transient_shape_change:
                self.walk_candidate_since = None
                self.walk_since = None
            result["locomotion"] = axis(
                "locomotion",
                "moving" if feet_rows and foot_motion > 0.085 else "unknown",
                "存在脚位调整或位移，尚不足以确认连续行走",
            )
        # Travel direction is separate from body rotation. No path claim at
        # rest, during depth-only travel, or without two substantial segments.
        path = [r for r in self.history if t - r["t"] <= 1.1]
        for i in range(len(path) - 1, -1, -1):
            if not path[i]["eligible"]:
                path = path[i + 1 :]
                break
        current_translation = (
            result["locomotion"]["label"] in {"starting", "walking", "moving"} and coherent and travel > 0.075
        )
        if current_translation and len(path) >= 6 and t - path[0]["t"] >= 0.65:
            middle = path[len(path) // 2]
            v1 = (middle["hips"] - path[0]["hips"]) / scale
            v2 = (row["hips"] - middle["hips"]) / scale
            if min(np.linalg.norm(v1), np.linalg.norm(v2)) > 0.12:
                cosine = float(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))
                result["path_change"] = axis(
                    "path_change",
                    "changing" if cosine < 0.65 else "straight",
                    "两个连续图像位移段的方向比较；不是身体朝向或地面真实转角",
                )
                m["path_direction_cosine"] = cosine
        result["metrics"] = {
            k: round(v, 5) if isinstance(v, (float, np.floating)) else v for k, v in m.items()
        }
        return result

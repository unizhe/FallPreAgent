"""Causal COCO17 adapter for the frozen OpenMMLab ST-GCN++ NTU60 model.

This module is an additional evidence branch. It never changes existing action
labels or fall events. Closed-set classification scores are not fall risks.
Sampling uses only observations already available at the window end; it never
wraps the end of a short motion back to its beginning.
"""
from __future__ import annotations

import hashlib
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fallpre.runtime_resources import action_dependency_path, resource_path

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = 'stgcnpp-ntu60-2d-coco17-joint'
MODEL_ROOT = ROOT / 'third_party/algorithms/mmaction2'
DEFAULT_CONFIG = MODEL_ROOT / 'configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'
DEFAULT_CHECKPOINT = ROOT / 'models/stgcn/stgcnpp-ntu60-2d-joint.pth'
DEFAULT_LABELS = MODEL_ROOT / 'tools/data/skeleton/label_map_ntu60.txt'
CHECKPOINT_SHA256 = '86e1e77a7b27dd53c6ccc1600a4cc3c321ec8e04d17f25a41fbbb7896026945a'
CONFIG_SHA256 = '31bd05179ebe52ccb9ecb194f2e9e8a804b4765d2172a124b646bdc169fd1c82'
LABEL_MAP_SHA256 = '4b6fc7042fd3a58e15eb70b8d021e2b6c9b2ca2ca996df2b7e98434189b7f0b5'
PREPROCESSING_HOLD = 'coco17-xy-score/image-normalized/zero-person2/past-hold100/single-clip-v1'
PREPROCESSING = 'coco17-xy-score/image-normalized/zero-person2/linear-supported100/single-clip-v1'
BODY = np.arange(5, 17)
RELEVANT_CLASSES = {'pickup': 5, 'sitting_down': 7, 'standing_up': 8,
                    'bow': 34, 'staggering': 41, 'falling': 42}
LABELS_ZH = dict(enumerate([
    '喝水', '吃东西', '刷牙', '梳头', '丢下物品', '捡东西', '投掷', '坐下', '从坐姿起身',
    '拍手', '阅读', '写字', '撕纸', '穿外套', '脱外套', '穿鞋', '脱鞋', '戴眼镜', '摘眼镜',
    '戴帽子', '摘帽子', '欢呼', '挥手', '踢物品', '伸手入口袋', '单脚跳', '跳起', '打电话',
    '使用手机 / 平板', '键盘打字', '指向物品', '自拍', '看手表', '搓手', '点头 / 鞠躬',
    '摇头', '擦脸', '敬礼', '双掌合十', '双手交叉示停', '打喷嚏 / 咳嗽', '踉跄', '跌倒动作',
    '摸头', '摸胸部', '摸背', '摸颈部', '恶心 / 呕吐动作', '扇风', '击打他人', '踢他人',
    '推他人', '拍他人背部', '指向他人', '拥抱', '递物品', '触碰他人口袋', '握手',
    '两人相向行走', '两人相背行走',
]))


@dataclass(frozen=True)
class TemporalConfig:
    window_s: float = 2.5
    min_span_s: float = 1.5
    min_samples: int = 8
    max_gap_s: float = .45
    inference_interval_s: float = .4
    sample_count: int = 100
    max_result_age_s: float = 1.0
    joint_confidence: float = .25
    min_body_joints: int = 6
    min_body_score: float = .4
    resampling: str = 'linear_supported'

    def __post_init__(self):
        finite = all(math.isfinite(v) for v in (
            self.window_s, self.min_span_s, self.min_samples, self.max_gap_s,
            self.inference_interval_s, self.max_result_age_s, self.joint_confidence,
            self.min_body_score, self.min_body_joints))
        if not (finite and self.window_s >= self.min_span_s > 0 and self.min_samples >= 2
                and self.max_gap_s > 0 and self.inference_interval_s > 0
                and self.sample_count == 100 and self.max_result_age_s > 0
                and 0 <= self.joint_confidence <= 1
                and 0 <= self.min_body_score <= 1 and 1 <= self.min_body_joints <= 12
                and self.resampling in {'past_hold', 'linear_supported'}):
            raise ValueError('Invalid ST-GCN++ temporal configuration')


@dataclass(frozen=True)
class PreparedWindow:
    """An immutable snapshot suitable for inference in a background worker."""

    keypoints: np.ndarray  # Real observations: (T, 17, 3), pixel x/y + score.
    source_times: np.ndarray
    width: int
    height: int
    target_epoch: Any
    generation: int
    metadata: dict


class TemporalPoseBuffer:
    """One associated person's real observations, bounded by source time.

    The caller supplies the existing track epoch; this adapter never selects a
    larger box or borrows another person's skeleton. Missing/ambiguous current
    observations suppress predictions immediately, while a gap of at most
    max_gap_s may retain history for the same confirmed identity.
    """

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
        self.samples = deque()
        self.generation = 0
        self.target_epoch = None
        self.dimensions = None
        self.last_t = None
        self.last_prepared_t = None
        self.current_valid = False
        self._status = self._make_status('warming_up', '等待连续人体骨架')

    @property
    def status(self):
        return dict(self._status)

    def _make_status(self, state, reason):
        span = self.samples[-1][0] - self.samples[0][0] if len(self.samples) > 1 else 0.
        return dict(model_id=MODEL_ID, status=state, reason=reason,
                    ready=state == 'ready', generation=self.generation,
                    target_epoch=self.target_epoch, source_time=self.last_t,
                    observations=len(self.samples), span_s=round(span, 6),
                    required_span_s=self.config.min_span_s,
                    required_observations=self.config.min_samples)

    def reset(self, reason='时序证据已重置'):
        self.samples.clear()
        self.generation += 1
        self.target_epoch = None
        self.dimensions = None
        self.last_prepared_t = None
        self.current_valid = False
        self._status = self._make_status('warming_up', reason)

    def _unavailable(self, state, reason):
        self.current_valid = False
        self._status = self._make_status(state, reason)
        return self.status

    def observe(self, person, t, width, height, target_epoch, camera_stable=True):
        t = float(t)
        if not math.isfinite(t) or t < 0 or self.last_t is not None and t <= self.last_t:
            self.reset('输入时间不递增，已清除时序证据')
            raise ValueError('ST-GCN++ source timestamps must be finite and strictly increasing')
        self.last_t = t
        if (not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer))
                or width <= 0 or height <= 0):
            self.reset('输入尺寸无效')
            return self._unavailable('invalid_input', '输入尺寸无效')
        dimensions = int(width), int(height)
        if not camera_stable:
            self.reset('摄像头移动，重新积累当前机位的骨架')
            return self._unavailable('camera_unstable', '摄像头移动，时序证据暂停')
        if self.samples and t - self.samples[-1][0] > self.config.max_gap_s + 1e-9:
            self.reset('目标观测间隔过长，未拼接中断前后的动作')
        if self.dimensions is not None and self.dimensions != dimensions:
            self.reset('输入画面尺寸发生变化')
        if target_epoch is not None and self.target_epoch is not None and target_epoch != self.target_epoch:
            self.reset('目标身份发生变化')
        if person is None or target_epoch is None:
            return self._unavailable('target_unavailable', '当前目标未确认或不可见')
        try:
            points = np.asarray(person['keypoints'], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return self._unavailable('pose_unreliable', '骨架数据无效')
        if points.shape != (17, 3) or not np.isfinite(points).all():
            return self._unavailable('pose_unreliable', '需要完整的 COCO17 点格式与有效数值')
        if np.any((points[:, 2] < 0) | (points[:, 2] > 1)):
            return self._unavailable('pose_unreliable', '关键点置信度范围无效')
        visible = ((points[:, 2] >= self.config.joint_confidence)
                   & (points[:, 0] > 0) & (points[:, 0] < width)
                   & (points[:, 1] > 0) & (points[:, 1] < height))
        body_visible = int(visible[BODY].sum())
        body_score = float(points[BODY, 2].mean())
        if (body_visible < self.config.min_body_joints or body_score < self.config.min_body_score
                or not visible[[5, 6]].any() or not visible[[11, 12]].any()):
            return self._unavailable('pose_unreliable', '当前躯干与四肢可见证据不足')
        self.target_epoch, self.dimensions = target_epoch, dimensions
        # All finite original points/scores are retained, including weak joints.
        self.samples.append((t, points.copy(), body_visible, body_score))
        while self.samples and self.samples[0][0] < t - self.config.window_s - 1e-9:
            self.samples.popleft()
        self.current_valid = True
        span = self.samples[-1][0] - self.samples[0][0]
        ready = len(self.samples) >= self.config.min_samples and span >= self.config.min_span_s - 1e-9
        self._status = self._make_status('ready' if ready else 'warming_up',
                                         '连续骨架可供时序分类' if ready else '正在积累连续动作窗口')
        return self.status

    def prepare(self, force=False):
        """Return a past-only snapshot, at most once per inference interval."""
        if not self.current_valid or not self._status['ready']:
            return None
        if (not force and self.last_prepared_t is not None
                and self.last_t - self.last_prepared_t < self.config.inference_interval_s - 1e-9):
            return None
        rows = list(self.samples)
        times = np.asarray([r[0] for r in rows], dtype=np.float64)
        points = np.stack([r[1] for r in rows]).astype(np.float32)
        points.setflags(write=False)
        times.setflags(write=False)
        metadata = dict(start_s=float(times[0]), end_s=float(times[-1]),
                        span_s=float(times[-1] - times[0]), observations=len(rows),
                        observed_hz=float((len(rows)-1) / (times[-1]-times[0])),
                        max_observation_gap_s=float(np.max(np.diff(times))),
                        min_visible_body_joints=min(r[2] for r in rows),
                        mean_body_score=float(np.mean([r[3] for r in rows])),
                        requested_window_s=self.config.window_s,
                        sample_count=self.config.sample_count,
                        resampling=self.config.resampling,
                        causal=True, temporal_wrap=False)
        self.last_prepared_t = self.last_t
        return PreparedWindow(points, times, *self.dimensions, self.target_epoch,
                              self.generation, metadata)

    def is_result_current(self, result, t=None):
        """Reject stale async results after gaps, track changes or source seeks."""
        now = self.last_t if t is None else float(t)
        if not self.current_valid or now is None or not isinstance(result, dict):
            return False
        end = result.get('window', {}).get('end_s')
        return bool(result.get('generation') == self.generation
                    and result.get('target_epoch') == self.target_epoch
                    and isinstance(end, (int, float)) and math.isfinite(end)
                    and 0 <= now-end <= self.config.max_result_age_s + 1e-9)


def prepare_tensor_hold(window: PreparedWindow):
    """Return (1, 1, 2, 100, 17, 3), matching the checkpoint's joint stream.

    Coordinates use official PreNormalize2D, scores use GenSkeFeat(j), and the
    absent second person uses FormatGCNInput's zero padding. The only intentional
    test-pipeline change is one deterministic physical-time sample sequence.
    Low-FPS inputs repeat earlier measured poses; unique counts disclose this.
    """
    times = window.source_times
    if (len(times) < 2 or np.any(np.diff(times) <= 0)
            or window.keypoints.shape != (len(times), 17, 3)):
        raise ValueError('Invalid temporal window')
    grid = np.linspace(times[0], times[-1], 100)
    indices = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times) - 1)
    samples = window.keypoints[indices].copy()
    samples[..., 0] = (samples[..., 0] - window.width / 2) / (window.width / 2)
    samples[..., 1] = (samples[..., 1] - window.height / 2) / (window.height / 2)
    tensor = np.zeros((1, 1, 2, 100, 17, 3), dtype=np.float32)
    tensor[0, 0, 0] = samples
    metadata = dict(window.metadata, unique_sampled_observations=int(np.unique(indices).size),
                    repeated_samples=int(100 - np.unique(indices).size),
                    sampling='uniform source-time grid, preceding observation; no interpolation or wrap',
                    preprocessing=PREPROCESSING_HOLD, resampling='past_hold')
    return tensor, metadata


def prepare_tensor_linear(window: PreparedWindow):
    """Estimate supported joint samples inside history already received.

    Interpolation is not an additional camera observation and does not repair
    missing joints. Both adjacent endpoints must be visible, sufficiently
    confident, and at most .45 s apart. No future endpoint or extrapolation is
    used. Scores take the lower endpoint confidence, never an invented increase.
    """
    _, metadata = prepare_tensor_hold(window)
    times = window.source_times
    grid = np.linspace(times[0], times[-1], 100)
    left = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times)-1)
    right = np.minimum(left+1, len(times)-1)
    gaps = times[right] - times[left]
    alpha = np.divide(grid-times[left], gaps, out=np.zeros_like(grid), where=gaps > 0)
    a, b = window.keypoints[left], window.keypoints[right]

    def valid(points):
        return (np.isfinite(points).all(-1) & (points[..., 2] >= .25)
                & (points[..., 0] > 0) & (points[..., 0] < window.width)
                & (points[..., 1] > 0) & (points[..., 1] < window.height))

    interpolate = (valid(a) & valid(b) & (gaps[:, None] <= .45)
                   & (alpha[:, None] > 1e-8) & (alpha[:, None] < 1-1e-8))
    samples = a.copy()
    xy = a[..., :2] + alpha[:, None, None] * (b[..., :2]-a[..., :2])
    samples[..., :2] = np.where(interpolate[..., None], xy, samples[..., :2])
    samples[..., 2] = np.where(interpolate, np.minimum(a[..., 2], b[..., 2]), samples[..., 2])
    samples[..., 0] = (samples[..., 0]-window.width/2) / (window.width/2)
    samples[..., 1] = (samples[..., 1]-window.height/2) / (window.height/2)
    tensor = np.zeros((1, 1, 2, 100, 17, 3), dtype=np.float32)
    tensor[0, 0, 0] = samples
    metadata.update(
        preprocessing=PREPROCESSING, resampling='linear_supported',
        sampling='Linear interpolation within already observed history; supported joints only, max gap .45s',
        preceding_observation_reuses=metadata.pop('repeated_samples'),
        interpolated_sample_count=int(interpolate.any(1).sum()),
        interpolated_joint_count=int(interpolate.sum()), real_observations=len(times),
        synthetic_samples_are_not_observations=True)
    return tensor, metadata


def prepare_tensor(window: PreparedWindow):
    """Dispatch a declared protocol; old manual snapshots remain reproducible."""
    protocol = window.metadata.get('resampling', 'past_hold')
    if protocol == 'linear_supported':
        return prepare_tensor_linear(window)
    if protocol == 'past_hold':
        return prepare_tensor_hold(window)
    raise ValueError('Unknown ST-GCN++ resampling protocol')


class STGCNRuntime:
    """Frozen CPU model. Instantiate once; calls are serialized for safety."""

    def __init__(self, checkpoint=None, config=None, labels=None, threads=2):
        self.lock = threading.Lock()
        checkpoint = resource_path('stgcn-ntu60') if checkpoint is None else checkpoint
        config = resource_path('mmaction-config') if config is None else config
        labels = resource_path('ntu60-labels') if labels is None else labels
        checkpoint, config, labels = Path(checkpoint), Path(config), Path(labels)
        for path in (checkpoint, config, labels):
            if not path.is_file():
                raise FileNotFoundError(f'ST-GCN++ required artifact missing: {path}')
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checksum != CHECKPOINT_SHA256:
            raise ValueError('ST-GCN++ checkpoint is not the verified NTU60 2D COCO17 joint weight')
        config_checksum = hashlib.sha256(config.read_bytes()).hexdigest()
        label_checksum = hashlib.sha256(labels.read_bytes()).hexdigest()
        if config_checksum != CONFIG_SHA256 or label_checksum != LABEL_MAP_SHA256:
            raise ValueError('ST-GCN++ configuration or labels differ from the verified COCO17 joint model')
        # Optional MMAction dependencies remain in a dedicated import directory;
        # this does not install/downgrade the running YOLO environment.
        for path in (MODEL_ROOT, action_dependency_path()):
            if path is not None and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import torch
        from mmaction.apis import init_recognizer
        from mmengine.runner import load_checkpoint

        self.torch = torch
        torch.set_num_threads(threads)
        self.model = init_recognizer(str(config), device='cpu')
        load_checkpoint(self.model, str(checkpoint), map_location='cpu', strict=True)
        self.labels = labels.read_text().splitlines()
        if len(self.labels) != 60:
            raise ValueError('Expected 60 NTU60 labels')
        self.metadata = dict(id=MODEL_ID, label='ST-GCN++ · NTU60 2D / COCO17 / joint',
                             device='cpu', sha256=checksum, weights_sha256=checksum,
                             config_sha256=config_checksum,
                             labels_sha256=label_checksum,
                             parameters=sum(p.numel() for p in self.model.parameters()),
                             preprocessing=PREPROCESSING, training=False,
                             score_semantics='closed-set NTU60 classification; not fall probability')

    def predict(self, window: PreparedWindow):
        tensor, metadata = prepare_tensor(window)
        start = time.perf_counter()
        with self.lock, self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(tensor), mode='tensor', stage='head')
            scores = logits.softmax(-1)[0].cpu().numpy()
        elapsed = (time.perf_counter() - start) * 1000
        if scores.shape != (60,) or not np.isfinite(scores).all():
            raise RuntimeError('ST-GCN++ returned invalid class scores')
        order = np.argsort(-scores, kind='stable')
        top3 = [dict(index=int(i), label=self.labels[i], label_zh=LABELS_ZH.get(int(i), self.labels[i]),
                     score=float(scores[i])) for i in order[:3]]
        return dict(model_id=MODEL_ID, status='ready', generation=window.generation,
                    target_epoch=window.target_epoch, source_time=float(window.source_times[-1]),
                    source_time_s=float(window.source_times[-1]),
                    window_start_s=metadata['start_s'], window_end_s=metadata['end_s'],
                    observed_frames=metadata['observations'],
                    window=metadata, top3=top3, all_scores=[float(v) for v in scores],
                    relevant_scores={name: float(scores[i]) for name, i in RELEVANT_CLASSES.items()},
                    inference_ms=elapsed, score_semantics=self.metadata['score_semantics'],
                    model=dict(self.metadata, preprocessing=metadata['preprocessing'],
                               resampling=metadata['resampling']))

"""Local camera/video test page with switchable pose models and fall events."""
import argparse
import hashlib
import json
import math
import re
import secrets
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/ultralytics'))
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from fallpre.runtime_resources import resource_path
from fallpre.weekly.action_v7 import CompletionModel
from fallpre.weekly.perception_semantics import decision_context
from fallpre.weekly.webcam_actions import LiveActions
from fallpre.weekly.webcam_live_setup import LiveSetup
from fallpre.weekly.webcam_behavior import BehaviorAxes, apply_auxiliary_bend
from fallpre.weekly.webcam_fall import LiveFallDetector
from fallpre.weekly.webcam_fall_handoff import LiveFallHandoff
from fallpre.weekly.webcam_fall_target import LiveFallSubject
from fallpre.weekly.webcam_pose_recovery import LivePoseRecovery
from fallpre.weekly.webcam_quality import LiveCameraQuality
from fallpre.weekly.webcam_rgb_fall import LiveRgbFall
from fallpre.weekly.webcam_rgb_model import RgbFallBackend, model_available, model_spec
from fallpre.weekly.webcam_risk_points import LiveRiskTransitionPoints

POSE_MODELS = {
    'yolo26n-pose': 'YOLO26n-Pose',
    'yolo11n-pose': 'YOLO11n-Pose',
}
ACTION_PIPELINE = dict(id='live-actions-v4', label='实时动作：正面弯腰参照、倒卧观察与机位检查',
                       revision='2026-10-28.19')
# The recording protocol revision remains compatible with the frozen UI.
# Capture the implementation at process startup so a live status response can
# distinguish an old loaded service from newer files waiting for a restart.
_implementation_files = [Path(__file__)] + sorted((ROOT / 'src/fallpre/weekly').glob('*.py'))
ACTION_PIPELINE.update(
    implementation_revision='2026-11-09.output-separation',
    implementation_sha256=hashlib.sha256(b''.join(
        str(p.relative_to(ROOT)).encode() + b'\0' + p.read_bytes()
        for p in _implementation_files)).hexdigest())
COMPARISON_REPORT = ROOT / 'runs/low-fall-fix-20261027/index.html'
COMPARISON_REPORT_URL = 'http://127.0.0.1:8799/'


class LiveSession:
    def __init__(self, action_model, model_id, source_type, fall_mode='pose', temporal_service=None,
                 fusion_enabled=False, fall_temporal_enabled=False, fall_event_fusion=False):
        self.actions = LiveActions(action_model)
        self.behavior = BehaviorAxes()
        self.risk_points = LiveRiskTransitionPoints()
        self.fall = LiveFallDetector()
        self.fall_target = LiveFallSubject()
        self.pose_recovery = LivePoseRecovery()
        self.quality = LiveCameraQuality()
        self.live_setup = LiveSetup()
        self.model_id = model_id
        self.source_type = source_type
        self.fall_mode = fall_mode
        self.temporal = temporal_service.session() if temporal_service is not None else None
        self.fusion = None
        if fusion_enabled:
            from fallpre.weekly.stgcn_fusion import ConstrainedSTGCNFusion
            self.fusion = ConstrainedSTGCNFusion()
        self.fall_temporal = None
        if fall_temporal_enabled or fall_event_fusion:
            from fallpre.weekly.stgcn_fall_evidence import FallTemporalEvidence
            self.fall_temporal = FallTemporalEvidence()
        self.fall_event_fusion = None
        if fall_event_fusion:
            from fallpre.weekly.fall_event_fusion import TemporalFallEventFusion
            self.fall_event_fusion = TemporalFallEventFusion()
        self.rgb_fall = (LiveFallHandoff() if (model_spec() or {}).get('strategy') == 'verified_handoff'
                         else LiveRgbFall()) if fall_mode == 'rgb' else None

    @property
    def last_t(self):
        return self.actions.last_t

    def close(self):
        if self.temporal is not None:
            self.temporal.close()

    def update(self, image, result, source_time, recover_infer=None, rgb_result=None):
        if self.fall_mode == 'rgb' and (
                not isinstance(rgb_result, dict) or not isinstance(rgb_result.get('detections'), list)
                or not isinstance(rgb_result.get('model'), dict)
                or not isinstance(rgb_result.get('inference_ms'), (int, float))
                or not math.isfinite(rgb_result['inference_ms']) or rgb_result['inference_ms'] < 0):
            raise ValueError('独立倒地模式缺少有效的本帧图像检测结果')
        if isinstance(self.rgb_fall, LiveFallHandoff) and (
                not isinstance(rgb_result.get('raw_candidates'), dict)
                or not all(isinstance(rgb_result['raw_candidates'].get(k), list) for k in ('persons', 'fallen'))):
            raise ValueError('独立倒地模式缺少当帧双检测候选')
        quality = self.quality.update(image, result['persons'], source_time)
        action = self.actions.update(result['persons'], source_time, result['width'], result['height'],
                                     quality['camera_stable'])
        selection = self.fall_target.prepare(result['persons'], source_time, result['width'], result['height'],
                                             camera_stable=quality['camera_stable'])
        recovery = dict(executed=False, accepted=False, inference_passes=0, elapsed_ms=0.,
                        reason=selection['reason_code'])
        if selection['recovery_allowed'] and recover_infer is not None:
            proposals, recovery = self.pose_recovery.recover(image, selection['anchor_box'], recover_infer)
            selection = self.fall_target.finish(proposals, source_time, result['width'], result['height'])
            recovery['accepted'] = selection['source'] == 'recovered'
            if proposals and not recovery['accepted']:
                recovery['reason'] = selection['reason_code']
        observation = dict(person=selection['person'], source=selection['source'], recovery=recovery,
                           selection={key: selection[key] for key in
                                      ('raw_index', 'target_epoch', 'state', 'reason', 'reason_code')})
        same_action_target = selection['source'] == 'raw' and selection['raw_index'] == action['target_index']
        fall = self.fall.update(selection['person'], source_time, result['width'], result['height'],
                                selection['target_epoch'], raw_label=action['raw_label'] if same_action_target else 'unknown',
                                camera_stable=quality['camera_stable'], quality_reason=quality['reason'])
        fall['summary'] = self.fall.export()
        result.update(action=action, fall=fall, quality=quality, fall_observation=observation,
                      recovery_ms=recovery['elapsed_ms'], source_type=self.source_type, source_time=source_time,
                      fall_mode=self.fall_mode)
        if self.fall_mode == 'rgb':
            current = self.rgb_fall.update(
                rgb_result['raw_candidates'] if isinstance(self.rgb_fall, LiveFallHandoff) else rgb_result['detections'],
                source_time, result['width'], result['height'],
                pose_person=selection['person'], pose_epoch=selection['target_epoch'],
                camera_stable=quality['camera_stable'], quality_reason=quality['reason'])
            result.update(pose_fall=fall, pose_fall_observation=observation,
                          fall=current['fall'], fall_observation=current['observation'],
                          rgb_diagnostics=current['diagnostics'], rgb_detections=rgb_result['detections'],
                          rgb_raw_candidates=rgb_result.get('raw_candidates', {}),
                          rgb_inference_ms=rgb_result['inference_ms'], fall_model=rgb_result['model'])
            result['fall']['summary'] = self.rgb_fall.export()

        result['live_setup'], result['lying_observation'] = self.live_setup.update(result, source_time, rgb_result)

        # Independent descriptive axes never feed back into action or fall rules.
        # Only the current associated action subject may supply motion evidence.
        behavior_index = action.get('target_index')
        behavior_epoch = action.get('selection', {}).get('target_epoch')
        behavior_person = (result['persons'][behavior_index]
                           if isinstance(behavior_index, int)
                           and 0 <= behavior_index < len(result['persons']) else None)
        result['behavior'] = self.behavior.update(
            behavior_person, source_time, behavior_epoch, action,
            result['width'], result['height'], camera_stable=quality['camera_stable'])
        result['behavior'].update(target_index=behavior_index, target_epoch=behavior_epoch)
        if behavior_index is None:
            apply_auxiliary_bend(result['behavior'], action, result['persons'], source_time,
                                 result['width'], result['height'], quality['camera_stable'])
        result['risk_points'] = self.risk_points.update(
            source_time, action, result['behavior'],
            result['fall'] if self.fall_mode == 'rgb' else {}, result.get('rgb_diagnostics', {}),
            camera_stable=quality['camera_stable'])
        result['decision_context'] = decision_context(result, source_time)


class PoseEngine:
    def __init__(self, device='cpu', imgsz=640, model_id='yolo26n-pose', fall_mode='pose', temporal_enabled=False,
                 fusion_enabled=False, fall_temporal_enabled=False, fall_event_fusion=False):
        if fall_event_fusion and fall_mode != 'rgb':
            raise ValueError('Event fusion requires RGB handoff mode')
        fall_temporal_enabled = fall_temporal_enabled or fall_event_fusion
        self.fall_event_fusion_enabled = fall_event_fusion
        if (fusion_enabled or fall_temporal_enabled) and not temporal_enabled:
            raise ValueError('融合旁路需要启用 ST-GCN++ 时序分支')
        self.fusion_enabled = fusion_enabled
        self.fall_temporal_enabled = fall_temporal_enabled
        if model_id not in POSE_MODELS:
            raise ValueError('不支持的姿态模型')
        self.device, self.imgsz = device, imgsz
        self.default_model = model_id
        if fall_mode not in {'pose', 'rgb'}:
            raise ValueError('不支持的跌倒模式')
        self.default_fall_mode = fall_mode
        self.rgb_backend = None
        torch.set_num_threads(2)
        self.models, self.model_metadata = {}, {}
        self.action_model = CompletionModel.load(resource_path('action-v7'))
        self.streams = {}
        self.lock = threading.Lock()
        self.load_model(model_id)
        if fall_mode == 'rgb':
            self.rgb_backend = RgbFallBackend(device)
        self.temporal_service = None
        if temporal_enabled:
            from fallpre.weekly.stgcn_temporal import STGCNRuntime
            from fallpre.weekly.webcam_temporal import TemporalService
            self.temporal_service = TemporalService(STGCNRuntime())

    def temporal_status(self):
        service = getattr(self, 'temporal_service', None)
        return service.status() if service is not None else dict(enabled=False, ready=False, mode='disabled')

    def fusion_status(self):
        from dataclasses import asdict
        from fallpre.weekly.stgcn_fusion import FusionConfig
        enabled = getattr(self, 'fusion_enabled', False)
        return dict(enabled=enabled, mode='shadow' if enabled else 'disabled',
                    changes_primary_decision=False, changes_alarm=False,
                    config=asdict(FusionConfig()) if enabled else None)

    def fall_temporal_status(self):
        enabled = getattr(self, 'fall_temporal_enabled', False)
        return dict(enabled=enabled, mode='shadow' if enabled else 'disabled',
                    changes_primary_decision=False, changes_alarm=False,
                    purpose='fall-process evidence; observation quality and event ownership remain separate',
                    report_url='http://127.0.0.1:8811/')

    def fall_event_fusion_status(self):
        enabled = getattr(self, 'fall_event_fusion_enabled', False)
        return dict(enabled=enabled, mode='limited_primary' if enabled else 'disabled',
                    changes_alarm=enabled, policy='st-supported-handoff-dwell-v1',
                    baseline_confirmation_s=1., supported_confirmation_s=.6,
                    baseline_output='baseline_fall' if enabled else 'fall',
                    shared_pose_evidence=True)

    def stop_stream(self, stream_id):
        entry = self.streams.pop(stream_id, None)
        if entry is not None:
            entry[0].close()

    def close(self):
        for key in list(self.streams):
            self.stop_stream(key)
        if self.temporal_service is not None:
            self.temporal_service.close()

    def available_models(self):
        return [dict(id=name, label=label, available=(ROOT / 'models' / f'{name}.pt').is_file())
                for name, label in POSE_MODELS.items()]

    def load_model(self, model_id):
        if model_id not in POSE_MODELS:
            raise ValueError('不支持的姿态模型')
        if model_id in self.models:
            return self.models[model_id]
        weights = ROOT / 'models' / f'{model_id}.pt'
        if not weights.is_file():
            raise ValueError(f'本地缺少 {weights.name}，请先运行 fallpre download-models')
        lock = json.loads((ROOT / 'models.lock.json').read_text())
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if lock.get(weights.name, {}).get('sha256') != digest:
            raise ValueError(f'{weights.name} 校验失败，请检查本地权重')
        model = YOLO(str(weights))
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, conf=.25,
                      iou=.45, device=self.device, verbose=False)
        # Ultralytics CPU predictor setup resets torch's thread count.
        torch.set_num_threads(2)
        self.models[model_id] = model
        self.model_metadata[model_id] = dict(id=model_id, label=POSE_MODELS[model_id], sha256=digest,
                                            end2end=bool(getattr(model.model, 'end2end', False)))
        return model

    def infer(self, image, model_id=None):
        model_id = model_id or self.default_model
        model = self.load_model(model_id)
        start = time.perf_counter()
        r = model.predict(image, imgsz=self.imgsz, conf=.25, iou=.45,
                          device=self.device, verbose=False)[0]
        persons = []
        if r.keypoints is not None:
            for box, score, points in zip(r.boxes.xyxy.cpu().tolist(),
                                         r.boxes.conf.cpu().tolist(), r.keypoints.data.cpu().tolist()):
                persons.append(dict(box=box, confidence=score, keypoints=points))
        return dict(persons=persons, inference_ms=(time.perf_counter()-start)*1000,
                    width=image.shape[1], height=image.shape[0], device=self.device,
                    model=dict(self.model_metadata[model_id]))

    def infer_live(self, image, stream_id, source_time, model_id=None, source_type='camera', fall_mode=None):
        model_id = model_id or self.default_model
        fall_mode = fall_mode or getattr(self, 'default_fall_mode', 'pose')
        if model_id not in POSE_MODELS or source_type not in {'camera', 'video'} or fall_mode not in {'pose', 'rgb'}:
            raise ValueError('不支持的模型或输入来源')
        if not math.isfinite(source_time) or source_time < 0:
            raise ValueError('输入时间戳必须是非负有限数')
        now = time.monotonic()
        for key, entry in list(self.streams.items()):
            if now - entry[1] >= 30:
                self.stop_stream(key)
        if stream_id not in self.streams:
            if len(self.streams) >= 8:
                raise ValueError('摄像头会话过多，请关闭旧页面后稍候重试')
            self.streams[stream_id] = (LiveSession(self.action_model, model_id, source_type, fall_mode,
                                                 getattr(self, 'temporal_service', None),
                                                 getattr(self, 'fusion_enabled', False),
                                                 getattr(self, 'fall_temporal_enabled', False),
                                                 getattr(self, 'fall_event_fusion_enabled', False)), now)
        state, _ = self.streams[stream_id]
        if state.model_id != model_id or state.source_type != source_type or state.fall_mode != fall_mode:
            raise ValueError('切换模型、跌倒模式或输入来源前请停止并重新启动测试')
        if state.last_t is not None and source_time <= state.last_t:
            raise ValueError('重复或倒退的摄像头时间戳，请重新启动')
        result = self.infer(image, model_id)
        start = time.perf_counter()
        rgb_result = None
        if fall_mode == 'rgb':
            if self.rgb_backend is None:
                self.rgb_backend = RgbFallBackend(self.device)
            rgb_result = self.rgb_backend.infer(image)
        state.update(image, result, source_time,
                     recover_infer=(lambda view: self.infer(view, model_id)['persons']) if fall_mode == 'pose' else None,
                     rgb_result=rgb_result)
        result['action_ms'] = (time.perf_counter() - start) * 1000
        if state.temporal is not None:
            result['temporal'] = state.temporal.update(result, source_time)
        if state.fusion is not None:
            result['fused_action'] = state.fusion.update(result, source_time)
        if state.fall_temporal is not None:
            result['fall_temporal_evidence'] = state.fall_temporal.update(result, source_time)
        if state.fall_event_fusion is not None and fall_mode == 'rgb':
            candidate = state.fall_event_fusion.update(result, source_time)
            result['baseline_fall'] = result['fall']
            result['baseline_fall_observation'] = result['fall_observation']
            result['fused_fall'] = candidate
            result['fall'] = candidate['fall']
            result['fall']['summary'] = state.fall_event_fusion.manager.export()
            result['fall_observation'] = candidate['observation']
            result['tool_event'] = candidate['tool_event']
            result['decision_context'] = candidate['decision_context']
        self.streams[stream_id] = (state, now)
        return result


def decode_frame(body):
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法解码摄像头图像')
    h, w = image.shape[:2]
    if not (16 <= h <= 1080 and 16 <= w <= 1920):
        raise ValueError('图像尺寸超出范围，请重新启动摄像头')
    return image


def make_handler(engine, port, token):
    allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
    allowed_origins = {f'http://{h}' for h in allowed_hosts}
    page = (ROOT / 'assets/webcam/index.html').read_text().replace('__SESSION_TOKEN__', token)
    recording_script = (ROOT / 'assets/webcam/recording.js').read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # camera pixels and request contents are never logged

        def reply(self, code, payload, kind='application/json; charset=utf-8'):
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Permissions-Policy', 'camera=(self), microphone=()')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def valid_host(self):
            return self.headers.get('Host') in allowed_hosts

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, dict(error='只接受本机访问'))
            if self.path == '/':
                return self.reply(200, page, 'text/html; charset=utf-8')
            if self.path == '/recording.js':
                return self.reply(200, recording_script, 'text/javascript; charset=utf-8')
            if self.path == '/comparison':
                report = COMPARISON_REPORT
                if report.is_file():
                    # Keep the report and its videos on the same origin; the
                    # camera page deliberately restricts media to its origin.
                    self.send_response(302)
                    self.send_header('Location', COMPARISON_REPORT_URL)
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                return self.reply(404, dict(error='对照报告尚未生成'))
            if self.path == '/comparison-previous':
                report = ROOT / 'runs/personal-fix-20261023/index.html'
                if report.is_file():
                    return self.reply(200, report.read_text(), 'text/html; charset=utf-8')
                return self.reply(404, dict(error='上一轮对照报告不可用'))
            if self.path in ('/comparison/video.webm', '/comparison/old.webm', '/comparison/new.webm', '/comparison/natural.webm'):
                video = ROOT / 'runs/personal-capture-20261023/source/fallpre-camera-2026-10-23T12-58-43-783Z-73738905.webm'
                if self.path == '/comparison/new.webm':
                    video = ROOT / 'runs/personal-retest-20261023-233739/source/fallpre-camera-2026-10-23T15-37-39-866Z-1ce8952c.webm'
                if self.path == '/comparison/natural.webm':
                    video = ROOT / 'runs/personal-retest-20261024-145114/source/fallpre-camera-2026-10-24T06-51-14-969Z-9430e45e.webm'
                if not video.is_file():
                    return self.reply(404, dict(error='本地录像不可用'))
                self.send_response(200)
                self.send_header('Content-Type', 'video/webm')
                self.send_header('Content-Length', str(video.stat().st_size))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
                self.end_headers()
                try:
                    with video.open('rb') as handle:
                        shutil.copyfileobj(handle, self.wfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == '/status':
                return self.reply(200, dict(ready=True, model=POSE_MODELS[engine.default_model],
                                           default_model=engine.default_model, models=engine.available_models(),
                                           default_fall_mode=engine.default_fall_mode,
                                           rgb_fall_available=model_available(),
                                           fall_model=model_spec(),
                                           fall_modes=[dict(id='rgb', label='独立图像倒地检测', available=model_available()),
                                                       dict(id='pose', label='仅骨架规则（对照）', available=True)],
                                           device=engine.device,
                                           imgsz=engine.imgsz, recording=False,
                                           recording_capability='browser_raw_video',
                                           action_model='V7 起坐 + 独立行为维度 + 独立实时跌倒事件', diagnostics='v5',
                                           action_pipeline=ACTION_PIPELINE,
                                           temporal_model=engine.temporal_status(),
                                           fusion=engine.fusion_status(),
                                           fall_temporal=engine.fall_temporal_status(),
                                           fall_event_fusion=engine.fall_event_fusion_status(),
                                           personal_video_finetuning=False,
                                           comparison_report_url=(COMPARISON_REPORT_URL if COMPARISON_REPORT.is_file() else None),
                                           external_alerts=False))
            return self.reply(404, dict(error='Not found'))

        def do_POST(self):
            if (not self.valid_host() or self.headers.get('Origin') not in allowed_origins
                    or self.headers.get('X-Session-Token') != token):
                return self.reply(403, dict(error='请从本机演示页面启动'))
            if self.path not in {'/infer', '/stop'}:
                return self.reply(404, dict(error='Not found'))
            stream_id = self.headers.get('X-Stream-Id', '')
            if not re.fullmatch(r'[a-zA-Z0-9-]{16,64}', stream_id):
                return self.reply(400, dict(error='无效会话，请刷新页面'))
            if self.path == '/stop':
                with engine.lock:
                    engine.stop_stream(stream_id)
                return self.reply(200, dict(stopped=True))
            try:
                model_id = self.headers.get('X-Pose-Model', engine.default_model)
                source_type = self.headers.get('X-Source-Type', 'camera')
                fall_mode = self.headers.get('X-Fall-Mode', engine.default_fall_mode)
                if model_id not in POSE_MODELS or source_type not in {'camera', 'video'} or fall_mode not in {'pose', 'rgb'}:
                    raise ValueError('Invalid model/source')
                source_time = float(self.headers.get('X-Source-Time', 'nan'))
                if not math.isfinite(source_time) or source_time < 0:
                    raise ValueError('Invalid source time')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2_000_000:
                    return self.reply(413, dict(error='图像大小超出限制'))
                if self.headers.get('Content-Type', '').split(';')[0] != 'image/jpeg':
                    return self.reply(415, dict(error='需要 JPEG 图像'))
                image = decode_frame(self.rfile.read(size))
            except (ValueError, cv2.error):
                return self.reply(400, dict(error='无效图像，请重新启动摄像头'))
            if not engine.lock.acquire(blocking=False):
                return self.reply(429, dict(error='推理忙，请只在一个页面启动摄像头'))
            try:
                self.reply(200, engine.infer_live(image, stream_id, source_time, model_id, source_type, fall_mode))
            except ValueError as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception as exc:
                print(f'Inference failed: {type(exc).__name__}', flush=True)
                self.reply(500, dict(error='模型推理失败，请检查终端后重试'))
            finally:
                engine.lock.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--pose-model', choices=list(POSE_MODELS), default='yolo26n-pose')
    parser.add_argument('--fall-mode', choices=['rgb', 'pose'], default='rgb')
    parser.add_argument('--no-temporal', action='store_true', help='Disable the optional ST-GCN++ evidence branch')
    parser.add_argument('--fusion-shadow', action='store_true',
                        help='Expose constrained fused_action candidate; never change primary action or fall')
    parser.add_argument('--fall-temporal-shadow', action='store_true',
                        help='Expose constrained fall-process evidence; preserve primary alarm and lifecycle')
    parser.add_argument('--fall-event-fusion', action='store_true',
                        help='Opt in to ST-supported RGB event confirmation; retain baseline_fall')
    args = parser.parse_args()
    if (args.fusion_shadow or args.fall_temporal_shadow or args.fall_event_fusion) and args.no_temporal:
        parser.error('Temporal fusion requires the temporal branch')
    print(f'Loading local {POSE_MODELS[args.pose_model]}…', flush=True)
    engine = PoseEngine(args.device, args.imgsz, args.pose_model, args.fall_mode,
                        temporal_enabled=not args.no_temporal, fusion_enabled=args.fusion_shadow,
                        fall_temporal_enabled=args.fall_temporal_shadow, fall_event_fusion=args.fall_event_fusion)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine, args.port, secrets.token_urlsafe(32)))
    server.daemon_threads = True
    print(f'Ready: http://127.0.0.1:{args.port}/ — camera and optional browser recording start only from the page.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == '__main__':
    main()

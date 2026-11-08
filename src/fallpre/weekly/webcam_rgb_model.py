"""Pinned local image-posture detector, separate from the pose/action models."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG = ROOT / 'configs/rgb-fall-model.json'


def model_spec():
    if not MODEL_CONFIG.is_file():
        return None
    return json.loads(MODEL_CONFIG.read_text())


def model_available():
    spec = model_spec()
    return bool(spec and (ROOT / spec['path']).is_file()
                and (not spec.get('person_model') or (ROOT / spec['person_model']['path']).is_file()))


def box_overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def match_current_persons(persons, fallen, minimum_overlap=.50, minimum_fall_confidence=.60):
    """Fall labels support CURRENT person boxes; they never supply geometry.

    Keep every current person, including weak competitors. A furniture-shaped
    fall detection without an independent matching person produces no candidate.
    No confidence is increased, and absence of a fall label does not mean standing.
    """
    detections = []
    for index, person in enumerate(persons):
        matches = [(i, f) for i, f in enumerate(fallen)
                   if box_overlap(person['box'], f['box']) >= minimum_overlap]
        support = max(matches, key=lambda item: item[1]['confidence']) if matches else None
        score = support[1]['confidence'] if support else 0.
        detections.append(dict(box=list(person['box']), confidence=person['confidence'],
                               label='fallen' if score >= minimum_fall_confidence else 'person',
                               fall_confidence=score, person_index=index,
                               fall_index=support[0] if support else None,
                               evidence_kind='person_plus_fall',
                               support_iou=box_overlap(person['box'], support[1]['box']) if support else 0.))
    return detections


def verified_path(spec):
    path = (ROOT / spec['path']).resolve()
    if not path.is_relative_to(ROOT / 'models') or not path.is_file():
        raise ValueError('本地缺少独立倒地模型，请选择骨架模式')
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != spec['sha256']:
        raise ValueError('独立倒地模型校验失败')
    return path


class RgbFallBackend:
    """Loads only the declared local weight; never downloads at runtime.

    A frame-wise fallen label is posture evidence. Temporal confirmation,
    identity continuity and scene stability are enforced by LiveRgbFall.
    """

    def __init__(self, device='cpu'):
        import torch
        from ultralytics import YOLO

        spec = model_spec()
        if spec is None:
            raise ValueError('独立倒地模型尚未配置，请选择骨架模式')
        self.model = YOLO(str(verified_path(spec)), task='detect')
        self.labels = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.person_model = None
        if spec.get('person_model'):
            if set(self.labels.values()) != {'fall-detected'}:
                raise ValueError('独立倒地模型类别与锁定配置不符')
            self.person_model = YOLO(str(verified_path(spec['person_model'])), task='detect')
            if self.person_model.names.get(0) != 'person':
                raise ValueError('通用人体模型类别不符')
        elif set(self.labels.values()) != {'fallen', 'sitting', 'standing'}:
            raise ValueError('独立倒地模型类别与锁定配置不符')
        self.device = device
        self.metadata = {key: spec[key] for key in ('id', 'label', 'sha256', 'revision', 'source', 'license', 'training_overlap')}
        if self.person_model is not None:
            self.metadata['person_model'] = dict(spec['person_model'])
        for net in (self.model, self.person_model):
            if net is not None:
                net.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, conf=.25,
                            iou=.45, device=device, verbose=False)
        torch.set_num_threads(2)

    def infer(self, image):
        start = time.perf_counter()
        result = self.model.predict(image, imgsz=640, conf=.25, iou=.45,
                                    device=self.device, verbose=False)[0]
        detections = [dict(box=box, confidence=float(score), label=self.labels[int(label)])
                      for box, score, label in zip(result.boxes.xyxy.cpu().tolist(),
                                                   result.boxes.conf.cpu().tolist(),
                                                   result.boxes.cls.cpu().tolist())]
        raw_candidates = dict(fallen=detections)
        if self.person_model is not None:
            persons = self.person_model.predict(image, imgsz=640, conf=.25, iou=.45,
                                                classes=[0], device=self.device, verbose=False)[0]
            people = [dict(box=box, confidence=float(score))
                      for box, score in zip(persons.boxes.xyxy.cpu().tolist(), persons.boxes.conf.cpu().tolist())]
            raw_candidates['persons'] = people
            detections = match_current_persons(people, detections)
        return dict(detections=detections, model=dict(self.metadata), raw_candidates=raw_candidates,
                    inference_ms=(time.perf_counter()-start)*1000)

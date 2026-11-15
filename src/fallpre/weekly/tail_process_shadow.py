"""Frozen, experimental tail-process head on the live asynchronous ST worker."""
import hashlib
import time
from pathlib import Path

import numpy as np

from fallpre.weekly.fall_event_fusion import TemporalFallEventFusion
from fallpre.weekly.stgcn_fall_evidence import (
    FallEvidenceConfig,
    FallTemporalEvidence,
    fall_geometry_features,
)
from fallpre.weekly.stgcn_temporal import (
    LABELS_ZH,
    MODEL_ID,
    RELEVANT_CLASSES,
    STGCNRuntime,
    prepare_tensor,
)

GEOMETRY_COLUMNS = (4, 5, 8, 9, 10, 11, 12, 13, 15, 16)


def segment_geometry(window):
    times, points = window.source_times, window.keypoints
    edges = np.linspace(times[0], times[-1], 4)
    vector = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (times >= start - 1e-9) & (times <= end + 1e-9)
        geometry = fall_geometry_features(points[mask], times[mask], window.width,
                                          window.height, window_s=100.)
        if geometry["vector"] is None:
            return None
        vector.extend(geometry["vector"][i] for i in GEOMETRY_COLUMNS)
    return np.asarray(vector, dtype=np.float32)


class TailProcessRuntime(STGCNRuntime):
    def __init__(self):
        super().__init__()
        self.active_head = None

    def load_head(self, path):
        path = Path(path)
        state = self.torch.load(path, map_location="cpu", weights_only=True)
        if state.get("variant") != "tail_joint" or state.get("geometry_enabled") is not True:
            raise ValueError("Expected an existing tail_joint geometry checkpoint")
        values = {key: state[key].numpy().copy() for key in ("mean", "scale")}
        values.update({key: state["head"][key].numpy().copy() for key in ("weight", "bias")})
        if (values["mean"].shape != (286,) or values["scale"].shape != (286,)
                or values["weight"].shape != (2, 286) or values["bias"].shape != (2,)
                or not all(np.isfinite(x).all() for x in values.values())
                or np.any(values["scale"] <= 0)):
            raise ValueError("Invalid tail process head")
        self.active_head = values, hashlib.sha256(path.read_bytes()).hexdigest(), state["heldout"]

    def predict(self, window):
        start = time.perf_counter()
        values, digest, heldout = self.active_head
        tensor, meta = prepare_tensor(window)
        with self.lock, self.torch.inference_mode():
            backbone, _ = self.model.extract_feat(self.torch.from_numpy(tensor), stage="backbone")
            assert backbone.shape[2:] == (256, 25, 17)
            feature = backbone[:, :, :, -8:, :].mean((1, 3, 4))[0].cpu().numpy()
            scores = self.model.cls_head(backbone).softmax(-1)[0].cpu().numpy()
        if scores.shape != (60,) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid ST class scores")
        geometry = segment_geometry(window)
        head = dict(status="unavailable", reason="segmented_geometry_unavailable")
        if geometry is not None:
            z = (np.r_[feature, geometry] - values["mean"]) / values["scale"]
            logits = values["weight"] @ z + values["bias"]
            score = float(np.exp(logits[1] - np.logaddexp(*logits)))
            head = dict(status="ready", fall_process_score=score, window_end_s=meta["end_s"],
                        target_epoch=window.target_epoch, generation=window.generation,
                        weights_sha256=digest, training_group_excluded=heldout)
        order = np.argsort(-scores, kind="stable")[:3]
        top = [dict(index=int(i), label=self.labels[i], label_zh=LABELS_ZH.get(int(i), self.labels[i]),
                    score=float(scores[i])) for i in order]
        completed = time.perf_counter()
        return dict(status="ready", model_id=MODEL_ID, window=meta, generation=window.generation,
                    target_epoch=window.target_epoch, window_start_s=meta["start_s"],
                    window_end_s=meta["end_s"], source_time_s=meta["end_s"],
                    source_time=meta["end_s"], observed_frames=meta["observations"],
                    top3=top, all_scores=scores.tolist(),
                    relevant_scores={name: float(scores[i]) for name, i in RELEVANT_CLASSES.items()},
                    inference_ms=(completed-start)*1000, completed_monotonic_s=completed,
                    model=dict(self.metadata, preprocessing=meta["preprocessing"], resampling=meta["resampling"]),
                    score_semantics=self.metadata["score_semantics"], process_head=head)


class TailShadow:
    def __init__(self):
        self.evidence = FallTemporalEvidence(FallEvidenceConfig(minimum_score=.8))
        self.events = TemporalFallEventFusion()

    def update(self, result, timestamp):
        temporal = result.get("temporal") or {}
        head = temporal.get("process_head") or {"status": "unavailable"}
        evidence = self.evidence.update(result, timestamp, process_model=head)
        candidate = self.events.update(dict(result, fall_temporal_evidence=evidence), timestamp)
        return dict(mode="shadow", changes_alarm=False, head=head, evidence=evidence,
                    simulated_fall=candidate["fall"], temporal_accelerated=candidate["temporal_accelerated"])

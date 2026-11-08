"""Compare fresh YOLO11/26 frames through the same frozen live application pipeline.

This is a source-clock replay and descriptive integration audit, not an accuracy
benchmark. No observations are reconstructed from old pose caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import platform
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolo11n-pose", "yolo26n-pose")
DEFAULT_CLIPS = (
    "urfd-fall-01-original", "urfd-adl-04", "urfd-adl-07",
    "utk-s01_e01-transition", "utk-s02_e01-transition",
    "utk-s03_e01-transition", "utk-s10_e01-transition", "tum-sitting-independent",
)
LINKS = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2))
LIMITATIONS = [
    "这是一套完整处理链的描述性回放，不是独立测试集准确率、跌倒召回率或临床验证。",
    "原始 URFD fall-01 使用配套 CSV 时钟；UTK 保留 manifest 的非等间隔 source_times_s（其物理时钟仍待核准）；ADL 使用派生视频名义时钟；TUM 使用逐帧源文件名时间戳。",
    "只有一个独立跌倒事件。若选择低清 fall-01，它与原始版本属于同一事件，不能计为两个独立样本。未纳入时钟不明的 fall-02 和 MultiCam 稀疏帧。",
    "动作区间多为已有暂定标注，部分接地/倒地边界只有人工确认区间，没有精确事件真值；候选与确认时刻是程序输出，不能直接当成检测延迟。",
    "10 Hz 网格选择此前最近的真实源帧，同一源帧只推理一次；保留原观测时间，不补帧、不伪造更高采样率。",
    "匹配浏览器 960×720 上限和 JPEG 质量 85，并调用相同解码函数；OpenCV 与浏览器 JPEG 编码器并不保证字节完全一致。两模型收到完全相同的解码像素。",
    "耗时为本机顺序执行的 Python 推理与后处理，不含真实摄像头采集、HTTP、浏览器绘制及排队；视频按源时间播放，不能据此宣称实时帧率。",
    "有框、锁定目标和高置信关节仅表示管线有可用输入；未做逐帧独立身份和骨架真值审定，因此不等于检出率或定位精度。",
    "本次正式回放期间冻结 V7 权重与全部阈值。模型与跌倒分支已在同源样本上做过探索性筛选，因此不是盲测；不同模型的分数分布仍可能改变目标获取和动作门控。",
    "独立图像模型的训练样本和划分不能完整核验，不能排除与当前公开视频重叠；本回放不能据此证明新模型的泛化能力。",
]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_times(entry):
    if "source_times_s" in entry:
        times = np.asarray(entry["source_times_s"], dtype=float)
    else:
        times = np.arange(entry["frame_count"], dtype=float) / entry["fps"]
    if (len(times) != entry["frame_count"] or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError(f"Invalid source clock for {entry['clip_id']}")
    return times


def sample_indices(times, hz=10.0):
    """Past-only grid sampling; retain original timestamps and unique source frames."""
    times = np.asarray(times, dtype=float)
    if not len(times) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("Need nonempty observations and a positive finite sample rate")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must strictly increase")
    ticks = times[0] + np.arange(math.floor((times[-1]-times[0])*hz+1e-8)+1)/hz
    # Permit only numerical equality, never a future observation.
    indices = np.searchsorted(times, ticks+1e-9, side="right")-1
    return list(dict.fromkeys(int(i) for i in indices if i >= 0))


def load_entries(clip_ids=DEFAULT_CLIPS):
    path = ROOT / "configs/action-v7-dataset.json"
    manifest = json.loads(path.read_text())
    entries = {e["clip_id"]: dict(e) for e in manifest["entries"]}
    survey = ROOT / "runs/dataset-survey-20261019"
    csv_path = survey / "urfd-fall-01-data.csv"
    with csv_path.open() as handle:
        clock = [float(row[1])/1000 for row in csv.reader(handle)]
    entries["urfd-fall-01-original"] = dict(
        clip_id="urfd-fall-01-original", source_group_id="urfd-fall-01", dataset="URFD",
        zip_path=str(survey / "urfd-fall-01-rgb.zip"), frame_count=len(clock),
        fps=30.0, source_times_s=clock, duration_s=clock[-1]+float(np.median(np.diff(clock))),
        clock_basis="official paired CSV milliseconds, relative to first observation",
        clock_file=str(csv_path), clock_sha256=digest(csv_path),
        source=dict(url="https://fenix.ur.edu.pl/~mkepski/ds/uf.html"),
        annotation_basis="One existing public fall event; no exact event ground truth is scored.",
        segments=[], state_segments=[])
    reviewed = ROOT / "runs/abc-v7-20261020/annotations/annotations.reviewed.json"
    if reviewed.exists():
        item = next((e for e in json.loads(reviewed.read_text())["entries"]
                     if e["clip_id"] == "urfd-fall-01-original"), None)
        if item:
            entries["urfd-fall-01-original"]["existing_event_landmarks"] = item.get("event_landmarks", {})
            entries["urfd-fall-01-original"]["annotation_file_sha256"] = digest(reviewed)
    selected = []
    for cid in clip_ids:
        if cid not in (*DEFAULT_CLIPS, "urfd-fall-01"):
            raise ValueError(f"Clip is outside the declared audit: {cid}")
        e = entries[cid]
        if cid == "tum-sitting-independent":
            stamps = [float(Path(name).stem) for name in e["source"]["source_frames"]]
            e["source_times_s"] = [t-stamps[0] for t in stamps]
            e["clock_basis"] = "TUM source RGB filenames, relative to first observation"
        elif "clock_basis" not in e:
            e["clock_basis"] = (
                "manifest source_times_s, original frame-number differences / 30; physical clock not verified"
                if "source_times_s" in e else "derived video nominal frame_index / container FPS; physical clock not verified")
        key = "zip_path" if e.get("zip_path") else "video_path"
        e[key] = str((ROOT/e[key]).resolve())
        e["input_sha256"] = digest(e[key])
        if e.get("video_sha256") and e["input_sha256"] != e["video_sha256"]:
            raise ValueError(f"Source checksum changed: {cid}")
        source_times(e)
        selected.append(e)
    return selected


def decoded_frames(entry):
    if entry.get("zip_path"):
        with zipfile.ZipFile(entry["zip_path"]) as archive:
            names = sorted(n for n in archive.namelist() if n.lower().endswith(".png"))
            if len(names) != entry["frame_count"]:
                raise ValueError("Unexpected number of PNG frames")
            for name in names:
                frame = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Could not decode {name}")
                yield frame
    else:
        cap = cv2.VideoCapture(entry["video_path"])
        if not cap.isOpened():
            raise ValueError(f"Could not open {entry['video_path']}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


def browser_jpeg(frame):
    height, width = frame.shape[:2]
    scale = min(1, 960/width, 720/height)
    size = (int(math.floor(width*scale+.5)), int(math.floor(height*scale+.5)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) if size != (width, height) else frame
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


def target_info(result):
    action = result.get("action", {})
    persons = result.get("persons", [])
    index = action.get("target_index")
    available = isinstance(index, int) and 0 <= index < len(persons)
    selection = action.get("selection", {})
    return dict(available=available, index=index,
                epoch=selection.get("target_epoch", action.get("target_epoch")),
                selection_state=selection.get("state"),
                reset_reasons=action.get("reset_reasons", []))


def fall_target_info(result):
    """Read independent fall observations, with explicit legacy-action fallback."""
    observation = result.get("fall_observation")
    if observation is None:
        action = target_info(result)
        persons = result.get("persons", [])
        return dict(available=action["available"], index=action["index"], epoch=action["epoch"],
                    selection_state=action["selection_state"], source="legacy_action_target",
                    person=persons[action["index"]] if action["available"] else None,
                    recovery={}, reason_code=None)
    selection = observation.get("selection", {})
    person = observation.get("person")
    return dict(available=person is not None, person=person, index=selection.get("raw_index"),
                epoch=selection.get("target_epoch"), selection_state=selection.get("state"),
                source=observation.get("source"), recovery=observation.get("recovery", {}),
                reason_code=selection.get("reason_code"))


def pose_fall_summary(rows, model_id):
    """Keep the diagnostic pose-only branch separate from the displayed RGB fall."""
    observed = [(row, row["models"][model_id]) for row in rows
                if "pose_fall" in row["models"][model_id]]
    if not observed:
        return None
    states, sources, transitions, last = Counter(), Counter(), [], None
    available, accepted, executed = 0, 0, 0
    for row, result in observed:
        fall = result["pose_fall"]
        observation = result.get("pose_fall_observation", {})
        state = fall.get("state", "missing")
        states[state] += 1
        sources[str(observation.get("source"))] += 1
        available += observation.get("person") is not None
        recovery = observation.get("recovery", {})
        accepted += bool(recovery.get("accepted"))
        executed += bool(recovery.get("executed"))
        if state != last:
            transitions.append(dict(t_s=row["source_time_s"], frame=row["source_frame_id"], state=state,
                                    reason=fall.get("reason"), event_id=fall.get("event_id")))
            last = state
    return dict(frames=len(observed), states=dict(states), target_available_frames=available,
                sources=dict(sources), recovery_executed_frames=executed, recovery_accepted_frames=accepted,
                first_candidate_s=next((x["t_s"] for x in transitions if x["state"] == "suspected"), None),
                first_confirmation_s=next((x["t_s"] for x in transitions if x["state"] == "confirmed"), None),
                transitions=transitions)


def referenced_weights(config, root=ROOT):
    """Find every explicitly referenced local model weight, including nested specs."""
    found = set()
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and "://" not in value and Path(value).suffix.lower() in {".pt", ".pth", ".onnx", ".ckpt"}:
            path = (root/value).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Referenced model must be within the project")
            if not path.is_file():
                raise FileNotFoundError(f"Configured model missing: {path}")
            found.add(path)
    visit(config)
    return sorted(found)


def baseline_comparison(rows, prior_rows):
    """Exact audit of unchanged observations and actions, never tolerance matching."""
    keys = [(r["source_frame_id"], r["source_time_s"]) for r in rows]
    prior_keys = [(r["source_frame_id"], r["source_time_s"]) for r in prior_rows]
    if keys != prior_keys:
        raise ValueError("Baseline source frame/time sequence differs")
    differences = []
    for row, prior in zip(rows, prior_rows):
        if row["input_jpeg_sha256"] != prior["input_jpeg_sha256"]:
            differences.append(dict(frame=row["source_frame_id"], field="input_jpeg_sha256"))
        for model in MODELS:
            for field in ("persons", "action"):
                if row["models"][model].get(field) != prior["models"][model].get(field):
                    differences.append(dict(frame=row["source_frame_id"], model=model, field=field))
    fields = ("target_available_frames", "fall_target_available_frames", "fall_target_epoch_changes",
              "fall_states", "first_candidate_s", "first_confirmation_s", "recovery_accepted_frames")
    summary = {}
    for model in MODELS:
        before, after = summarize(prior_rows, model), summarize(rows, model)
        summary[model] = dict(prior={k:before[k] for k in fields}, current={k:after[k] for k in fields})
    return dict(compared_frames=len(rows), exact_input_raw_action_match=not differences,
                mismatches=differences, models=summary,
                legacy_fall_target_basis="Prior fall target is action target when fall_observation is absent")


def summarize(rows, model_id):
    results = [r["models"][model_id] for r in rows]
    count = len(results)
    def distribution(fn):
        return dict(Counter(str(fn(r)) for r in results))
    def timing(field):
        values = [r[field] for r in results if isinstance(r.get(field), (int, float))]
        return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95))) if values else None
    transitions = []
    last = None
    for row, result in zip(rows, results):
        fall = result.get("fall", {})
        state = fall.get("state", "missing")
        if state != last:
            transitions.append(dict(t_s=row["source_time_s"], frame=row["source_frame_id"], state=state,
                                    label=fall.get("label"), reason=fall.get("reason"),
                                    event_id=fall.get("event_id")))
            last = state
    reset_reasons = Counter(reason for r in results for reason in target_info(r)["reset_reasons"])
    epochs = [target_info(r)["epoch"] for r in results]
    epoch_changes = sum(a is not None and b is not None and a != b for a, b in zip(epochs, epochs[1:]))
    fall_targets = [fall_target_info(r) for r in results]
    fall_epochs = [x["epoch"] for x in fall_targets]
    recovery = [x["recovery"] for x in fall_targets]
    fall_states = distribution(lambda r: r.get("fall", {}).get("state", "missing"))
    return dict(frames=count, any_pose_frames=sum(bool(r.get("persons")) for r in results),
                multiple_pose_frames=sum(len(r.get("persons", [])) > 1 for r in results),
                target_available_frames=sum(target_info(r)["available"] for r in results),
                target_available_fraction=sum(target_info(r)["available"] for r in results)/count if count else 0,
                target_epoch_changes=epoch_changes, target_reset_reasons=dict(reset_reasons),
                action_target_available_frames=sum(target_info(r)["available"] for r in results),
                action_target_epoch_changes=epoch_changes, action_target_reset_reasons=dict(reset_reasons),
                fall_target_available_frames=sum(x["available"] for x in fall_targets),
                fall_target_available_fraction=sum(x["available"] for x in fall_targets)/count if count else 0,
                fall_target_epoch_changes=sum(a is not None and b is not None and a != b
                                              for a, b in zip(fall_epochs, fall_epochs[1:])),
                fall_target_sources=dict(Counter(str(x["source"]) for x in fall_targets)),
                fall_target_reason_codes=dict(Counter(str(x["reason_code"]) for x in fall_targets)),
                recovery_executed_frames=sum(bool(x.get("executed")) for x in recovery),
                recovery_accepted_frames=sum(bool(x.get("accepted")) for x in recovery),
                recovery_inference_passes=sum(x.get("inference_passes", 0) for x in recovery),
                recovery_reasons=dict(Counter(str(x.get("reason")) for x in recovery if x.get("executed"))),
                action_labels=distribution(lambda r: r.get("action", {}).get("label")),
                raw_action_labels=distribution(lambda r: r.get("action", {}).get("raw_label")),
                phases=distribution(lambda r: r.get("action", {}).get("phase")),
                action_decision_codes=distribution(lambda r: r.get("action", {}).get("decision_code")),
                fall_states=fall_states, fall_reasons=distribution(lambda r: r.get("fall", {}).get("reason")),
                fall_modes=distribution(lambda r: r.get("fall_mode", "legacy_pose")),
                pose_fall=pose_fall_summary(rows, model_id),
                rgb=dict(inference_frames=sum(isinstance(r.get("rgb_inference_ms"), (int, float)) for r in results),
                         any_detection_frames=sum(bool(r.get("rgb_detections")) for r in results),
                         detection_labels=dict(Counter(str(d.get("label")) for r in results
                                                       for d in r.get("rgb_detections", [])))),
                quality_codes=distribution(lambda r: r.get("quality", {}).get("code")),
                camera_stable=distribution(lambda r: r.get("quality", {}).get("camera_stable")),
                first_candidate_s=next((x["t_s"] for x in transitions if x["state"] == "suspected"), None),
                first_confirmation_s=next((x["t_s"] for x in transitions if x["state"] == "confirmed"), None),
                fall_transitions=transitions, timing={field:timing(field) for field in
                    ("inference_ms", "action_ms", "fall_ms", "quality_ms", "recovery_ms",
                     "rgb_inference_ms", "rgb_person_inference_ms", "audit_pipeline_ms")})


def comparison_stats(rows):
    pairs = Counter()
    changes = []
    for row in rows:
        left, right = [row["models"][m] for m in MODELS]
        a, b = left["action"].get("label"), right["action"].get("label")
        pairs[f"{a} → {b}"] += 1
        if a != b or left.get("fall", {}).get("state") != right.get("fall", {}).get("state"):
            changes.append(dict(source_time_s=row["source_time_s"], source_frame_id=row["source_frame_id"],
                                action=[a, b], fall=[left.get("fall", {}).get("state"), right.get("fall", {}).get("state")]))
    return dict(action_pairs=dict(pairs), differing_frames=len(changes), differences=changes)


def render_panel(image, result, name, timestamp):
    panel = np.full((650, 640, 3), (24, 27, 32), np.uint8)
    scale = min(640/image.shape[1], 460/image.shape[0])
    size = (round(image.shape[1]*scale), round(image.shape[0]*scale))
    picture = cv2.resize(image, size)
    xoff, yoff = (640-size[0])//2, 66+(460-size[1])//2
    fall = result.get("fall", {})
    state = fall.get("state", "unobservable")
    color = {"normal":(120, 220, 150), "suspected":(50, 200, 255),
             "confirmed":(80, 90, 255)}.get(state, (175, 175, 180))
    target = target_info(result)["index"]
    fall_target = fall_target_info(result)
    def skeleton(person, tint):
        points = np.asarray(person.get("keypoints", []), float)
        if points.shape != (17, 3):
            return  # RGB boxes carry no pose; never synthesize joints.
        for a, b in LINKS:
            if np.isfinite(points[[a, b]]).all() and min(points[a, 2], points[b, 2]) >= .30:
                cv2.line(picture, tuple((points[a, :2]*scale).astype(int)),
                         tuple((points[b, :2]*scale).astype(int)), tint, 2)
    for i, person in enumerate(result.get("persons", [])):
        chosen = i == target
        tint = (255, 180, 75) if chosen else (105, 105, 115)
        box = np.asarray(person["box"])*scale
        cv2.rectangle(picture, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), tint, 2 if chosen else 1)
        if chosen:
            cv2.putText(picture, "A", tuple(box[:2].astype(int)+[0, -6]), cv2.FONT_HERSHEY_SIMPLEX, .6, tint, 2)
        skeleton(person, tint)
    if fall_target["available"]:
        person = fall_target["person"]
        box = np.asarray(person["box"])*scale
        cv2.rectangle(picture, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), color, 3)
        cv2.putText(picture, "F:"+str(fall_target["source"]), tuple(box[:2].astype(int)+[25, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
        skeleton(person, color)
    panel[yoff:yoff+size[1], xoff:xoff+size[0]] = picture
    action = result.get("action", {})
    quality = result.get("quality", {})
    lines = [(f"{name} | source {timestamp:.3f}s", 26, (240, 240, 240)),
             (f"{fall.get('label', 'UNKNOWN')}  action={action.get('label')}", 54, color),
             (f"raw={action.get('raw_label')} phase={action.get('phase')}", 554, (225, 225, 225)),
             (f"A={target}/{target_info(result)['epoch']} F={fall_target['source']}/{fall_target['epoch']}", 578, (225, 225, 225)),
             (f"camera={quality.get('camera_stable')} pose={result.get('inference_ms', 0):.1f} "
              + (f"RGB={result['rgb_inference_ms']:.1f}ms" if 'rgb_inference_ms' in result
                 else f"recovery={result.get('recovery_ms', 0):.1f}ms"), 602, (225, 225, 225)),
             (f"candidate={fall.get('candidate_duration_s', 0) or 0:.2f}s | {state}", 626, color)]
    for text, y, tint in lines:
        cv2.putText(panel, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, .54, tint, 1, cv2.LINE_AA)
    return panel


def write_report(out, report):
    esc = html.escape
    def compact(value):
        return esc(json.dumps(value, ensure_ascii=False))
    cards = []
    for clip in report["clips"]:
        rows = []
        for model, stats in clip["models"].items():
            rows.append(f"<tr><th>{esc(model)}</th><td>{stats['any_pose_frames']}/{stats['frames']}</td>"
                        f"<td>{stats['action_target_available_frames']}/{stats['frames']}</td>"
                        f"<td>{stats['fall_target_available_frames']}/{stats['frames']}</td><td>{stats['fall_target_epoch_changes']}</td>"
                        f"<td>{stats['recovery_accepted_frames']}/{stats['recovery_executed_frames']}</td>"
                        f"<td>{compact(stats['fall_states'])}</td><td>{compact(stats['action_labels'])}</td>"
                        f"<td>{stats['first_candidate_s'] if stats['first_candidate_s'] is not None else '—'}</td>"
                        f"<td>{stats['first_confirmation_s'] if stats['first_confirmation_s'] is not None else '—'}</td></tr>")
        branch_rows = []
        for model, stats in clip["models"].items():
            pose = stats.get("pose_fall")
            rgb_time = stats["timing"].get("rgb_inference_ms")
            branch_rows.append(f"<tr><th>{esc(model)}</th><td>{compact(stats['fall_target_sources'])}</td>"
                               f"<td>{compact(pose['states']) if pose else '—'}</td>"
                               f"<td>{compact(rgb_time) if rgb_time else '—'}</td></tr>")
        cid = esc(clip["clip_id"])
        cards.append(f"<section><h2>{cid}</h2><p>{esc(clip['clock_basis'])}</p>"
                     f"<p>{clip['source_frames']} 个源帧 → {clip['sampled_frames']} 次观测；"
                     f"时段 {clip['first_time_s']:.3f}–{clip['last_time_s']:.3f} 秒。两模型动作/跌倒状态不同的观测：{clip['comparison']['differing_frames']}。</p>"
                     f"<div class='scroll'><table><thead><tr><th>模型</th><th>有任意框</th><th>动作目标可用</th><th>跌倒目标可用</th><th>跌倒epoch变化</th><th>补检接受/执行</th>"
                     f"<th>跌倒状态计数</th><th>显示动作计数</th><th>首次候选(s)</th><th>首次确认(s)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
                     f"<div class='scroll'><table><thead><tr><th>模型</th><th>显示跌倒目标的来源</th><th>旁路骨架跌倒状态</th><th>独立图像推理耗时</th></tr></thead>"
                     f"<tbody>{''.join(branch_rows)}</tbody></table></div>"
                     f"<video controls preload='metadata' src='{cid}/comparison.mp4'></video>"
                     f"<p><a href='{cid}/frames.jsonl'>逐帧原始结果</a> · <a href='{cid}/summary.json'>本段汇总</a></p>"
                     f"<details><summary>状态变化、耗时、动作对照及限制</summary><pre>{compact(clip)}</pre></details></section>")
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO11 / YOLO26 全链路回放对照</title><style>
body{{margin:0;background:#f3f5f8;color:#18212b;font:16px/1.7 system-ui,-apple-system,sans-serif}}main{{max-width:1420px;margin:auto;padding:36px 24px}}
h1{{font-size:32px;line-height:1.3}}h2{{font-size:22px}}section{{background:white;padding:24px;margin:26px 0;border-radius:14px;border:1px solid #dce2e9}}
.note{{background:#fff4db;border-left:4px solid #c78b16;padding:16px 22px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #e0e4ea}}th{{white-space:nowrap}}.scroll{{overflow:auto}}video{{display:block;width:100%;max-width:1100px;margin:22px auto;background:#15191e;border-radius:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}a{{color:#1659a7}}li{{margin:8px 0}}
</style><main><h1>YOLO11 / YOLO26：同输入、同规则的完整处理链对照</h1>
<p>冻结 V7 与全部阈值。每个模型都从公开原始图像重新推理；使用当前应用的目标选择、动作、画面质量与跌倒状态逻辑。</p>
<p>视频中 A 为动作目标，F 为独立跌倒目标；仅绘制当前观测，补检来源单独标记。原 target_* 字段保留动作目标语义。</p>
<p>本次跌倒模式：<strong>{esc(report['execution'].get('fall_mode', 'pose'))}</strong>。RGB 模式的框没有骨架点；旁路 pose_fall 单独汇总，不作为 RGB 分支的确认。独立图像模型来源和训练重叠限制见来源文件。</p>
<div class="note"><strong>这些数字描述程序行为，不代表准确率。</strong> 有框不代表是真人，目标可用不代表身份正确；确认时刻不等于检测延迟。</div>
<p>样本 {len(report['clips'])} 段；原始独立跌倒事件 1 个。<a href="summary.json">完整 JSON 汇总</a> · <a href="manifest.json">来源与模型校验</a></p>
<details open><summary>阅读限制</summary><ul>{''.join('<li>'+esc(x)+'</li>' for x in LIMITATIONS)}</ul></details>
{''.join(cards)}<section><h2>与既有回放的输入和动作一致性</h2><pre>{compact(report.get('baseline', '未指定基线'))}</pre></section>
<section><h2>执行信息</h2><pre>{compact(report['execution'])}</pre></section></main></html>"""
    (out/"index.html").write_text(content)


def load_server():
    spec = importlib.util.spec_from_file_location("live_compare_server", ROOT/"scripts/webcam_pose_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(entries, out, hz=10.0, device="cpu", baseline=None, fall_mode="pose"):
    import torch
    sys.path.insert(0, str(ROOT/"src"))
    from fallpre.weekly.human import open_video_writer
    server = load_server()
    out.mkdir(parents=True, exist_ok=False)
    frozen = ROOT/"runs/action-v7/training/model.json"
    watched = [frozen, ROOT/"scripts/webcam_pose_demo.py", ROOT/"scripts/compare_live_models.py"]
    watched += sorted((ROOT/"src/fallpre/weekly").glob("webcam*.py"))
    watched += sorted((ROOT/"src/fallpre/weekly").glob("action_v*.py"))
    watched += [ROOT/"models.lock.json", ROOT/"configs/action-v7-dataset.json"]
    watched += [ROOT/"models"/(m+".pt") for m in MODELS]
    rgb_spec = None
    if fall_mode == "rgb":
        config_path = ROOT/"configs/rgb-fall-model.json"
        rgb_spec = json.loads(config_path.read_text())
        watched += [config_path, *referenced_weights(rgb_spec)]
    checksums = {str(p.relative_to(ROOT)):digest(p) for p in watched}
    engines = {m:server.PoseEngine(device=device, imgsz=640, model_id=m, fall_mode=fall_mode) for m in MODELS}
    # Ultralytics predictor setup can reset thread count; set it after both initializations.
    torch.set_num_threads(2)
    for engine in engines.values():
        for _ in range(4):
            engine.infer(np.zeros((480, 640, 3), np.uint8))
            if fall_mode == "rgb":
                engine.rgb_backend.infer(np.zeros((480, 640, 3), np.uint8))
    if torch.get_num_threads() != 2:
        raise RuntimeError("Unexpected PyTorch CPU thread count")
    execution = dict(device=device, imgsz=640, conf=.25, iou=.45, torch_version=torch.__version__,
                     torch_threads=torch.get_num_threads(), platform=platform.platform(), sample_hz=hz,
                     input_max=[960, 720], jpeg_quality=85, fall_mode=fall_mode, rgb_model=rgb_spec,
                     wall_start_unix=time.time(), fresh_pose_inference=True, no_old_pose_cache=True,
                     timing_scope="local infer_live call; excludes JPEG encode/decode, HTTP, browser and capture",
                     action_ms_scope="server field includes action, camera-quality, fall and recovery/RGB work; not isolated V7 latency")
    (out/"manifest.json").write_text(json.dumps(dict(entries=entries, checksums=checksums,
        execution=execution, limitations=LIMITATIONS), ensure_ascii=False, indent=2, allow_nan=False))
    reports, baseline_reports = [], {}
    for entry in entries:
        cid = entry["clip_id"]
        dest = out/cid
        dest.mkdir()
        times = source_times(entry)
        selected = set(sample_indices(times, hz))
        for engine in engines.values():
            engine.streams.clear()
        writer, codec = open_video_writer(dest/"comparison.mp4", 20, (1280, 650))
        rows, last_overlay, video_index = [], None, 0
        decoded = 0
        try:
            with (dest/"frames.jsonl").open("w") as log:
                for i, frame in enumerate(decoded_frames(entry)):
                    decoded += 1
                    if i not in selected:
                        continue
                    encoded = browser_jpeg(frame)
                    image = server.decode_frame(encoded)
                    t = float(times[i])
                    row = dict(source_frame_id=i+1, original_frame_id=entry.get("source_original_frame_ids", list(range(1, len(times)+1)))[i],
                               source_time_s=t, input_jpeg_sha256=hashlib.sha256(encoded).hexdigest(),
                               decoded_shape=list(image.shape), models={})
                    order = MODELS if len(rows)%2 == 0 else MODELS[::-1]
                    for model in order:
                        start = time.perf_counter()
                        result = engines[model].infer_live(image, f"audit-{cid}-{model}", t,
                                                          model_id=model, source_type="video", fall_mode=fall_mode)
                        result["audit_pipeline_ms"] = (time.perf_counter()-start)*1000
                        row["models"][model] = result
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                    rows.append(row)
                    overlay = np.hstack([render_panel(image, row["models"][m], m, t) for m in MODELS])
                    # A source-time preview repeats only prior rendered observations; no repeated inference.
                    if last_overlay is not None:
                        while times[0]+video_index/20 < t-1e-9:
                            writer.write(last_overlay)
                            video_index += 1
                    last_overlay = overlay
                if decoded != entry["frame_count"]:
                    raise ValueError(f"Decoded {decoded} of {entry['frame_count']} frames: {cid}")
                end = float(times[-1]+np.median(np.diff(times)))
                while times[0]+video_index/20 < end-1e-9:
                    writer.write(last_overlay)
                    video_index += 1
        finally:
            writer.release()
        summary = dict(clip_id=cid, clock_basis=entry["clock_basis"], source_frames=decoded,
                       sampled_frames=len(rows), first_time_s=float(times[0]), last_time_s=float(times[-1]),
                       max_sample_gap_s=max(np.diff([r["source_time_s"] for r in rows]), default=0),
                       models={m:summarize(rows, m) for m in MODELS}, comparison=comparison_stats(rows),
                       video=dict(codec=codec, fps=20, frames=video_index, purpose="source-time replay, not inference throughput"),
                       annotation_basis=entry.get("annotation_basis"), limitations=LIMITATIONS)
        if baseline is not None:
            prior_path = baseline/cid/"frames.jsonl"
            prior_rows = [json.loads(line) for line in prior_path.read_text().splitlines()]
            summary["baseline"] = baseline_comparison(rows, prior_rows)
            baseline_reports[cid] = summary["baseline"]
        # Verify the final playable artifact decodes fully.
        cap = cv2.VideoCapture(str(dest/"comparison.mp4"))
        verified = 0
        while cap.read()[0]:
            verified += 1
        cap.release()
        if verified != video_index:
            raise RuntimeError(f"Comparison video incomplete: {cid}")
        summary["video"]["decoded_frames_verified"] = verified
        (dest/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        reports.append(summary)
        print(json.dumps(dict(clip=cid, frames=len(rows), fall={m:summary['models'][m]['fall_states'] for m in MODELS}), ensure_ascii=False), flush=True)
    if any(digest(ROOT/p) != value for p, value in checksums.items()):
        raise RuntimeError("Code or weights changed during the comparison; rerun before publishing")
    execution["wall_elapsed_s"] = time.time()-execution["wall_start_unix"]
    execution["all_code_and_weight_checksums_unchanged"] = True
    report = dict(execution=execution, clips=reports, limitations=LIMITATIONS)
    if baseline is not None:
        report["baseline"] = dict(path=str(baseline), clips=baseline_reports,
                                exact_input_raw_action_match=all(x["exact_input_raw_action_match"]
                                                                for x in baseline_reports.values()))
    (out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    write_report(out, report)
    print(str(out/"index.html"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/live-fall-compare-20261022")
    parser.add_argument("--clip", action="append")
    parser.add_argument("--hz", type=float, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--fall-mode", default="pose", choices=["pose", "rgb"],
                        help="Explicit fall branch; keeps the chosen pose model and V7 action unchanged")
    parser.add_argument("--baseline", type=Path, help="Prior output directory for exact JPEG/raw/action comparison")
    parser.add_argument("--plan", action="store_true", help="Validate source files and print sample counts; no model loading")
    args = parser.parse_args()
    entries = load_entries(args.clip or DEFAULT_CLIPS)
    if args.plan:
        plans = [dict(clip=e["clip_id"], source_frames=e["frame_count"],
                      observations=len(sample_indices(source_times(e), args.hz)),
                      first_s=float(source_times(e)[0]), last_s=float(source_times(e)[-1]),
                      clock=e["clock_basis"]) for e in entries]
        print(json.dumps(dict(clips=plans, observations_per_model=sum(p["observations"] for p in plans),
                              model_calls=2*sum(p["observations"] for p in plans)), indent=2))
        return
    run(entries, args.output.resolve(), args.hz, args.device,
        args.baseline.resolve() if args.baseline else None, args.fall_mode)


if __name__ == "__main__":
    main()

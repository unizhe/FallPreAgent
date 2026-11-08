"""Render synchronized A/B/C media from immutable logs and original RGB.

No inference, training, annotation generation, or pose interpolation occurs.
The manifest records source clocks separately from the illustrative 30 FPS
MultiCam preview. Missing display frames hold the entire previous observation.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallpre.weekly.human import LINKS, open_video_writer, sha256  # noqa: E402

WIDTH, HEIGHT, FPS = 1920, 720, 30
GROUPS = {
    "A": ROOT / "runs/public-samples-20261020",
    "B": ROOT / "runs/verified-tracking-20261020-v2",
    "C": ROOT / "runs/evidence-recovery-20261020",
}
TITLES = {
    "A": "A  legacy / off",
    "B": "B  verified / off",
    "C": "C  verified / evidence",
}
COLORS = {"A": (77, 178, 245), "B": (235, 178, 91), "C": (154, 220, 113)}
CASES = [
    dict(id="urfd-lowres", clip_id="urfd-fall-01", first=95, last=145,
         title="低清 URFD：跌倒后人体观测与动作输出", review_frames=[113, 117, 125]),
    dict(id="urfd-original-partial", clip_id="urfd-fall-01-original", first=105, last=145,
         title="原始 URFD：局部人体恢复，134–135 帧动作仍未知", review_frames=[114, 134, 135]),
    dict(id="utk-seated", clip_id="utk-s10_e01-transition", first=642, last=818,
         title="UTK s10：暂定坐姿与起身内段", review_frames=[654, 730, 806]),
    dict(id="multicam-empty", clip_id="multicam-chute01-cam8", first=449, last=550,
         title="MultiCam：空房中的椅子假目标", review_frames=[449, 501, 550]),
    dict(id="multicam-fall", clip_id="multicam-chute01-cam8", first=1070, last=1310,
         title="MultiCam：跌倒及倒地后的目标选择", review_frames=[1109, 1135, 1285]),
    dict(id="multicam-door", clip_id="multicam-chute01-cam8", first=78, last=112,
         title="MultiCam：门口遮挡、局部观测与出画边界", review_frames=[89, 98, 111]),
]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def source_frames(entry):
    if entry.get("zip_path"):
        with zipfile.ZipFile(entry["zip_path"]) as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith(".png")):
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                assert image is not None, name
                yield image
    else:
        capture = cv2.VideoCapture(entry["video_path"])
        try:
            assert capture.isOpened(), entry["video_path"]
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                yield image
        finally:
            capture.release()


def normalized(row, group):
    if group == "A":
        poses = row["instances"]
        observed = row["target_observed"]
        chosen = next((int(i) for i, tid in row["detection_track_ids"].items()
                       if tid == row["target_id"]), None) if observed else None
        result = dict(target_id=row["target_id"], observed=observed, chosen=chosen,
                      prediction=row["prediction"], reason=row["tracking_reason"],
                      accepted=set(range(len(poses))))
    else:
        poses = row["pose_instances"]
        after = row["after"]
        result = dict(target_id=after["target_id"], observed=after["observed"],
                      chosen=after["target_index"], prediction=after["prediction"],
                      reason=after["reason"], presence_only=after.get("presence_only", False),
                      accepted={a["index"] for a in after["admission"] if a["accepted"]})
    assert bool(result["chosen"] is not None) == bool(result["observed"]), (group, row)
    result["poses"] = poses
    pose = poses[result["chosen"]] if result["chosen"] is not None else None
    result["joints"] = sum(k[2] >= .2 for k in pose["kpts"]) if pose else 0
    result["backend"] = pose.get("pose_backend", "YOLO") if pose else "none"
    return result


def put(canvas, text, position, scale=.59, color=(230, 235, 241), width=None, thick=1):
    if width:
        measured = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        if measured > width:
            scale *= width / measured
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def panel(image, data, group):
    out = np.full((600, 640, 3), (29, 26, 24), np.uint8)
    put(out, TITLES[group], (15, 29), .76, COLORS[group], thick=2)
    rgb = image.copy()
    for index, pose in enumerate(data["poses"]):
        selected = index == data["chosen"]
        color = (105, 235, 145) if selected else (65, 65, 230) if index not in data["accepted"] else (130, 137, 143)
        x1, y1, x2, y2 = map(int, pose["box"])
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2 if selected else 1)
        if not selected:
            continue
        keypoints = pose["kpts"]
        for a, b in LINKS:
            if min(keypoints[a][2], keypoints[b][2]) >= .2:
                cv2.line(rgb, tuple(map(int, keypoints[a][:2])),
                         tuple(map(int, keypoints[b][:2])), color, 2)
        for x, y, confidence in keypoints:
            if confidence >= .2:
                cv2.circle(rgb, (int(x), int(y)), 2, (245, 242, 114), -1)
    scale = min(624 / rgb.shape[1], 390 / rgb.shape[0])
    rgb = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
    x, y = (640 - rgb.shape[1]) // 2, 43 + (390 - rgb.shape[0]) // 2
    out[y:y + rgb.shape[0], x:x + rgb.shape[1]] = rgb
    prediction = data["prediction"]
    put(out, f"Action: {prediction['label']}", (15, 461), .70, COLORS[group], 610, 2)
    put(out, f"Phase: {prediction['phase']}", (15, 489), .61, width=610)
    put(out, f"Target: {data['target_id']}  observed: {str(data['observed']).lower()}  joints: {data['joints']}/17",
        (15, 518), .57, width=610)
    state = "PRESENCE ONLY; action withheld" if data.get("presence_only") else prediction.get("quality", "frame audit")
    put(out, f"Pose: {data['backend']} | {state}", (15, 545), .54, width=610)
    put(out, f"Tracking: {data['reason']}", (15, 572), .53, width=610)
    return out


def render_frame(image, rows, entry):
    out = np.full((HEIGHT, WIDTH, 3), (20, 18, 17), np.uint8)
    row = rows["A"]
    put(out, entry["clip_id"] + " | fixed V7 | synchronized A/B/C", (18, 30), .81, thick=2)
    if entry.get("frame_only"):
        clock = f"Original frame {row['original_frame_id']} | FRAME ONLY - physical FPS unknown | illustrative preview 30 FPS; action disabled"
    else:
        clock = f"Original frame {row['original_frame_id']} | source clock {row['source_time_s']:.3f} s | decoded observation {row['decoded_frame_1based']}"
    put(out, clock, (18, 62), .65, (149, 208, 240))
    for n, group in enumerate(GROUPS):
        out[80:680, n * 640:(n + 1) * 640] = panel(image, normalized(rows[group], group), group)
        if n:
            cv2.line(out, (n * 640, 80), (n * 640, 680), (90, 80, 70), 1)
    put(out, "Green: selected current pose | Red: rejected candidate | Reference labels provisional; no human GT | Display holds prior observations; no pose interpolation",
        (18, 708), .57, width=1884)
    return out


def artifact_metadata(path, entry, selected, frame_count, codec, *, title, case_id=None):
    clock = ("frame_only_preview_30fps_not_physical_time" if entry.get("frame_only")
             else "source_time_30hz_display_hold_previous_observation")
    result = dict(file=str(path.relative_to(path.parents[2])), clip_id=entry["clip_id"],
                  title=title, frames=frame_count, width=WIDTH, height=HEIGHT, fps=FPS,
                  codec=codec, bytes=path.stat().st_size, sha256=sha256(path),
                  source_observations=len(selected),
                  source_frame_range=[selected[0]["original_frame_id"], selected[-1]["original_frame_id"]],
                  source_time_range_s=([selected[0]["source_time_s"], selected[-1]["source_time_s"]]
                                       if not entry.get("frame_only") else None),
                  source_preview_range_s=[selected[0]["preview_time_s"], selected[-1]["preview_time_s"]],
                  clock=clock, labels_human_reviewed=False)
    # path is output/media/{full,cases}/file; use output-relative paths.
    result["file"] = "media/" + str(path.relative_to(path.parents[1]))
    if case_id:
        result["id"] = case_id
    return result


def validate(path, expected):
    cap = cv2.VideoCapture(str(path))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    count = 0
    shapes = set()
    while True:
        ok, image = cap.read()
        if not ok:
            break
        count += 1
        shapes.add(tuple(image.shape[:2]))
    cap.release()
    assert count == declared == expected, (path, count, declared, expected)
    assert shapes == {(HEIGHT, WIDTH)} and abs(fps - FPS) < .01
    return dict(file=str(path), decoded_frames=count, declared_frames=declared, expected_frames=expected,
                width=WIDTH, height=HEIGHT, fps=fps,
                fourcc="".join(chr((codec >> (8 * i)) & 255) for i in range(4)), complete_decode=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/abc-v7-20261020/media")
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ("full", "cases", "review"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((GROUPS["A"] / "manifest.json").read_text())
    result = dict(
        version=1, groups={g: dict(label=TITLES[g], log_root=str(p)) for g, p in GROUPS.items()},
        rendering="Original RGB decoded once. Cached pose, target and V7 predictions; no inference or training.",
        reference_labels="Not used by renderer. All review labels remain provisional; no human GT.",
        interpolation="None. Display repeats the entire previous observation at source-time gaps.",
        full_videos=[], cases=[], review_images=[], input_hashes={})
    for entry in manifest["entries"]:
        cid = entry["clip_id"]
        logs = {g: read_jsonl(p / cid / "frames.jsonl") for g, p in GROUPS.items()}
        source_path = Path(entry.get("zip_path", entry.get("video_path")))
        assert sha256(source_path) == entry["input_sha256"], source_path
        result["input_hashes"][cid] = dict(
            source=entry["input_sha256"],
            logs={g: sha256(p / cid / "frames.jsonl") for g, p in GROUPS.items()})
        assert all(len(rows) == entry["frame_count"] for rows in logs.values())
        for i, row in enumerate(logs["A"]):
            for g in ("B", "C"):
                assert logs[g][i]["original_frame_id"] == row["original_frame_id"]
                assert logs[g][i]["source_time_s"] == row["source_time_s"]
        path = out / "full" / f"{cid}.mp4"
        writer, codec = open_video_writer(path, FPS, (WIDTH, HEIGHT))
        selection = [c for c in CASES if c["clip_id"] == cid]
        case_states = []
        for case in selection:
            selected = [r for r in logs["A"] if case["first"] <= r["original_frame_id"] <= case["last"]]
            assert selected
            cpath = out / "cases" / f"{case['id']}.mp4"
            cwriter, ccodec = open_video_writer(cpath, FPS, (WIDTH, HEIGHT))
            case_states.append(dict(case=case, selected=selected, path=cpath, writer=cwriter,
                                    codec=ccodec, count=0, previous=None, review=[]))
        previous, count, n_observed = None, 0, 0
        review_ids = {f for c in selection for f in c["review_frames"]}
        for i, image in enumerate(source_frames(entry)):
            assert i < len(logs["A"]), cid
            rows = {g: values[i] for g, values in logs.items()}
            row = rows["A"]
            canvas = render_frame(image, rows, entry)
            goal = i if entry.get("frame_only") else round(row["source_time_s"] * FPS)
            assert goal >= count, (cid, goal, count)
            while count < goal:
                writer.write(previous if previous is not None else canvas)
                count += 1
            writer.write(canvas)
            count += 1
            previous = canvas
            n_observed += 1
            for state in case_states:
                case = state["case"]
                fid = row["original_frame_id"]
                if case["first"] <= fid <= case["last"]:
                    cgoal = round((row["preview_time_s"] - state["selected"][0]["preview_time_s"]) * FPS)
                    assert cgoal >= state["count"]
                    while state["count"] < cgoal:
                        state["writer"].write(state["previous"] if state["previous"] is not None else canvas)
                        state["count"] += 1
                    state["writer"].write(canvas)
                    state["count"] += 1
                    state["previous"] = canvas
            if row["original_frame_id"] in review_ids:
                review_path = out / "review" / f"{cid}-{row['original_frame_id']:06d}.jpg"
                assert cv2.imwrite(str(review_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
                result["review_images"].append("media/review/" + review_path.name)
            if n_observed % 400 == 0:
                print(f"{cid}: {n_observed} source observations rendered", flush=True)
        writer.release()
        assert n_observed == entry["frame_count"]
        result["full_videos"].append(artifact_metadata(
            path, entry, logs["A"], count, codec, title=f"{cid}：完整 A/B/C 对照"))
        for state in case_states:
            state["writer"].release()
            result["cases"].append(artifact_metadata(
                state["path"], entry, state["selected"], state["count"], state["codec"],
                title=state["case"]["title"], case_id=state["case"]["id"]))
        (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"{cid}: complete {n_observed} observations, {count} display frames", flush=True)
    validations = []
    for item in result["full_videos"] + result["cases"]:
        path = out.parent / item["file"]
        validations.append(validate(path, item["frames"]))
        print(f"Full decode passed: {path.name}", flush=True)
    (out / "validation.json").write_text(json.dumps(dict(
        all_passed=True, videos=len(validations), checks=validations), indent=2))
    print(f"Finished: {out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Drive the local video Agent API with real frames and record correlated debug events."""
import argparse
import hashlib
import json
import re
import time
import uuid
from html import escape
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import cv2


def select(obj, *keys):
    return {key: obj.get(key) for key in keys}


def matched_ready(summary):
    history = summary.get("result") or {}
    projection = summary["active_projection"]
    current = projection.get("current") or {}
    return (history.get("provider") == "aliyun" and bool(history.get("request_id"))
            and history.get("http_status") == 200 and history.get("status") == "ready"
            and projection.get("current_applicable") is True and current.get("status") == "ready"
            and current.get("decision_id") == history["decision_id"]
            and history.get("fall_probability") is not None
            and current.get("fall_probability") == history["fall_probability"])


def write_result_page(directory, summary):
    history = summary.get("result") or {}
    decision_id = history.get("decision_id")
    stages = {}
    with (directory / "trace.jsonl").open() as trace:
        for line in trace:
            event = json.loads(line)
            if decision_id and event.get("decision_id") == decision_id:
                stages[event["stage"]] = event
    pose = stages.get("yolo_pose", {})
    temporal = stages.get("stgcn_window", {})
    head = stages.get("fall_head", {})
    gate = stages.get("agent_registration", {})
    window = history.get("source_window") or {}
    score = history.get("fall_probability")
    display = "—" if score is None else f"{score * 100:g}%"
    verified = matched_ready(summary)
    rows = [
        ("视频文件", summary["video"]),
        ("决策窗口（秒）", f'{window.get("start_s", "—")} – {window.get("end_s", "—")}'),
        ("YOLO 人数", pose.get("person_count")),
        ("ST-GCN++ 观测帧数", temporal.get("observed_frames")),
        ("跌倒识别头原始分数", head.get("head_raw_score")),
        ("任务调度原因", gate.get("schedule_reason")),
        ("决策模型", history.get("model")),
        ("HTTP 状态", history.get("http_status")),
        ("客户端耗时（毫秒）", history.get("client_latency_ms")),
        ("阿里云请求 ID", history.get("request_id")),
        ("决策 ID", decision_id),
        ("运行 ID", summary["run_id"]),
    ]
    body = "".join(f"<tr><th>{label}</th><td>{escape(str(value)) if value is not None else '—'}</td></tr>"
                   for label, value in rows)
    state = "运行时当前值与持久历史已对齐" if verified else "尚未验证当前值与历史一致"
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>视频 Agent 联调结果</title>
<style>body{{font:16px/1.6 system-ui;color:#182f38;background:#f4f7f6;max-width:780px;margin:4vh auto;padding:20px}}
h1{{font-size:1.55rem}}strong{{font-size:3.5rem;color:#186257}}th,td{{text-align:left;padding:.4rem .8rem;
border-bottom:1px solid #ced9d6;overflow-wrap:anywhere}}th{{width:13em}}table{{background:white;width:100%}}
a{{color:#115d77}}</style><main><h1>视频 Agent 联调结果</h1><strong>{display}</strong>
<p>阿里云返回的窗口跌倒概率 · 未校准模型估计</p><p>{state}。此页保存了当次运行的查询记录。</p>
<table>{body}</table><p><a href="result.json">查看接口结果 JSON</a> ·
<a href="trace.jsonl">查看逐节点日志</a></p></main></html>"""
    (directory / "result.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--right-rgb-panel", action="store_true",
                        help="Read the right-hand RGB image in an original URFD depth/RGB container")
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        parser.error("Cannot decode video or determine source FPS")
    args.out.mkdir(parents=True, exist_ok=False)
    origin = f"http://127.0.0.1:{args.port}"
    opener = build_opener(ProxyHandler({}))
    with opener.open(origin + "/", timeout=15) as response:
        token = re.search(r"const token='([^']+)'", response.read().decode()).group(1)
    headers = {"Origin": origin, "X-Session-Token": token, "X-Client-Session-Id": uuid.uuid4().hex}

    def api(path, body=None, extra=None):
        fields = dict(headers)
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            fields["Content-Type"] = "application/json"
        fields.update(extra or {})
        with opener.open(Request(origin + path, data=body, headers=fields), timeout=90) as resp:
            return json.load(resp)

    run = api("/api/v1/agent/runs", {"request_id": uuid.uuid4().hex, "source_type": "video"})
    run_path = "/api/v1/agent/runs/" + run["run_id"]
    result = None
    latest = {}
    cursor = 0
    with (args.out / "trace.jsonl").open("x") as log:
        def emit(stage, **values):
            log.write(json.dumps({"stage": stage, "run_id": run["run_id"], "time_unix": time.time(), **values}) + "\n")
            log.flush()

        def poll():
            nonlocal latest, result, cursor
            latest = api(run_path + "/latest")
            while True:
                page = api(run_path + f"/decisions?after_id={cursor}&limit=100")
                for row in page["items"]:
                    cursor = row["cursor"]
                    decision = row["result"]
                    meta = decision.get("provider_metadata") or {}
                    item = {**select(row, "decision_id", "task_id", "source_time"),
                            **select(decision, "status", "provider", "model", "fall_probability",
                                     "probability_kind", "source_window", "reason_codes"),
                            **select(meta, "request_id", "http_status", "input_tokens", "client_latency_ms",
                                     "diagnostic_code", "field_path")}
                    emit("decision_history", **item)
                    if item["provider"] == "aliyun" and (item["request_id"] or item["status"] == "unavailable"):
                        result = item
                        if item["status"] == "ready":
                            return
                if not page["has_more"]:
                    break
                cursor = page["next_after_id"]

        try:
            emit("video_open", video=args.video.name, fps=fps, provider=run["provider_status"]["provider"])
            index = 0
            submitted = 0
            reached_eof = False
            interval = max(1, round(fps / 10))
            while result is None:
                ok, frame = capture.read()
                if not ok:
                    reached_eof = True
                    break
                t = index / fps
                index += 1
                if (index - 1) % interval:
                    continue
                if args.right_rgb_panel:
                    frame = frame[:, frame.shape[1] // 2:]
                # Match the input canvas used by the production video page.
                height, width = frame.shape[:2]
                scale = min(1.0, 960 / width, 720 / height)
                if scale < 1:
                    frame = cv2.resize(frame, (max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5))),
                                       interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                data = encoded.tobytes()
                frame_headers = {"Content-Type": "image/jpeg", "X-Agent-Run-Id": run["run_id"],
                                 "X-Stream-Id": run["stream_id"], "X-Source-Type": "video",
                                 "X-Source-Time": str(t), "X-Pose-Model": "yolo26n-pose", "X-Fall-Mode": "rgb"}
                visual = api("/infer", data, frame_headers)
                submitted += 1
                agent = visual["agent"]
                trace = agent.get("trace") or {}
                ids = {"source_time_s": t, **select(agent, "snapshot_id", "decision_id", "task_id")}
                emit("video_frame", **ids, frame_sha256=hashlib.sha256(data).hexdigest())
                emit("yolo_pose", **ids, **select(trace, "yolo_model_version", "person_count", "pose_valid_fraction"))
                emit("stgcn_window", **ids, **select(trace, "temporal_status", "temporal_score", "source_window", "observed_frames"))
                emit("fall_head", **ids, **select(trace, "head_status", "head_version", "head_raw_score", "event_state"))
                emit("agent_registration", **ids, **select(agent, "schedule_reason", "missing_reasons", "status"))
                if (run["provider_status"]["provider"] == "aliyun" and agent.get("task_id")
                        and trace.get("temporal_status") == "ready" and trace.get("head_status") == "ready"):
                    # Hold the source clock here so the asynchronous result can be seen for its own input.
                    deadline = time.monotonic() + 22
                    while time.monotonic() < deadline:
                        poll()
                        if result or latest["projection"].get("latest_task_status") not in {"pending", "running"}:
                            break
                        time.sleep(.15)
            if result is None:
                poll()
            # Capture the current view while the source is still active, then stop normally.
            latest = api(run_path + "/latest")
            projection = latest["projection"]
            current = projection.get("current") or {}
            emit("latest_projection", current_applicable=projection.get("current_applicable"),
                 **select(current, "status", "decision_id", "fall_probability", "source_window"))
            summary = {"run_id": run["run_id"], "video": args.video.name, "result": result,
                       "provider": run["provider_status"]["provider"], "sampled_frames": submitted,
                       "reached_eof": reached_eof, "rgb_panel": "right" if args.right_rgb_panel else "full",
                       "active_projection": select(projection, "current_applicable", "current", "stale_reason_codes"),
                       "http_budget": latest["provider_status"]["http_budget"]}
            (args.out / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            write_result_page(args.out, summary)
        finally:
            capture.release()
            api(run_path + "/stop", {})
    print(json.dumps({"run_id": run["run_id"], "result": result, "output": str(args.out)}, ensure_ascii=False))
    return 0 if matched_ready(summary) else 2


if __name__ == "__main__":
    raise SystemExit(main())

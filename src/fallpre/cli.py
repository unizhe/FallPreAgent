"""Replay first CLI; live side effects require explicit deployment flags."""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import RuleConfig, load_rules, load_site


def _json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _output_dir(argument, prefix):
    path = Path(argument or f"runs/{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(args):
    from .models import download_models
    _json(download_models(args.models))


def calibrate(args):
    from .geometry.calibration import fit_calibration
    from .schemas import new_id
    site, rules = load_site(args.input), load_rules(args.rules)
    matrix, error = fit_calibration(site.calibration_image_points, site.calibration_world_points,
                                    site.validation_image_points, site.validation_world_points,
                                    rules.calibration_max_error_m)
    site = site.model_copy(update={"homography": matrix, "validation_error_m": error,
                                   "calibration_validated": True, "calibration_id": new_id("calibration")})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(site.model_dump_json(indent=2) + "\n")
    _json({"site": str(Path(args.output).resolve()), "max_validation_error_m": error,
           "next": "Measure ankle-proxy error and register footprint/background before live geometry use"})


def tools_schema(args):
    from .service import MVPService
    service = MVPService(load_rules(args.rules), dry_run=True)
    result = service.tools.list_tools()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    else:
        _json(result)
    service.close()


def demo(args):
    from .demo import run_demo
    directory = _output_dir(args.output, "demo")
    if args.trace:
        from .showcase import run_showcase
        _json(run_showcase(directory))
    else:
        _json(run_demo(directory))


def _overlay(image, snapshot, humans):
    import cv2
    canvas = image.copy()
    for human in humans:
        x1, y1, x2, y2 = map(int, human.bbox_px)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 210, 80), 2)
        cv2.putText(canvas, f"track={human.track_id}", (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 210, 80), 1)
        for x, y, confidence in human.keypoints:
            if confidence >= 0.5:
                cv2.circle(canvas, (int(x), int(y)), 2, (50, 180, 255), -1)
    lines = [f"t={snapshot.t_s:.2f}s binding={snapshot.binding_status}",
             f"behavior={snapshot.behavior.value} fall={snapshot.fall.phase}",
             f"spatial={snapshot.spatial.ttc_status} TTC={snapshot.spatial.ttc_s}",
             f"camera_stable={snapshot.quality.camera_stable}"]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 24+i*23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 50, 240), 1)
    return canvas


def smoke_models(args):
    import cv2

    from .models import BackpackDetector
    from .perception.human import HumanPerceptor
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Cannot read {args.image}")
    output = _output_dir(args.output, "model-smoke")
    pose = HumanPerceptor(str(Path(args.models) / "yolo11n-pose.pt"), device=args.device, rules=RuleConfig())
    detector = BackpackDetector(str(Path(args.models) / "yolo11n.pt"), device=args.device)
    timings, tracks = [], []
    for i in range(3):
        begin = time.perf_counter()
        humans = pose.infer(image, i / 10, i)
        timings.append(time.perf_counter() - begin)
        tracks.append([h.track_id for h in humans])
    detections = detector.infer(image)
    result = {"image": str(Path(args.image).resolve()), "device": args.device,
              "pose_frames": len(timings), "people_in_last_frame": len(humans),
              "keypoints_per_person": [len(h.keypoints) for h in humans], "track_ids": tracks,
              "pose_wall_seconds": timings, "backpack_detections": detections,
              "tracker": "bytetrack.yaml", "scope": "model loading/inference only, not accuracy or deployment FPS"}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "humans.json").write_text(json.dumps([h.model_dump(mode="json") for h in humans], indent=2) + "\n")
    from .schemas import Snapshot
    canvas = _overlay(image, Snapshot(t_s=0.2, person_id="unbound", binding_id="none",
                                      binding_status="UNBOUND"), humans)
    cv2.imwrite(str(output / "pose-overlay.jpg"), canvas)
    _json(result)


def reference_frame(args):
    import cv2

    from .capture import VideoSource
    from .geometry.calibration import Calibration
    source_arg = args.camera if args.camera is not None else args.video
    with VideoSource(source_arg) as source:
        frame = source.read(timeout=5)
        if frame is None:
            raise RuntimeError("No frame captured")
        image = Calibration(load_site(args.site)).undistort(frame.image) if args.site else frame.image
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(args.output, image):
            raise RuntimeError("Could not write background image")
    _json({"reference": str(Path(args.output).resolve()), "image_size": [image.shape[1], image.shape[0]],
           "next": "Set background_reference and select a static background ROI in your site config"})


def run(args):
    import cv2

    from .capture import VideoSource
    from .pipeline import VideoPipeline
    from .service import MVPService
    site, rules = load_site(args.site), load_rules(args.rules)
    if site.synthetic:
        raise ValueError("Synthetic calibration cannot be used on camera/recorded images; use a measured site config")
    live = args.camera is not None
    if args.enable_actions and not live:
        raise ValueError("Replay is always dry-run; live channels are available only with --camera")
    handler = None
    if args.enable_actions:
        from .adapters.channels import ConfiguredChannels
        url = os.environ.get(args.webhook_env) if args.webhook_env else None
        if args.webhook_env and (url is None or not url.strip()):
            raise ValueError(f"Configured webhook environment variable is empty: {args.webhook_env}")
        handler = ConfiguredChannels(speech=args.speech, webhook_url=url, voice=args.speech_voice)
    elif args.speech or args.webhook_env:
        raise ValueError("Channels require --enable-actions; otherwise omit channel flags for dry-run")
    output = _output_dir(args.output, "live" if live else "replay")
    database = output / "events.sqlite3"
    if database.exists():
        raise FileExistsError("Output contains an existing session; select a new output directory")
    pipeline = VideoPipeline(site, rules, args.track_id, str(Path(args.models) / "yolo11n-pose.pt"),
                             str(Path(args.models) / "yolo11n.pt"), args.device, "camera" if live else "video")
    if live:
        pipeline.warmup()
    service = MVPService(rules, db_path=str(database), dry_run=not args.enable_actions, handler=handler)
    source = VideoSource(args.camera if live else args.video, args.fps)
    result_queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    worker = None
    writer = None
    frames = 0
    timestamps = set()
    start = time.perf_counter()
    def infer_live():
        try:
            while not stop.is_set():
                frame = source.read()
                if frame is None:
                    if source.ended:
                        break
                    continue
                result = (*pipeline.process(frame), frame.timestamp_source)
                while not stop.is_set():
                    try:
                        result_queue.put(result, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as error:
            result_queue.put(error)
        finally:
            source.ended = True
    if live:
        worker = threading.Thread(target=infer_live, name="perception", daemon=True)
        worker.start()
    state_stream = (output / "snapshots.jsonl").open("w")
    decision_stream = (output / "decisions.jsonl").open("w")
    seen_decisions = set()
    def record_decisions(decisions):
        for decision in decisions:
            if decision.decision_id not in seen_decisions:
                seen_decisions.add(decision.decision_id)
                decision_stream.write(decision.model_dump_json() + "\n")
    try:
        while args.max_frames is None or frames < args.max_frames:
            if live:
                record_decisions(service.tick(source.now()))
                try:
                    result = result_queue.get(timeout=0.1)
                except queue.Empty:
                    if source.ended and (worker is None or not worker.is_alive()):
                        unresolved = [e for e in service.store.list_events()
                                      if e.outcome == "WAIT" or (e.status == "ACTIVE" and not e.confirmed_alarm)]
                        if not unresolved:
                            break
                    continue
                if isinstance(result, Exception):
                    raise result
                snapshot, image, humans, timestamp_kind = result
            else:
                frame = source.read()
                if frame is None:
                    break
                snapshot, image, humans = pipeline.process(frame)
                timestamp_kind = frame.timestamp_source
            timestamps.add(timestamp_kind)
            decisions = service.ingest_snapshot(snapshot, auto_execute=True)
            pipeline.core.feedback(decisions)
            state_stream.write(snapshot.model_dump_json() + "\n")
            record_decisions(decisions)
            frames += 1
            if args.overlay or args.show:
                canvas = _overlay(image, snapshot, humans)
                if args.overlay:
                    if writer is None:
                        writer = cv2.VideoWriter(str(output / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 source.fps if 0 < source.fps <= 120 else 15,
                                                 (canvas.shape[1], canvas.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create overlay video")
                    writer.write(canvas)
                if args.show:
                    cv2.imshow("FallPre MVP - q to stop", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if worker:
            worker.join(timeout=2)
        source.close()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        state_stream.close()
        decision_stream.close()
        service.close()
    result = {"output": str(output.resolve()), "frames_processed": frames,
              "wall_seconds": time.perf_counter()-start, "dry_run": not args.enable_actions,
              "binding_status": pipeline.core.binding_status, "session_id": pipeline.core.session_id,
              "timestamp_sources": sorted(timestamps), "created_at": datetime.now(timezone.utc).isoformat(),
              "source_type": "camera" if live else "video", "rules": rules.model_dump(),
              "notes": ["End of replay preserves unresolved events; no fabricated post-EOF observations",
                        "Live overlay timing is approximate; source timestamps are in snapshots.jsonl"]}
    (output / "session.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _json(result)


def main():
    parser = argparse.ArgumentParser(description="FallPre MVP: offline replay and explicitly configured live perception")
    commands = parser.add_subparsers(dest="command", required=True)
    def common_models(p):
        p.add_argument("--models", default="models")
        p.add_argument("--device", default="cpu")
    def source_flags(p):
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--video")
        group.add_argument("--camera", type=int)
    p = commands.add_parser("download-models")
    p.add_argument("--models", default="models")
    p.set_defaults(func=download)
    p = commands.add_parser("calibrate")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rules")
    p.set_defaults(func=calibrate)
    p = commands.add_parser("tools-schema")
    p.add_argument("--output")
    p.add_argument("--rules")
    p.set_defaults(func=tools_schema)
    p = commands.add_parser("demo")
    p.add_argument("--output")
    p.add_argument("--trace", action="store_true", help="Record actual scripted Agent tool calls for visual replay")
    p.set_defaults(func=demo)
    p = commands.add_parser("smoke-models")
    common_models(p)
    p.add_argument("--image", default="third_party/ultralytics/ultralytics/assets/bus.jpg")
    p.add_argument("--output")
    p.set_defaults(func=smoke_models)
    p = commands.add_parser("reference-frame")
    source_flags(p)
    p.add_argument("--site")
    p.add_argument("--output", required=True)
    p.set_defaults(func=reference_frame)
    p = commands.add_parser("run")
    source_flags(p)
    common_models(p)
    p.add_argument("--site", required=True)
    p.add_argument("--rules")
    p.add_argument("--track-id", type=int)
    p.add_argument("--fps", type=float)
    p.add_argument("--output")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--overlay", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--enable-actions", action="store_true")
    p.add_argument("--speech", action="store_true")
    p.add_argument("--speech-voice", default="Tingting", help="Installed macOS Chinese voice")
    p.add_argument("--webhook-env", help="Environment variable containing the configured HTTPS endpoint")
    p.set_defaults(func=run)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

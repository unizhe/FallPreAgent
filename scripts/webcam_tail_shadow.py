"""Opt-in local camera/video service for the frozen tail candidate."""
import argparse
import json
import secrets
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from webcam_pose_demo import PoseEngine, make_handler

from fallpre.care_loop import live_perception_payload
from fallpre.runtime_resources import resource_path
from fallpre.weekly.tail_process_shadow import TailProcessRuntime, TailShadow
from fallpre.weekly.webcam_temporal import TemporalService


class TailShadowEngine(PoseEngine):
    def __init__(self, head, log, *, device="cpu", established_fusion=False):
        self.shadow_streams = {}
        self.shadow_log = log
        self.log_error = None
        super().__init__(device=device, fall_mode="rgb", temporal_enabled=False)
        self.runtime = TailProcessRuntime()
        self.runtime.load_head(head)
        self.temporal_service = TemporalService(self.runtime)
        self.fall_temporal_enabled = established_fusion
        self.fall_event_fusion_enabled = established_fusion

    def temporal_status(self):
        status = super().temporal_status()
        status["tail_process_shadow"] = dict(
            enabled=True, changes_alarm=False, mode="shadow",
            checkpoint_sha256=self.runtime.active_head[1],
            development_heldout_group=self.runtime.active_head[2],
            logging="error" if self.log_error else "local_jsonl",
        )
        return status

    def stop_stream(self, stream_id):
        self.shadow_streams.pop(stream_id, None)
        super().stop_stream(stream_id)

    def infer_live(self, image, stream_id, source_time, model_id=None,
                   source_type="camera", fall_mode=None):
        started = time.perf_counter()
        result = super().infer_live(image, stream_id, source_time, model_id, source_type, fall_mode)
        if result["fall_mode"] != "rgb":
            return result
        raw = result
        if "baseline_fall" in result:
            raw = dict(result, fall=result["baseline_fall"],
                       fall_observation=result["baseline_fall_observation"])
        try:
            shadow = self.shadow_streams.get(stream_id)
            if shadow is None:
                shadow = self.shadow_streams[stream_id] = TailShadow()
            candidate = shadow.update(raw, source_time)
        except Exception as error:
            self.shadow_streams.pop(stream_id, None)
            result["tail_process_shadow"] = dict(mode="shadow", status="error",
                                                 changes_alarm=False, error=type(error).__name__)
            return result
        result["tail_process_shadow"] = candidate
        if not self.log_error:
            head = candidate["head"]
            evidence = candidate["evidence"]
            temporal = result["temporal"]
            alarm = result["fall"]
            simulated = candidate["simulated_fall"]
            log = dict(stream_id=stream_id, source_type=source_type, source_time_s=source_time,
                       observed_wall_s=time.time(),
                       processing_ms=(time.perf_counter()-started)*1000,
                       current_action=result["action"]["label"], primary_state=alarm["state"],
                       primary_new_event=alarm["new_event"], primary_event_id=alarm["current_event_id"],
                       shadow_state=simulated["state"], shadow_new_event=simulated["new_event"],
                       shadow_event_id=simulated["current_event_id"], head=head,
                       inference_ms=temporal.get("inference_ms"),
                       result_age_s=temporal.get("result_age_s"),
                       association=evidence["association"], reason_code=evidence["code"],
                       support=evidence["support"], changes_alarm=False)
            try:
                log["perception"] = live_perception_payload(stream_id, result)
                self.shadow_log.write(json.dumps(log, allow_nan=False) + "\n")
                self.shadow_log.flush()
            except (OSError, ValueError) as error:
                self.log_error = type(error).__name__
                candidate["logging_error"] = self.log_error
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--head", type=Path,
                        help="Explicit experimental checkpoint; default is the verified handoff checkpoint")
    parser.add_argument("--log", type=Path, required=True, help="A new local JSONL file")
    parser.add_argument("--fall-event-fusion", action="store_true",
                        help="Keep the established NTU-based primary fusion enabled if already in use")
    args = parser.parse_args()
    try:
        head = args.head if args.head is not None else resource_path("fall-head-tail-joint")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    with args.log.open("x", encoding="utf-8") as log:
        engine = TailShadowEngine(head, log, device=args.device,
                                  established_fusion=args.fall_event_fusion)
        try:
            with ThreadingHTTPServer(("127.0.0.1", args.port),
                                     make_handler(engine, args.port, secrets.token_urlsafe(32))) as server:
                server.daemon_threads = True
                print(f"Shadow service http://127.0.0.1:{args.port}/", flush=True)
                server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            engine.close()


if __name__ == "__main__":
    main()

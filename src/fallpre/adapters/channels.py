"""Configured channels; each channel is attempted independently with fixed content."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PHRASES = {
    "OBSTACLE_REMINDER": "前方有障碍物，请减速，确认脚下安全。",
    "STOP_REMINDER": "请停下，站稳后再继续。",
    "FALL_ALARM": "检测到符合报警规则的跌倒事件，请照护人员立即查看。",
    "REQUEST_CHECK": "当前风险事件仍待核查，尚未确认跌倒，请照护人员查看。",
}


class ChannelDeliveryError(RuntimeError):
    def __init__(self, failures, receipts):
        # Do not put endpoint URLs, response bodies or credentials into logs.
        super().__init__("; ".join(f"{name}:{kind}" for name, kind in failures.items()))
        self.delivery_receipt = {"receipts": receipts, "failures": failures, "partial": bool(receipts)}


class ConfiguredChannels:
    def __init__(self, *, speech: bool = False, webhook_url: str | None = None,
                 cache_dir: str | Path = "runs/audio-cache", voice: str = "Tingting"):
        if not speech and not webhook_url:
            raise ValueError("No execution channel configured")
        if speech and platform.system() != "Darwin":
            raise ValueError("This MVP speech backend currently requires macOS say/afplay")
        if webhook_url and not webhook_url.startswith("https://"):
            raise ValueError("Webhook must use HTTPS")
        self.speech, self.webhook_url = speech, webhook_url
        self.voice = voice
        self.cache_dir = Path(cache_dir)
        self._lock = threading.RLock()
        self._receipts: dict[str, dict] = {}

    def _send_webhook(self, action_type, decision, key, phrase):
        payload = {"event_id": decision.event_id, "person_id": decision.person_id,
                   "action_type": action_type, "message": phrase,
                   "confirmed_alarm": decision.confirmed_alarm, "idempotency_key": key}
        request = urllib.request.Request(self.webhook_url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Idempotency-Key": key}, method="POST")
        with urllib.request.urlopen(request, timeout=8) as response:
            return {"http_status": response.status, "meaning": "endpoint acknowledged; human receipt unknown"}

    def _play_speech(self, phrase):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (hashlib.sha256(f"{self.voice}\n{phrase}".encode()).hexdigest()[:16] + ".aiff")
        if not path.exists():
            partial = path.with_suffix(".partial.aiff")
            subprocess.run(["say", "-v", self.voice, "-o", str(partial), phrase], check=True, timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            partial.replace(path)
        subprocess.run(["afplay", str(path)], check=True, timeout=25,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"status": "PLAYBACK_COMPLETED", "audio_file": str(path.resolve())}

    def __call__(self, action_type, decision, idempotency_key):
        phrase = PHRASES.get(action_type)
        if phrase is None:
            raise ValueError(f"No fixed phrase for action {action_type}")
        jobs = {}
        if self.webhook_url and action_type in ("FALL_ALARM", "REQUEST_CHECK"):
            jobs["webhook"] = lambda: self._send_webhook(action_type, decision, idempotency_key, phrase)
        if self.speech:
            jobs["speech"] = lambda: self._play_speech(phrase)
        if not jobs:
            raise RuntimeError("This action needs a speech channel; webhook accepts alarms/checks only")
        # Successful channels are not repeated during in-process retries. A process
        # crash still requires endpoint idempotency; do not promise exactly-once I/O.
        with self._lock:
            receipts = self._receipts.setdefault(idempotency_key, {})
            pending = {name: job for name, job in jobs.items() if name not in receipts}
            failures = {}
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fallpre-channel") as pool:
                futures = {pool.submit(job): name for name, job in pending.items()}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        receipts[name] = future.result()
                    except Exception as error:  # independent I/O boundary
                        failures[name] = type(error).__name__
            if failures:
                raise ChannelDeliveryError(failures, dict(receipts))
            return {"channel": "+".join(sorted(receipts)), "receipts": dict(receipts)}

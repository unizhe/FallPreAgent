"""One capture per source. Live capture drops old frames instead of queuing latency."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    frame_id: int
    t_s: float
    timestamp_source: str


class VideoSource:
    def __init__(self, source: str | int, fps_override: float | None = None):
        self.live = isinstance(source, int)
        if not self.live and not Path(source).is_file():
            raise FileNotFoundError(source)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.fps = fps_override or float(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.live and self.fps <= 0:
            self.cap.release()
            raise ValueError("Video has no usable frame rate; supply --fps")
        self.start = time.monotonic()
        self.frame_id = 0
        self.last_t: float | None = None
        self.ended = False
        self.stop_event = threading.Event()
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self.thread: threading.Thread | None = None
        if self.live:
            self.thread = threading.Thread(target=self._capture_live, daemon=True, name="video-capture")
            self.thread.start()

    def now(self) -> float:
        return time.monotonic() - self.start if self.live else (self.last_t or 0.0)

    def _capture_live(self):
        try:
            while not self.stop_event.is_set():
                ok, image = self.cap.read()
                if not ok:
                    break
                frame = Frame(image, self.frame_id, self.now(), "capture_monotonic")
                self.frame_id += 1
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(frame)
        finally:
            # Release on the owning reader thread, never concurrently with read().
            self.cap.release()
            self.ended = True

    def read(self, timeout: float = 0.1) -> Frame | None:
        if self.live:
            try:
                return self.frames.get(timeout=timeout)
            except queue.Empty:
                return None
        ok, image = self.cap.read()
        if not ok:
            self.ended = True
            return None
        pts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        # Some OpenCV decoders expose no PTS. Record the CFR fallback explicitly.
        if self.last_t is None or pts > self.last_t:
            t_s, kind = pts, "decoder_pts"
        elif pts == 0.0:
            t_s, kind = self.frame_id / self.fps, "cfr_fps_fallback"
        else:
            raise ValueError("Non-monotonic decoder PTS; split the recording into separate sessions")
        if self.last_t is not None and t_s <= self.last_t:
            raise ValueError("Source timestamps did not advance")
        self.last_t = t_s
        result = Frame(image, self.frame_id, t_s, kind)
        self.frame_id += 1
        return result

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        else:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

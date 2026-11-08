"""Official weights and a small adapter for supported environment detections."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

MODEL_NAMES = ("yolo11n-pose.pt", "yolo11n.pt", "yolo26n-pose.pt")
RELEASE = "v8.4.0"


def download_models(destination: str | Path = "models") -> dict:
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = Path(__file__).resolve().parents[2] / "models.lock.json"
    expected = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    result = {}
    for name in MODEL_NAMES:
        path = directory / name
        url = f"https://github.com/ultralytics/assets/releases/download/{RELEASE}/{name}"
        if not path.exists():
            partial = path.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "FallPreAgent-MVP/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            partial.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in expected and digest != expected[name]["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}; file left for inspection")
        result[name] = {"url": url, "sha256": digest, "size_bytes": path.stat().st_size}
    (directory / "download-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class BackpackDetector:
    def __init__(self, weights: str, device: str = "cpu"):
        from ultralytics import YOLO
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Missing {weights}; run fallpre download-models first")
        self.model = YOLO(weights)
        self.device = device
        self.class_id = next((i for i, label in self.model.names.items() if label == "backpack"), None)
        if self.class_id is None:
            raise ValueError("Detector weights must contain the COCO backpack class")

    def infer(self, frame) -> list[dict]:
        result = self.model.predict(frame, classes=[self.class_id], conf=0.25,
                                    device=self.device, verbose=False)[0]
        return [{"bbox_px": tuple(map(float, box.xyxy[0].tolist())),
                 "score": float(box.conf[0]), "class_name": "backpack"}
                for box in result.boxes]

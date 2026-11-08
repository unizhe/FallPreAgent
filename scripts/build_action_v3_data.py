"""Construct derived videos and immutable split/clock/annotation metadata."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from prepare_action_v2 import crop
from prepare_action_v3 import labels, sha

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "runs/action-v3/inputs"


def video_info(p):
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Several original Weizmann AVIs under-report frame count in their headers.
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    if fps <= 0 or n <= 0:
        raise ValueError(f"Invalid video {p}")
    return fps, n


def entry(cid, p, split, segments, source, **kwargs):
    fps, n = video_info(p)
    return dict(
        clip_id=cid,
        source_group_id=cid,
        split=split,
        video_path=str(p.relative_to(ROOT)),
        video_sha256=sha(p),
        fps=fps,
        frame_count=n,
        duration_s=n / fps,
        segments=segments,
        source=source,
        **kwargs,
    )


def segment(a, b, label):
    return dict(start_s=a, end_s=b, label=label)


def contact(p, dest, frames=None):
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = frames if frames is not None else np.linspace(0, n - 1, 12, dtype=int)
    canvas = np.zeros((3 * 160, 4 * 220, 3), np.uint8)
    for i, f in enumerate(frames[:12]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (220, 140))
        y, x = divmod(i, 4)
        canvas[y * 160 : y * 160 + 140, x * 220 : (x + 1) * 220] = img
        cv2.putText(
            canvas,
            f"frame {f + 1} / {f / fps:.2f}s",
            (x * 220 + 4, y * 160 + 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )
    cap.release()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), canvas)


def build():
    old = json.loads((ROOT / "configs/action-v2-dataset.json").read_text())
    entries = []
    for e in old["entries"]:
        e = dict(e)
        e["legacy_v2"] = True
        if e["split"] == "test":
            e["split"] = "dev"
        e["previous_split"] = next(x["split"] for x in old["entries"] if x["clip_id"] == e["clip_id"])
        entries.append(e)
    acq = json.loads((OUT / "utk/acquisition.json").read_text())
    ll = labels()
    for group, a in ll.items():
        subject = int(group[1:3])
        split = "train" if subject <= 5 else "dev" if subject <= 7 else "test"
        all_rows = sorted(
            [r for r in acq["records"] if r["group"] == group], key=lambda r: r["source_frame_id"]
        )
        for kind, lo, hi, actions in [
            ("walk", a["walk"][0] - 30, a["walk"][1] + 30, ["walk"]),
            ("transition", a["sitDown"][0] - 30, a["standUp"][1] + 30, ["sitDown", "standUp"]),
        ]:
            rows = [r for r in all_rows if lo <= r["source_frame_id"] <= hi]
            first = rows[0]["source_frame_id"]
            times = [(r["source_frame_id"] - first) / 30 for r in rows]
            duration = times[-1] + 1 / 30
            fps = len(rows) / duration
            p = OUT / "utk/videos" / f"{group}-{kind}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
                for r in rows:
                    image = cv2.imread(str(ROOT / r["path"]))
                    assert image.shape[:2] == (480, 640)
                    writer.write(image)
                writer.release()
            mapping = {"walk": "walking", "sitDown": "sitting_down", "standUp": "standing_up"}
            segs = [
                segment(
                    max(0, (a[act][0] - first) / 30),
                    min(duration, (a[act][1] + 1 - first) / 30),
                    mapping[act],
                )
                for act in actions
            ]
            # Shared boundary frame belongs to the later action, explicitly recorded.
            for left, right in zip(segs, segs[1:]):
                left["end_s"] = min(left["end_s"], right["start_s"])
            states = []
            if kind == "transition":
                start = (a["sitDown"][1] + 4 - first) / 30
                end = (a["standUp"][0] - 3 - first) / 30
                if end > start:
                    segs.append(segment(start, end, "background"))
                    states = [dict(start_s=start, end_s=end, state="seated")]
                start = (a["standUp"][1] + 4 - first) / 30
                if duration > start:
                    segs.append(segment(start, duration, "background"))
            e = entry(
                "utk-" + group + "-" + kind,
                p,
                split,
                sorted(segs, key=lambda x: x["start_s"]),
                dict(
                    url=acq["url"],
                    labels_url="https://cvrc.ece.utexas.edu/KinectDatasets/actionLabel.txt",
                    labels_sha256=acq["labels_sha256"],
                ),
                dataset="UTKinect",
                subject_id=f"utk-s{subject:02}",
                parent_source_id="utk-" + group,
                source_times_s=times,
                source_original_frame_ids=[r["source_frame_id"] for r in rows],
                state_segments=states,
                annotation_basis="Official action intervals; intervening seated and post-stand states inferred from official ordering with margins, agent raw-contact review before training confirms seated middle frames; no human adjudication. No pickUp-to-bending mapping.",
            )
            # Source clock is authoritative; video FPS is only CFR preview timing.
            e["duration_s"] = duration
            entries.append(e)
            if kind == "transition":
                contact(p, ROOT / "runs/action-v3/review" / f"{group}.jpg")
    train = {"daria", "denis", "eli", "ido", "ira"}
    dev = {"lena"}
    test = {"lyova", "moshe", "shahar"}
    for action in ["walk", "bend"]:
        files = sorted((OUT / "weizmann" / action).glob("*.avi"))
        if not files and "--stage" not in sys.argv:
            raise ValueError(f"Missing Weizmann {action}")
        for p in files:
            subject = p.stem.split("_")[0].lower()
            split = (
                "train"
                if action == "robust_view" or subject in train
                else "dev"
                if subject in dev
                else "test"
                if subject in test
                else None
            )
            if split is None:
                raise ValueError(f"Unregistered subject {subject}")
            fps, n = video_info(p)
            entries.append(
                entry(
                    "weiz-" + p.stem,
                    p,
                    split,
                    [segment(0, n / fps, "bending" if action == "bend" else "walking")],
                    dict(
                        url="https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html",
                        mirror_url="https://www.kaggle.com/datasets/wendguan/weizmann",
                        file_sha256=sha(p),
                        acquisition_sha256=sha(OUT / "weizmann/acquisition.json"),
                    ),
                    dataset="Weizmann",
                    subject_id="weiz-" + subject if action != "robust_view" else "weiz-viewpoint-unverified",
                    annotation_basis="Publisher clip-level class; applied as weak temporal labels, not a frame-annotated action onset benchmark.",
                )
            )
    states = {}
    for r in csv.reader((ROOT / "runs/weekly-20261015/labels/urfall-cam0-falls.csv").open()):
        states.setdefault(r[0], {})[int(r[1])] = int(r[2])
    for n in range(17, 31):
        name = f"fall-{n:02}"
        src = OUT / "urfd" / f"{name}-cam0.mp4"
        p = src.with_name(f"{name}-rgb.mp4")
        crop(src, p)
        fps, count = video_info(p)
        official = {str(k): v for k, v in states[name].items() if k <= count}
        segs = []
        for state, label in [(0, "falling"), (1, "background")]:
            frames = sorted(int(f) for f, v in official.items() if v == state)
            if frames:
                first = last = frames[0]
                for f in frames[1:]:
                    if f != last + 1:
                        segs.append(segment((first - 1) / fps, last / fps, label))
                        first = f
                    last = f
                segs.append(segment((first - 1) / fps, last / fps, label))
        entries.append(
            entry(
                "urfd-" + name,
                p,
                "train" if n <= 20 else "dev" if n <= 22 else "test",
                sorted(segs, key=lambda s: s["start_s"]),
                dict(url=f"https://fenix.ur.edu.pl/~mkepski/ds/data/{name}-cam0.mp4", sha256=sha(src)),
                dataset="URFD",
                official_frame_states=official,
                annotation_basis="Official state 0 falling, 1 lying/background, -1 unknown class / negative binary falling.",
            )
        )
    result = dict(
        schema="fallpre.action_v3.dataset.v1",
        protocol_sha256=sha(ROOT / "configs/action-v3-protocol.json"),
        entries=entries,
        independent_human_annotators=0,
        test_independence="UTKinect and standard Weizmann subject-disjoint by publisher IDs; URFD new videos only; legacy_v2 never test",
    )
    target = ROOT / (
        "runs/action-v3/preparation-manifest.json"
        if "--stage" in sys.argv
        else "configs/action-v3-dataset.json"
    )
    if target.exists():
        raise FileExistsError("Do not overwrite a prepared annotation manifest")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(Counter(e["split"] for e in entries))


if __name__ == "__main__":
    from collections import Counter

    build()

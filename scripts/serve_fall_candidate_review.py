"""Local human review of fourteen fall candidates, with versioned JSON saves."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from serve_stgcn_fusion_report import handler as media_handler  # noqa: E402

from fallpre.weekly.stgcn_fall_head import guard_destination, reviewed_regions  # noqa: E402

SOURCE = ROOT / 'runs/stgcn-fall-20261107/data'
DEFAULT_OUTPUT = ROOT / 'runs/stgcn-fall-review-20261107'
KINDS = {'simulated_fall', 'real_fall', 'intentional_lying', 'uncertain'}
QUALITIES = {'observable', 'occluded', 'unobservable', 'uncertain'}
TIMES = ('onset_s', 'landing_s', 'recovery_start_s', 'recovery_end_s')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


class ReviewStore:
    def __init__(self, output, source=SOURCE):
        guard_destination(output, ROOT / 'runs')
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((source / 'manifest.json').read_text())
        self.candidates = json.loads((source / 'review-candidates.json').read_text())['events']
        self.clips = {c['id']: c for c in self.manifest['clips']}
        self.source_hash = sha(source / 'review-candidates.json')
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.draft = output / 'draft.json'
        self.media = output / 'media'
        guard_destination(self.draft, ROOT / 'runs')
        guard_destination(self.media, ROOT / 'runs')
        self.media.mkdir(exist_ok=True)
        for cid in {e['clip_id'] for e in self.candidates}:
            video = self.clips[cid]['video']
            path = Path(video['path'])
            if sha(path) != video['sha256']:
                raise ValueError('Original video changed: ' + cid)
            link = self.media / (cid + '.webm')
            if link.is_symlink():
                if link.resolve() != path.resolve():
                    raise ValueError('Unexpected media link')
            elif link.exists():
                raise ValueError('Refuse to replace existing media')
            else:
                link.symlink_to(os.path.relpath(path, self.media))

    def revision(self):
        return sha(self.draft) if self.draft.exists() else None

    def bootstrap(self):
        with self.lock:
            draft = json.loads(self.draft.read_text()) if self.draft.exists() else None
            if draft and draft.get('candidate_source_sha256') != self.source_hash:
                raise ValueError('Draft belongs to a different candidate set')
            candidates = []
            counters = {}
            for e in self.candidates:
                cid = e['clip_id']
                counters[cid] = counters.get(cid, 0) + 1
                duration = self.clips[cid]['duration_s']
                initial = {k: e[k] if e[k] is not None and e[k] <= duration else None for k in TIMES}
                candidates.append(dict(e, initial_times=initial, duration_s=duration,
                    title=self.clips[cid]['title'], short_id={'new': 'A', 'natural': 'B', 'fall': 'C'}[cid]
                    + str(counters[cid]), media_url='/media/' + cid + '.webm'))
            return dict(candidates=candidates, draft=draft, revision=self.revision(),
                        token=self.token, output_directory=str(self.output), source_sha256=self.source_hash)

    def validate(self, payload, final=False):
        entries = payload.get('entries')
        reviewer = payload.get('reviewed_by', '')
        if not isinstance(reviewer, str) or len(reviewer) > 100:
            raise ValueError('请填写有效复核人称呼')
        if not isinstance(entries, list) or len(entries) != len(self.candidates):
            raise ValueError('需要完整的 14 个候选记录')
        results = []
        for index, (entry, original) in enumerate(zip(entries, self.candidates, strict=True)):
            if not isinstance(entry, dict) or entry.get('event_id') != original['event_id'] \
                    or entry.get('clip_id') != original['clip_id']:
                raise ValueError('候选身份或顺序发生变化')
            done = entry.get('confirmed') is True
            kind = entry.get('action_type', '')
            quality = entry.get('quality', {})
            resolution = entry.get('boundary_resolution', '')
            if kind not in KINDS | {''} or not isinstance(quality, dict) \
                    or any(v not in QUALITIES | {''} for v in quality.values()):
                raise ValueError('无效动作或观测质量')
            if resolution not in {'', 'confirmed', 'unresolved'}:
                raise ValueError('无效时间边界确认状态')
            if final and not done:
                raise ValueError(f'第 {index + 1} 段尚未确认')
            if done and (not reviewer.strip() or kind not in KINDS or resolution == ''
                         or any(quality.get(k) not in QUALITIES for k in ('process', 'lying', 'recovery'))):
                raise ValueError(f'第 {index + 1} 段缺少复核人、动作、可见性或边界结论')
            duration = self.clips[original['clip_id']]['duration_s']
            times = {}
            for key in TIMES:
                value = entry.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                         or not math.isfinite(value) or not 0 <= value <= duration):
                    raise ValueError(f'第 {index + 1} 段时间超出视频范围')
                times[key] = value
            uncertainty = entry.get('boundary_uncertainty_s', .2)
            if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)) \
                    or not math.isfinite(uncertainty) or not 0 <= uncertainty <= duration:
                raise ValueError('请填写有效时间误差')
            note = entry.get('note', '')
            if not isinstance(note, str) or len(note) > 3000:
                raise ValueError('备注过长')
            clean = dict(clip_id=original['clip_id'], event_id=original['event_id'],
                         action_type=kind, quality={k: quality.get(k, '') for k in ('process', 'lying', 'recovery')},
                         boundary_resolution=resolution, confirmed=done, note=note,
                         boundary_uncertainty_s=uncertainty, **times)
            # Confirmed boundaries must satisfy the exact training reader contract.
            if done and resolution == 'confirmed':
                reviewed_regions({'events': [self.event(clean, reviewer)]}, self.manifest)
            results.append(clean)
        return dict(schema_version=1, reviewed_by=reviewer.strip(), entries=results,
                    candidate_source_sha256=self.source_hash)

    def event(self, entry, reviewer):
        original = next(e for e in self.candidates if e['clip_id'] == entry['clip_id']
                        and e['event_id'] == entry['event_id'])
        kind = entry['action_type']
        return dict(entry, video_sha256=original['video_sha256'],
                    event_kind='fall' if kind in {'simulated_fall', 'real_fall'} else kind,
                    demonstration=kind == 'simulated_fall' if kind in {'simulated_fall', 'real_fall'} else True,
                    review_status='human_confirmed' if entry['boundary_resolution'] == 'confirmed'
                    else 'human_reviewed_unresolved', reviewed_by=reviewer,
                    reviewed_at=datetime.now(timezone.utc).isoformat(),
                    uncertainty_s=dict.fromkeys(('onset', 'landing', 'recovery_start', 'recovery_end'),
                                                entry['boundary_uncertainty_s']))

    def save(self, payload, final=False):
        with self.lock:
            if payload.get('revision') != self.revision():
                raise FileExistsError('另一个页面已更新草稿，请刷新后继续，避免覆盖')
            clean = self.validate(payload, final)
            clean['saved_at'] = datetime.now(timezone.utc).isoformat()
            if final:
                events = [self.event(e, clean['reviewed_by']) for e in clean['entries']]
                document = dict(schema_version=1, events=events, non_fall_intervals=[], clip_identities=[],
                                candidate_source_sha256=self.source_hash,
                                note='Unresolved boundaries excluded from training; no automatic negative intervals.')
                reviewed_regions(document, self.manifest)
            guard_destination(self.draft, ROOT / 'runs')
            tmp = self.output / ('.draft-' + secrets.token_hex(8))
            with tmp.open('x') as f:
                f.write(dumps(clean))
            os.replace(tmp, self.draft)
            result = dict(revision=self.revision(), saved_path=str(self.draft))
            if final:
                name = 'reviews-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') \
                    + '-' + secrets.token_hex(4) + '.json'
                dest = self.output / name
                with dest.open('x') as f:
                    f.write(dumps(document))
                result.update(saved_path=str(dest), download_url='/results/' + name,
                              confirmed_events=sum(e['review_status'] == 'human_confirmed' for e in events),
                              unresolved_events=sum(e['review_status'] != 'human_confirmed' for e in events))
            return result


def handler(store, port):
    class Handler(media_handler(store.output, port)):
        def reply(self, status, value):
            body = dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

        def do_GET(self):
            if not self.valid_host():
                return self.send_error(403)
            path = urlsplit(self.path).path
            if path == '/api/review':
                return self.reply(200, store.bootstrap())
            if path.startswith('/results/'):
                name = path.removeprefix('/results/')
                if '/' in name or not name.startswith('reviews-') or not name.endswith('.json'):
                    return self.send_error(404)
                file = store.output / name
                if not file.is_file() or file.is_symlink():
                    return self.send_error(404)
                return self.reply(200, json.loads(file.read_text()))
            return super().do_GET()

        def do_POST(self):
            if not self.valid_host() or self.headers.get('Origin') != 'http://' + self.headers.get('Host', '') \
                    or self.headers.get('X-Review-Token') != store.token:
                return self.send_error(403)
            if self.path not in {'/api/draft', '/api/final'}:
                return self.send_error(404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 200_000:
                    raise ValueError('请求大小无效')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('复核内容格式无效')
                return self.reply(200, store.save(payload, self.path == '/api/final'))
            except FileExistsError as error:
                return self.reply(409, {'error': str(error)})
            except (ValueError, TypeError, KeyError) as error:
                return self.reply(400, {'error': str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8812)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    store = ReviewStore(args.output.absolute())
    page = args.output / 'index.html'
    guard_destination(page, ROOT / 'runs')
    page.write_text((ROOT / 'assets/fall-review/index.html').read_text())
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(store, args.port))
    print(f'Human review: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

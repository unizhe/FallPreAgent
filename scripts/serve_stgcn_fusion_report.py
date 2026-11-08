"""Read-only localhost A/B report server. No directory browsing or uploads."""
import argparse
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def handler(directory, port):
    allowed = {'/', '/index.html', '/clips.json', '/B/metrics.json', '/B/protocol.json',
               '/data/review-queue.json', '/data/visual-review.json', '/data/labels.provisional.json',
               '/data/reviews.template.json', '/data/training-gate.json', '/data/review.html', '/data/README.md',
               *{f'/data/review-contact-{i}.jpg' for i in range(1, 5)}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.deliver(False)

        def do_HEAD(self):
            self.deliver(True)

        def deliver(self, head):
            if self.headers.get('Host') not in {f'127.0.0.1:{port}', f'localhost:{port}'}:
                self.send_error(403)
                return
            path = unquote(urlsplit(self.path).path)
            if not (path in allowed or re.fullmatch(r'/(B|media)/(clip0[1-6]|old|new|natural|fall)\.(jsonl|mp4|webm)', path)):
                self.send_error(404)
                return
            file = directory / ('index.html' if path == '/' else path[1:])
            if not file.is_file():
                self.send_error(404)
                return
            size, start = file.stat().st_size, 0
            end, partial = size-1, False
            if self.headers.get('Range'):
                m = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers['Range'])
                if not m or not any(m.groups()):
                    self.send_error(416)
                    return
                if m[1]:
                    start, end = int(m[1]), min(int(m[2]) if m[2] else end, end)
                else:
                    start = max(0, size-int(m[2]))
                if start > end or start >= size:
                    self.send_error(416)
                    return
                partial = True
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end-start+1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if head:
                return
            try:
                with file.open('rb') as source:
                    source.seek(start)
                    remaining = end-start+1
                    while remaining:
                        chunk = source.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8808)
    parser.add_argument('--directory', type=Path, default=ROOT / 'runs/stgcn-fusion-20261106')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(args.directory.resolve(), args.port))
    print(f'Fusion report: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()

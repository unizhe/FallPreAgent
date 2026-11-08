"""Start isolated localhost candidate/report; never stop an occupied service."""
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/stgcn-fall-20261107'


def main():
    if not (OUT/'B/metrics.json').is_file():
        raise RuntimeError('Complete offline fall replay before starting candidate')
    for port in (8810, 8811):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} already occupied; inspect it, never terminate automatically')
    commands = {
        '8810': [str(ROOT/'.venv/bin/python'), '-B', 'scripts/webcam_pose_demo.py', '--port', '8810',
                 '--pose-model', 'yolo26n-pose', '--fall-mode', 'rgb', '--fall-temporal-shadow'],
        '8811': [str(ROOT/'.venv/bin/python'), '-B', 'scripts/serve_stgcn_fusion_report.py',
                 '--port', '8811', '--directory', str(OUT)],
    }
    processes = {}
    for port, command in commands.items():
        with (OUT/f'server-{port}.log').open('ab') as log:
            processes[port] = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=log, start_new_session=True)
    record = {p:dict(pid=v.pid,command=commands[p]) for p,v in processes.items()}
    (OUT/'servers.json').write_text(json.dumps(record, indent=2)+'\n')
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(90):
        if any(p.poll() is not None for p in processes.values()):
            raise RuntimeError('New candidate/report exited; inspect its log')
        try:
            with client.open('http://127.0.0.1:8810/status',timeout=2) as response:
                status=json.load(response)
            if status.get('ready'):
                assert status['fall_temporal']['mode'] == 'shadow'
                (OUT/'candidate-live-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                return
        except OSError:
            pass
        time.sleep(.5)
    raise RuntimeError('Candidate startup timeout; inspect new PIDs/logs, existing services remain untouched')


if __name__ == '__main__':
    main()

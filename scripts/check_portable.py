"""Run the shareable contract tests without private videos, weights or experiments."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    'test_end_to_end.py', 'test_policy_tools.py', 'test_cli_configuration.py',
    'test_perception_chain.py', 'test_stgcn_fusion.py', 'test_stgcn_fall_evidence.py',
    'test_fall_event_fusion.py', 'test_webcam_fall_handoff.py',
    'test_stgcn_temporal.py', 'test_webcam_temporal.py',
    'test_runtime_resources.py', 'test_source_handoff.py',
    'test_binary_fall_evidence.py', 'test_care_loop.py', 'test_event_interface.py',
    'test_event_store.py', 'test_perception_semantics.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-ui', action='store_true', help='Also run Node recording tests; no browser launched')
    args = parser.parse_args()
    node = shutil.which('node')
    if args.with_ui and node is None:
        parser.error('--with-ui needs Node.js on PATH')
    subprocess.run([sys.executable, '-m', 'pytest', '-o', 'addopts=', '-q',
                    *[str(ROOT / 'tests' / name) for name in TESTS]],
                   cwd=ROOT, check=True)
    if args.with_ui:
        subprocess.run([node, '--test', 'tests/webcam_recording_core.cjs', 'tests/webcam_recording_ui.cjs'],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

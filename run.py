#!/usr/bin/env python3
"""
PropEdge V10.0 — Master Orchestrator
======================================
Repo:       git@github.com:iamwerewolf1007/PropEdgeV10.0.git
Working dir: ~/Documents/GitHub/PropEdgeV10.0

Commands:
  python3 run.py setup      — Configure Git + SSH + install launchd agents
  python3 run.py generate   — Train all V10.0 models + build season JSONs (~4 min)
  python3 run.py 0          — Run Batch 0  (grade + retrain, 07:30 UK)
  python3 run.py 1          — Run Batch 1  (predict, 08:00 UK)
  python3 run.py 2          — Run Batch 2  (predict, 18:00 UK)
  python3 run.py 3          — Run Batch 3  (pre-tip, 21:30 UK)
  python3 run.py all        — Run B0 then B2

Batch schedule (launchd agents, UK time):
  06:00 — batch0_grade.py   (grade, append game logs, retrain all 4 models)
  08:00 — batch_predict.py 1
  18:00 — batch_predict.py 2
  21:30 — batch3_dynamic.sh (pre-tip, dynamic timing)
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def run_batch(n: int):
    if n == 0:
        subprocess.run([sys.executable, str(ROOT / 'batch0_grade.py')], cwd=ROOT)
    else:
        subprocess.run([sys.executable, str(ROOT / 'batch_predict.py'), str(n)], cwd=ROOT)


def setup():
    """Configure Git remote, SSH key, and install all 4 launchd agents."""
    import os, textwrap

    repo_dir = Path.home() / 'Documents' / 'GitHub' / 'PropEdgeV10.0'
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Git remote
    git_remote = 'git@github.com:iamwerewolf1007/PropEdgeV10.0.git'
    subprocess.run(['git','init'], cwd=repo_dir, capture_output=True)
    result = subprocess.run(['git','remote','get-url','origin'], cwd=repo_dir, capture_output=True)
    if result.returncode != 0:
        subprocess.run(['git','remote','add','origin',git_remote], cwd=repo_dir)
        print(f"  ✓ Git remote set to {git_remote}")
    else:
        print(f"  ✓ Git remote already configured")

    # launchd agents
    agents = [
        ('com.propedge.v10.batch0',  '30 7 * * *',  [sys.executable, str(ROOT/'batch0_grade.py')]),
        ('com.propedge.v10.batch1',  '0 8 * * *',  [sys.executable, str(ROOT/'batch_predict.py'),'1']),
        ('com.propedge.v10.batch2',  '0 18 * * *', [sys.executable, str(ROOT/'batch_predict.py'),'2']),
        ('com.propedge.v10.batch3',  '30 21 * * *',[sys.executable, str(ROOT/'batch_predict.py'),'3']),
    ]
    launch_agents_dir = Path.home() / 'Library' / 'LaunchAgents'
    launch_agents_dir.mkdir(parents=True, exist_ok=True)

    for label, schedule, prog_args in agents:
        minute, hour = schedule.split()[:2]
        plist = textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
              "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
                <key>Label</key>
                <string>{label}</string>
                <key>ProgramArguments</key>
                <array>
                    {''.join(f'<string>{a}</string>' for a in prog_args)}
                </array>
                <key>StartCalendarInterval</key>
                <dict>
                    <key>Hour</key>   <integer>{hour}</integer>
                    <key>Minute</key> <integer>{minute}</integer>
                </dict>
                <key>WorkingDirectory</key>
                <string>{ROOT}</string>
                <key>StandardOutPath</key>
                <string>{ROOT}/logs/{label}.stdout.log</string>
                <key>StandardErrorPath</key>
                <string>{ROOT}/logs/{label}.stderr.log</string>
                <key>EnvironmentVariables</key>
                <dict>
                    <key>PATH</key>
                    <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
                </dict>
            </dict>
            </plist>
        """)
        plist_path = launch_agents_dir / f"{label}.plist"
        plist_path.write_text(plist)
        subprocess.run(['launchctl','load','-w',str(plist_path)], capture_output=True)
        print(f"  ✓ launchd agent installed: {label}")

    (ROOT / 'logs').mkdir(exist_ok=True)
    print("\nSetup complete. launchd agents will run on schedule.")
    print("Run 'python3 run.py generate' next to train models and build season JSONs.")


def generate():
    """Train all V10.0 models, then generate both season JSON files."""
    print("="*60)
    print("PropEdge V10.0 — Full Generate")
    print("Step 1: Train all 4 V10.0 models")
    print("="*60)

    result = subprocess.run([sys.executable, '-c', '''
import sys; sys.path.insert(0, ".")
from config import (FILE_GL_2425, FILE_GL_2526, FILE_H2H,
                    FILE_MODEL, FILE_TRUST, FILE_SEG_MODELS,
                    FILE_Q_MODELS, FILE_CALIBRATOR)
from model_trainer import train_and_save
print("Training V10.0 models (projection + segment + quantile + calibrator)...")
train_and_save(
    FILE_GL_2425, FILE_GL_2526, FILE_H2H,
    FILE_MODEL, FILE_TRUST,
    segment_file=FILE_SEG_MODELS,
    quantile_file=FILE_Q_MODELS,
    calibrator_file=FILE_CALIBRATOR,
)
print("All models trained and saved.")
'''], cwd=ROOT)

    if result.returncode != 0:
        print("Model training failed — aborting season JSON generation.")
        sys.exit(1)

    print("\nStep 2: Generate season JSONs")
    subprocess.run([sys.executable, str(ROOT / 'generate_season_json.py')], cwd=ROOT)
    print("\nGenerate complete.")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if   cmd == 'setup':    setup()
    elif cmd == 'generate': generate()
    elif cmd == 'all':      run_batch(0); run_batch(2)
    elif cmd in ('0','1','2','3'): run_batch(int(cmd))
    else: print(f"Unknown command: {cmd}"); print(__doc__)


if __name__ == '__main__': main()

"""
run_hidden.py
=============
Window-less launcher for the scheduled tasks. Run it with pythonw.exe, which has no console:

    D:\\trader_env\\Scripts\\pythonw.exe D:\\Quant_Project\\run_hidden.py <log file> <script.py or exe> [args...]

It runs the command with no window at all (a .py file runs under the venv's python.exe),
appends its output to the log file (relative paths are under D:\\Quant_Project), waits for it
and exits with its exit code. That way the task still counts as running while the job runs,
so "do not start a new instance" keeps working.

Why: the tasks used to run "cmd.exe /c python ...", which opened a console window every
time (every 20 minutes for the Reddit collector). Closing such a window kills the job
(exit code 0xC000013A, seen on the news watcher and the nightly jobs in October 2026).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

PROJECT = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"D:\trader_env\Scripts\python.exe"
CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    if len(sys.argv) < 3:
        return 2
    log = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(PROJECT, sys.argv[1])
    cmd = sys.argv[2:]
    if cmd[0].lower().endswith(".py"):
        cmd = [PYTHON, cmd[0] if os.path.isabs(cmd[0]) else os.path.join(PROJECT, cmd[0])] + cmd[1:]
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="ascii:replace")
    with open(log, "a", encoding="ascii", errors="replace") as f:
        f.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} start: {' '.join(cmd[1:] if cmd[0] == PYTHON else cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           env=env, creationflags=CREATE_NO_WINDOW)
        f.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} exit code {p.returncode}\n")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())

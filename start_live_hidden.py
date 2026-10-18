"""
start_live_hidden.py
====================
Hidden replacement for start_titan_q.bat in the 09:10 task TitanQ_Live_Trading. Run with
pythonw.exe (no console):

    D:\\trader_env\\Scripts\\pythonw.exe D:\\Quant_Project\\start_live_hidden.py

It starts, without any window and only if not already running:
- run_live.py (the trading loop under its supervisor; its log is logs\\titanq_*.log);
- angel_feed.py --record (the Angel tick recorder; logs\\angel_record_*.log);
- dashboard_server.py (http://localhost:5000).
Console output goes to logs\\console_<name>.log. Then it exits; the three keep running.

The .bat opened three console windows, and closing any of them stopped that part of the
system mid-session. start_titan_q.bat is kept for starting by hand with visible windows.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime

PROJECT = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"D:\trader_env\Scripts\python.exe"
CREATE_NO_WINDOW, DETACHED, NEW_GROUP = 0x08000000, 0x00000008, 0x00000200
JOBS = (("run_live.py",), ("angel_feed.py", "--record"), ("dashboard_server.py",))


def running(script: str) -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                          "ForEach-Object { $_.CommandLine }"],
                         capture_output=True, text=True, creationflags=CREATE_NO_WINDOW).stdout
    return any(script in line for line in out.splitlines())


def main() -> None:
    logs = os.path.join(PROJECT, "logs")
    os.makedirs(logs, exist_ok=True)
    for job in JOBS:
        if running(job[0]):
            continue
        name = job[0].replace(".py", "")
        f = open(os.path.join(logs, f"console_{name}.log"), "a", encoding="ascii", errors="replace")
        f.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} start\n")
        f.flush()
        subprocess.Popen([PYTHON, os.path.join(PROJECT, job[0]), *job[1:]], cwd=PROJECT, stdout=f,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=dict(os.environ, PYTHONIOENCODING="ascii:replace"),
                         creationflags=CREATE_NO_WINDOW | DETACHED | NEW_GROUP, close_fds=True)


if __name__ == "__main__":
    main()

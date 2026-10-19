"""
procs.py
========
Small process helper shared by the long background jobs (fundamentals.py, llm_backfill.py,
angel_history.py): is another copy of this script already running? (No psutil in the venv.)
"""

from __future__ import annotations

import os
import subprocess


def other_instance_running(script: str, ignore_args=("--report", "--count")) -> bool:
    """True if another python process runs `script`.

    This process and all of its ancestors are ignored: the venv's python.exe is a launcher
    with a child process, and the scheduled tasks start jobs through run_hidden.py, whose own
    command line names the script. (Ignoring only the parent made every task-started run
    think a copy was already running and exit, 2026-10-18/19.)"""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | ForEach-Object { "
                          "\"$($_.ProcessId)|$($_.ParentProcessId)|$($_.Name)|$($_.CommandLine)\" }"],
                         capture_output=True, text=True, creationflags=0x08000000).stdout
    procs = {}
    for line in out.splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            procs[int(parts[0])] = (int(parts[1]), parts[2].lower(), parts[3])
    mine, pid = set(), os.getpid()
    while pid in procs and pid not in mine:            # this process and its ancestors
        mine.add(pid)
        pid = procs[pid][0]
    mine.add(pid)
    for pid, (_ppid, name, cmd) in procs.items():
        if (pid not in mine and name.startswith("python") and script in cmd
                and "run_hidden.py" not in cmd and not any(a in cmd for a in ignore_args)):
            return True
    return False

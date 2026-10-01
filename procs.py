"""
procs.py
========
Small process helper shared by the long background jobs (fundamentals.py, llm_backfill.py):
is another copy of this script already running? (No psutil in the venv.)
"""

from __future__ import annotations

import os
import subprocess


def other_instance_running(script: str, ignore_args=("--report", "--count")) -> bool:
    """True if another python process runs `script`. The venv's python.exe is a launcher
    with a child process, so this process and its parent are ignored."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                          "ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"],
                         capture_output=True, text=True, creationflags=0x08000000).stdout
    mine = {os.getpid(), os.getppid()}
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if (pid.isdigit() and int(pid) not in mine and script in cmd
                and not any(a in cmd for a in ignore_args)):
            return True
    return False

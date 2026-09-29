"""
run_live.py
===========
Supervisor for the live paper-trading loop, so a session needs nobody at the keyboard.

    D:\\trader_env\\Scripts\\python.exe run_live.py        (start_titan_q.bat runs this at 09:10)

It runs main_trading_loop.py and, until 15 minutes after the close:
- exit code 75 (a restart was requested through restart_trading_loop.request, e.g. to
  apply a code fix): starts it again after 5 seconds, on the current code;
- any other non-zero exit (a crash): starts it again after 60 seconds;
- exit code 0 (the session ended, a weekend or holiday, or another live instance holds
  the lock): stops.
At most 12 restarts a day, so a loop that dies at once cannot spin forever.
Log: logs\\supervisor_YYYY-MM-DD.log (the loop keeps its own titanq_YYYY-MM-DD.log).

Only the paper broker exists; nothing here can place a real order.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

from market_data import Session, now_ist
from titanq_config import LOGS_DIR, PROJECT_DIR

LOOP = os.path.join(PROJECT_DIR, "main_trading_loop.py")
RESTART_EXIT_CODE = 75
MAX_RESTARTS = 12
log = logging.getLogger("supervisor")


def main() -> None:
    session = Session()
    restarts = 0
    while True:
        t0 = time.time()
        log.info("starting main_trading_loop.py (run %d)", restarts + 1)
        rc = subprocess.call([sys.executable, LOOP], cwd=PROJECT_DIR)
        now = now_ist()
        end = session.bounds(now.date())[1] + timedelta(minutes=15)
        log.info("main_trading_loop.py exited with code %d after %.0f s", rc, time.time() - t0)
        if rc == 0:
            log.info("normal exit; supervisor done")
            return
        if now.weekday() >= 5 or now >= end:
            log.info("after the session; not restarting")
            return
        restarts += 1
        if restarts > MAX_RESTARTS:
            log.error("%d restarts today; giving up until tomorrow's scheduled start", MAX_RESTARTS)
            return
        wait = 5 if rc == RESTART_EXIT_CODE else 60
        log.warning("%s; restarting in %d s (restart %d of %d)",
                    "restart requested" if rc == RESTART_EXIT_CODE else "the loop crashed",
                    wait, restarts, MAX_RESTARTS)
        time.sleep(wait)


if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(LOGS_DIR, datetime.now().strftime("supervisor_%Y-%m-%d.log")),
                            encoding="utf-8")], force=True)
    main()

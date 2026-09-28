"""
angel_history.py
================
Downloads historical 1-minute candles from Angel One SmartAPI (read-only) for
the tradable universe plus NIFTY, for the intraday research (M1, VPIN gate test).

    D:\\trader_env\\Scripts\\python.exe angel_history.py                 all tradable stocks, 2020 -> yesterday
    D:\\trader_env\\Scripts\\python.exe angel_history.py --start 2023-01-01 --limit 50

Measured on 2026-09-28: 1-minute history reaches back to at least 2020 (January 2016
was empty); a request returns at most 8,000 rows, so it asks for 25 calendar days
at a time (~17 trading days, ~6,500 bars).

Rate limit, learned the hard way on 2026-09-28: at ~2.5 requests a second the API
began answering "Access denied because of exceeding access rate" after about 5,000
requests in the first hour, and refused everything for ~50 minutes from 20:55. It
behaves like a cap of roughly 5,000 requests an hour. So:
- the pace is 1.2 requests a second (4,320 an hour) across 4 worker threads;
- a rate-limit answer pauses ALL workers together (5 minutes, doubling up to 30 while
  it persists), and does not count as a failed attempt, so no chunk is given up
  because of it. Network failures (including certificate errors: the Wi-Fi network
  intercepts HTTPS to finance sites at times) are handled the same way. Certificate
  checks are never switched off.

Behaviour:
- resumable: every finished chunk is a file (written atomically), and an existing
  file is never fetched again;
- the stocks the research needs first: NIFTY/BANKNIFTY, the 50 live stocks and every
  month's top-100 by liquidity (intraday_research.monthly_top), all windows newest
  first; then the rest of the tradable universe. When the first group is complete,
  data/intraday_hist_priority_done.json is written (the research chain waits for it);
- paused 08:55-15:50 IST on weekdays, so it never competes with the live recorder
  (a second login during the session could end the recorder's session).

Files: data/intraday_hist/<symbol>/<from>_<to>.parquet
       columns ts (IST), open, high, low, close, volume.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dtime, timedelta
from typing import List, Set, Tuple

import pandas as pd
import requests

from titanq_config import DATA_DIR, PROJECT_DIR, UNIVERSE
from market_data import IST, now_ist
from universe import load_universe

log = logging.getLogger("angel_history")

HIST_DIR = os.path.join(DATA_DIR, "intraday_hist")
PRIORITY_MARKER = os.path.join(DATA_DIR, "intraday_hist_priority_done.json")
CHUNK_DAYS = 25
RATE_PER_SEC = 1.2
WORKERS = 4
PAUSE_FIRST, PAUSE_MAX = 300.0, 1800.0
INDEX_TOKENS = {"NIFTY50_IDX": "99926000", "BANKNIFTY_IDX": "99926009"}

Job = Tuple[str, str, date, date]


class RateLimiter:
    def __init__(self, per_sec: float) -> None:
        self.gap = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.gap
        time.sleep(max(0.0, t - time.monotonic()))


def in_quiet_hours(now: datetime) -> bool:
    """True during the trading window, when the downloader must pause."""
    return now.weekday() < 5 and dtime(8, 55) <= now.time() <= dtime(15, 50)


def chunks(start: date, end: date) -> List[Tuple[date, date]]:
    out, a = [], start
    while a <= end:
        b = min(a + timedelta(days=CHUNK_DAYS - 1), end)
        out.append((a, b))
        a = b + timedelta(days=1)
    return list(reversed(out))              # newest first


def chunk_path(symbol: str, a: date, b: date) -> str:
    return os.path.join(HIST_DIR, symbol, f"{a:%Y%m%d}_{b:%Y%m%d}.parquet")


def priority_symbols() -> Set[str]:
    """Indices, the 50 live stocks and every month's top 100 (M1's pool)."""
    from intraday_research import monthly_top
    tops = monthly_top()
    return set(INDEX_TOKENS) | {s[:-3] for s in UNIVERSE} | set().union(*tops.values())


def is_rate_limit(msg: str) -> bool:
    m = msg.lower()
    return "exceeding access rate" in m or "access denied" in m or "too many requests" in m


class Downloader:
    def __init__(self) -> None:
        from angel_feed import login
        self._login = login
        self.api = None
        self.lock = threading.Lock()
        self.rate = RateLimiter(RATE_PER_SEC)
        self.done = self.failed = self.empty = self.rate_limited = 0
        self.pause_until = 0.0
        self.pause_len = PAUSE_FIRST
        self.relogin()

    def relogin(self) -> None:
        with self.lock:
            self.api, _jwt, _feed, _client = self._login()
            log.info("logged in")

    def _wait_pause(self) -> None:
        while True:
            with self.lock:
                left = self.pause_until - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 30.0))

    def _hit_pause(self, why: str) -> None:
        with self.lock:
            now = time.monotonic()
            if now >= self.pause_until:              # start one shared pause, not one per thread
                self.pause_until = now + self.pause_len
                log.warning("%s: all workers pause %.0f min", why, self.pause_len / 60)
                self.pause_len = min(self.pause_len * 2, PAUSE_MAX)

    def _ok(self) -> None:
        with self.lock:
            self.pause_len = PAUSE_FIRST

    def fetch(self, symbol: str, token: str, a: date, b: date) -> None:
        path = chunk_path(symbol, a, b)
        if os.path.exists(path):
            return
        params = {"exchange": "NSE", "symboltoken": token, "interval": "ONE_MINUTE",
                  "fromdate": f"{a:%Y-%m-%d} 09:15", "todate": f"{b:%Y-%m-%d} 15:30"}
        attempt, r = 0, None
        while attempt < 6:
            while in_quiet_hours(now_ist()):
                time.sleep(60)
            self._wait_pause()
            self.rate.wait()
            try:
                r = self.api.getCandleData(params)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                # includes SSLError: the Wi-Fi network intercepts HTTPS to finance sites at
                # times (seen 2026-09-28 from 20:57). Wait for the network, do not give up.
                self._hit_pause(f"network problem ({type(exc).__name__})")
                continue
            except Exception as exc:
                r = {"status": False, "message": str(exc)}
            if r and r.get("status"):
                rows = r.get("data") or []
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
                if len(df):
                    df["ts"] = pd.to_datetime(df["ts"])
                    self.done += 1
                else:
                    self.empty += 1           # before listing or before the API's history
                tmp = path + ".partial"
                df.to_parquet(tmp, compression="zstd", index=False)
                os.replace(tmp, path)
                self._ok()
                return
            msg = str((r or {}).get("message", ""))
            if is_rate_limit(msg):
                self.rate_limited += 1
                self._hit_pause("rate limit")  # shared pause; not a failed attempt
                continue
            attempt += 1
            if "token" in msg.lower() or "session" in msg.lower() or "invalid" in msg.lower():
                self.relogin()
            time.sleep(2.0 * attempt)
        self.failed += 1
        log.warning("gave up on %s %s..%s: %s", symbol, a, b, (r or {}).get("message"))


def run_jobs(dl: Downloader, jobs: List[Job], label: str) -> None:
    t0, last = time.time(), time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(dl.fetch, *j) for j in jobs]
        for i, f in enumerate(futures):
            f.result()
            if time.time() - last > 60:
                last = time.time()
                log.info("%s: %d/%d (%d with data, %d empty, %d failed, %d rate-limit answers), %.0f min",
                         label, i + 1, len(jobs), dl.done, dl.empty, dl.failed, dl.rate_limited,
                         (time.time() - t0) / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--limit", type=int, default=0, help="only the N most liquid stocks (0 = all tradable)")
    a = ap.parse_args()

    uni = load_universe()
    tr = uni[uni["tier"] == "tradable"].sort_values("median_value_60d", ascending=False)
    if a.limit:
        tr = tr.head(a.limit)
    targets = list(INDEX_TOKENS.items()) + [(r.symbol.replace("-EQ", ""), r.token) for r in tr.itertuples()]
    end = now_ist().date() - timedelta(days=1)
    windows = chunks(date.fromisoformat(a.start), end)
    prio = priority_symbols()
    first = [(s, t, w0, w1) for (w0, w1) in windows for (s, t) in targets if s in prio]
    rest = [(s, t, w0, w1) for (w0, w1) in windows for (s, t) in targets if s not in prio]
    todo = lambda js: [j for j in js if not os.path.exists(chunk_path(j[0], j[2], j[3]))]
    log.info("%d symbols x %d windows; research set %d symbols: %d chunks to fetch; rest: %d to fetch",
             len(targets), len(windows), len({j[0] for j in first}), len(todo(first)), len(todo(rest)))

    dl = Downloader()
    for sweep in range(3):                       # research set first, with up to 2 re-sweeps
        missing = todo(first)
        if not missing:
            break
        run_jobs(dl, missing, f"research set (sweep {sweep + 1})")
    missing = todo(first)
    with open(PRIORITY_MARKER + ".partial", "w", encoding="ascii") as f:
        json.dump({"finished": now_ist().isoformat(timespec="seconds"), "symbols": len({j[0] for j in first}),
                   "chunks": len(first), "missing": len(missing), "start": a.start, "end": str(end)}, f, indent=2)
    os.replace(PRIORITY_MARKER + ".partial", PRIORITY_MARKER)
    log.info("research set complete: %d chunks still missing; marker %s", len(missing), PRIORITY_MARKER)

    run_jobs(dl, todo(rest), "rest of universe")
    log.info("finished: %d with data, %d empty, %d failed, %d rate-limit answers",
             dl.done, dl.empty, dl.failed, dl.rate_limited)


if __name__ == "__main__":
    logs = os.path.join(PROJECT_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(logs, "angel_history.log"), encoding="utf-8")],
                        force=True)
    main()

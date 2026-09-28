"""
angel_history.py
================
Downloads historical 1-minute candles from Angel One SmartAPI (read-only) for
the tradable universe plus NIFTY, for the intraday research (M1).

    D:\\trader_env\\Scripts\\python.exe angel_history.py                 all tradable stocks, 2020 -> yesterday
    D:\\trader_env\\Scripts\\python.exe angel_history.py --start 2023-01-01 --limit 50

Measured on 2026-09-28: 1-minute history reaches back to at least 2020 (January 2016
was empty); a request returns at most 8,000 rows, so it asks for 25 calendar days
at a time (~17 trading days, ~6,500 bars); about one request per second.

Behaviour:
- resumable: every finished chunk is a file, and an existing file is never fetched again;
- newest chunks first, so recent data is usable while older data is still downloading;
- paused 08:55-15:50 IST on weekdays, so it never competes with the live recorder
  (a second login during the session could end the recorder's session);
- rate-limited to about 2.5 requests a second across 8 worker threads (a request
  takes 2-3 s, so fewer workers cannot reach the cap), with backoff on errors and a
  fresh login if the session expires.

Files: data/intraday_hist/<symbol>/<from>_<to>.parquet
       columns ts (IST), open, high, low, close, volume.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dtime, timedelta
from typing import List, Tuple

import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR
from market_data import IST, now_ist
from universe import load_universe

log = logging.getLogger("angel_history")

HIST_DIR = os.path.join(DATA_DIR, "intraday_hist")
CHUNK_DAYS = 25
RATE_PER_SEC = 2.5
INDEX_TOKENS = {"NIFTY50_IDX": "99926000", "BANKNIFTY_IDX": "99926009"}


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


class Downloader:
    def __init__(self) -> None:
        from angel_feed import login
        self._login = login
        self.api = None
        self.lock = threading.Lock()
        self.rate = RateLimiter(RATE_PER_SEC)
        self.done = self.failed = self.empty = 0
        self.relogin()

    def relogin(self) -> None:
        with self.lock:
            self.api, _jwt, _feed, _client = self._login()
            log.info("logged in")

    def fetch(self, symbol: str, token: str, a: date, b: date) -> None:
        path = os.path.join(HIST_DIR, symbol, f"{a:%Y%m%d}_{b:%Y%m%d}.parquet")
        if os.path.exists(path):
            return
        while in_quiet_hours(now_ist()):
            time.sleep(60)
        params = {"exchange": "NSE", "symboltoken": token, "interval": "ONE_MINUTE",
                  "fromdate": f"{a:%Y-%m-%d} 09:15", "todate": f"{b:%Y-%m-%d} 15:30"}
        for attempt in range(6):
            self.rate.wait()
            try:
                r = self.api.getCandleData(params)
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
                df.to_parquet(path, compression="zstd", index=False)
                return
            msg = str((r or {}).get("message", ""))
            if "token" in msg.lower() or "session" in msg.lower() or "invalid" in msg.lower():
                self.relogin()
            time.sleep(2.0 * (attempt + 1))
        self.failed += 1
        log.warning("gave up on %s %s..%s: %s", symbol, a, b, (r or {}).get("message"))


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
    jobs = [(s, t, w0, w1) for (w0, w1) in windows for (s, t) in targets]   # newest window for all first
    log.info("%d symbols x %d windows = %d requests (existing files are skipped)",
             len(targets), len(windows), len(jobs))
    dl = Downloader()
    t0, last = time.time(), time.time()
    # each request takes 2-3 s, so several workers are needed to reach the rate cap
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(dl.fetch, *j) for j in jobs]
        for i, f in enumerate(futures):
            f.result()
            if time.time() - last > 60:
                last = time.time()
                log.info("%d/%d done (%d with data, %d empty, %d failed), %.0f min elapsed", i + 1,
                         len(jobs), dl.done, dl.empty, dl.failed, (time.time() - t0) / 60)
    log.info("finished: %d with data, %d empty, %d failed in %.0f min", dl.done, dl.empty, dl.failed,
             (time.time() - t0) / 60)


if __name__ == "__main__":
    logs = os.path.join(PROJECT_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(logs, "angel_history.log"), encoding="utf-8")],
                        force=True)
    main()

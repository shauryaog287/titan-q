"""
backup_to_drive.py
==================
Backs up Titan-Q's market data to Google Drive (G:, 5 TB) as a FEW LARGE files,
never thousands of small ones (Drive's sync handles big files well and small
files badly).

    D:\\trader_env\\Scripts\\python.exe backup_to_drive.py            back up what changed
    D:\\trader_env\\Scripts\\python.exe backup_to_drive.py --dry-run  show what would be written

What goes where (G:\\My Drive\\Titan-Q_History):
    minute\\<SYMBOL>.parquet   all Angel 1-minute candles for one symbol (data/intraday_hist)
    daily\\daily_all.parquet   every daily bar in data/daily, long format (symbol, date, OHLCV)
    intraday_1m\\<date>.parquet archived Yahoo 1-minute sessions (data/intraday), copied as is

Code, models and reports are in git (GitHub), and tick recordings are archived by
angel_feed.py, so they are not repeated here. A file is rewritten only when its
source is newer. Nothing is ever deleted from Drive by this script.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time

import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR

log = logging.getLogger("backup")

DRIVE_ROOT = r"G:\My Drive\Titan-Q_History"


def _newest_mtime(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getmtime(path)
    times = [os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path)]
    return max(times) if times else 0.0


def _stale(target: str, source_mtime: float) -> bool:
    return not os.path.exists(target) or os.path.getmtime(target) < source_mtime


def backup_minutes(dry: bool) -> int:
    src_root = os.path.join(DATA_DIR, "intraday_hist")
    dst_root = os.path.join(DRIVE_ROOT, "minute")
    n = 0
    if not os.path.isdir(src_root):
        return 0
    for sym in sorted(os.listdir(src_root)):
        src = os.path.join(src_root, sym)
        dst = os.path.join(dst_root, f"{sym}.parquet")
        if not os.path.isdir(src) or not _stale(dst, _newest_mtime(src)):
            continue
        n += 1
        if dry:
            continue
        parts = []
        for f in os.listdir(src):
            if f.endswith(".parquet"):
                try:
                    parts.append(pd.read_parquet(os.path.join(src, f)))
                except Exception as exc:          # a chunk still being written by angel_history.py
                    log.warning("skipped unreadable %s\\%s: %s", sym, f, exc)
        parts = [p for p in parts if len(p)]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True).drop_duplicates("ts").sort_values("ts")
        os.makedirs(dst_root, exist_ok=True)
        tmp = dst + ".partial"
        df.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, dst)
    return n


def backup_daily(dry: bool) -> int:
    src = os.path.join(DATA_DIR, "daily")
    dst = os.path.join(DRIVE_ROOT, "daily", "daily_all.parquet")
    if not os.path.isdir(src) or not _stale(dst, _newest_mtime(src)):
        return 0
    if dry:
        return 1
    frames = []
    for f in os.listdir(src):
        if f.endswith(".parquet"):
            d = pd.read_parquet(os.path.join(src, f))
            d = d.reset_index().rename(columns={d.index.name or "index": "date"})
            d["symbol"] = f[:-8]
            frames.append(d)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    pd.concat(frames, ignore_index=True).to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, dst)
    return 1


def backup_yahoo_sessions(dry: bool) -> int:
    src = os.path.join(DATA_DIR, "intraday")
    dst_root = os.path.join(DRIVE_ROOT, "intraday_1m")
    n = 0
    if not os.path.isdir(src):
        return 0
    for f in sorted(os.listdir(src)):
        s, d = os.path.join(src, f), os.path.join(dst_root, f)
        if f.endswith(".parquet") and _stale(d, os.path.getmtime(s)):
            n += 1
            if not dry:
                os.makedirs(dst_root, exist_ok=True)
                shutil.copy2(s, d)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(os.path.dirname(DRIVE_ROOT)):
        log.error("Google Drive (G:\\My Drive) is not available; nothing backed up")
        return
    t0 = time.time()
    m = backup_minutes(a.dry_run)
    d = backup_daily(a.dry_run)
    y = backup_yahoo_sessions(a.dry_run)
    verb = "would write" if a.dry_run else "wrote"
    log.info("%s %d minute-history files, %d daily file, %d Yahoo sessions to %s (%.0fs)",
             verb, m, d, y, DRIVE_ROOT, time.time() - t0)


if __name__ == "__main__":
    logs = os.path.join(PROJECT_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(logs, "backup.log"), encoding="utf-8")],
                        force=True)
    main()

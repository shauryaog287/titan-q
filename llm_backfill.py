"""
llm_backfill.py
===============
Labels historical NSE filings with the local model (Ollama, qwen2.5:7b on the GPU) for EV1,
reports/preregistration_ev1_ai_filings.md. Labels only: no returns are computed here.

    D:\\trader_env\\Scripts\\python.exe llm_backfill.py                  development window (2017-2020)
    D:\\trader_env\\Scripts\\python.exe llm_backfill.py --window confirm only after development qualifies
    D:\\trader_env\\Scripts\\python.exe llm_backfill.py --count          how many filings are in scope

- Resumable: labels go to data/events_hist/ev1_<window>.parquet, saved every 200 filings;
  filings already labelled are skipped.
- Runs only on mains power (pauses while on battery) and keeps the laptop awake while
  working (lid closing still sleeps it).
- Runs during market hours too (since 2026-10-19), at below-normal CPU priority; the model runs
  on the GPU, which the trading loop does not use.
- About 1.3 s a filing on the RTX 3050.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import logging
import os
import time
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR

log = logging.getLogger("llm_backfill")

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b"
OUT_DIR = os.path.join(DATA_DIR, "events_hist")
WINDOWS = {"dev": ("2017-01-01", "2021-01-01"), "confirm": ("2024-10-01", "2026-09-01")}
KEEP = ("Updates", "Press Release", "General Updates", "Bagging/Receiving of orders/contracts", "Acquisition",
        "Credit Rating", "Action(s) taken or orders passed", "Action(s) initiated or orders passed",
        "Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company", "Change in Auditors",
        "News Verification", "Clarification", "Reply to Clarification Sought", "News Clarification")
MIN_VALUE, MIN_PRICE, MIN_HIST = 1e7, 20.0, 250          # NJ1's liquidity definition
SAVE_EVERY = 200

# The prompt is part of the pre-registration: do not edit it once labelling has started.
PROMPT = """You are an equity analyst covering Indian listed companies. Read the NSE filing below
and answer with JSON only, using exactly these keys:
"event_type": one of {types};
"direction": "positive", "negative" or "neutral" for the company's shareholders;
"materiality": "high", "medium" or "low" (how much it could move the share price);
"value_cr": the rupee value in crore if the filing states one (a number), else null.
Use "routine" for compliance notices and procedural intimations.

Filing ({desc}): {text}"""


def schema(types) -> Dict:
    return {"type": "object", "required": ["event_type", "direction", "materiality", "value_cr"],
            "properties": {"event_type": {"type": "string", "enum": list(types)},
                           "direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                           "materiality": {"type": "string", "enum": ["high", "medium", "low"]},
                           "value_cr": {"type": ["number", "null"]}}}


def in_scope(window: str) -> pd.DataFrame:
    """Filings of the window in the KEEP categories, of stocks liquid on the trading day they
    are attached to, de-duplicated. Columns: key, symbol, yahoo, an_dt, desc, text."""
    from order_research import ANN_DIR
    from universe import load_universe
    from whole_market_research import load_panels
    lo, hi = WINDOWS[window]
    parts = []
    for year in sorted(os.listdir(ANN_DIR)):
        if not (int(lo[:4]) <= int(year) <= int(hi[:4])):
            continue
        for f in sorted(os.listdir(os.path.join(ANN_DIR, year))):
            if f.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(ANN_DIR, year, f))
                if len(df) and {"symbol", "desc", "an_dt", "text"} <= set(df.columns):
                    parts.append(df[["symbol", "desc", "an_dt", "text"]])
    a = pd.concat(parts, ignore_index=True)
    a["t"] = pd.to_datetime(a["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    a = a[a["desc"].isin(KEEP) & (a["t"] >= lo) & (a["t"] < hi)].dropna(subset=["t"])
    uni = load_universe()
    ticker = dict(zip(uni["symbol"].str.replace("-EQ", "", regex=False), uni["yahoo"]))
    a["yahoo"] = a["symbol"].map(ticker)
    a = a.dropna(subset=["yahoo"])
    px, vol = load_panels()
    stocks = [c for c in px.columns if c != INDEX_SYMBOL]
    P, V = px[stocks], vol[stocks]
    liq = ((P * V).shift(1).rolling(60, min_periods=40).median() >= MIN_VALUE) & \
          (P.shift(1) >= MIN_PRICE) & (P.shift(1).notna().cumsum() >= MIN_HIST)
    days = px.index.values
    # the trading day the filing is attached to: that day if broadcast before 15:30, else the next
    after = (a["t"].dt.hour * 60 + a["t"].dt.minute) >= 15 * 60 + 30
    d = a["t"].dt.normalize() + pd.to_timedelta(after.astype(int), unit="D")
    row = np.searchsorted(days, d.values, side="left")
    ok = row < len(days)
    a, row = a[ok].copy(), row[ok]
    col = {s: i for i, s in enumerate(stocks)}
    L = liq.to_numpy()
    a["liquid"] = [bool(L[r, col[y]]) if y in col else False for r, y in zip(row, a["yahoo"])]
    a = a[a["liquid"]]
    a["text"] = a["text"].fillna("").astype(str)
    a["key"] = [hashlib.sha1(f"{s}|{de}|{tx}".encode("utf-8", "replace")).hexdigest()[:20]
                for s, de, tx in zip(a["symbol"], a["desc"], a["text"])]
    a = a.drop_duplicates("key").sort_values("t")
    return a[["key", "symbol", "yahoo", "an_dt", "desc", "text"]].reset_index(drop=True)


def label(desc: str, text: str, types) -> Optional[Dict]:
    from news_analyser import ascii_text
    body = {"model": MODEL, "prompt": PROMPT.format(types=", ".join(types), desc=desc, text=ascii_text(text)[:1500]),
            "format": schema(types), "stream": False,
            "options": {"temperature": 0, "num_ctx": 1024, "num_predict": 120}}
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=180)
        r.raise_for_status()
        import json
        out = json.loads(r.json().get("response", "{}"))
    except (requests.RequestException, ValueError) as exc:
        log.warning("model call failed: %s", type(exc).__name__)
        return None
    v = out.get("value_cr")
    return {"event_type": out.get("event_type"), "direction": out.get("direction"),
            "materiality": out.get("materiality"),
            "value_cr": float(v) if isinstance(v, (int, float)) else np.nan}


def keep_awake(on: bool) -> None:
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def may_run() -> Optional[str]:
    """None if labelling may run now, else the reason to wait. Since 2026-10-19 it also runs
    during market hours (the user's choice: two weeks gave only 2,000 of 41,437 labels); the
    work is on the GPU, which the trading loop does not use, and this process runs at low
    CPU priority (lower_priority)."""
    from alerts import on_battery
    if on_battery():
        return "on battery"
    return None


def lower_priority() -> None:
    """Below-normal CPU priority for this process, so the live trading processes come first."""
    try:
        BELOW_NORMAL = 0x00004000
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL)
    except Exception:
        pass


def save(path: str, rows: List[Dict]) -> None:
    old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
    out = out.drop_duplicates("key")
    out.to_parquet(path + ".partial", index=False)
    os.replace(path + ".partial", path)


def run(window: str) -> None:
    from llm_reader import EVENT_TYPES
    lower_priority()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"ev1_{window}.parquet")
    todo = in_scope(window)
    done = set(pd.read_parquet(path)["key"]) if os.path.exists(path) else set()
    todo = todo[~todo["key"].isin(done)]
    log.info("%s window: %d filings to label (%d already done)", window, len(todo), len(done))
    rows: List[Dict] = []
    t0, n = time.time(), 0
    for r in todo.itertuples(index=False):
        while (why := may_run()) is not None:
            if rows:
                save(path, rows)
                rows = []
            keep_awake(False)
            log.info("paused (%s)", why)
            time.sleep(300)
        keep_awake(True)
        ev = None
        for attempt in range(3):
            ev = label(r.desc, r.text, EVENT_TYPES)
            if ev is not None:
                break
            time.sleep(30 * (attempt + 1))              # Ollama restarting, GPU busy
        if ev is None:
            continue
        rows.append({"key": r.key, "symbol": r.symbol, "yahoo": r.yahoo, "an_dt": r.an_dt, "desc": r.desc,
                     **ev, "labelled_at": datetime.now().isoformat(timespec="seconds")})
        n += 1
        if len(rows) >= SAVE_EVERY:
            save(path, rows)
            rows = []
            rate = n / (time.time() - t0)
            log.info("%d/%d labelled (%.2f s each; about %.1f h left)", n, len(todo), 1 / rate,
                     (len(todo) - n) / rate / 3600)
    if rows:
        save(path, rows)
    keep_awake(False)
    log.info("%s window finished: %d labelled this run", window, n)
    left = len(in_scope(window)) - len(pd.read_parquet(path)) if os.path.exists(path) else -1
    report = os.path.join(PROJECT_DIR, "reports", "ev1_ai_filings.md")
    if window == "dev" and left <= 0 and not os.path.exists(report):
        # the pre-registered development decision is mechanical: run it once and tell the user
        import re
        import subprocess
        import sys
        subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "ev1_research.py")], cwd=PROJECT_DIR,
                       capture_output=True, creationflags=0x08000000)
        if os.path.exists(report):
            with open(report, encoding="ascii") as f:
                dec = re.findall(r"\*\*Decision: ([^*]+)\*\*", f.read())
            log.info("EV1 development: %s", dec[-1] if dec else "no decision line")
            try:
                from alerts import send
                send("EV1 development result", (dec[-1] if dec else "see reports/ev1_ai_filings.md")
                     + " (reports/ev1_ai_filings.md)", priority="default", tags="microscope", repeat_minutes=0)
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=sorted(WINDOWS), default="dev")
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    if a.count:
        s = in_scope(a.window)
        print(f"{a.window}: {len(s):,} filings in scope")
        print(s["desc"].value_counts().to_string())
        return
    from procs import other_instance_running
    if other_instance_running("llm_backfill.py"):
        log.info("another llm_backfill.py is running; exiting")
        return
    run(a.window)


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(PROJECT_DIR, "logs", "llm_backfill.log"), encoding="utf-8")], force=True)
    main()

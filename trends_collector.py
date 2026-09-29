"""
trends_collector.py
===================
Retail attention from Google Trends: how much India is searching "<company> share price"
for the stocks we hold or plan to hold. Search attention is the best-studied free proxy
for retail interest (Da, Engelberg and Gao, "In Search of Attention", 2011): attention
spikes tend to come before short-term overpricing and later underperformance.

    D:\\trader_env\\Scripts\\python.exe trends_collector.py            collect today's snapshot (scheduled 20:40)
    D:\\trader_env\\Scripts\\python.exe trends_collector.py --report   attention spikes of the latest snapshot

Each run stores the full 90-day daily series AS SEEN THAT DAY (Google rescales and
revises its numbers), in data/trends/trends_<date>.parquet, copied to G:. A backtest
must only use each day's own snapshot. One search term per request (best resolution),
spaced out; Google throttles bursts (pytrends is unofficial).

Attention ratio = mean of the last 5 weekday values / median of the previous 60 weekdays.
OBSERVE ONLY: logged by the trading loop, no trade depends on it until a pre-registered
test says it should.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from typing import Dict, List, Optional

import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR
from market_data import now_ist

log = logging.getLogger("trends")

TRENDS_DIR = os.path.join(DATA_DIR, "trends")
DRIVE_DIR = r"G:\My Drive\Titan-Q_History\trends"
GAP_SECONDS = 30.0          # Google answered 429 after ~13 requests 10 s apart (2026-09-29)
BACKOFF = (120.0, 300.0, 600.0)
SPIKE_RATIO = 2.0


def search_terms(symbols: List[str]) -> Dict[str, str]:
    """NSE symbol -> the Google search people make before trading it."""
    from news_analyser import ALIASES, Tagger
    names = Tagger(symbols).names
    out = {}
    for s in symbols:
        # the short name people type ("TCS share price", "SBI share price"), else the company name
        base = ALIASES[s][0] if s in ALIASES else names.get(s)
        if base:
            out[s] = f"{base} share price"
    return out


def fetch(terms: Dict[str, str]) -> pd.DataFrame:
    from pytrends.request import TrendReq
    t = TrendReq(hl="en-IN", tz=-330, timeout=(10, 30))
    rows = []
    for i, (sym, term) in enumerate(sorted(terms.items())):
        for attempt in range(len(BACKOFF) + 1):
            try:
                t.build_payload([term], timeframe="today 3-m", geo="IN")
                df = t.interest_over_time()
                if len(df) and term in df:
                    for d, v in df[term].items():
                        rows.append({"symbol": sym, "term": term, "date": pd.Timestamp(d).date(), "value": int(v)})
                break
            except Exception as exc:
                if attempt == len(BACKOFF):
                    log.warning("Google keeps refusing (%s); stopping with %d of %d terms",
                                type(exc).__name__, i, len(terms))
                    return pd.DataFrame(rows)
                log.warning("%s: %s (attempt %d); waiting %.0f s", term, type(exc).__name__, attempt + 1,
                            BACKOFF[attempt])
                time.sleep(BACKOFF[attempt])
        time.sleep(GAP_SECONDS)
        if (i + 1) % 20 == 0:
            log.info("%d/%d terms", i + 1, len(terms))
    return pd.DataFrame(rows)


def collect() -> Optional[str]:
    from news_analyser import focus_symbols
    terms = search_terms(focus_symbols())
    df = fetch(terms)
    if not len(df):
        log.warning("no Trends data collected")
        return None
    day = now_ist().date()
    df["snapshot"] = str(day)
    os.makedirs(TRENDS_DIR, exist_ok=True)
    path = os.path.join(TRENDS_DIR, f"trends_{day}.parquet")
    df.to_parquet(path + ".partial", index=False)
    os.replace(path + ".partial", path)
    log.info("%d terms, %d rows -> %s", df["symbol"].nunique(), len(df), path)
    if os.path.isdir(os.path.dirname(os.path.dirname(DRIVE_DIR))):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        shutil.copy2(path, os.path.join(DRIVE_DIR, os.path.basename(path)))
    return path


def attention(max_age_days: int = 4) -> Dict[str, Dict]:
    """symbol -> {ratio, last5, base} from the newest snapshot (weekdays only)."""
    if not os.path.isdir(TRENDS_DIR):
        return {}
    files = sorted(f for f in os.listdir(TRENDS_DIR) if f.startswith("trends_") and f.endswith(".parquet"))
    if not files:
        return {}
    snap = pd.Timestamp(files[-1][7:17]).date()
    if (now_ist().date() - snap).days > max_age_days:
        return {}
    df = pd.read_parquet(os.path.join(TRENDS_DIR, files[-1]))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.weekday < 5]
    out = {}
    for sym, g in df.groupby("symbol"):
        v = g.sort_values("date")["value"].astype(float)
        if len(v) < 30:
            continue
        last5, base = float(v.iloc[-5:].mean()), float(v.iloc[-65:-5].median())
        if base > 0:
            out[sym] = {"ratio": round(last5 / base, 2), "last5": round(last5, 1), "base": base,
                        "snapshot": str(snap)}
    return out


def report() -> None:
    att = attention(max_age_days=30)
    print(f"Retail search attention ({len(att)} stocks), highest first; ratio = last 5 weekdays / "
          "previous 60-weekday median:")
    for s, v in sorted(att.items(), key=lambda kv: -kv[1]["ratio"]):
        flag = "  <-- attention spike" if v["ratio"] >= SPIKE_RATIO else ""
        print(f"  {s:12s} x{v['ratio']:.2f} (last5 {v['last5']:.0f} vs base {v['base']:.0f}){flag}")


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(PROJECT_DIR, "logs", "trends.log"), encoding="utf-8")], force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else collect()

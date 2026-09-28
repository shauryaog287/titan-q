"""
universe.py
===========
The whole-NSE universe: every NSE cash-market equity that Angel One lists (series
EQ, about 2,700), mapped to Yahoo tickers for daily history, with liquidity
statistics that decide what is tradable.

    D:\\trader_env\\Scripts\\python.exe universe.py      build or refresh

Outputs
    data/universe/nse_universe.parquet   one row per stock (not in git; rebuilt daily)
    reports/universe_summary.md          counts per tier, coverage

Tiers, from the stock's last 60 trading days of Yahoo data:
    tradable   median daily traded value >= Rs. 10 crore, price >= Rs. 20,
               and at least 2 years of history
    watch      median daily traded value >= Rs. 1 crore
    illiquid   the rest: recorded by the Angel feed, never traded

Research must decide membership POINT IN TIME, from data available at each
formation date (liquid_at()), not from these current tiers. The list itself is
today's listed stocks, so delisted names are missing: survivorship bias remains.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR
from market_data import DailyStore

log = logging.getLogger("universe")

UNIVERSE_DIR = os.path.join(DATA_DIR, "universe")
UNIVERSE_FILE = os.path.join(UNIVERSE_DIR, "nse_universe.parquet")
TRADABLE_VALUE = 10e7          # Rs. 10 crore median daily traded value
WATCH_VALUE = 1e7              # Rs. 1 crore
MIN_PRICE = 20.0
MIN_HISTORY_DAYS = 2 * 252


def liquidity_stats(df: pd.DataFrame, window: int = 60) -> Dict[str, float]:
    tail = df.iloc[-window:]
    value = (tail["Close"] * tail["Volume"]).median()
    return {"first_date": df.index[0], "last_date": df.index[-1], "n_days": int(len(df)),
            "last_close": float(df["Close"].iloc[-1]), "median_value_60d": float(value)}


def tier(row) -> str:
    if (row["median_value_60d"] >= TRADABLE_VALUE and row["last_close"] >= MIN_PRICE
            and row["n_days"] >= MIN_HISTORY_DAYS):
        return "tradable"
    if row["median_value_60d"] >= WATCH_VALUE:
        return "watch"
    return "illiquid"


def liquid_at(daily: Dict[str, pd.DataFrame], asof: pd.Timestamp, window: int = 60,
              min_value: float = TRADABLE_VALUE, min_price: float = MIN_PRICE,
              min_history: int = MIN_HISTORY_DAYS) -> list:
    """Stocks that were tradable ON `asof`, using only data dated before it."""
    out = []
    for s, df in daily.items():
        if s == INDEX_SYMBOL:
            continue
        past = df[df.index < asof]
        if len(past) < max(min_history, window):
            continue
        tail = past.iloc[-window:]
        if (tail["Close"] * tail["Volume"]).median() >= min_value and past["Close"].iloc[-1] >= min_price:
            out.append(s)
    return out


def build_universe(refresh: bool = True) -> pd.DataFrame:
    from angel_feed import load_scrip_master, nse_equities

    t0 = time.time()
    eq = nse_equities(load_scrip_master())
    store = DailyStore()
    daily = store.load(eq["yahoo"].tolist() + [INDEX_SYMBOL], refresh=refresh)
    rows = []
    for r in eq.itertuples(index=False):
        df = daily.get(r.yahoo)
        base = {"symbol": r.symbol, "token": r.token, "name": r.name, "yahoo": r.yahoo,
                "tick_size": r.tick_size, "has_daily": df is not None and len(df) > 0}
        if base["has_daily"]:
            base.update(liquidity_stats(df))
        rows.append(base)
    uni = pd.DataFrame(rows)
    ok = uni["has_daily"]
    uni.loc[ok, "tier"] = uni[ok].apply(tier, axis=1)
    uni.loc[~ok, "tier"] = "no_yahoo_data"
    os.makedirs(UNIVERSE_DIR, exist_ok=True)
    uni.to_parquet(UNIVERSE_FILE, index=False)
    write_summary(uni, time.time() - t0)
    return uni


def write_summary(uni: pd.DataFrame, seconds: float) -> None:
    counts = uni["tier"].value_counts()
    tr = uni[uni["tier"] == "tradable"]
    lines = ["# NSE universe", "",
             f"Built {pd.Timestamp.now():%Y-%m-%d %H:%M} by universe.py in {seconds:.0f}s from Angel One's "
             "instrument list (NSE, series EQ) and Yahoo daily history.", "",
             "| Tier | Stocks | Rule |", "|---|---|---|",
             f"| tradable | {counts.get('tradable', 0)} | median 60-day traded value >= Rs. 10 crore, "
             f"price >= Rs. 20, >= 2 years of history |",
             f"| watch | {counts.get('watch', 0)} | median 60-day traded value >= Rs. 1 crore |",
             f"| illiquid | {counts.get('illiquid', 0)} | the rest (recorded, not traded) |",
             f"| no Yahoo data | {counts.get('no_yahoo_data', 0)} | ticker not found on Yahoo |",
             f"| **total** | **{len(uni)}** | |", ""]
    if len(tr):
        lines += [f"Tradable tier: median daily traded value Rs. "
                  f"{tr['median_value_60d'].median() / 1e7:,.1f} crore; history starts "
                  f"{pd.to_datetime(tr['first_date']).min():%Y-%m-%d} at the earliest.", ""]
    with open(os.path.join(PROJECT_DIR, "reports", "universe_summary.md"), "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(UNIVERSE_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    uni = build_universe()
    with open(os.path.join(PROJECT_DIR, "reports", "universe_summary.md"), encoding="ascii") as f:
        print(f.read())

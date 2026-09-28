"""
forward_test.py
===============
Daily after-close update of the pre-registered forward test
(reports/forward_test/README.md): the seven-signal composite, top 30 stocks from
the whole liquid NSE market, rebalanced monthly on a separate paper account.

    D:\\trader_env\\Scripts\\python.exe forward_test.py            update for the latest close
    D:\\trader_env\\Scripts\\python.exe forward_test.py --report   print the running record

main_trading_loop.py starts it automatically after each live session, and the scheduled
task TitanQ_Forward_Test runs it at 16:30 on weekdays as a fallback. It is idempotent:
running it twice for the same close changes nothing, and a lock file stops two copies
from running at once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from datetime import datetime, time as dtime
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

from titanq_config import CONFIG, INDEX_SYMBOL, PROJECT_DIR
from market_data import IST, DailyStore, now_ist
from paper_broker import LIVE_ACCOUNT_FILE, PaperOrderRejected, PaperTradingBroker
from universe import load_universe
from whole_market_research import eligible_at, signals_at

log = logging.getLogger("forward_test")

FT_DIR = os.path.join(PROJECT_DIR, "reports", "forward_test")
LOG_FILE = os.path.join(FT_DIR, "composite_top30.csv")
STATE_FILE = os.path.join(FT_DIR, "state.json")
ACCOUNT_FILE = os.path.join(PROJECT_DIR, "paper_account_shadow.json")
TOP_N = 30


def load_panels(refresh: bool = True):
    uni = load_universe()
    names = uni.loc[uni["has_daily"], "yahoo"].tolist()
    # max age 3 hours: files fetched by the morning session must be re-read after the close
    daily = DailyStore().load(names + [INDEX_SYMBOL], refresh=refresh, max_age_hours=3.0)
    px = pd.DataFrame({s: d["Close"] for s, d in daily.items()}).sort_index()
    vol = pd.DataFrame({s: d["Volume"] for s, d in daily.items() if s != INDEX_SYMBOL}).reindex(px.index)
    return px, vol


def composite_scores(px: pd.DataFrame, vol: pd.DataFrame, a: int):
    """COMPOSITE7 at row a (data up to and including row a), highest first."""
    value = px.drop(columns=[INDEX_SYMBOL]) * vol
    stocks = eligible_at(px, value, a)
    sig = signals_at(px, a, stocks)
    z = (sig - sig.mean()) / sig.std()
    return z.mean(axis=1).dropna().sort_values(ascending=False), stocks


def _load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="ascii") as f:
            return json.load(f)
    return {"formations": [], "bench": None}


def _save_state(state: Dict) -> None:
    with open(STATE_FILE, "w", encoding="ascii") as f:
        json.dump(state, f, indent=1, default=float)


def _live_equity(close: pd.Series):
    if not os.path.exists(LIVE_ACCOUNT_FILE):
        return None
    with open(LIVE_ACCOUNT_FILE, "r", encoding="ascii") as f:
        acct = json.load(f)
    return float(acct["cash"] + sum(q * float(close.get(s, np.nan)) for s, q in acct["positions"].items()
                                    if np.isfinite(close.get(s, np.nan))))


def rebalance(broker: PaperTradingBroker, picks: List[str], close: pd.Series, day) -> int:
    loop = asyncio.new_event_loop()
    trades = 0

    def send(sym, side, qty, k):
        nonlocal trades
        try:
            loop.run_until_complete(broker.place_order(SimpleNamespace(
                symbol=sym, side=side, quantity=int(qty), client_order_id=f"ft-{day}-{k}"), "FORWARD"))
            trades += 1
        except PaperOrderRejected as exc:
            log.warning("rejected: %s", exc)

    equity = broker.equity()
    target = {s: equity / len(picks) for s in picks}
    k = 0
    for s, q in sorted(broker.positions.items()):               # sells first
        px = float(close.get(s, np.nan))
        keep = int(target.get(s, 0.0) // px) if np.isfinite(px) and px > 0 else q
        if q > keep:
            send(s, "SELL", q - keep, k)
            k += 1
    for s in picks:                                             # then buys
        px = float(close.get(s, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        want = int(target[s] // (px * 1.0025)) - broker.positions.get(s, 0)
        if want > 0 and want * px >= CONFIG.min_trade_value / 4:
            send(s, "BUY", want, k)
            k += 1
    loop.close()
    return trades


def update(force: bool = False) -> None:
    os.makedirs(FT_DIR, exist_ok=True)
    px, vol = load_panels(refresh=True)
    i = len(px) - 1
    day = px.index[i].date()
    if day != now_ist().date() and not force:
        log.info("latest close is %s, not today (%s): nothing to update", day, now_ist().date())
        return
    logged = pd.read_csv(LOG_FILE)["date"].astype(str).tolist() if os.path.exists(LOG_FILE) else []
    if str(day) in logged:
        log.info("%s already recorded", day)
        return
    close = px.iloc[i].drop(labels=[INDEX_SYMBOL])
    state = _load_state()
    broker = PaperTradingBroker(CONFIG.initial_capital, price_fn=lambda s: close.get(s),
                                clock_fn=lambda: datetime.combine(day, dtime(15, 30), IST),
                                state_file=ACCOUNT_FILE)
    fees_before = broker.fees_paid
    first_run = not state["formations"]
    new_month = px.index[i].month != px.index[i - 1].month
    trades = 0
    if first_run or new_month:
        scores, eligible = composite_scores(px, vol, i - 1)       # data up to the previous close
        picks = scores.index[:TOP_N].tolist()
        trades = rebalance(broker, picks, close, day)
        entry = {s: float(close[s]) for s in eligible if np.isfinite(close.get(s, np.nan))}
        level = state["bench"]["level"] if state["bench"] else 100.0
        state["bench"] = {"level0": level, "level": level, "entry": entry}
        state["formations"].append({"date": str(day), "signal_date": str(px.index[i - 1].date()),
                                    "n_eligible": len(eligible), "picks": picks,
                                    "scores": {s: round(float(scores[s]), 4) for s in picks}})
        log.info("%s: rebalanced to top %d of %d eligible (%d trades): %s ...", day, len(picks),
                 len(eligible), trades, ", ".join(picks[:5]))
    b = state["bench"]
    rel = [float(close[s]) / p for s, p in b["entry"].items() if np.isfinite(close.get(s, np.nan))]
    b["level"] = b["level0"] * float(np.mean(rel)) if rel else b["level"]
    eq = broker.equity()
    row = {"date": str(day), "equity": round(eq, 2), "cash": round(broker.capital, 2),
           "holdings": len(broker.positions), "trades": trades,
           "fees_today": round(broker.fees_paid - fees_before, 2), "fees_total": round(broker.fees_paid, 2),
           "bench_ew_level": round(b["level"], 4), "nifty": round(float(px[INDEX_SYMBOL].iloc[i]), 2),
           "live_v31_equity": _live_equity(close)}
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
    broker.save()
    _save_state(state)
    log.info("%s: forward equity Rs. %s (%d holdings), EW benchmark %.2f", day, f"{eq:,.2f}",
             len(broker.positions), b["level"])


def report() -> None:
    if not os.path.exists(LOG_FILE):
        print("no forward record yet")
        return
    df = pd.read_csv(LOG_FILE)
    first, last = df.iloc[0], df.iloc[-1]
    strat = last["equity"] / first["equity"] - 1 if len(df) > 1 else 0.0
    print(f"forward test {first['date']} .. {last['date']} ({len(df)} days)")
    print(f"  composite top-30: Rs. {last['equity']:,.2f} ({100 * (last['equity'] / CONFIG.initial_capital - 1):+.2f}% "
          f"since start, fees Rs. {last['fees_total']:,.2f})")
    print(f"  equal-weight universe: {100 * (last['bench_ew_level'] / 100 - 1):+.2f}% since the first rebalance")
    print(f"  NIFTY: {100 * (last['nifty'] / first['nifty'] - 1):+.2f}% over the record")


def _single_instance():
    """Non-blocking OS lock, so the post-session launch and the 16:30 scheduled task
    (TitanQ_Forward_Test) can never record the same close twice. None if held."""
    f = open(os.path.join(PROJECT_DIR, "forward_test.lock"), "a+")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true", help="record even if the latest close is not today")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        lock = _single_instance()
        if lock is None:
            log.info("another forward-test update is running; nothing to do")
        else:
            update(force=a.force)

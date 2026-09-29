"""
corporate_actions.py
====================
Keeps the paper accounts honest about two things the market does to real holdings:

- Splits, bonus issues and cash dividends (Yahoo "actions"): booked once per holding
  held before the ex-date, via PaperTradingBroker.apply_corporate_action.
- Circuit limits (NSE price bands, sec_list.csv: 2/5/10/20% or none for F&O stocks):
  a paper buy at the upper limit or a sell at the lower limit is rejected, as it would
  not fill on the exchange (PaperTradingBroker.limits_fn).

    D:\\trader_env\\Scripts\\python.exe corporate_actions.py     offline self-test

Used by main_trading_loop.py (live account, each morning) and forward_test.py (shadow
account, each evening).
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from titanq_config import DATA_DIR
from market_data import now_ist

log = logging.getLogger("corporate_actions")

REF_DIR = os.path.join(DATA_DIR, "reference")
BANDS_FILE = os.path.join(REF_DIR, "sec_list.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

Action = Tuple[str, str, str, float]            # (yahoo symbol, ex-date YYYY-MM-DD, kind, value)


def fetch_actions(yahoo_symbols: Sequence[str], days: int = 30) -> List[Action]:
    """Splits/bonuses and dividends with an ex-date in the last `days` days."""
    import yfinance as yf
    syms = sorted(set(yahoo_symbols))
    if not syms:
        return []
    df = yf.download(syms, period="3mo", actions=True, auto_adjust=False, group_by="ticker",
                     progress=False, threads=True)
    since = pd.Timestamp(now_ist().date() - timedelta(days=days))
    out: List[Action] = []
    for s in syms:
        try:
            sub = df[s] if len(syms) > 1 else df
        except KeyError:
            continue
        for col, kind in (("Stock Splits", "split"), ("Dividends", "dividend")):
            if col not in sub:
                continue
            ser = sub[col].dropna()
            for d, v in ser[(ser > 0) & (ser.index >= since)].items():
                out.append((s, str(pd.Timestamp(d).date()), kind, float(v)))
    return out


def apply_to_account(broker, days: int = 30) -> List[str]:
    """Book recent actions on every current holding of a PaperTradingBroker."""
    held = [s for s, q in broker.positions.items() if q]
    try:
        actions = fetch_actions(held, days)
    except Exception as exc:
        log.warning("corporate actions not checked (%s)", type(exc).__name__)
        return []
    today = str(now_ist().date())
    msgs = [m for s, d, k, v in actions if d <= today
            for m in [broker.apply_corporate_action(s, d, k, v)] if m]
    log.info("corporate actions: %d found for %d holdings, %d booked", len(actions), len(held), len(msgs))
    return msgs


def load_bands(max_age_hours: float = 20.0) -> Dict[str, Optional[float]]:
    """NSE symbol -> price band as a fraction (0.05 = 5%), None for "No Band"."""
    os.makedirs(REF_DIR, exist_ok=True)
    if not (os.path.exists(BANDS_FILE) and time.time() - os.path.getmtime(BANDS_FILE) < 3600 * max_age_hours):
        try:
            r = requests.get("https://nsearchives.nseindia.com/content/equities/sec_list.csv", headers=UA, timeout=30)
            r.raise_for_status()
            with open(BANDS_FILE, "w", encoding="ascii", newline="") as f:
                f.write(r.content.decode("utf-8", "replace").encode("ascii", "replace").decode("ascii"))
        except Exception as exc:
            log.warning("price bands not refreshed (%s)", type(exc).__name__)
    if not os.path.exists(BANDS_FILE):
        return {}
    b = pd.read_csv(BANDS_FILE)
    b.columns = [c.strip() for c in b.columns]
    b = b[b["Series"].astype(str).str.strip().isin(["EQ", "BE", "BZ"])]
    out: Dict[str, Optional[float]] = {}
    for sym, band in zip(b["Symbol"].astype(str).str.strip(), b["Band"].astype(str).str.strip()):
        try:
            out[sym] = float(band) / 100.0
        except ValueError:
            out[sym] = None                              # "No Band" (F&O stocks)
    return out


def circuit_limits(prev_close: Dict[str, float], bands: Dict[str, Optional[float]]) -> Dict[str, Tuple[float, float]]:
    """Yahoo symbol -> (lower, upper) for today, from yesterday's close and the band."""
    out = {}
    for ysym, pc in prev_close.items():
        band = bands.get(ysym[:-3] if ysym.endswith(".NS") else ysym)
        if band and pc and pc > 0:
            out[ysym] = (pc * (1 - band), pc * (1 + band))
    return out


def _test() -> None:
    import asyncio
    import tempfile
    from types import SimpleNamespace
    from paper_broker import PaperOrderRejected, PaperTradingBroker
    with tempfile.TemporaryDirectory() as tmp:
        price = {"AAA.NS": 100.0, "BBB.NS": 50.0}
        b = PaperTradingBroker(100_000.0, price_fn=price.get, state_file=os.path.join(tmp, "acct.json"),
                               clock_fn=lambda: pd.Timestamp("2026-09-01 10:00", tz="Asia/Kolkata").to_pydatetime())
        buy = lambda s, q, k: asyncio.run(b.place_order(SimpleNamespace(symbol=s, side="BUY", quantity=q,
                                                                        client_order_id=k), "T"))
        buy("AAA.NS", 101, "c1")
        cost = 101 * b.avg_cost["AAA.NS"]
        cash0 = b.capital
        # a 1:2 split after the purchase: 101 -> 202 shares, same total cost
        assert b.apply_corporate_action("AAA.NS", "2026-09-10", "split", 2.0)
        assert b.positions["AAA.NS"] == 202 and abs(202 * b.avg_cost["AAA.NS"] - cost) < 1e-6
        assert b.apply_corporate_action("AAA.NS", "2026-09-10", "split", 2.0) is None      # once only
        # an odd ratio leaves a fraction, paid in cash
        b.positions["AAA.NS"], b.avg_cost["AAA.NS"] = 3, 100.0
        assert b.apply_corporate_action("AAA.NS", "2026-09-11", "split", 1.5)
        assert b.positions["AAA.NS"] == 4 and abs(b.capital - (cash0 + 0.5 * 100.0)) < 1e-6
        # dividend on the holding; nothing for an ex-date on or before the purchase date
        c1 = b.capital
        assert b.apply_corporate_action("AAA.NS", "2026-09-12", "dividend", 2.5)
        assert abs(b.capital - c1 - 10.0) < 1e-9 and abs(b.dividends_received - 10.0) < 1e-9
        assert b.apply_corporate_action("AAA.NS", "2026-09-01", "dividend", 2.5) is None
        assert b.apply_corporate_action("BBB.NS", "2026-09-12", "dividend", 1.0) is None   # not held
        b2 = PaperTradingBroker(1.0, state_file=os.path.join(tmp, "acct.json"))
        assert len(b2.ca_applied) == 3 and abs(b2.dividends_received - 10.0) < 1e-9
        # circuit limits: a buy at the upper limit and a sell at the lower one are rejected
        limits = circuit_limits({"BBB.NS": 50.0, "AAA.NS": 100.0}, {"BBB": 0.05, "AAA": None})
        assert "AAA.NS" not in limits and limits["BBB.NS"] == (47.5, 52.5)
        b.limits_fn = limits.get
        price["BBB.NS"] = 52.5
        try:
            buy("BBB.NS", 10, "c2")
            raise AssertionError("buy at the upper circuit was filled")
        except PaperOrderRejected:
            pass
        price["BBB.NS"] = 51.0
        buy("BBB.NS", 10, "c3")
        assert b.positions["BBB.NS"] == 10
    print("[ok] corporate actions: split keeps total cost, odd ratio pays cash in lieu, dividends credited, "
          "each booked once and persisted, none before the purchase; circuit: buy at the upper limit rejected")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _test()

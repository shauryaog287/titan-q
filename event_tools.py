"""
event_tools.py
==============
Shared machinery for the event studies (EAP1, FB1): price panels, the NJ1 liquidity rule,
excess returns against the equal-weight liquid universe, and month-clustered summaries.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from titanq_config import INDEX_SYMBOL

MIN_VALUE, MIN_PRICE, MIN_HIST = 1e7, 20.0, 250          # NJ1's liquidity definition


class Panels:
    def __init__(self) -> None:
        from universe import load_universe
        from whole_market_research import load_panels
        px, vol = load_panels()
        stocks = [c for c in px.columns if c != INDEX_SYMBOL]
        P, V = px[stocks], vol[stocks]
        med60 = (P * V).shift(1).rolling(60, min_periods=40).median()
        liq = (med60 >= MIN_VALUE) & (P.shift(1) >= MIN_PRICE) & (P.shift(1).notna().cumsum() >= MIN_HIST)
        self.days = P.index
        self.P, self.F = P.to_numpy(), P.ffill().to_numpy()
        self.VAL = (P * V).to_numpy()                       # daily traded value (Rs.)
        self.L, self.M = liq.to_numpy(), med60.to_numpy()
        self.col = {s: i for i, s in enumerate(stocks)}
        uni = load_universe()
        self.yahoo = dict(zip(uni["symbol"].str.replace("-EQ", "", regex=False), uni["yahoo"]))

    def row_on_or_after(self, ts) -> int:
        return int(np.searchsorted(self.days.values, np.datetime64(pd.Timestamp(ts).normalize()), side="left"))

    def known_row(self, ts) -> int:
        """The trading day whose close is the first close after time ts (15:30 cut-off)."""
        ts = pd.Timestamp(ts)
        d = ts.normalize() + pd.Timedelta(days=1 if ts.hour * 60 + ts.minute >= 15 * 60 + 30 else 0)
        return self.row_on_or_after(d)

    def liquid(self, row: int, yahoo: str) -> bool:
        j = self.col.get(yahoo)
        return j is not None and 0 <= row < len(self.days) and bool(self.L[row, j])

    def excess(self, entry: int, exit_: int, yahoo: str) -> float:
        """Stock return from the close of `entry` to the close of `exit_`, minus the equal-weight
        return of the stocks liquid at `entry` over the same days."""
        j = self.col.get(yahoo)
        if j is None or not (0 <= entry < exit_ < len(self.days)):
            return float("nan")
        r_s = self.F[exit_, j] / self.P[entry, j] - 1
        pool = self.L[entry]
        with np.errstate(all="ignore"):
            r = self.F[exit_, pool] / self.P[entry, pool] - 1
        r = r[np.isfinite(r)]
        return float(r_s - r.mean()) if np.isfinite(r_s) and len(r) else float("nan")

    def new_listing_liquid(self, first_row: int, entry: int, yahoo: str, min_days: int = 10) -> bool:
        """For recent IPOs (no 250-day history): median traded value from `first_row` to the
        day before `entry` >= Rs. 1 crore over >= min_days days, and entry close >= Rs. 20."""
        j = self.col.get(yahoo)
        if j is None or entry <= first_row or entry >= len(self.days):
            return False
        v = self.VAL[first_row:entry, j]
        v = v[np.isfinite(v)]
        px = self.P[entry, j]
        return len(v) >= min_days and float(np.median(v)) >= MIN_VALUE and np.isfinite(px) and px >= MIN_PRICE

    def size_third(self, row: int, yahoo: str) -> str:
        j = self.col.get(yahoo)
        if j is None:
            return "unknown"
        pool = self.M[row][self.L[row]]
        pool = pool[np.isfinite(pool)]
        v = self.M[row, j]
        if not len(pool) or not np.isfinite(v):
            return "unknown"
        q1, q2 = np.quantile(pool, [1 / 3, 2 / 3])
        return "small" if v <= q1 else "mid" if v <= q2 else "large"


def summary(x: pd.Series, dates: pd.Series, freq: str = "M") -> Dict:
    """Mean, median, share > 0 and a t clustered by calendar month (freq "M") or week ("W")
    of the values x (dates = event dates)."""
    ok = x.notna()
    x, dates = x[ok], pd.to_datetime(dates[ok])
    if len(x) < 2:
        return {"events": int(len(x))}
    by = x.groupby(dates.dt.to_period(freq)).mean()
    sd = by.std(ddof=1)
    return {"events": int(len(x)), "months": int(len(by)), "mean": float(x.mean()), "median": float(x.median()),
            "t": float(by.mean() / sd * math.sqrt(len(by))) if len(by) > 1 and sd > 0 else 0.0,
            "hit": float((x > 0).mean())}


def line(name: str, s: Dict) -> str:
    if s.get("events", 0) < 2:
        return f"| {name} | {s.get('events', 0)} | - | - | - | - |"
    return (f"| {name} | {s['events']:,} | {100 * s['mean']:+.2f}% | {100 * s['median']:+.2f}% | "
            f"{s['t']:+.2f} | {100 * s['hit']:.1f}% |")


HEAD = ["| Group | Events | Mean | Median | Month t | Share > 0 |", "|---|---|---|---|---|---|"]

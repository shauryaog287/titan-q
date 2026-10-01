"""
ev1_research.py
===============
Implements reports/preregistration_ev1_ai_filings.md (EV1), development only: do filings the
local AI rates positive/high drift up (H1), and negative/high-or-medium drift down (H2)?

    D:\\trader_env\\Scripts\\python.exe ev1_research.py      after llm_backfill.py has labelled the dev window

There is deliberately no confirmation path yet: the confirmation window is labelled and run
only if a hypothesis qualifies here, after this decision is committed.
Output: reports/ev1_ai_filings.md and .json
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
LABELS = os.path.join(DATA_DIR, "events_hist", "ev1_dev.parquet")
MIN_VALUE, MIN_PRICE, MIN_HIST = 1e7, 20.0, 250
EXCLUDE_TYPES = {"results", "routine", "market_commentary"}
COST_H1 = 0.005
MERGE_DAYS = 10
H1_MIN = dict(events=200, mean=0.010, t=2.0)
H2_MIN = dict(events=200, mean=-0.010, t=-2.0)


def panels():
    from whole_market_research import load_panels
    px, vol = load_panels()
    stocks = [c for c in px.columns if c != INDEX_SYMBOL]
    P, V = px[stocks], vol[stocks]
    value = P * V
    med60 = value.shift(1).rolling(60, min_periods=40).median()
    liq = (med60 >= MIN_VALUE) & (P.shift(1) >= MIN_PRICE) & (P.shift(1).notna().cumsum() >= MIN_HIST)
    return P, P.ffill(), liq, med60


def entry_rows(labels: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    t = pd.to_datetime(labels["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    after = (t.dt.hour * 60 + t.dt.minute) >= 15 * 60 + 30
    d = t.dt.normalize() + pd.to_timedelta(after.astype(int), unit="D")
    return np.searchsorted(days.values, d.values, side="left")


def measure(ev: pd.DataFrame, P, F, liq, med60, horizons=(5, 20, 60)) -> pd.DataFrame:
    """ev has columns yahoo, row; adds ex<h> (excess vs the liquid equal-weight pool) and size third."""
    L = liq.to_numpy()
    Pv, Fv = P.to_numpy(), F.to_numpy()
    col = {s: i for i, s in enumerate(P.columns)}
    out = ev.copy()
    n = len(P)
    for h in horizons:
        R = np.full_like(Pv, np.nan)
        R[:n - h] = Fv[h:] / Pv[:n - h] - 1
        Rm = np.where(L, R, np.nan)
        with np.errstate(all="ignore"):
            ew = np.nanmean(Rm, axis=1)
        vals = []
        for r, s in zip(out["row"], out["yahoo"]):
            j = col.get(s)
            vals.append(R[r, j] - ew[r] if j is not None and r + h < n else np.nan)
        out[f"ex{h}"] = vals
    M = med60.to_numpy()
    thirds = []
    for r, s in zip(out["row"], out["yahoo"]):
        j = col.get(s)
        pool = M[r][L[r]]
        pool = pool[np.isfinite(pool)]
        if j is None or not len(pool) or not np.isfinite(M[r, j]):
            thirds.append("unknown")
            continue
        q1, q2 = np.quantile(pool, [1 / 3, 2 / 3])
        thirds.append("small" if M[r, j] <= q1 else "mid" if M[r, j] <= q2 else "large")
    out["size"] = thirds
    return out


def merge(ev: pd.DataFrame) -> pd.DataFrame:
    keep, last = [], {}
    for i, r in ev.sort_values("row").iterrows():
        if r["row"] - last.get(r["yahoo"], -10 ** 9) < MERGE_DAYS:
            continue
        last[r["yahoo"]] = r["row"]
        keep.append(i)
    return ev.loc[keep]


def summary(ev: pd.DataFrame, h: int, cost: float, days: pd.DatetimeIndex) -> Dict:
    e = ev.dropna(subset=[f"ex{h}"])
    if len(e) < 2:
        return {"events": int(len(e))}
    x = e[f"ex{h}"] - cost
    month = days[e["row"].to_numpy()].to_period("M")
    by = x.groupby(np.asarray(month)).mean()
    sd = by.std(ddof=1)
    return {"events": int(len(e)), "months": int(len(by)), "mean": float(x.mean()), "median": float(x.median()),
            "t": float(by.mean() / sd * math.sqrt(len(by))) if len(by) > 1 and sd > 0 else 0.0,
            "hit": float((x > 0).mean())}


def main() -> None:
    lab = pd.read_parquet(LABELS)
    P, F, liq, med60 = panels()
    days = P.index
    lab["row"] = entry_rows(lab, days)
    lab = lab[(lab["row"] < len(days))]
    col = {s: i for i, s in enumerate(P.columns)}
    L = liq.to_numpy()
    lab = lab[[bool(L[r, col[s]]) if s in col else False for r, s in zip(lab["row"], lab["yahoo"])]]
    lab = lab[~lab["event_type"].isin(EXCLUDE_TYPES)]
    h1 = merge(lab[(lab["direction"] == "positive") & (lab["materiality"] == "high")])
    h2 = merge(lab[(lab["direction"] == "negative") & lab["materiality"].isin(["high", "medium"])])
    h1, h2 = measure(h1, P, F, liq, med60), measure(h2, P, F, liq, med60)
    s1, s2 = summary(h1, 20, COST_H1, days), summary(h2, 20, 0.0, days)
    ok1 = s1.get("events", 0) >= H1_MIN["events"] and s1["mean"] >= H1_MIN["mean"] and s1["t"] >= H1_MIN["t"]
    ok2 = s2.get("events", 0) >= H2_MIN["events"] and s2["mean"] <= H2_MIN["mean"] and s2["t"] <= H2_MIN["t"]

    def line(name, s):
        if s.get("events", 0) < 2:
            return f"| {name} | {s.get('events', 0)} | - | - | - | - |"
        return (f"| {name} | {s['events']:,} | {100 * s['mean']:+.2f}% | {100 * s['median']:+.2f}% | "
                f"{s['t']:+.2f} | {100 * s['hit']:.1f}% |")
    head = ["| Group | Events | Mean | Median | Month t | Share > 0 |", "|---|---|---|---|---|---|"]
    out: Dict = {"labels": int(len(lab)), "H1_20d": s1, "H2_20d": s2, "H1_qualifies": ok1, "H2_qualifies": ok2}
    Lr = ["# EV1: filings read by the local AI (development, 2017-2020)", "",
          "Implements reports/preregistration_ev1_ai_filings.md. Excess = the stock's return from the entry close "
          "minus the equal-weight liquid universe over the same days; H1 net of 0.50% cost, H2 without cost. "
          f"{len(lab):,} labelled filings of liquid stocks in scope (results, routine and market commentary left "
          "out); a stock's events less than 10 trading days apart count once.", "",
          "## Primary (20 trading days)", ""] + head + [
          line("**H1: positive, high (net)**", s1), line("**H2: negative, high or medium**", s2), "",
          f"Rules: H1 >= {H1_MIN['events']} events, mean net >= +1.0%, t >= 2.0. "
          f"H2 >= {H2_MIN['events']} events, mean <= -1.0%, t <= -2.0.", "",
          f"**Decision: H1 {'qualifies; label and run the confirmation window once' if ok1 else 'does not qualify'}; "
          f"H2 {'qualifies; label and run the confirmation window once' if ok2 else 'does not qualify'}.**", ""]
    Lr += ["## Also reported (not part of the decision)", ""]
    for name, ev, cost in (("H1", h1, COST_H1), ("H2", h2, 0.0)):
        Lr += [f"### {name} by horizon", ""] + head
        for h in (5, 20, 60):
            s = summary(ev, h, cost, days)
            out[f"{name}_{h}d"] = s
            Lr.append(line(f"{h} days", s))
        Lr += ["", f"### {name} by size third (60-day median traded value within the liquid pool), 20 days", ""] + head
        for sz in ("small", "mid", "large"):
            s = summary(ev[ev["size"] == sz], 20, cost, days)
            out[f"{name}_size_{sz}"] = s
            Lr.append(line(sz, s))
        Lr += ["", f"### {name} by event type, 20 days", ""] + head
        for et, g in sorted(ev.groupby("event_type"), key=lambda kv: -len(kv[1])):
            s = summary(g, 20, cost, days)
            out[f"{name}_type_{et}"] = s
            Lr.append(line(et, s))
        valued = ev["value_cr"].notna()
        Lr += ["", f"### {name}: rupee value stated or not, 20 days", ""] + head + [
            line("value stated", summary(ev[valued], 20, cost, days)),
            line("no value", summary(ev[~valued], 20, cost, days)), ""]
    with open(os.path.join(REPORT_DIR, "ev1_ai_filings.md"), "w", encoding="ascii") as f:
        f.write("\n".join(Lr) + "\n")
    with open(os.path.join(REPORT_DIR, "ev1_ai_filings.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(Lr))


if __name__ == "__main__":
    main()

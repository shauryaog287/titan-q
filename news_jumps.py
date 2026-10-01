"""
news_jumps.py
=============
Implements reports/preregistration_news_jumps.md (NJ1): do jumps backed by an NSE filing keep
going over the next week, and do no-news jumps fade?

    D:\\trader_env\\Scripts\\python.exe news_jumps.py                development only
    D:\\trader_env\\Scripts\\python.exe news_jumps.py --reveal-test  once, after the decision is committed

Needs the announcement archive (order_research.py --fetch) and the Yahoo daily panels.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from titanq_config import INDEX_SYMBOL, PROJECT_DIR

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2017-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-08-31")
JUMP, VOL_X, MIN_VALUE, MIN_PRICE, MIN_HIST = 0.05, 3.0, 1e7, 20.0, 250
COST = 0.005
MIN_EXCESS, MIN_T, MIN_EVENTS = 0.005, 2.0, 150
ROUTINE = {"Trading Window", "Shareholders meeting", "Copy of Newspaper Publication",
           "Analysts/Institutional Investor Meet/Con. Call Updates", "Closure of Trading Window",
           "Certificate under SEBI (Depositories and Participants) Regulations, 2018",
           "Compliances-Certificate under Reg. 74 (5) of SEBI (DP) Regulations, 2018"}


def news_times(ticker: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Yahoo ticker -> sorted broadcast times (int64 ns, IST wall clock) of non-routine filings."""
    from order_research import ANN_DIR
    parts = []
    for year in sorted(os.listdir(ANN_DIR)):
        for f in sorted(os.listdir(os.path.join(ANN_DIR, year))):
            if f.endswith(".parquet"):
                parts.append(pd.read_parquet(os.path.join(ANN_DIR, year, f), columns=["symbol", "desc", "an_dt"]))
    a = pd.concat(parts, ignore_index=True)
    a = a[~a["desc"].isin(ROUTINE)]
    a["t"] = pd.to_datetime(a["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    a["y"] = a["symbol"].map(ticker)
    a = a.dropna(subset=["t", "y"])
    print(f"non-routine announcements mapped to tickers: {len(a):,}")
    return {y: np.sort(g["t"].to_numpy().astype("datetime64[ns]").astype(np.int64)) for y, g in a.groupby("y")}


def events(px: pd.DataFrame, vol: pd.DataFrame, times: Dict[str, np.ndarray], lo: str, hi: str,
           up: bool = True, jump: float = JUMP) -> pd.DataFrame:
    stocks = [c for c in px.columns if c != INDEX_SYMBOL]
    P, V = px[stocks], vol[stocks]
    value = P * V
    liq = (value.shift(1).rolling(60, min_periods=40).median() >= MIN_VALUE) & \
          (P.shift(1) >= MIN_PRICE) & (P.shift(1).notna().cumsum() >= MIN_HIST)
    ret = P / P.shift(1) - 1
    big = (ret >= jump) if up else (ret <= -jump)
    hot = V >= VOL_X * V.shift(1).rolling(20, min_periods=15).mean()
    cond = big & hot & liq
    days = px.index
    filled = P.ffill()
    last: Dict[str, int] = {}
    out = []
    for t in np.where(cond.any(axis=1).to_numpy())[0]:
        d = days[t]
        if not (pd.Timestamp(lo) <= d <= pd.Timestamp(hi)) or t < 1 or t + 20 >= len(days):
            continue
        w0 = (days[t - 1] + pd.Timedelta(hours=15, minutes=30)).value
        w1 = (d + pd.Timedelta(hours=15, minutes=30)).value
        pool = list(liq.columns[liq.iloc[t].to_numpy(dtype=bool)])
        for s in cond.columns[cond.iloc[t].to_numpy(dtype=bool)]:
            if t - last.get(s, -99) < 10:
                continue
            last[s] = t
            ts = times.get(s)
            news = bool(ts is not None and bisect.bisect_right(ts, w1) > bisect.bisect_right(ts, w0))
            row = {"stock": s, "date": d.date(), "news": news, "jump": float(ret.iat[t, stocks.index(s)])}
            for h in (1, 5, 20):
                rs = filled[s].iloc[t + h] / P[s].iloc[t] - 1
                bench = (filled[pool].iloc[t + h] / P[pool].iloc[t] - 1).mean()
                row[f"ex{h}"] = float(rs - bench) if np.isfinite(rs) and np.isfinite(bench) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def summary(ev: pd.DataFrame, h: int = 5, cost: float = COST) -> Dict:
    e = ev.dropna(subset=[f"ex{h}"]) if len(ev) else ev
    if not len(e):
        return {"events": 0, "mean": float("nan"), "t": 0.0, "hit": float("nan")}
    x = e[f"ex{h}"] - cost
    by = x.groupby(e["date"]).mean()
    sd = by.std(ddof=1)
    return {"events": int(len(e)), "dates": int(len(by)), "mean": float(x.mean()),
            "t": float(by.mean() / sd * math.sqrt(len(by))) if len(by) > 1 and sd > 0 else 0.0,
            "hit": float((x > 0).mean())}


def block(px, vol, times, lo, hi, title) -> Tuple[Dict, List[str]]:
    ev = events(px, vol, times, lo, hi)
    news, nonews = ev[ev["news"]] if len(ev) else ev, ev[~ev["news"]] if len(ev) else ev
    res = {"primary_news_5d": summary(news, 5),
           "no_news_5d": summary(nonews, 5),
           "news_1d": summary(news, 1), "news_20d": summary(news, 20),
           "no_news_1d": summary(nonews, 1), "no_news_20d": summary(nonews, 20),
           "news_jump_10pct_5d": summary(news[news["jump"] >= 0.10], 5) if len(news) else summary(news)}
    dn = events(px, vol, times, lo, hi, up=False)
    res["down_news_5d"] = summary(dn[dn["news"]], 5, cost=0.0) if len(dn) else summary(dn)
    res["down_no_news_5d"] = summary(dn[~dn["news"]], 5, cost=0.0) if len(dn) else summary(dn)

    def line(name, s):
        if not s["events"]:
            return f"| {name} | 0 | - | - | - |"
        return f"| {name} | {s['events']:,} | {100 * s['mean']:+.2f}% | {s['t']:+.2f} | {100 * s['hit']:.1f}% |"
    L = [f"## {title}", "", "| Group | Events | Mean excess (net of 0.5% for buys) | Date-clustered t | Hit rate |",
         "|---|---|---|---|---|", line("**Primary: news-backed jumps, 5 days**", res["primary_news_5d"])]
    L += [line(k, v) for k, v in res.items() if k != "primary_news_5d"]
    L += ["", "Down moves are shown without cost (they are not trades here)."]
    return res, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal-test", action="store_true")
    a = ap.parse_args()
    from universe import load_universe
    from whole_market_research import load_panels
    uni = load_universe()
    ticker = dict(zip(uni["symbol"].str.replace("-EQ", "", regex=False), uni["yahoo"]))
    times = news_times(ticker)
    px, vol = load_panels()
    dev, L_dev = block(px, vol, times, *DEV, "Development (event dates 2017-01 to 2020-12)")
    s = dev["primary_news_5d"]
    ok = s["events"] >= MIN_EVENTS and s["mean"] >= MIN_EXCESS and s["t"] >= MIN_T
    L = ["# NJ1: news-backed vs no-news jumps", "", "Implements reports/preregistration_news_jumps.md. "
         "Jump = +5% close-to-close on 3x the 20-day volume, liquid stocks (>= Rs. 1 crore/day). "
         "Excess vs the equal-weight liquid universe from the close of the jump day.", ""] + L_dev
    L += ["", f"Rule: news-backed 5-day net excess >= +{100 * MIN_EXCESS:.1f}%, t >= {MIN_T:.1f}, >= {MIN_EVENTS} events.",
          "", f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**", ""]
    out = {"development": dev, "qualifies": ok}
    if a.reveal_test and ok:
        test, L_test = block(px, vol, times, *TEST, "Test (event dates 2021-01 to 2026-08, revealed once)")
        passed = test["primary_news_5d"]["events"] > 0 and test["primary_news_5d"]["mean"] > 0
        L += L_test + ["", f"**Test: {'passed' if passed else 'failed'}.**", ""]
        out["test"], out["test_passed"] = test, passed
    with open(os.path.join(REPORT_DIR, "news_jumps.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "news_jumps.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

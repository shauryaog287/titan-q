"""
delivery_research.py
====================
Implements reports/preregistration_delivery.md (DEL1): does the delivery percentage (the
share of traded shares actually delivered, not squared off intraday) carry information?

    D:\\trader_env\\Scripts\\python.exe delivery_research.py --fetch        download NSE MTO files (resumable)
    D:\\trader_env\\Scripts\\python.exe delivery_research.py                development only (DEL1a and DEL1b)
    D:\\trader_env\\Scripts\\python.exe delivery_research.py --reveal-test  once, after the decisions are committed

Data: NSE "Security-wise Delivery Position" files, one per trading day since 2016,
saved as data/delivery/<year>/<date>.parquet (symbol, series, traded, deliverable, pct).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR

DELIV_DIR = os.path.join(DATA_DIR, "delivery")
PANEL = os.path.join(DATA_DIR, "delivery_panel.parquet")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2016-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-08-31")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
GAP_SECONDS = 2.0


# ===========================================================================
# DOWNLOAD
# ===========================================================================

def parse_mto(text: str) -> pd.DataFrame:
    """Record-type 20 lines: 20,<sr>,<symbol>,<series>,<traded>,<deliverable>,<pct>."""
    rows = []
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7 and p[0] == "20":
            try:
                rows.append((p[2], p[3], float(p[4]), float(p[5]), float(p[6])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["symbol", "series", "traded", "deliverable", "pct"])


def day_path(d: date) -> str:
    return os.path.join(DELIV_DIR, str(d.year), f"{d}.parquet")


def fetch(start: str = "2016-01-01") -> None:
    from market_data import DailyStore
    idx = DailyStore().load([INDEX_SYMBOL], refresh=False)[INDEX_SYMBOL].index
    days = [d.date() for d in pd.to_datetime(idx) if d >= pd.Timestamp(start)]
    todo = [d for d in days if not os.path.exists(day_path(d))]
    print(f"{len(days)} trading days, {len(todo)} to fetch", flush=True)
    s = requests.Session()
    s.headers.update(UA)
    missing = 0
    for i, d in enumerate(todo):
        url = f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d:%d%m%Y}.DAT"
        for attempt in range(3):
            try:
                r = s.get(url, timeout=30)
                break
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
        else:
            continue
        if r.status_code == 200:
            df = parse_mto(r.text)
            os.makedirs(os.path.dirname(day_path(d)), exist_ok=True)
            df.to_parquet(day_path(d) + ".partial", index=False)
            os.replace(day_path(d) + ".partial", day_path(d))
        else:
            missing += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} ({d}), {missing} not available", flush=True)
        time.sleep(GAP_SECONDS)
    print(f"done; {missing} days had no file", flush=True)


def delivery_panel(rebuild: bool = False) -> pd.DataFrame:
    """dates x Yahoo symbols of delivery % (EQ series only)."""
    if os.path.exists(PANEL) and not rebuild:
        return pd.read_parquet(PANEL)
    from universe import load_universe
    uni = load_universe()
    ymap = dict(zip(uni["symbol"].str.replace("-EQ", "", regex=False), uni["yahoo"]))
    parts = {}
    for year in sorted(os.listdir(DELIV_DIR)):
        for f in sorted(os.listdir(os.path.join(DELIV_DIR, year))):
            if f.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(DELIV_DIR, year, f))
                df = df[df["series"] == "EQ"]
                parts[pd.Timestamp(f[:10])] = pd.Series(df["pct"].to_numpy(), index=df["symbol"].map(ymap)).dropna()
    panel = pd.DataFrame({d: s[~s.index.duplicated()] for d, s in parts.items()}).T.sort_index()
    panel.to_parquet(PANEL)
    return panel


# ===========================================================================
# DEL1a: ABNORMAL DELIVERY, MONTHLY
# ===========================================================================

def make_signal_fn(D: pd.DataFrame):
    """signals_at-compatible function: ABN_DELIV and DELIV_LEVEL at row a of px."""
    def fn(px: pd.DataFrame, a: int, stocks: List[str]) -> pd.DataFrame:
        rows = D.reindex(px.index[max(0, a - 249):a + 1])
        cols = [s for s in stocks if s in rows.columns]
        w = rows[cols]
        last20, prev = w.iloc[-20:].mean(), w.iloc[:-20].mean()
        ok = w.notna().sum() >= 150
        out = pd.DataFrame(index=stocks)
        out["ABN_DELIV"] = (last20 - prev).where(ok)
        out["DELIV_LEVEL"] = last20.where(ok)
        return out
    return fn


def del1a(px, vol, D, period) -> Dict:
    from whole_market_research import monthly_study, summarise
    df = monthly_study(px, vol, signal_names=("ABN_DELIV", "DELIV_LEVEL"), signal_fn=make_signal_fn(D))
    rows = summarise(df, period, ("ABN_DELIV", "DELIV_LEVEL"))
    return {"rows": rows, "months": int(len(df.loc[period[0]:period[1]]))}


# ===========================================================================
# DEL1b: RETAIL-FRENZY REVERSAL, DAILY EVENTS
# ===========================================================================

def del1b_events(px, vol, D, lo: str, hi: str, jump: float = 0.05, low_delivery: bool = True,
                 horizons=(5, 10, 20)) -> pd.DataFrame:
    from whole_market_research import eligible_at
    value = px.drop(columns=[INDEX_SYMBOL]) * vol
    stocks = [c for c in px.columns if c != INDEX_SYMBOL and c in D.columns]
    P, V = px[stocks], vol[stocks]
    Dd = D.reindex(px.index)[stocks]
    ret = P / P.shift(1) - 1
    vavg = V.shift(1).rolling(20, min_periods=15).mean()
    q = Dd.shift(1).rolling(250, min_periods=150).quantile(0.75 if not low_delivery else 0.25)
    big = ret >= jump if jump > 0 else ret <= jump
    cond = big & (V >= 3 * vavg) & ((Dd < q) if low_delivery else (Dd > q))
    days = px.index
    filled = P.ffill()
    elig_cache: Dict[int, set] = {}
    last_event: Dict[str, int] = {}
    out = []
    for t in np.where(cond.any(axis=1).to_numpy())[0]:
        d = days[t]
        if not (pd.Timestamp(lo) <= d <= pd.Timestamp(hi)) or t < 260 or t + 1 + max(horizons) >= len(days):
            continue
        hits = cond.columns[cond.iloc[t].to_numpy(dtype=bool)]
        if t not in elig_cache:
            elig_cache[t] = set(eligible_at(px, value, t))
        elig = elig_cache[t]
        ew_cols = [s for s in stocks if s in elig]
        for s in hits:
            if s not in elig or t - last_event.get(s, -99) < 10:
                continue
            last_event[s] = t
            row = {"stock": s, "date": d.date()}
            for h in horizons:
                a, b = t + 1, t + 1 + h
                rs = filled[s].iloc[b] / P[s].iloc[a] - 1
                bench = (filled[ew_cols].iloc[b] / P[ew_cols].iloc[a] - 1).mean()
                row[f"ex{h}"] = float(rs - bench) if np.isfinite(rs) and np.isfinite(bench) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def summarise_events(ev: pd.DataFrame, h: int = 10) -> Dict:
    if not len(ev) or f"ex{h}" not in ev:
        return {"events": 0, "mean": float("nan"), "t": 0.0, "dates": 0, "neg_rate": float("nan")}
    e = ev.dropna(subset=[f"ex{h}"])
    by = e.groupby("date")[f"ex{h}"].mean()
    sd = by.std(ddof=1)
    return {"events": int(len(e)), "dates": int(len(by)), "mean": float(e[f"ex{h}"].mean()),
            "t": float(by.mean() / sd * math.sqrt(len(by))) if len(by) > 1 and sd > 0 else 0.0,
            "neg_rate": float((e[f"ex{h}"] < 0).mean())}


def del1b(px, vol, D, period) -> Dict:
    main = del1b_events(px, vol, D, *period)
    res = {"primary_10d": summarise_events(main, 10), "5d": summarise_events(main, 5),
           "20d": summarise_events(main, 20)}
    res["high_delivery_jump_10d"] = summarise_events(del1b_events(px, vol, D, *period, low_delivery=False), 10)
    res["low_delivery_drop_10d"] = summarise_events(del1b_events(px, vol, D, *period, jump=-0.05), 10)
    return res


# ===========================================================================
# REPORT
# ===========================================================================

def a_qualifies(rows) -> bool:
    return bool(rows[0]["qualifies"])


def b_qualifies(s: Dict) -> bool:
    return s["events"] >= 100 and s["mean"] <= -0.01 and s["t"] <= -2.5


def ev_line(name: str, s: Dict) -> str:
    if not s["events"]:
        return f"| {name} | 0 | - | - | - |"
    return (f"| {name} | {s['events']:,} ({s['dates']:,} dates) | {100 * s['mean']:+.2f}% | {s['t']:+.2f} | "
            f"{100 * s['neg_rate']:.1f}% |")


def block(px, vol, D, period, title) -> Tuple[Dict, List[str]]:
    from whole_market_research import table
    a = del1a(px, vol, D, period)
    b = del1b(px, vol, D, period)
    L = table(a["rows"], f"{title}: DEL1a, abnormal delivery (monthly)")
    L += [f"### {title}: DEL1b, retail-frenzy reversal (events)", "",
          "| Events | N | Mean excess | Date-clustered t | Share negative |", "|---|---|---|---|---|",
          ev_line("**Primary: +5% on 3x volume, low delivery, next 10 days**", b["primary_10d"]),
          ev_line("same, 5 days", b["5d"]), ev_line("same, 20 days", b["20d"]),
          ev_line("+5% on 3x volume, HIGH delivery, 10 days", b["high_delivery_jump_10d"]),
          ev_line("-5% on 3x volume, low delivery, 10 days", b["low_delivery_drop_10d"]), ""]
    return {"DEL1a": a["rows"], "DEL1b": b}, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--reveal-test", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    from whole_market_research import load_panels
    px, vol = load_panels()
    D = delivery_panel(rebuild=a.rebuild)
    print(f"delivery panel: {D.shape[0]} days x {D.shape[1]} stocks, {D.index.min().date()} .. {D.index.max().date()}")
    dev, L_dev = block(px, vol, D, DEV, "Development 2016-2020")
    qa, qb = a_qualifies(dev["DEL1a"]), b_qualifies(dev["DEL1b"]["primary_10d"])
    L = ["# DEL1: delivery percentage", "", "Implements reports/preregistration_delivery.md.", ""] + L_dev
    L += [f"**Decision DEL1a: {'qualifies; reveal its test once' if qa else 'does not qualify; test sealed'}.**", "",
          f"**Decision DEL1b: {'qualifies; reveal its test once' if qb else 'does not qualify; test sealed'}.**", ""]
    out = {"development": dev, "DEL1a_qualifies": qa, "DEL1b_qualifies": qb}
    if a.reveal_test and (qa or qb):
        test, L_test = block(px, vol, D, TEST, "Test 2021-2026 (revealed once)")
        L += L_test
        if qa:
            r = test["DEL1a"][0]
            out["DEL1a_test_passed"] = bool(r["dec_excess_ann"] > 0 and r["mean_ic"] > 0)
            L.append(f"**Test DEL1a: {'passed' if out['DEL1a_test_passed'] else 'failed'}.**")
        if qb:
            out["DEL1b_test_passed"] = bool(test["DEL1b"]["primary_10d"]["mean"] < 0)
            L.append(f"**Test DEL1b: {'passed' if out['DEL1b_test_passed'] else 'failed'}.**")
        out["test"] = test
    with open(os.path.join(REPORT_DIR, "delivery.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "delivery.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

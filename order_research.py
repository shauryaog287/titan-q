"""
order_research.py
=================
Implements reports/preregistration_order_wins.md (ORD1): do stocks drift up after they
announce an order win?

    D:\\trader_env\\Scripts\\python.exe order_research.py --fetch        download NSE announcements, 2017 onward (resumable)
    D:\\trader_env\\Scripts\\python.exe order_research.py                development only
    D:\\trader_env\\Scripts\\python.exe order_research.py --reveal-test  once, after the decision is committed

The download keeps EVERY announcement (one file per day under data/announcements/<year>/),
so later studies (auditor resignations and so on) can reuse it without downloading again.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR

ANN_DIR = os.path.join(DATA_DIR, "announcements")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2017-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-08-31")
HOLD = 20
MIN_EXCESS, MIN_T, MIN_EVENTS = 0.010, 2.0, 100

ORDER = re.compile(
    r"bag(ged|ging)? (an |a |the )?(new )?(order|contract)s?"
    r"|receiv(ed|ing|es)? (an |a |the )?(new )?(order|contract|letter of award|work order)s?"
    r"|letter of (award|acceptance|intent)"
    r"|work order"
    r"|(order|contract)s? (worth|valued at|of value|aggregating)"
    r"|award(ed)? (of )?(an |a |the )?(order|contract)", re.I)
EXCLUDE = re.compile(r"cancel|terminat|withdraw|rumour", re.I)
PSU_DEFENCE_NAMES = re.compile(r"bharat|hindustan|garden reach|mazagon|cochin shipyard|beml", re.I)
VALUE = re.compile(r"Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(crore|cr\b)", re.I)


def day_path(d: date) -> str:
    return os.path.join(ANN_DIR, str(d.year), f"{d}.parquet")


def fetch(start: str = "2017-01-01") -> None:
    from disclosure_radar import NSE, _rows
    import disclosure_radar
    disclosure_radar.GAP_SECONDS = 5.0
    end = date.today() - timedelta(days=1)
    days = [date.fromisoformat(start) + timedelta(days=k)
            for k in range((end - date.fromisoformat(start)).days + 1)]
    todo = [d for d in days if not os.path.exists(day_path(d))]
    print(f"{len(days)} days, {len(todo)} to fetch", flush=True)
    nse = NSE()
    for i, d in enumerate(todo):
        rows = nse.json("https://www.nseindia.com/api/corporate-announcements?index=equities"
                        f"&from_date={d:%d-%m-%Y}&to_date={d:%d-%m-%Y}")
        if rows is None:
            continue                                  # failed: retried on the next run
        df = pd.DataFrame([{"symbol": r.get("symbol"), "desc": r.get("desc"), "an_dt": r.get("an_dt"),
                            "text": (r.get("attchmntText") or "")[:1500], "file": r.get("attchmntFile")}
                           for r in _rows(rows)])
        os.makedirs(os.path.dirname(day_path(d)), exist_ok=True)
        df.to_parquet(day_path(d) + ".partial", index=False)
        os.replace(day_path(d) + ".partial", day_path(d))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} ({d})", flush=True)
    print("done", flush=True)


def load_order_wins() -> pd.DataFrame:
    parts = []
    for year in sorted(os.listdir(ANN_DIR)):
        for f in sorted(os.listdir(os.path.join(ANN_DIR, year))):
            if f.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(ANN_DIR, year, f))
                if len(df) and "desc" in df.columns:                # empty days have no columns
                    parts.append(df)
    a = pd.concat(parts, ignore_index=True)
    text = a["desc"].fillna("") + " " + a["text"].fillna("")
    hit = text.str.contains(ORDER) & ~text.str.contains(EXCLUDE) & \
        (a["desc"].fillna("") != "Clarification - Financial results")
    w = a[hit].copy()
    w["broadcast"] = pd.to_datetime(w["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    v = w["text"].str.extract(VALUE)[0].str.replace(",", "", regex=False)
    w["value"] = pd.to_numeric(v, errors="coerce").fillna(0.0)          # Rs. crore when stated
    print(f"announcements: {len(a):,}; order wins by the pre-registered pattern: {len(w):,}")
    return w.dropna(subset=["broadcast"])


def block(w, px, R, ticker, lo, hi, title):
    from insider_research import measure, summarise, to_events
    ev = to_events(w, px, ticker)
    main = summarise(measure(ev, R, lo, hi, HOLD))
    res = {"primary": main,
           "5 days": summarise(measure(ev, R, lo, hi, 5)),
           "60 days": summarise(measure(ev, R, lo, hi, 60))}
    valued = w[w["value"] > 0]
    if len(valued):
        med = valued["value"].median()
        res["stated value above median"] = summarise(measure(to_events(valued[valued["value"] > med], px, ticker),
                                                             R, lo, hi, HOLD))
        res["stated value at or below median"] = summarise(measure(
            to_events(valued[valued["value"] <= med], px, ticker), R, lo, hi, HOLD))
    names = dict(zip(ticker.keys(), ticker.keys()))
    psu = w[w["symbol"].map(lambda s: bool(PSU_DEFENCE_NAMES.search(str(s))) or str(s) in
                            {"BEL", "HAL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "BEML", "NBCC",
                             "RVNL", "IRCON", "RAILTEL", "NTPC", "POWERGRID", "SJVN", "NHPC"})]
    res["government-owned / defence names"] = summarise(measure(to_events(psu, px, ticker), R, lo, hi, HOLD))

    def line(name, s):
        if not s["events"]:
            return f"| {name} | 0 | - | - | - | - |"
        return (f"| {name} | {s['events']:,} | {100 * s['mean_excess']:+.2f}% | {100 * s['median_excess']:+.2f}% | "
                f"{s['month_t']:+.2f} | {100 * s['hit_rate']:.1f}% |")
    L = [f"## {title}", "",
         "| Events | N | Mean net excess | Median | Month-clustered t | Hit rate |", "|---|---|---|---|---|---|",
         line(f"**Primary: order wins, {HOLD} days**", main)] + [line(k, v) for k, v in res.items() if k != "primary"]
    return res, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--reveal-test", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        logging.basicConfig(level=logging.WARNING)
        fetch()
        return
    from insider_research import Returns
    from universe import load_universe
    from whole_market_research import load_panels
    w = load_order_wins()
    uni = load_universe()
    ticker = dict(zip(uni["symbol"].str.replace("-EQ", "", regex=False), uni["yahoo"]))
    px, vol = load_panels()
    R = Returns(px, vol)
    dev, L_dev = block(w, px, R, ticker, *DEV, "Development (entries 2017-01 to 2020-12)")
    s = dev["primary"]
    ok = s["events"] >= MIN_EVENTS and s["mean_excess"] >= MIN_EXCESS and s["month_t"] >= MIN_T
    L = ["# ORD1: order-win announcements", "", "Implements reports/preregistration_order_wins.md. Net excess = "
         f"{HOLD}-day return from the close after the announcement date, minus the equal-weight eligible universe, "
         "minus 0.50% cost.", ""] + L_dev
    L += ["", f"Rule: mean net excess >= +{100 * MIN_EXCESS:.1f}%, month t >= {MIN_T:.1f}, >= {MIN_EVENTS} events.", "",
          f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**", ""]
    out = {"development": dev, "qualifies": ok}
    if a.reveal_test and ok:
        test, L_test = block(w, px, R, ticker, *TEST, "Test (entries 2021-01 onward, revealed once)")
        passed = test["primary"]["events"] > 0 and test["primary"]["mean_excess"] > 0
        L += L_test + ["", f"**Test: {'passed' if passed else 'failed'}.**", ""]
        out["test"], out["test_passed"] = test, passed
    with open(os.path.join(REPORT_DIR, "order_wins.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "order_wins.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

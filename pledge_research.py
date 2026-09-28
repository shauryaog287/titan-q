"""
pledge_research.py
==================
Implements reports/preregistration_pledge_invocations.md (INS2): do stocks whose
promoters' pledged shares are invoked by lenders underperform over the next 60
trading days? If so, an invocation becomes a hold-nothing red flag.

    D:\\trader_env\\Scripts\\python.exe pledge_research.py                development only
    D:\\trader_env\\Scripts\\python.exe pledge_research.py --reveal-test  once, after the decision is committed

Uses the PIT disclosures downloaded by insider_research.py --fetch.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from insider_research import (DEV, REPORT_DIR, TEST, Returns, load_disclosures, measure, summarise,
                              to_events)
from universe import load_universe
from whole_market_research import load_panels

HOLD = 60
PROMOTERS = ("Promoters", "Promoter Group")
MAX_EXCESS, MAX_T, MIN_EVENTS = -0.02, -2.0, 30


def invocations(d: pd.DataFrame, cats: Optional[Tuple[str, ...]] = PROMOTERS) -> pd.DataFrame:
    m = ((d["tdpTransactionType"] == "Pledge Invoke") | (d["acqMode"] == "Invocation of pledge")) \
        & d["broadcast"].notna()
    if cats is not None:
        m &= d["personCategory"].isin(cats)
    return d[m]


def transactions(d: pd.DataFrame, tx: str) -> pd.DataFrame:
    return d[(d["tdpTransactionType"] == tx) & d["personCategory"].isin(PROMOTERS) & d["broadcast"].notna()]


def stats(m: pd.DataFrame) -> Dict[str, float]:
    s = summarise(m)
    s["neg_rate"] = float((m["excess"] < 0).mean()) if len(m) else float("nan")
    return s


def qualifies(s: Dict) -> bool:
    return s["events"] >= MIN_EVENTS and s["mean_excess"] <= MAX_EXCESS and s["month_t"] <= MAX_T


def row(name: str, s: Dict) -> str:
    if not s["events"]:
        return f"| {name} | 0 | - | - | - | - |"
    return (f"| {name} | {s['events']:,} | {100 * s['mean_excess']:+.2f}% | {100 * s['median_excess']:+.2f}% | "
            f"{s['month_t']:+.2f} | {100 * s['neg_rate']:.1f}% |")


def block(d, px, R, ticker, lo, hi, title) -> Tuple[Dict, List[str]]:
    inv = to_events(invocations(d), px, ticker)
    res = {"primary": stats(measure(inv, R, lo, hi, HOLD, cost=0.0)),
           "invocations, 20 days": stats(measure(inv, R, lo, hi, 20, cost=0.0)),
           "invocations, 120 days": stats(measure(inv, R, lo, hi, 120, cost=0.0)),
           "invocations, any person category": stats(measure(
               to_events(invocations(d, None), px, ticker), R, lo, hi, HOLD, cost=0.0)),
           "new promoter pledges": stats(measure(
               to_events(transactions(d, "Pledge"), px, ticker), R, lo, hi, HOLD, cost=0.0)),
           "promoter pledge releases": stats(measure(
               to_events(transactions(d, "Pledge Revoke"), px, ticker), R, lo, hi, HOLD, cost=0.0))}
    L = [f"## {title}", "",
         "| Events | N | Mean excess | Median | Month-clustered t | Share negative |", "|---|---|---|---|---|---|",
         row(f"**Primary: promoter pledge invocations, {HOLD} days**", res["primary"])]
    L += [row(k, v) for k, v in res.items() if k != "primary"]
    return res, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal-test", action="store_true")
    a = ap.parse_args()
    d = load_disclosures()
    uni = load_universe()
    ticker = dict(zip(uni["symbol"].str.replace("-EQ", "", regex=False), uni["yahoo"]))
    px, vol = load_panels()
    R = Returns(px, vol)
    dev, L_dev = block(d, px, R, ticker, *DEV, "Development (entries 2016-01 to 2020-12)")
    ok = qualifies(dev["primary"])
    L = ["# INS2: promoter pledge invocations as a red flag", "",
         "Implements reports/preregistration_pledge_invocations.md. Excess = stock return over the next "
         f"{HOLD} trading days (from the close after NSE's broadcast date) minus the equal-weight eligible "
         "universe; no cost (the flag works by not holding).", ""] + L_dev
    L += ["", f"Rule: mean excess <= {100 * MAX_EXCESS:.1f}%, month-clustered t <= {MAX_T:.1f}, >= {MIN_EVENTS} events.",
          "", f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**", ""]
    out = {"development": dev, "qualifies": ok, "test_revealed": bool(a.reveal_test and ok)}
    if a.reveal_test and ok:
        test, L_test = block(d, px, R, ticker, *TEST, "Test (entries 2021-01 onward, revealed once)")
        passed = test["primary"]["events"] > 0 and test["primary"]["mean_excess"] < 0
        L += L_test + ["", f"**Test: {'passed; adopt the red flag' if passed else 'failed; not adopted'}.**", ""]
        out["test"], out["test_passed"] = test, passed
    with open(os.path.join(REPORT_DIR, "pledge_invocations.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "pledge_invocations.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

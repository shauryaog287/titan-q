"""
fo_research.py
==============
Implements reports/preregistration_fo_positioning.md (FO1): is retail ("Client") crowding in
NIFTY index futures a contrarian signal for the next week?

    D:\\trader_env\\Scripts\\python.exe fo_research.py --fetch        download participant OI files (resumable)
    D:\\trader_env\\Scripts\\python.exe fo_research.py                development only
    D:\\trader_env\\Scripts\\python.exe fo_research.py --reveal-test  once, after the decision is committed

Data: NSE fao_participant_oi_DDMMYYYY.csv (contracts by Client/DII/FII/Pro), one per
trading day since 2016, saved as data/fo_participant/<year>/<date>.parquet.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR

FO_DIR = os.path.join(DATA_DIR, "fo_participant")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2016-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-08-31")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
GAP_SECONDS = 2.0
MIN_DIFF, MIN_T, MIN_CROWDED = 0.003, 2.0, 40


def day_path(d: date) -> str:
    return os.path.join(FO_DIR, str(d.year), f"{d}.parquet")


def parse(text: str) -> pd.DataFrame:
    lines = text.replace('""', '"').splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Client Type"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [c.strip() for c in df.columns]
    df["Client Type"] = df["Client Type"].astype(str).str.strip()
    return df


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
        url = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv"
        r = None
        for attempt in range(3):
            try:
                r = s.get(url, timeout=30)
                break
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
        if r is not None and r.status_code == 200:
            try:
                df = parse(r.text)
                os.makedirs(os.path.dirname(day_path(d)), exist_ok=True)
                df.to_parquet(day_path(d) + ".partial", index=False)
                os.replace(day_path(d) + ".partial", day_path(d))
            except Exception:
                missing += 1
        else:
            missing += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} ({d}), {missing} not available", flush=True)
        time.sleep(GAP_SECONDS)
    print(f"done; {missing} days had no usable file", flush=True)


def _num(x) -> float:
    """Some days' files write numbers with thousands separators ("80,776.00")."""
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return float("nan")


def panel() -> pd.DataFrame:
    """Per date: Client and FII net index-futures ratios, Client index put/call (longs)."""
    rows = {}
    for year in sorted(os.listdir(FO_DIR)):
        for f in sorted(os.listdir(os.path.join(FO_DIR, year))):
            if not f.endswith(".parquet"):
                continue
            df = pd.read_parquet(os.path.join(FO_DIR, year, f)).set_index("Client Type")
            rec = {}
            for who in ("Client", "FII"):
                if who in df.index:
                    L, S = _num(df.loc[who, "Future Index Long"]), _num(df.loc[who, "Future Index Short"])
                    rec[f"{who}_net"] = (L - S) / (L + S) if L + S > 0 else np.nan
            if "Client" in df.index:
                c = _num(df.loc["Client", "Option Index Call Long"])
                rec["Client_pcr"] = _num(df.loc["Client", "Option Index Put Long"]) / c if c > 0 else np.nan
            rows[pd.Timestamp(f[:10])] = rec
    return pd.DataFrame(rows).T.sort_index()


def weekly(P: pd.DataFrame, nifty: pd.Series, col: str, lo: str, hi: str) -> pd.DataFrame:
    """One row per week (last trading day t): percentile of col vs its previous 250 days and
    the NIFTY return from close t+1 to close t+6."""
    s = P[col].reindex(nifty.index)
    pct = s.rolling(251, min_periods=201).apply(lambda w: (w[:-1] < w[-1]).mean() if np.isfinite(w[-1]) else np.nan,
                                                raw=True)
    idx = nifty.index
    week = pd.Series(idx.to_period("W"), index=idx)
    last = week != week.shift(-1)
    out = []
    for t in np.where(last.to_numpy())[0]:
        d = idx[t]
        if not (pd.Timestamp(lo) <= d <= pd.Timestamp(hi)) or t + 6 >= len(idx) or not np.isfinite(pct.iloc[t]):
            continue
        out.append({"date": d.date(), "pct": float(pct.iloc[t]),
                    "fwd": float(nifty.iloc[t + 6] / nifty.iloc[t + 1] - 1)})
    return pd.DataFrame(out)


def compare(w: pd.DataFrame, group_hi: bool) -> Dict:
    """normal minus crowded (crowded = pct >= 0.8 if group_hi else pct <= 0.2)."""
    from scipy import stats
    crowd = w["pct"] >= 0.8 if group_hi else w["pct"] <= 0.2
    normal = (w["pct"] > 0.2) & (w["pct"] < 0.8)
    a, b = w.loc[normal, "fwd"], w.loc[crowd, "fwd"]
    if len(a) < 5 or len(b) < 5:
        return {"weeks": int(len(w)), "crowded": int(len(b)), "diff": float("nan"), "t": 0.0}
    t = stats.ttest_ind(a, b, equal_var=False).statistic
    return {"weeks": int(len(w)), "crowded": int(len(b)), "normal": int(len(a)),
            "mean_normal": float(a.mean()), "mean_crowded": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "t": float(t)}


def block(P, nifty, period, title) -> Tuple[Dict, List[str]]:
    wc = weekly(P, nifty, "Client_net", *period)
    wf = weekly(P, nifty, "FII_net", *period)
    wp = weekly(P, nifty, "Client_pcr", *period)
    res = {"primary_client_crowded_long": compare(wc, True),
           "client_crowded_short": compare(wc, False),
           "fii_crowded_long": compare(wf, True),
           "fii_crowded_short": compare(wf, False),
           "client_high_put_call": compare(wp, True)}

    def line(name, r, note):
        if not np.isfinite(r["diff"]):
            return f"| {name} | {r['crowded']} | - | - | - | {note} |"
        return (f"| {name} | {r['crowded']} of {r['weeks']} | {100 * r['mean_crowded']:+.2f}% | "
                f"{100 * r['mean_normal']:+.2f}% | {100 * r['diff']:+.2f}% (t {r['t']:+.2f}) | {note} |")
    L = [f"## {title}", "",
         "| Group (vs normal weeks) | Crowded weeks | NIFTY next week, crowded | Normal | Normal minus crowded | Reading |",
         "|---|---|---|---|---|---|",
         line("**Primary: retail net long (Client >= 80th pct)**", res["primary_client_crowded_long"],
              "positive = fade retail"),
         line("retail net short (Client <= 20th pct)", res["client_crowded_short"], "negative = fade retail"),
         line("FII net long (>= 80th pct)", res["fii_crowded_long"], "negative = follow FII"),
         line("FII net short (<= 20th pct)", res["fii_crowded_short"], "positive = follow FII"),
         line("retail put/call high (>= 80th pct)", res["client_high_put_call"], "negative = fade fear"), ""]
    return res, L


def qualifies(r: Dict) -> bool:
    return r["crowded"] >= MIN_CROWDED and np.isfinite(r["diff"]) and r["diff"] >= MIN_DIFF and r["t"] >= MIN_T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--reveal-test", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    from market_data import DailyStore
    nifty = DailyStore().load([INDEX_SYMBOL], refresh=False)[INDEX_SYMBOL]["Close"].dropna()
    nifty.index = pd.to_datetime(nifty.index)
    P = panel()
    print(f"participant panel: {len(P)} days, {P.index.min().date()} .. {P.index.max().date()}")
    dev, L_dev = block(P, nifty, DEV, "Development 2016-2020 (weekly, NIFTY return close t+1 to t+6)")
    ok = qualifies(dev["primary_client_crowded_long"])
    L = ["# FO1: retail and FII positioning in NIFTY index futures", "",
         "Implements reports/preregistration_fo_positioning.md.", ""] + L_dev
    L += [f"Rule: normal minus retail-crowded-long >= +{100 * MIN_DIFF:.2f}%/week, Welch t >= {MIN_T:.1f}, "
          f">= {MIN_CROWDED} crowded weeks.", "",
          f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**", ""]
    out = {"development": dev, "qualifies": ok}
    if a.reveal_test and ok:
        test, L_test = block(P, nifty, TEST, "Test 2021-2026 (revealed once)")
        passed = test["primary_client_crowded_long"]["diff"] > 0
        L += L_test + [f"**Test: {'passed' if passed else 'failed'}.**", ""]
        out["test"], out["test_passed"] = test, bool(passed)
    with open(os.path.join(REPORT_DIR, "fo_positioning.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "fo_positioning.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

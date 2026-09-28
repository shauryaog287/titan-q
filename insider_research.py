"""
insider_research.py
===================
Implements reports/preregistration_insider_buys.md (INS1): do open-market purchases
by promoters, directors and key managers, as disclosed to NSE under SEBI's PIT
regulations, predict the stock's excess return over the next 60 trading days?

    D:\\trader_env\\Scripts\\python.exe insider_research.py --fetch        download disclosures (cached per month)
    D:\\trader_env\\Scripts\\python.exe insider_research.py                development only
    D:\\trader_env\\Scripts\\python.exe insider_research.py --reveal-test  once, after the decision is committed

Only published disclosures are used, and a position is entered only after the
disclosure was public. Downloads are slow on purpose (one request every few seconds).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR
from universe import load_universe
from whole_market_research import eligible_at, load_panels

PIT_DIR = os.path.join(DATA_DIR, "disclosures", "pit")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
FIRST_MONTH = "2016-01"
DEV = ("2016-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-12-31")
HOLD = 60
COST = 0.005                                   # round trip (pre-registered)
MIN_VALUE = 1_000_000.0                        # Rs. 10 lakh
BIG_VALUE = 10_000_000.0                       # Rs. 1 crore (secondary)
INSIDERS = ("Promoters", "Promoter Group", "Director", "Key Managerial Personnel")
PROMOTERS = ("Promoters", "Promoter Group")
MIN_EXCESS, MIN_T, MIN_EVENTS = 0.010, 2.0, 100
GAP_SECONDS = 3.0


# ===========================================================================
# DOWNLOAD
# ===========================================================================

class NSESession:
    """Browser-like session for NSE's public filings pages, with a pause between requests."""

    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
               "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
               "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"}

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update(self.HEADERS)
        self.last = 0.0
        self.warm()

    def warm(self) -> None:
        self._pace()
        self.s.get("https://www.nseindia.com/", timeout=30)

    def _pace(self) -> None:
        time.sleep(max(0.0, self.last + GAP_SECONDS - time.time()))
        self.last = time.time()

    def get_json(self, url: str):
        for attempt in range(4):
            self._pace()
            try:
                r = self.s.get(url, timeout=60)
                if r.status_code == 200:
                    return r.json()
            except (requests.RequestException, ValueError):
                pass
            time.sleep(10.0 * (attempt + 1))
            self.warm()                          # fresh cookies
        raise RuntimeError(f"NSE request failed 4 times: {url}")


def fetch() -> None:
    os.makedirs(PIT_DIR, exist_ok=True)
    nse = NSESession()
    today = pd.Timestamp.today()
    months = pd.period_range(FIRST_MONTH, today.strftime("%Y-%m"), freq="M")
    for p in months:
        path = os.path.join(PIT_DIR, f"{p}.json")
        recent = p >= (today - pd.DateOffset(months=2)).to_period("M")
        if os.path.exists(path) and not recent:
            continue
        a, b = p.start_time, min(p.end_time, today)
        url = ("https://www.nseindia.com/api/corporates-pit?index=equities"
               f"&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}")
        data = (nse.get_json(url) or {}).get("data", [])
        with open(path + ".partial", "w", encoding="ascii") as f:
            json.dump(data, f)
        os.replace(path + ".partial", path)
        print(f"  {p}: {len(data)} disclosures", flush=True)


# ===========================================================================
# EVENTS
# ===========================================================================

def load_disclosures() -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(PIT_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(PIT_DIR, f), encoding="ascii") as fh:
                rows += json.load(fh)
    d = pd.DataFrame(rows)
    d["value"] = pd.to_numeric(d["secVal"], errors="coerce")
    d["broadcast"] = pd.to_datetime(d["date"], format="%d-%b-%Y %H:%M", errors="coerce")
    return d.drop_duplicates(subset=["symbol", "date", "acqName", "secAcq", "acqMode", "tdpTransactionType"])


def select(d: pd.DataFrame, side: str, cats=INSIDERS, min_value: float = MIN_VALUE) -> pd.DataFrame:
    mode, tx = ("Market Purchase", "Buy") if side == "buy" else ("Market Sale", "Sell")
    m = ((d["acqMode"] == mode) & (d["tdpTransactionType"] == tx) & d["personCategory"].isin(cats)
         & (d["secType"] == "Equity Shares") & (d["value"] >= min_value) & d["broadcast"].notna())
    return d[m]


def to_events(sel: pd.DataFrame, px: pd.DataFrame, ticker: Dict[str, str]) -> pd.DataFrame:
    """One row per (stock, entry row): entry = first trading day after the broadcast date."""
    days = px.index
    sel = sel.assign(yahoo=sel["symbol"].map(ticker))
    unmapped = int(sel["yahoo"].isna().sum())
    sel = sel.dropna(subset=["yahoo"])
    sel = sel[sel["yahoo"].isin(px.columns)]
    a = days.searchsorted(sel["broadcast"].dt.normalize() + pd.Timedelta(days=1), side="left")
    ev = pd.DataFrame({"yahoo": sel["yahoo"].to_numpy(), "a": a, "value": sel["value"].to_numpy()})
    ev = ev[ev["a"] < len(days)]
    ev = ev.groupby(["yahoo", "a"], as_index=False)["value"].sum()
    ev.attrs["unmapped"] = unmapped
    return ev


# ===========================================================================
# STUDY
# ===========================================================================

class Returns:
    def __init__(self, px: pd.DataFrame, vol: pd.DataFrame) -> None:
        self.px = px
        self.value = px.drop(columns=[INDEX_SYMBOL]) * vol
        self.filled = px.ffill()
        self._elig: Dict[int, List[str]] = {}

    def eligible(self, a: int) -> List[str]:
        if a not in self._elig:
            self._elig[a] = eligible_at(self.px, self.value, a)
        return self._elig[a]

    def window(self, a: int, h: int, stocks) -> pd.Series:
        """Return from the close of row a to the close of row a + h (last price if it stopped)."""
        return self.filled.iloc[a + h][stocks] / self.px.iloc[a][stocks] - 1.0


def measure(ev: pd.DataFrame, R: Returns, lo: str, hi: str, h: int) -> pd.DataFrame:
    days = R.px.index
    out = []
    for e in ev.itertuples():
        if e.a < 253 or e.a + h >= len(days):
            continue
        d = days[e.a]
        if not (pd.Timestamp(lo) <= d <= pd.Timestamp(hi)):
            continue
        elig = R.eligible(e.a)
        if e.yahoo not in elig:
            continue
        rs = R.window(e.a, h, [e.yahoo]).iloc[0]
        bench = R.window(e.a, h, elig).mean()
        if not np.isfinite(rs) or not np.isfinite(bench):
            continue
        out.append({"stock": e.yahoo, "entry": d.date(), "month": str(d.to_period("M")),
                    "value": e.value, "excess": rs - bench - COST})
    return pd.DataFrame(out)


def summarise(m: pd.DataFrame) -> Dict[str, float]:
    if not len(m):
        return {"events": 0, "mean_excess": float("nan"), "month_t": 0.0, "months": 0, "hit_rate": float("nan")}
    mm = m.groupby("month")["excess"].mean()
    sd = mm.std(ddof=1)
    return {"events": int(len(m)), "stocks": int(m["stock"].nunique()), "months": int(len(mm)),
            "mean_excess": float(m["excess"].mean()), "median_excess": float(m["excess"].median()),
            "month_t": float(mm.mean() / sd * math.sqrt(len(mm))) if len(mm) > 1 and sd > 0 else 0.0,
            "hit_rate": float((m["excess"] > 0).mean())}


def qualifies(s: Dict) -> bool:
    return s["events"] >= MIN_EVENTS and s["mean_excess"] >= MIN_EXCESS and s["month_t"] >= MIN_T


def row(name: str, s: Dict) -> str:
    if not s["events"]:
        return f"| {name} | 0 | - | - | - | - |"
    return (f"| {name} | {s['events']:,} | {100 * s['mean_excess']:+.2f}% | {100 * s['median_excess']:+.2f}% | "
            f"{s['month_t']:+.2f} | {100 * s['hit_rate']:.1f}% |")


def period_block(d: pd.DataFrame, px: pd.DataFrame, R: Returns, ticker: Dict[str, str], lo: str, hi: str,
                 title: str) -> (Dict, List[str]):
    buys = to_events(select(d, "buy"), px, ticker)
    main = summarise(measure(buys, R, lo, hi, HOLD))
    L = [f"## {title}", "", f"Disclosures that could not be mapped to a current ticker: {buys.attrs['unmapped']:,}.", "",
         "| Events | N | Mean net excess | Median | Month-clustered t | Hit rate |", "|---|---|---|---|---|---|",
         row(f"**Primary: insider buys, {HOLD} days**", main)]
    extra = {
        "insider buys, 20 days": summarise(measure(buys, R, lo, hi, 20)),
        "insider buys, 120 days": summarise(measure(buys, R, lo, hi, 120)),
        "promoters / promoter group only": summarise(measure(
            to_events(select(d, "buy", PROMOTERS), px, ticker), R, lo, hi, HOLD)),
        "directors / key managers only": summarise(measure(
            to_events(select(d, "buy", ("Director", "Key Managerial Personnel")), px, ticker), R, lo, hi, HOLD)),
        "buys of Rs. 1 crore or more": summarise(measure(
            to_events(select(d, "buy", INSIDERS, BIG_VALUE), px, ticker), R, lo, hi, HOLD)),
        "insider SALES (negative = signal)": summarise(measure(
            to_events(select(d, "sell"), px, ticker), R, lo, hi, HOLD)),
    }
    L += [row(k, v) for k, v in extra.items()]
    return {"primary": main, **extra}, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--reveal-test", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    d = load_disclosures()
    uni = load_universe()
    ticker = dict(zip(uni["symbol"].str.replace("-EQ", "", regex=False), uni["yahoo"]))
    px, vol = load_panels()
    R = Returns(px, vol)
    print(f"disclosures: {len(d):,} ({d['broadcast'].min():%Y-%m} .. {d['broadcast'].max():%Y-%m})", flush=True)
    dev, L_dev = period_block(d, px, R, ticker, *DEV, "Development (entries 2016-01 to 2020-12)")
    ok = qualifies(dev["primary"])
    L = ["# INS1: insider open-market purchases (SEBI PIT disclosures)", "",
         "Implements reports/preregistration_insider_buys.md. Net excess = stock return from the close after the "
         f"disclosure date over {HOLD} trading days, minus the equal-weight eligible universe, minus 0.50% cost.", ""]
    L += L_dev
    L += ["", f"Rule: mean net excess >= +{100 * MIN_EXCESS:.1f}%, month-clustered t >= {MIN_T:.1f}, "
          f">= {MIN_EVENTS} events.", "",
          f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**", ""]
    out = {"development": dev, "qualifies": ok, "test_revealed": bool(a.reveal_test and ok)}
    if a.reveal_test and ok:
        test, L_test = period_block(d, px, R, ticker, *TEST, "Test (entries 2021-01 onward, revealed once)")
        passed = test["primary"]["events"] > 0 and test["primary"]["mean_excess"] > 0
        L += L_test + ["", f"**Test: {'passed; start a shadow forward test' if passed else 'failed'}.**", ""]
        out["test"], out["test_passed"] = test, passed
    with open(os.path.join(REPORT_DIR, "insider_buys.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "insider_buys.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""
fundamentals.py
===============
Point-in-time quarterly fundamentals from NSE's own filings, for the liquid universe.

    D:\\trader_env\\Scripts\\python.exe fundamentals.py            both stages (resumable; scheduled nightly)
    D:\\trader_env\\Scripts\\python.exe fundamentals.py --index    stage 1 only
    D:\\trader_env\\Scripts\\python.exe fundamentals.py --report   coverage summary

Stage 1, the list of results filings for each stock (NSE corporates-financial-results API,
one request per stock):
- written to data/fundamentals/index/<SYMBOL>.json;
- refreshed when older than 20 days.

Stage 2, the figures from each filing's XBRL file (nsearchives):
- one filing per stock and quarter: the consolidated one if the company files it, else the
  standalone one, and the FIRST broadcast (what the market saw first, not later
  revisions);
- quarters ending 2015-04 onward;
- parsed into data/fundamentals/quarterly.parquet, one row per stock and quarter, with
  broadcast (the time the market learned it), period_from and period_to, consolidated,
  revenue, other_income, expenses, pbt, net_profit, profit_owners, paid_up_capital,
  face_value, eps_basic and the xbrl URL.

Banks file a different form, so some of their fields stay empty.

Stocks: every ticker in data/industries.csv, i.e. every stock that was liquid at some month
end from 2016 on. Network problems (the school Wi-Fi, the night shutdown) pause the run; they
are not counted as failures. Nothing here trades; it is the raw material for pre-registered
earnings studies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from titanq_config import DATA_DIR, PROJECT_DIR

log = logging.getLogger("fundamentals")

FUND_DIR = os.path.join(DATA_DIR, "fundamentals")
INDEX_DIR = os.path.join(FUND_DIR, "index")
QUARTERLY = os.path.join(FUND_DIR, "quarterly.parquet")
FAILED = os.path.join(FUND_DIR, "xbrl_failed.json")
FIRST_QUARTER_END = "2015-04-01"
INDEX_MAX_AGE_DAYS = 20
XBRL_GAP = 0.6
SAVE_EVERY = 200
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124 Safari/537.36"}
FIELDS = {
    "revenue": "RevenueFromOperations", "other_income": "OtherIncome", "expenses": "Expenses",
    "pbt": "ProfitBeforeTax", "net_profit": "ProfitLossForPeriod",
    "profit_owners": "ProfitOrLossAttributableToOwnersOfParent",
    "paid_up_capital": "PaidUpValueOfEquityShareCapital", "face_value": "FaceValueOfEquityShareCapital",
    "eps_basic": "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "eps_basic_cont": "BasicEarningsLossPerShareFromContinuingOperations",
}


def symbols() -> List[str]:
    from universe import load_universe
    lab = pd.read_csv(os.path.join(DATA_DIR, "industries.csv"), keep_default_na=False)
    uni = load_universe()
    to_nse = dict(zip(uni["yahoo"], uni["symbol"].str.replace("-EQ", "", regex=False)))
    return sorted({to_nse[t] for t in lab["ticker"] if t in to_nse})


def network_ok() -> bool:
    try:
        requests.get("https://nsearchives.nseindia.com/", headers=UA, timeout=15)
        return True
    except requests.RequestException:
        return False


def wait_for_network() -> None:
    while not network_ok():
        log.info("no clean connection to NSE (VPN or night shutdown); waiting 5 min")
        time.sleep(300)


def build_index() -> None:
    import disclosure_radar
    from disclosure_radar import NSE, _rows
    disclosure_radar.GAP_SECONDS = 3.0
    os.makedirs(INDEX_DIR, exist_ok=True)
    syms = symbols()
    todo = [s for s in syms if not os.path.exists(os.path.join(INDEX_DIR, f"{s}.json")) or
            time.time() - os.path.getmtime(os.path.join(INDEX_DIR, f"{s}.json")) > INDEX_MAX_AGE_DAYS * 86400]
    log.info("index: %d stocks, %d to fetch", len(syms), len(todo))
    nse = None
    for i, s in enumerate(todo):
        rows = None
        for attempt in range(3):
            try:
                nse = nse or NSE()
                rows = nse.json("https://www.nseindia.com/api/corporates-financial-results?index=equities"
                                f"&symbol={requests.utils.quote(s)}&period=Quarterly")
            except Exception as exc:
                log.warning("%s: %s", s, type(exc).__name__)
                rows = None
            if rows is not None:
                break
            nse = None
            wait_for_network()
        if rows is None:
            continue
        keep = [{k: r.get(k) for k in ("symbol", "broadCastDate", "filingDate", "fromDate", "toDate", "consolidated",
                                       "audited", "bank", "relatingTo", "xbrl", "seqNumber")} for r in _rows(rows)]
        path = os.path.join(INDEX_DIR, f"{s}.json")
        with open(path + ".partial", "w", encoding="ascii") as f:
            json.dump(keep, f)
        os.replace(path + ".partial", path)
        if (i + 1) % 50 == 0:
            log.info("index: %d/%d", i + 1, len(todo))
    log.info("index done")


def wanted_filings() -> pd.DataFrame:
    """One filing per stock and quarter: consolidated if filed, else standalone; the first broadcast."""
    parts = []
    for f in os.listdir(INDEX_DIR):
        if f.endswith(".json"):
            with open(os.path.join(INDEX_DIR, f), encoding="ascii") as fh:
                parts.extend(json.load(fh))
    df = pd.DataFrame(parts)
    if not len(df):
        return df
    df = df[df["xbrl"].fillna("").str.endswith(".xml")].copy()
    df["to"] = pd.to_datetime(df["toDate"], format="%d-%b-%Y", errors="coerce")
    df["bc"] = pd.to_datetime(df["broadCastDate"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["to", "bc"])
    df = df[df["to"] >= FIRST_QUARTER_END]
    df["is_cons"] = df["consolidated"].fillna("").str.startswith("Consolidated")
    df = df.sort_values(["symbol", "to", "is_cons", "bc"], ascending=[True, True, False, True])
    return df.drop_duplicates(["symbol", "to"], keep="first").reset_index(drop=True)


def parse_xbrl(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for name, tag in FIELDS.items():
        m = re.search(rf"<in-bse-fin:{tag}\b[^>]*contextRef=\"OneD\"[^>]*>([^<]+)<", text)
        try:
            out[name] = float(m.group(1)) if m else None
        except ValueError:
            out[name] = None
    m = re.search(r"<in-bse-fin:NatureOfReportStandaloneConsolidated\b[^>]*>([^<]+)<", text)
    out["nature"] = m.group(1).strip() if m else None
    return out


def save(rows: List[Dict]) -> None:
    old = pd.read_parquet(QUARTERLY) if os.path.exists(QUARTERLY) else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
    out = out.drop_duplicates(["symbol", "period_to"], keep="last")
    out.to_parquet(QUARTERLY + ".partial", index=False)
    os.replace(QUARTERLY + ".partial", QUARTERLY)


def fetch_xbrl() -> None:
    want = wanted_filings()
    done = set(pd.read_parquet(QUARTERLY)["xbrl"]) if os.path.exists(QUARTERLY) else set()
    failed: Dict[str, int] = {}
    if os.path.exists(FAILED):
        with open(FAILED, encoding="ascii") as f:
            failed = json.load(f)
    todo = want[~want["xbrl"].isin(done) & want["xbrl"].map(lambda u: failed.get(u, 0) < 3)]
    log.info("xbrl: %d filings wanted, %d to fetch", len(want), len(todo))
    rows: List[Dict] = []
    s = requests.Session()
    for i, r in enumerate(todo.itertuples(index=False)):
        text = None
        while text is None:
            try:
                resp = s.get(r.xbrl, headers=UA, timeout=30)
                if resp.status_code == 200:
                    text = resp.text
                else:
                    failed[r.xbrl] = failed.get(r.xbrl, 0) + 1
                    break
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if rows:
                    save(rows)
                    rows = []
                wait_for_network()
        time.sleep(XBRL_GAP)
        if text is None:
            continue
        rows.append({"symbol": r.symbol, "broadcast": r.bc, "period_from": pd.to_datetime(r.fromDate, format="%d-%b-%Y",
                                                                                         errors="coerce"),
                     "period_to": r.to, "consolidated": bool(r.is_cons), "audited": r.audited, "bank": r.bank,
                     "xbrl": r.xbrl, **parse_xbrl(text)})
        if len(rows) >= SAVE_EVERY:
            save(rows)
            rows = []
            with open(FAILED, "w", encoding="ascii") as f:
                json.dump(failed, f)
            log.info("xbrl: %d/%d", i + 1, len(todo))
    if rows:
        save(rows)
    with open(FAILED, "w", encoding="ascii") as f:
        json.dump(failed, f)
    log.info("xbrl done")


def report() -> None:
    q = pd.read_parquet(QUARTERLY) if os.path.exists(QUARTERLY) else pd.DataFrame()
    n_idx = len([f for f in os.listdir(INDEX_DIR) if f.endswith(".json")]) if os.path.isdir(INDEX_DIR) else 0
    print(f"index files: {n_idx}; quarterly rows: {len(q):,}")
    if len(q):
        print(f"stocks: {q['symbol'].nunique()}; quarters {q['period_to'].min().date()} to {q['period_to'].max().date()}")
        print("share with revenue:", round(q["revenue"].notna().mean(), 3), "| with net profit:",
              round(q["net_profit"].notna().mean(), 3), "| consolidated:", round(q["consolidated"].mean(), 3))


def main() -> None:
    from procs import other_instance_running
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    if other_instance_running("fundamentals.py"):
        log.info("another fundamentals.py is running; exiting")
        return
    os.makedirs(FUND_DIR, exist_ok=True)
    wait_for_network()
    build_index()
    if not a.index:
        fetch_xbrl()
    report()


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(PROJECT_DIR, "logs", "fundamentals.log"), encoding="utf-8")], force=True)
    for noisy in ("radar", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    main()

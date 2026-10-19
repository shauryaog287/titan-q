"""
ipo_lockin_research.py
======================
Implements reports/preregistration_ipo1_lockin_expiry.md (IPO1): do new listings
underperform around their anchor investors' 30-day lock-in expiry?

    D:\\trader_env\\Scripts\\python.exe ipo_lockin_research.py --fetch   NSE past issues per year (2016 onward)
    D:\\trader_env\\Scripts\\python.exe ipo_lockin_research.py           development only

There is deliberately no --reveal-test path yet: it is added only if the development
decision qualifies and has been committed. Output: reports/ipo_lockin.md and .json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR

IPO_FILE = os.path.join(DATA_DIR, "ipo", "past_issues.parquet")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2017-01-01", "2021-12-31")
BEFORE, AFTER = 3, 10
MIN_EVENTS, MAX_MEAN, MAX_T = 100, -0.010, -2.0
COLS = ("symbol", "company", "companyName", "securityType", "ipoStartDate", "ipoEndDate", "listingDate", "issuePrice")


def fetch() -> None:
    import disclosure_radar
    from disclosure_radar import NSE, _rows
    disclosure_radar.GAP_SECONDS = 3.0
    nse, rows = NSE(), []
    for y in range(2016, date.today().year + 1):
        r = nse.json("https://www.nseindia.com/api/public-past-issues?index=equities"
                     f"&from_date=01-01-{y}&to_date=31-12-{y}")
        got = _rows(r) if r is not None else []
        rows += [{k: (str(x.get(k)).strip() if x.get(k) is not None else None) for k in COLS} for x in got]
        print(f"{y}: {len(got)} issues", flush=True)
    df = pd.DataFrame(rows, columns=COLS).drop_duplicates(["symbol", "listingDate"])
    os.makedirs(os.path.dirname(IPO_FILE), exist_ok=True)
    df.to_parquet(IPO_FILE, index=False)
    print(f"saved {len(df)} issues ({(df['securityType'] == 'EQ').sum()} mainboard)", flush=True)


def mainboard() -> pd.DataFrame:
    df = pd.read_parquet(IPO_FILE)
    df = df[df["securityType"] == "EQ"].copy()
    df["listing"] = pd.to_datetime(df["listingDate"], format="%d-%b-%Y", errors="coerce")
    df["close"] = pd.to_datetime(df["ipoEndDate"], format="%d-%b-%Y", errors="coerce")
    df["issue_price"] = pd.to_numeric(df["issuePrice"].str.replace(",", "", regex=False), errors="coerce")
    return df.dropna(subset=["listing"]).drop_duplicates("symbol")


def unlock_events(ipos: pd.DataFrame, tools, kind: str) -> pd.DataFrame:
    rows = []
    for r in ipos.itertuples(index=False):
        y = tools.yahoo.get(r.symbol)
        if y is None:
            continue
        if kind == "anchor30":
            u = r.listing + pd.Timedelta(days=29)
        elif kind == "anchor90":
            if pd.isna(r.close) or r.close < pd.Timestamp("2022-04-01"):
                continue
            u = r.listing + pd.Timedelta(days=89)
        else:                                                         # pre-IPO shareholders
            months = 6 if (pd.notna(r.close) and r.close >= pd.Timestamp("2021-08-13")) else 12
            u = r.listing + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        first = tools.row_on_or_after(r.listing)
        ur = tools.row_on_or_after(u)
        entry, exit_ = ur - BEFORE, ur + AFTER
        if first >= len(tools.days) or exit_ >= len(tools.days) or entry <= first:
            continue
        if not tools.new_listing_liquid(first, entry, y):
            continue
        j = tools.col[y]
        px = tools.P[entry, j]
        rows.append({"symbol": r.symbol, "yahoo": y, "unlock": tools.days[ur], "kind": kind,
                     "ex": tools.excess(entry, exit_, y), "pre": tools.excess(entry, ur - 1, y),
                     "post": tools.excess(ur - 1, exit_, y),
                     "above_issue": bool(np.isfinite(px) and pd.notna(r.issue_price) and px > r.issue_price)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    from event_tools import HEAD, Panels, line, summary
    tools = Panels()
    ipos = mainboard()

    def dev(ev):
        if not len(ev):
            return ev
        return ev[(ev["unlock"] >= DEV[0]) & (ev["unlock"] <= DEV[1])]
    a30 = dev(unlock_events(ipos, tools, "anchor30"))
    a90 = dev(unlock_events(ipos, tools, "anchor90"))
    pre = dev(unlock_events(ipos, tools, "preipo"))
    s = summary(a30["ex"], a30["unlock"])
    ok = s.get("events", 0) >= MIN_EVENTS and s["mean"] <= MAX_MEAN and s["t"] <= MAX_T

    def sm(ev, col="ex", mask=None):
        if not len(ev):
            return {"events": 0}
        e = ev if mask is None else ev[mask]
        return summary(e[col], e["unlock"])
    L = ["# IPO1: anchor lock-in expiry (development, unlocks 2017-2021)", "",
         "Implements reports/preregistration_ipo1_lockin_expiry.md. Unlock U = listing date + 29 days (next "
         "trading day if needed); window from the close of U-3 to the close of U+10 trading days; excess vs the "
         f"equal-weight liquid universe; no cost. {len(ipos)} mainboard IPOs on file.", "",
         "## Primary", ""] + HEAD + [line("**Anchor 30-day unlock, U-3 to U+10**", s), "",
         f"Rule: >= {MIN_EVENTS} events, mean <= {100 * MAX_MEAN:+.1f}%, month t <= {MAX_T:+.1f}.", "",
         f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**",
         "", "## Also reported (not part of the decision)", ""] + HEAD + [
         line("Anchor 30: run-up part (U-3 to U-1)", sm(a30, "pre")),
         line("Anchor 30: after part (U-1 to U+10)", sm(a30, "post")),
         line("Anchor 30: above issue price at entry", sm(a30, "ex", a30["above_issue"] if len(a30) else None)),
         line("Anchor 30: at or below issue price", sm(a30, "ex", ~a30["above_issue"] if len(a30) else None)),
         line("Anchor 90-day unlock (issues from 2022-04)", sm(a90)),
         line("Pre-IPO shareholder unlock (6 or 12 months)", sm(pre)), ""]
    with open(os.path.join(REPORT_DIR, "ipo_lockin.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "ipo_lockin.json"), "w", encoding="ascii") as f:
        json.dump({"primary": s, "qualifies": ok}, f, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()

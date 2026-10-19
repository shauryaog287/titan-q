"""
runup_watch.py
==============
The forward test EAP2 (reports/preregistration_eap2_small_runup_forward.md): the small-stock
pre-results run-up, judged only on results meetings from 2026-10-20 onward. OBSERVE ONLY:
nothing here trades.

    D:\\trader_env\\Scripts\\python.exe runup_watch.py       update notices, log open windows, write the report

Scheduled as TitanQ_Runup_Watch (18:00 daily). Every run:
1. refreshes NSE board-meeting notices (earnings_premium.fetch: the last 45 days and the
   next 30);
2. writes data/board_meetings/open_windows_<date>.json, listing the small-third stocks whose
   pre-results window is open (results meeting within the next 5 trading days, date already
   public);
3. writes reports/runup_watch.md with the running forward results. They are information
   only until the pre-registered evaluation date (2027-04-30, or the first month end with
   300 trades, by 2027-10-31).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR

START = "2026-10-20"
EVAL_DATE, LAST_EVAL, MIN_TRADES = "2027-04-30", "2027-10-31", 300
OPEN_DIR = os.path.join(DATA_DIR, "board_meetings")
REPORT = os.path.join(PROJECT_DIR, "reports", "runup_watch.md")


def trading_days_ahead(n: int):
    from market_data import Session
    s, d, out = Session(), datetime.now().date(), []
    while len(out) < n:
        d += timedelta(days=1)
        if s.is_trading_day(d):
            out.append(d)
    return out


def open_windows(tools) -> list:
    """Small-third stocks with a results meeting in the next 5 trading days, date already public."""
    from earnings_premium import results_meetings
    ahead = trading_days_ahead(5)
    bm = results_meetings(str(datetime.now().date() + timedelta(days=1)), str(ahead[-1]))
    bm = bm[bm["announced"] <= pd.Timestamp.now()]
    last = len(tools.days) - 1
    out = []
    for r in bm.sort_values("meeting").itertuples(index=False):
        y = tools.yahoo.get(r.bm_symbol)
        if y is None or not tools.liquid(last, y):
            continue
        size = tools.size_third(last, y)
        if size != "small":
            continue
        before = [d for d in ahead if pd.Timestamp(d) < r.meeting]
        if not before:                     # no trading day left before the meeting: window closed
            continue
        out.append({"symbol": r.bm_symbol, "meeting": str(r.meeting.date()), "announced": str(r.announced),
                    "planned_exit": str(before[-1])})
    return out


def main() -> None:
    from earnings_premium import COST, fetch, results_meetings, trades
    from event_tools import HEAD, Panels, line, summary
    try:
        fetch()
    except Exception as exc:                                  # network: use the notices on file
        print(f"board-meeting refresh failed: {type(exc).__name__}", flush=True)
    tools = Panels()
    wins = open_windows(tools)
    os.makedirs(OPEN_DIR, exist_ok=True)
    with open(os.path.join(OPEN_DIR, f"open_windows_{datetime.now().date()}.json"), "w", encoding="ascii") as f:
        json.dump(wins, f, indent=1)
    bm = results_meetings(START, str(datetime.now().date() + timedelta(days=60)))
    ev = trades(bm, tools)
    if len(ev):
        ev = ev[ev["pre"].notna()].copy()
        ev["pre_net"] = ev["pre"] - COST
    rows = {sz: (summary(ev.loc[ev["size"] == sz, "pre_net"], ev.loc[ev["size"] == sz, "entry_date"], freq="W")
                 if len(ev) else {"events": 0}) for sz in ("small", "mid", "large")}
    n_small = rows["small"].get("events", 0)
    today = str(datetime.now().date())
    due = (today >= EVAL_DATE and n_small >= MIN_TRADES) or today >= LAST_EVAL
    L = ["# EAP2 forward watch: small-stock pre-results run-up (OBSERVE ONLY)", "",
         f"Updated {datetime.now():%Y-%m-%d %H:%M}. Rules: reports/preregistration_eap2_small_runup_forward.md. "
         f"Meetings from {START} only; entry up to 5 trading days before a results meeting once its date is public, "
         "exit at the close of the last trading day before it; net of 0.50% cost; t clustered by week.", "",
         f"**Status:** {'EVALUATION DUE: run the pre-registered decision' if due else f'collecting; no decision before {EVAL_DATE} (needs {MIN_TRADES} small-stock trades)'}. "
         f"Small-stock trades completed so far: {n_small}.", ""] + [h.replace("Month t", "Week t") for h in HEAD] + [
         line("**Small third (the tested group)**", rows["small"]),
         line("Mid third (comparison)", rows["mid"]),
         line("Large third (comparison)", rows["large"]), "",
         f"Pre-results windows open now in small stocks: {len(wins)}"
         + (": " + ", ".join(f"{w['symbol']} (meeting {w['meeting']})" for w in wins[:15]) if wins else "") + ".", ""]
    with open(REPORT, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()

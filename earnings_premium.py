"""
earnings_premium.py
===================
Implements reports/preregistration_eap1_earnings_premium.md (EAP1): buy up to 5 trading days
before a results board meeting (once its date is public), sell at the close of the first
trading day after it. Does that beat the equal-weight market, net of costs?

    D:\\trader_env\\Scripts\\python.exe earnings_premium.py --fetch   NSE board meetings 2017 onward (resumable)
    D:\\trader_env\\Scripts\\python.exe earnings_premium.py           development only

There is deliberately no --reveal-test path yet: it is added only if the development
decision qualifies and has been committed. Output: reports/earnings_premium.md and .json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR

BM_DIR = os.path.join(DATA_DIR, "board_meetings")
BM_FILE = os.path.join(BM_DIR, "board_meetings.parquet")
DONE_FILE = os.path.join(BM_DIR, "windows_done.json")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2017-01-01", "2020-12-31")
LEAD, COST, MERGE_DAYS = 5, 0.005, 20
MIN_TRADES, MIN_MEAN, MIN_T = 500, 0.0030, 2.0
KEEP = ("bm_symbol", "bm_date", "bm_purpose", "bm_desc", "bm_timestamp")


def fetch(start: str = "2017-01-01") -> None:
    """Board meetings by meeting date, 7-day windows from `start` to 30 days ahead."""
    import disclosure_radar
    from disclosure_radar import NSE, _rows
    disclosure_radar.GAP_SECONDS = 3.0
    os.makedirs(BM_DIR, exist_ok=True)
    done = set(json.load(open(DONE_FILE, encoding="ascii"))) if os.path.exists(DONE_FILE) else set()
    end = date.today() + timedelta(days=30)
    wins, a = [], date.fromisoformat(start)
    while a <= end:
        b = min(a + timedelta(days=6), end)
        wins.append((a, b))
        a = b + timedelta(days=1)
    recent = date.today() - timedelta(days=45)            # recent windows are refetched (new notices)
    todo = [(a, b) for a, b in wins if str(a) not in done or b >= recent]
    print(f"{len(wins)} windows, {len(todo)} to fetch", flush=True)
    old = pd.read_parquet(BM_FILE) if os.path.exists(BM_FILE) else pd.DataFrame(columns=KEEP)
    rows, nse = [], None
    for i, (a, b) in enumerate(todo):
        r = None
        for attempt in range(3):
            try:
                nse = nse or NSE()
                r = nse.json("https://www.nseindia.com/api/corporate-board-meetings?index=equities"
                             f"&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}")
            except Exception:
                r = None
            if r is not None:
                break
            nse = None
        if r is None:
            continue
        rows += [{k: (str(x.get(k)) if x.get(k) is not None else None) for k in KEEP} for x in _rows(r)]
        done.add(str(a))
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            df = pd.concat([old, pd.DataFrame(rows, columns=KEEP)], ignore_index=True).drop_duplicates()
            df.to_parquet(BM_FILE + ".partial", index=False)
            os.replace(BM_FILE + ".partial", BM_FILE)
            with open(DONE_FILE, "w", encoding="ascii") as f:
                json.dump(sorted(done), f)
            print(f"  {i + 1}/{len(todo)} ({len(df):,} notices)", flush=True)


def results_meetings(lo: str, hi: str) -> pd.DataFrame:
    bm = pd.read_parquet(BM_FILE)
    text = (bm["bm_purpose"].fillna("") + " " + bm["bm_desc"].fillna("")).str.lower()
    bm = bm[text.str.contains("result")].copy()
    bm["meeting"] = pd.to_datetime(bm["bm_date"], format="%d-%b-%Y", errors="coerce")
    bm["announced"] = pd.to_datetime(bm["bm_timestamp"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    bm = bm.dropna(subset=["meeting", "announced"])
    bm = bm[(bm["meeting"] >= lo) & (bm["meeting"] <= hi) & (bm["announced"] < bm["meeting"] + pd.Timedelta(days=1))]
    # the earliest announcement of each (stock, meeting date)
    return bm.sort_values("announced").drop_duplicates(["bm_symbol", "meeting"], keep="first")


def trades(bm: pd.DataFrame, tools) -> pd.DataFrame:
    rows, last = [], {}
    for r in bm.sort_values("meeting").itertuples(index=False):
        y = tools.yahoo.get(r.bm_symbol)
        if y is None:
            continue
        m_row = tools.row_on_or_after(r.meeting)            # meeting day, or the next trading day
        meet_is_trading = m_row < len(tools.days) and tools.days[m_row] == r.meeting
        exit_row = m_row + 1 if meet_is_trading else m_row  # first trading day AFTER the meeting date
        entry = max(tools.known_row(r.announced), m_row - LEAD)
        if not (entry <= m_row - 1 and exit_row < len(tools.days)) or not tools.liquid(entry, y):
            continue
        if entry - last.get(y, -10 ** 9) < MERGE_DAYS:
            continue
        last[y] = entry
        gross = tools.excess(entry, exit_row, y)
        rows.append({"symbol": r.bm_symbol, "yahoo": y, "meeting": r.meeting, "entry_date": tools.days[entry],
                     "gross": gross, "net": gross - COST,
                     "pre": tools.excess(entry, m_row - 1, y) if m_row - 1 > entry else float("nan"),
                     "ann": tools.excess(m_row - 1, exit_row, y),
                     "lead_days": (r.meeting - r.announced).days, "size": tools.size_third(entry, y)})
    return pd.DataFrame(rows)


def eap1b_confirmation() -> None:
    """reports/preregistration_eap1b_pre_results_runup.md: one run on meetings 2021-01 to 2026-08."""
    from event_tools import HEAD, Panels, line, summary
    tools = Panels()
    bm = results_meetings("2021-01-01", "2026-08-31")
    ev = trades(bm, tools)
    ev = ev[ev["pre"].notna()].copy()
    ev["pre_net"] = ev["pre"] - COST
    s = summary(ev["pre_net"], ev["entry_date"])
    ok = s.get("events", 0) > 1 and s["mean"] > 0 and s["t"] >= 2.0

    def sub(mask):
        return summary(ev.loc[mask, "pre_net"], ev.loc[mask, "entry_date"])
    L = ["# EAP1b: the pre-results run-up, confirmation on meetings 2021-01 to 2026-08 (run once)", "",
         "Implements reports/preregistration_eap1b_pre_results_runup.md. Entry as EAP1 (the later of the day the "
         "meeting date became public and 5 trading days before it); exit at the close of the last trading day "
         "before the meeting. Result = stock minus the equal-weight liquid universe, minus 0.50% cost. "
         f"{len(bm):,} results meetings announced in advance; {len(ev):,} trades in liquid stocks.", ""] + HEAD + [
         line("**Net result per trade (decisive)**", s), "",
         "Rule: mean net > 0 and month t >= 2.0.", "",
         f"**Decision: {'confirmed; observe-mode list, then a pre-registered shadow paper strategy' if ok else 'not confirmed; nothing changes'}.**",
         "", "## Also reported (not part of the decision)", ""] + HEAD + [
         line("Gross (no cost)", summary(ev["pre"], ev["entry_date"])),
         line("Small third (net)", sub(ev["size"] == "small")),
         line("Mid third (net)", sub(ev["size"] == "mid")),
         line("Large third (net)", sub(ev["size"] == "large")),
         line("Date announced > 7 days ahead (net)", sub(ev["lead_days"] > 7)),
         line("Date announced <= 7 days ahead (net)", sub(ev["lead_days"] <= 7)), ""]
    with open(os.path.join(REPORT_DIR, "pre_results_runup.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "pre_results_runup.json"), "w", encoding="ascii") as f:
        json.dump({"confirmation": s, "confirmed": ok, "trades": int(len(ev))}, f, indent=2)
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--eap1b-confirmation", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    if a.eap1b_confirmation:
        eap1b_confirmation()
        return
    from event_tools import HEAD, Panels, line, summary
    tools = Panels()
    bm = results_meetings(*DEV)
    ev = trades(bm, tools)
    s = summary(ev["net"], ev["entry_date"])
    ok = s.get("events", 0) >= MIN_TRADES and s["mean"] >= MIN_MEAN and s["t"] >= MIN_T

    def sub(mask):
        return summary(ev.loc[mask, "net"], ev.loc[mask, "entry_date"])
    L = ["# EAP1: the earnings announcement premium (development, meetings 2017-2020)", "",
         "Implements reports/preregistration_eap1_earnings_premium.md. Entry at the close of the later of the day "
         "the meeting date became public and 5 trading days before the meeting; exit at the close of the first "
         "trading day after it. Result = stock minus the equal-weight liquid universe, minus 0.50% cost. "
         f"{len(bm):,} results meetings announced in advance; {len(ev):,} trades in liquid stocks.", "",
         "## Primary", ""] + HEAD + [line("**Net result per trade**", s), "",
         f"Rule: >= {MIN_TRADES} trades, mean net >= +{100 * MIN_MEAN:.2f}%, month t >= {MIN_T:.1f}.", "",
         f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**",
         "", "## Also reported (not part of the decision)", ""] + HEAD + [
         line("Gross (no cost)", summary(ev["gross"], ev["entry_date"])),
         line("Pre-meeting part only (gross)", summary(ev["pre"], ev["entry_date"])),
         line("Announcement part only (gross)", summary(ev["ann"], ev["entry_date"])),
         line("Small third (net)", sub(ev["size"] == "small")),
         line("Mid third (net)", sub(ev["size"] == "mid")),
         line("Large third (net)", sub(ev["size"] == "large")),
         line("Date announced > 7 days ahead (net)", sub(ev["lead_days"] > 7)),
         line("Date announced <= 7 days ahead (net)", sub(ev["lead_days"] <= 7)), ""]
    with open(os.path.join(REPORT_DIR, "earnings_premium.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "earnings_premium.json"), "w", encoding="ascii") as f:
        json.dump({"primary": s, "qualifies": ok, "trades": int(len(ev)), "meetings": int(len(bm))}, f, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()

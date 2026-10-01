"""
weekly_report.py
================
A weekly summary of everything Titan-Q did, as a markdown page and a short phone alert.

    D:\\trader_env\\Scripts\\python.exe weekly_report.py            the 7 days ending today
    D:\\trader_env\\Scripts\\python.exe weekly_report.py --no-alert write the page only

Scheduled as TitanQ_Weekly_Report, Saturdays 10:30. Output: reports/weekly/weekly_<date>.md
It covers:
- the live paper account (v3.2) against NIFTY: the week's P&L, the best and worst holdings,
  trades and fees;
- the forward test (composite top-30) against the equal-weight market;
- NJ1b no-buy flags, market movers and what was behind them, and the local AI's most
  material readings;
- system health (VPN failures, recorder sessions) and research decisions taken this week.
Each section fails on its own: a missing input skips that section, not the report.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Callable, List

import pandas as pd

from titanq_config import DATA_DIR, LOGS_DIR, PROJECT_DIR

REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "weekly")
FT_CSV = os.path.join(PROJECT_DIR, "reports", "forward_test", "composite_top30.csv")
ACCOUNT = os.path.join(PROJECT_DIR, "paper_account.json")


def rs(x: float) -> str:
    return f"Rs. {x:,.0f}"


def pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def week_rows(start: date, end: date) -> pd.DataFrame:
    ft = pd.read_csv(FT_CSV)
    ft["date"] = pd.to_datetime(ft["date"]).dt.date
    return ft


def live_section(start: date, end: date, notes: List[str]) -> List[str]:
    ft = week_rows(start, end)
    live = ft.dropna(subset=["live_v31_equity"])
    before = live[live["date"] < start]
    inside = live[(live["date"] >= start) & (live["date"] <= end)]
    if not len(inside):
        return ["## Live paper account", "", "No trading days recorded this week.", ""]
    first_eq = float(before["live_v31_equity"].iloc[-1]) if len(before) else 500_000.0
    first_nifty = float(before["nifty"].iloc[-1]) if len(before) else float(inside["nifty"].iloc[0])
    last = inside.iloc[-1]
    wk = last["live_v31_equity"] / first_eq - 1
    nf = last["nifty"] / first_nifty - 1
    notes.append(f"live {pct(wk)} vs NIFTY {pct(nf)}")
    L = ["## Live paper account (v3.2: always invested, 3% daily loss halt)", "",
         f"- Equity at {last['date']}: **{rs(last['live_v31_equity'])}** "
         f"({pct(last['live_v31_equity'] / 500_000 - 1)} since the start).",
         f"- This week: **{pct(wk)}** ({rs(last['live_v31_equity'] - first_eq)}); NIFTY {pct(nf)}."]
    with open(ACCOUNT, encoding="ascii") as f:
        acct = json.load(f)
    trades = [t for t in acct.get("trades", []) if start <= date.fromisoformat(t["time"][:10]) <= end]
    L.append(f"- Trades this week: {len(trades)} (fees {rs(sum(t.get('fees', 0) for t in trades))}); "
             f"fees since the start {rs(acct.get('fees_paid', 0))}.")
    # contribution of each holding over the week (from the later of the week's start and the buy)
    from market_data import DailyStore
    pos = {s: q for s, q in acct["positions"].items() if q}
    d = DailyStore().load(list(pos), refresh=False)
    contrib = []
    for s, q in pos.items():
        c = d.get(s)
        if c is None or not len(c):
            continue
        c = c["Close"].dropna()
        c.index = pd.to_datetime(c.index).date
        prior = c[c.index < start]
        inwk = c[(c.index >= start) & (c.index <= end)]
        if not len(inwk):
            continue
        bought = [t for t in trades if t["ticker"] == s and t["side"] == "BUY"]
        base = float(bought[0]["price"]) if bought else (float(prior.iloc[-1]) if len(prior) else None)
        if base:
            contrib.append((s, q * (float(inwk.iloc[-1]) - base), float(inwk.iloc[-1]) / base - 1))
    if contrib:
        contrib.sort(key=lambda x: x[1])
        L += ["", "| Best this week | P&L | Move |", "|---|---|---|"]
        L += [f"| {s} | {rs(p)} | {pct(m)} |" for s, p, m in reversed(contrib[-5:])]
        L += ["", "| Worst this week | P&L | Move |", "|---|---|---|"]
        L += [f"| {s} | {rs(p)} | {pct(m)} |" for s, p, m in contrib[:5]]
    halts = [p for p in glob.glob(os.path.join(LOGS_DIR, "day_open_equity_*.json"))
             if start <= date.fromisoformat(os.path.basename(p)[len("day_open_equity_"):][:10]) <= end]
    L += ["", f"- Trading days with a recorded open: {len(halts)}."]
    return L + [""]


def forward_section(start: date, end: date, notes: List[str]) -> List[str]:
    ft = week_rows(start, end)
    inside = ft[(ft["date"] >= start) & (ft["date"] <= end)]
    if not len(inside):
        return []
    before = ft[ft["date"] < start]
    e0 = float(before["equity"].iloc[-1]) if len(before) else 500_000.0
    b0 = float(before["bench_ew_level"].iloc[-1]) if len(before) else 100.0
    last = inside.iloc[-1]
    notes.append(f"forward test {pct(last['equity'] / e0 - 1)} vs market {pct(last['bench_ew_level'] / b0 - 1)}")
    return ["## Forward test (composite top-30, whole market)", "",
            f"- This week: composite **{pct(last['equity'] / e0 - 1)}**, equal-weight market "
            f"{pct(last['bench_ew_level'] / b0 - 1)}.",
            f"- Since 2026-09-28: composite {pct(last['equity'] / 500_000 - 1)}, market "
            f"{pct(last['bench_ew_level'] / 100 - 1)}.", ""]


def signals_section(start: date, end: date, notes: List[str]) -> List[str]:
    L = ["## Signals and news", ""]
    nj = os.path.join(DATA_DIR, "movers", "nj1b_avoid.json")
    if os.path.exists(nj):
        with open(nj, encoding="ascii") as f:
            jumps = json.load(f)
        wk = {s: d for s, d in jumps.items() if start <= date.fromisoformat(d) <= end}
        from titanq_config import UNIVERSE
        ours = sorted(set(wk) & set(UNIVERSE))
        L.append(f"- NJ1b no-news jumps flagged: {len(wk)}" + (f"; in our 50 stocks: {', '.join(ours)}" if ours else "") + ".")
    movers = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "movers", "movers_*.parquet"))):
        d = date.fromisoformat(os.path.basename(p)[7:17])
        if start <= d <= end:
            movers.append(pd.read_parquet(p).assign(day=d))
    if movers:
        m = pd.concat(movers, ignore_index=True).sort_values("move_pct", key=lambda s: -s.abs())
        m = m.drop_duplicates(["yahoo", "day"])
        filing = (m["filings"].fillna("") != "").sum()
        news = ((m["filings"].fillna("") == "") & (m["news"].fillna("") != "")).sum()
        L.append(f"- Market movers (5%+ on 3x volume): {len(m)}; with an NSE filing {filing}, with news only "
                 f"{news}, with neither {len(m) - filing - news}.")
        notes.append(f"{len(m)} movers")
        L += ["", "| Biggest moves | Day | Move | Why |", "|---|---|---|---|"]
        for r in m.head(8).itertuples():
            why = (r.filings or r.news or "no news found")[:90].replace("|", "/")
            L.append(f"| {r.symbol} | {r.day} | {r.move_pct:+.1f}% | {why} |")
    ev = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "events", "events_*.parquet"))):
        d = date.fromisoformat(os.path.basename(p)[7:17])
        if start <= d <= end:
            ev.append(pd.read_parquet(p))
    if ev:
        e = pd.concat(ev, ignore_index=True)
        e = e[~e["event_type"].isin(["routine", "market_commentary"]) & (e["materiality"] == "high")]
        e = e.drop_duplicates(["symbols", "event_type"])
        L += ["", f"- The local AI rated {len(e)} company events as high materiality this week:"]
        for r in e.head(10).itertuples():
            L.append(f"  - {r.symbols}: {r.event_type}, {r.direction}: {r.summary}")
    return L + [""]


def health_section(start: date, end: date, notes: List[str]) -> List[str]:
    L = ["## System health", ""]
    vpn = os.path.join(LOGS_DIR, "vpn_switch.log")
    if os.path.exists(vpn):
        fails = conn = 0
        with open(vpn, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = date.fromisoformat(line[:10])
                except ValueError:
                    continue
                if start <= d <= end:
                    fails += ("no VPN got through" in line) or ("still not working" in line)
                    conn += (" connected" in line)
        L.append(f"- VPN: {conn} automatic (re)connections, {fails} failed rounds.")
        if fails:
            notes.append(f"{fails} VPN failures")
    sessions = []
    for p in glob.glob(os.path.join(LOGS_DIR, "angel_record_*.log")):
        d = date.fromisoformat(os.path.basename(p)[13:23])
        if start <= d <= end:
            with open(p, encoding="utf-8", errors="replace") as f:
                m = re.findall(r"session \S+: (\d+) rows", f.read())
            sessions.append((d, int(m[-1]) if m else 0))
    if sessions:
        L.append("- Price recorder: " + ", ".join(f"{d:%a %d} {n / 1e6:.1f}M rows" for d, n in sorted(sessions)) + ".")
    return L + [""]


def research_section(start: date, end: date, notes: List[str]) -> List[str]:
    L = ["## Research decisions this week", ""]
    found = 0
    for p in sorted(glob.glob(os.path.join(PROJECT_DIR, "reports", "*.md"))):
        if not (start <= date.fromtimestamp(os.path.getmtime(p)) <= end):
            continue
        with open(p, encoding="ascii", errors="replace") as f:
            text = f.read()
        dec = re.findall(r"\*\*Decision: ([^*]+)\*\*", text)
        test = re.findall(r"\*\*Test: ([^*]+)\*\*", text)
        if dec:
            title = text.splitlines()[0].lstrip("# ").strip()
            L.append(f"- {title}: development: {dec[-1].strip()}" + (f" Sealed test: {test[-1].strip()}" if test else ""))
            found += 1
    if not found:
        L.append("- None.")
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-alert", action="store_true")
    ap.add_argument("--end", default=None, help="last day of the week (YYYY-MM-DD), default today")
    a = ap.parse_args()
    end = date.fromisoformat(a.end) if a.end else date.today()
    start = end - timedelta(days=6)
    notes: List[str] = []
    L = [f"# Titan-Q weekly report, {start} to {end}", "",
         f"Written {datetime.now():%Y-%m-%d %H:%M}. Paper trading only; nothing here is a real order.", ""]
    sections: List[Callable] = [live_section, forward_section, signals_section, health_section, research_section]
    for sec in sections:
        try:
            L += sec(start, end, notes)
        except Exception as exc:
            L += [f"_{sec.__name__} skipped: {type(exc).__name__}: {str(exc)[:120]}_", ""]
    text = ("\n".join(L) + "\n").encode("ascii", "replace").decode("ascii")
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"weekly_{end}.md")
    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write(text)
    print(text)
    if not a.no_alert:
        from alerts import send
        send(f"weekly report {end}", "; ".join(notes) or "quiet week", priority="default", tags="bar_chart",
             repeat_minutes=0)


if __name__ == "__main__":
    main()

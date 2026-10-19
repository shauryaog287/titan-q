"""
fo_ban_research.py
==================
Implements reports/preregistration_fb1_fo_ban.md (FB1): do stocks entering the F&O ban
underperform over the next 10 trading days?

    D:\\trader_env\\Scripts\\python.exe fo_ban_research.py --fetch   daily ban lists 2017 onward (resumable)
    D:\\trader_env\\Scripts\\python.exe fo_ban_research.py           development only

There is deliberately no --reveal-test path yet: it is added only if the development
decision qualifies and has been committed. Output: reports/fo_ban.md and .json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import pandas as pd
import requests

from titanq_config import DATA_DIR, PROJECT_DIR

BAN_DIR = os.path.join(DATA_DIR, "fo_ban")
BAN_FILE = os.path.join(BAN_DIR, "ban_lists.json")          # {"YYYY-MM-DD": ["SYM", ...]}
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
URL = "https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{d:%d%m%Y}.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124 Safari/537.36"}
DEV = ("2017-01-01", "2020-12-31")
H, LOOKBACK = 10, 10
MIN_EVENTS, MAX_MEAN, MAX_T = 150, -0.010, -2.0


def parse(text: str):
    """Symbols in a ban-list file; [] for NIL; None if the file is not a ban list."""
    if "Securities in Ban" not in text:
        return None
    return [m.group(1).strip() for m in re.finditer(r"^\s*\d+\s*,\s*([A-Z0-9&\-]+)", text, re.M)]


def fetch() -> None:
    from whole_market_research import load_panels
    px, _ = load_panels()
    days = [d for d in px.index if d >= pd.Timestamp("2017-01-01")]
    os.makedirs(BAN_DIR, exist_ok=True)
    have = json.load(open(BAN_FILE, encoding="ascii")) if os.path.exists(BAN_FILE) else {}
    todo = [d for d in days if str(d.date()) not in have]
    print(f"{len(days)} trading days, {len(todo)} to fetch", flush=True)
    s = requests.Session()
    for i, d in enumerate(todo):
        for attempt in range(3):
            try:
                r = s.get(URL.format(d=d), headers=UA, timeout=20)
                syms = parse(r.text) if r.status_code == 200 else None
                if syms is not None:
                    have[str(d.date())] = syms
                break
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
        time.sleep(0.3)
        if (i + 1) % 200 == 0 or i + 1 == len(todo):
            with open(BAN_FILE + ".partial", "w", encoding="ascii") as f:
                json.dump(have, f)
            os.replace(BAN_FILE + ".partial", BAN_FILE)
            print(f"  {i + 1}/{len(todo)}", flush=True)
    print(f"done: {len(have)} days on file", flush=True)


def events(bans: dict, tools) -> pd.DataFrame:
    days = [d for d in sorted(bans)]
    rows = []
    for k, d in enumerate(days):
        prev = set().union(*[set(bans[x]) for x in days[max(0, k - LOOKBACK):k]]) if k else set()
        for sym in bans[d]:
            if sym in prev:
                continue
            y = tools.yahoo.get(sym)
            r = tools.row_on_or_after(d)
            if y is None or r >= len(tools.days) or str(tools.days[r].date()) != d:
                continue
            # spell length: consecutive list days from d
            n = 0
            while k + n < len(days) and sym in bans[days[k + n]]:
                n += 1
            exit_day = days[k + n] if k + n < len(days) else None
            ex_row = tools.row_on_or_after(exit_day) if exit_day else None
            rows.append({"symbol": sym, "yahoo": y, "date": d, "row": r, "spell": n,
                         "ex10": tools.excess(r, r + H, y), "ex5": tools.excess(r, r + 5, y),
                         "ex20": tools.excess(r, r + 20, y), "day0": tools.excess(r - 1, r, y),
                         "exit_ex10": tools.excess(ex_row, ex_row + H, y) if ex_row is not None else float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    from event_tools import HEAD, Panels, line, summary
    bans = json.load(open(BAN_FILE, encoding="ascii"))
    bans = {d: v for d, v in bans.items() if DEV[0] <= d <= DEV[1]}
    tools = Panels()
    ev = events(bans, tools)
    s = summary(ev["ex10"], ev["date"])
    ok = s.get("events", 0) >= MIN_EVENTS and s["mean"] <= MAX_MEAN and s["t"] <= MAX_T
    L = ["# FB1: stocks entering the F&O ban (development, 2017-2020)", "",
         "Implements reports/preregistration_fb1_fo_ban.md. Excess = the stock's return from the close of its first "
         "ban day minus the equal-weight liquid universe over the same days; no cost (an avoid flag). "
         f"{len(bans)} trading days of ban lists; an entry counts only if the stock was in none of the previous "
         f"{LOOKBACK} days' lists.", "", "## Primary", ""] + HEAD + [
        line("**Ban entry, next 10 trading days**", s), "",
        f"Rule: >= {MIN_EVENTS} entries, mean <= {100 * MAX_MEAN:+.1f}%, month t <= {MAX_T:+.1f}.", "",
        f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**",
        "", "## Also reported (not part of the decision)", ""] + HEAD + [
        line("First ban day itself (close t-1 to t, not tradeable)", summary(ev["day0"], ev["date"])),
        line("Ban entry, 5 days", summary(ev["ex5"], ev["date"])),
        line("Ban entry, 20 days", summary(ev["ex20"], ev["date"])),
        line("Ban exit, 10 days after the first day out", summary(ev["exit_ex10"], ev["date"])),
        line("Spells of 5+ days, entry 10 days", summary(ev.loc[ev["spell"] >= 5, "ex10"], ev.loc[ev["spell"] >= 5, "date"])),
        line("Spells under 5 days, entry 10 days", summary(ev.loc[ev["spell"] < 5, "ex10"], ev.loc[ev["spell"] < 5, "date"])),
        ""]
    with open(os.path.join(REPORT_DIR, "fo_ban.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "fo_ban.json"), "w", encoding="ascii") as f:
        json.dump({"primary": s, "qualifies": ok, "entries": int(len(ev))}, f, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""
disclosure_radar.py
===================
Daily collector of NSE's public disclosures, plus a red-flag watchlist.

    D:\\trader_env\\Scripts\\python.exe disclosure_radar.py                 today's snapshot + watchlist
    D:\\trader_env\\Scripts\\python.exe disclosure_radar.py --no-archive    skip the Google Drive copy

Everything comes from public NSE pages, one request every few seconds. It collects:
- the promoter pledge table (snapshot);
- the ASM and GSM surveillance lists: the exchange's own warnings about unusual
  trading or distress (GSM "IBC" = insolvency proceedings);
- the day's bulk and block deals;
- corporate announcements for today and yesterday (late filings), with the PDF
  attachments of auditor changes and of newspaper publications (the regional-language
  notices) for tradable stocks;
- FII/DII cash-market flows for the day (this builds a history going forward);
- insider-trading (PIT) disclosures of the last 7 days. The API has returned nothing
  after about 2026-04; the count is logged so it shows when that changes.

Raw files go to data/disclosures/daily/<date>/ and are copied to
G:\\My Drive\\Titan-Q_History\\disclosures\\<date>\\. The watchlist, reports/radar/<date>.md,
lists red flags for the live book, the forward-test holdings and the rest of the tradable
universe.

The flags are information only. No trading rule uses them: the one tested so far (INS2,
pledge invocations) did not qualify. Adding a flag to a strategy needs its own
pre-registered test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

from titanq_config import DATA_DIR, PROJECT_DIR, UNIVERSE
from market_data import now_ist
from universe import load_universe

log = logging.getLogger("radar")

DAILY_DIR = os.path.join(DATA_DIR, "disclosures", "daily")
RADAR_DIR = os.path.join(PROJECT_DIR, "reports", "radar")
DRIVE_DIR = r"G:\My Drive\Titan-Q_History\disclosures"
SHADOW_ACCOUNT = os.path.join(PROJECT_DIR, "paper_account_shadow.json")
GAP_SECONDS = 5.0                    # NSE closed connections from this PC on 2026-09-28 after
                                     # ~150 requests in 30 minutes; stay well below that
PLEDGE_FLAG = 0.50                   # pledged shares >= 50% of the promoters' holding
MAX_PDFS = 30                        # attachments per run, to stay polite


class NSE:
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
               "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
               "Referer": "https://www.nseindia.com/"}

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update(self.HEADERS)
        self.last = 0.0
        self.get("https://www.nseindia.com/")

    def get(self, url: str) -> Optional[requests.Response]:
        for attempt in range(3):
            time.sleep(max(0.0, self.last + GAP_SECONDS - time.time()))
            self.last = time.time()
            try:
                r = self.s.get(url, timeout=60)
                if r.status_code == 200:
                    return r
                log.warning("HTTP %s for %s", r.status_code, url)
            except requests.RequestException as exc:
                log.warning("%s for %s", type(exc).__name__, url)
            time.sleep(10.0 * (attempt + 1))
            if attempt == 1:
                try:
                    self.s.get("https://www.nseindia.com/", timeout=30)     # fresh cookies
                except requests.RequestException:
                    pass
        return None

    def json(self, url: str):
        r = self.get(url)
        try:
            return r.json() if r is not None else None
        except ValueError:
            return None


def _save_json(path: str, obj) -> None:
    with open(path + ".partial", "w", encoding="ascii") as f:
        json.dump(obj, f)
    os.replace(path + ".partial", path)


def _rows(obj) -> List[Dict]:
    if isinstance(obj, dict):
        obj = obj.get("data", [])
    return obj if isinstance(obj, list) else []


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return float("nan")


# ===========================================================================
# COLLECT
# ===========================================================================

def collect(nse: NSE, day: date, tradable: Set[str]) -> Dict[str, object]:
    out_dir = os.path.join(DAILY_DIR, str(day))
    os.makedirs(out_dir, exist_ok=True)
    got: Dict[str, object] = {}

    pledges = _rows(nse.json("https://www.nseindia.com/api/corporate-pledgedata?index=equities"))
    _save_json(os.path.join(out_dir, "pledges.json"), pledges)
    got["pledges"] = pledges

    asm = nse.json("https://www.nseindia.com/api/reportASM") or {}
    _save_json(os.path.join(out_dir, "asm.json"), asm)
    got["asm"] = [dict(r, list=k) for k in ("longterm", "shortterm") for r in _rows(asm.get(k, {}))]
    gsm = _rows(nse.json("https://www.nseindia.com/api/reportGSM"))
    _save_json(os.path.join(out_dir, "gsm.json"), gsm)
    got["gsm"] = gsm

    for name in ("bulk", "block"):
        r = nse.get(f"https://nsearchives.nseindia.com/content/equities/{name}.csv")
        if r is not None:                    # stored as ASCII (client names can carry other characters)
            with open(os.path.join(out_dir, f"{name}.csv"), "w", encoding="ascii", newline="") as f:
                f.write(r.content.decode("utf-8", "replace").encode("ascii", "replace").decode("ascii"))
        got[name] = r.text if r is not None else ""

    flows = _rows(nse.json("https://www.nseindia.com/api/fiidiiTradeReact"))
    _save_json(os.path.join(out_dir, "fii_dii.json"), flows)
    got["fii_dii"] = flows

    a, b = day - timedelta(days=7), day
    pit = _rows(nse.json("https://www.nseindia.com/api/corporates-pit?index=equities"
                         f"&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}"))
    _save_json(os.path.join(out_dir, "insider_trades_7d.json"), pit)
    got["pit"] = pit

    ann: List[Dict] = []
    for d in (day - timedelta(days=1), day):
        rows = _rows(nse.json("https://www.nseindia.com/api/corporate-announcements?index=equities"
                              f"&from_date={d:%d-%m-%Y}&to_date={d:%d-%m-%Y}"))
        _save_json(os.path.join(DAILY_DIR, str(d), "announcements.json") if d == day else
                   os.path.join(out_dir, "announcements_prev_day.json"), rows)
        ann += rows
    got["announcements"] = ann

    pdfs = [x for x in ann if x.get("symbol") in tradable and x.get("attchmntFile")
            and (is_auditor_change(x) or x.get("desc") == "Copy of Newspaper Publication")]
    pdf_dir = os.path.join(out_dir, "attachments")
    n = 0
    for x in pdfs[:MAX_PDFS]:
        url = x["attchmntFile"]
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{x['symbol']}_{os.path.basename(url)}")
        path = os.path.join(pdf_dir, name)
        if os.path.exists(path):
            continue
        r = nse.get(url)
        if r is not None:
            os.makedirs(pdf_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
            n += 1
    log.info("collected: %d pledge rows, %d ASM, %d GSM, %d announcements (%d attachments saved), "
             "%d insider-trade disclosures in 7 days, FII/DII rows %d",
             len(pledges), len(got["asm"]), len(gsm), len(ann), n, len(pit), len(flows))
    return got


def is_auditor_change(x: Dict) -> bool:
    text = f"{x.get('desc') or ''} {x.get('attchmntText') or ''}".lower()
    return "change in auditor" in text or ("auditor" in text and "resign" in text)


# ===========================================================================
# WATCHLIST
# ===========================================================================

def flags(got: Dict[str, object]):
    """(symbol -> list of red-flag descriptions, company -> pledged share of promoter holding)."""
    f: Dict[str, List[str]] = {}
    add = lambda s, msg: f.setdefault(s, []).append(msg)
    for r in got["asm"]:
        add(r.get("symbol"), f"on {r.get('survDesc') or 'ASM list'}")
    for r in got["gsm"]:
        add(r.get("symbol"), f"on GSM: {r.get('survDesc') or r.get('gsmStage')}")
    for x in got["announcements"]:
        if is_auditor_change(x):
            add(x.get("symbol"), f"auditor change announced {x.get('an_dt', '')[:11]}: "
                                 f"{(x.get('attchmntText') or '')[:140]}")
        elif x.get("desc") == "Credit Rating":
            add(x.get("symbol"), f"credit rating announcement {x.get('an_dt', '')[:11]} (check direction)")
    ticker = {}
    for p in got["pledges"]:
        held, pledged = _num(p.get("totPromoterHolding")), _num(p.get("numSharesPledged"))
        if held > 0 and pledged / held >= PLEDGE_FLAG:
            name = p.get("comName") or ""
            ticker[name] = pledged / held
    return f, ticker


def latest_flags(max_age_days: int = 7):
    """(date, symbol -> red flags) from the newest collected day, or (None, {}).
    Used by the trading loop in observe mode: the flags are logged, never traded on."""
    if not os.path.isdir(DAILY_DIR):
        return None, {}
    days = sorted(d for d in os.listdir(DAILY_DIR) if os.path.exists(os.path.join(DAILY_DIR, d, "asm.json")))
    if not days or (now_ist().date() - date.fromisoformat(days[-1])).days > max_age_days:
        return None, {}
    folder = os.path.join(DAILY_DIR, days[-1])

    def load(name, default):
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            return default
        with open(p, encoding="ascii") as fh:
            return json.load(fh)

    asm = load("asm.json", {})
    got = {"asm": [dict(r, list=k) for k in ("longterm", "shortterm") for r in _rows(asm.get(k, {}))],
           "gsm": _rows(load("gsm.json", [])),
           "announcements": _rows(load("announcements.json", [])) + _rows(load("announcements_prev_day.json", [])),
           "pledges": []}                   # pledges are keyed by company name, not symbol
    f, _ = flags(got)
    return date.fromisoformat(days[-1]), f


def write_watchlist(day: date, got: Dict[str, object], tradable: Set[str]) -> str:
    f, pledged = flags(got)
    live = {s[:-3] for s in UNIVERSE}
    try:
        with open(SHADOW_ACCOUNT, encoding="ascii") as fh:
            fwd = {s[:-3] for s, q in json.load(fh).get("positions", {}).items() if q}
    except (OSError, ValueError):
        fwd = set()
    groups = [("Live book (50 large caps)", live), ("Forward-test holdings", fwd),
              ("Other tradable stocks", tradable - live - fwd)]
    L = [f"# Disclosure radar, {day}", "",
         "Public NSE disclosures only. Information, not a trading rule: no flag has passed a "
         "pre-registered test (INS2, pledge invocations, did not).", ""]
    for title, names in groups:
        hits = sorted(s for s in f if s in names)
        L += [f"## {title}: {len(hits)} flagged", ""]
        L += [f"- **{s}**: " + "; ".join(f[s]) for s in hits] or ["- none"]
        L.append("")
    L += [f"## Promoter pledges of {int(PLEDGE_FLAG * 100)}% or more of their holding ({len(pledged)} companies)", ""]
    L += [f"- {n}: {100 * v:.0f}%" for n, v in sorted(pledged.items(), key=lambda kv: -kv[1])[:40]] or ["- none"]
    flows = got.get("fii_dii") or []
    if flows:
        L += ["", "## Institutional cash-market flows (Rs. crore)", ""]
        L += [f"- {r.get('category')}: net {r.get('netValue')} (buy {r.get('buyValue')}, sell {r.get('sellValue')}), "
              f"{r.get('date')}" for r in flows]
    L += ["", f"Insider-trade disclosures in the last 7 days: {len(got['pit'])} "
          "(the NSE API has returned none since about 2026-04).", ""]
    os.makedirs(RADAR_DIR, exist_ok=True)
    path = os.path.join(RADAR_DIR, f"{day}.md")
    text = "\n".join(L) + "\n"
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(text.encode("ascii", "replace").decode("ascii"))
    return path


def archive(day: date, report: str) -> None:
    if not os.path.isdir(os.path.dirname(os.path.dirname(DRIVE_DIR))):
        log.warning("Google Drive not available; not archived")
        return
    dst = os.path.join(DRIVE_DIR, str(day))
    shutil.copytree(os.path.join(DAILY_DIR, str(day)), dst, dirs_exist_ok=True)
    shutil.copy2(report, os.path.join(dst, "watchlist.md"))
    log.info("archived to %s", dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-archive", action="store_true")
    a = ap.parse_args()
    day = now_ist().date()
    uni = load_universe()
    tradable = set(uni.loc[uni["tier"] == "tradable", "symbol"].str.replace("-EQ", "", regex=False))
    try:
        nse = NSE()
    except requests.RequestException as exc:
        log.error("NSE unreachable (%s); nothing collected today", type(exc).__name__)
        return
    got = collect(nse, day, tradable)
    report = write_watchlist(day, got, tradable)
    log.info("watchlist: %s", report)
    if not a.no_archive:
        archive(day, report)


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "radar.log"), encoding="utf-8")],
                        force=True)
    main()

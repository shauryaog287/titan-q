"""
mover_scanner.py
================
Live scanner for big movers across the whole NSE market, with the news behind them.

    D:\\trader_env\\Scripts\\python.exe mover_scanner.py      runs 09:16-15:30 IST (scheduled task TitanQ_Movers)

Every minute, from the Angel recorder's latest prices for ~2,700 stocks
(data/ticks/latest_prices.json):
- move = last price vs the previous close (Yahoo daily cache, the last close before today);
- volume ratio = today's volume so far vs the 20-day median daily volume scaled by the share
  of the session elapsed;
- a MOVER is a stock up or down at least 5% with a volume ratio of at least 3, and a median
  daily traded value of at least Rs. 1 crore (illiquid names are left out). It is reported
  when it first qualifies, and again each time it crosses another 5% step (10%, 15%, ...).

Each mover is matched to today's NSE filings for that stock (one announcements request every
3 minutes, cached) and to headlines in the news archive tagged with it. Then it is:
- archived to data/movers/movers_<date>.parquet (time first seen, move, volume ratio, news);
- pushed to the phone (alerts.py): one alert per cycle listing the new movers, at most 20 an
  hour.

INFORMATION ONLY. Whether news-backed jumps keep going and no-news jumps reverse is tested in
NJ1/NJ2 (pre-registered) before anything trades on it. If the NJ1b confirmation passed
(reports/no_news_fade.json), up-jumps with no non-routine NSE filing since 15:30 on the
previous trading day are tagged "historically tends to fade".
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, time as dtime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR
from market_data import IST, DailyStore, now_ist

log = logging.getLogger("movers")

TICK_DIR = os.path.join(DATA_DIR, "ticks")
LATEST = os.path.join(TICK_DIR, "latest_prices.json")
TOKEN_MAP = os.path.join(TICK_DIR, "token_map.json")
MOVERS_DIR = os.path.join(DATA_DIR, "movers")
OPEN, CLOSE = dtime(9, 15), dtime(15, 30)
MIN_MOVE, MIN_VOL_RATIO, MIN_VALUE_CR, STEP = 5.0, 3.0, 1.0, 5.0
MAX_ALERTS_PER_HOUR = 20
FILINGS_EVERY = 180


def baseline(yahoo_symbols: List[str]) -> pd.DataFrame:
    """Per Yahoo ticker: previous close, 20-day median volume and traded value, from the
    daily cache (rows dated before today)."""
    today = pd.Timestamp(now_ist().date())
    daily = DailyStore().load(yahoo_symbols, refresh=False)
    rows = {}
    for s, d in daily.items():
        d = d[d.index < today].dropna(subset=["Close"])
        if len(d) < 20:
            continue
        tail = d.iloc[-20:]
        rows[s] = {"prev_close": float(d["Close"].iloc[-1]),
                   "med_vol": float(tail["Volume"].median()),
                   "med_value_cr": float((tail["Close"] * tail["Volume"]).median() / 1e7)}
    return pd.DataFrame(rows).T


def session_fraction(now: datetime) -> float:
    start = datetime.combine(now.date(), OPEN, IST)
    return min(1.0, max(1 / 375, (now - start).total_seconds() / (375 * 60)))


NJ1B_AVOID = os.path.join(MOVERS_DIR, "nj1b_avoid.json")


def record_nj1b(yahoo_symbols: List[str]) -> None:
    """Add today's no-news jumps to the no-buy list the trading loop reads (entries older
    than 14 days are dropped; the loop itself counts the 5 trading days)."""
    today = now_ist().date()
    try:
        with open(NJ1B_AVOID, encoding="ascii") as f:
            jumps = json.load(f)
    except (OSError, ValueError):
        jumps = {}
    jumps = {s: d for s, d in jumps.items() if (today - datetime.strptime(d, "%Y-%m-%d").date()).days <= 14}
    for s in yahoo_symbols:
        jumps[s] = str(today)
    os.makedirs(MOVERS_DIR, exist_ok=True)
    with open(NJ1B_AVOID + ".partial", "w", encoding="ascii") as f:
        json.dump(jumps, f, indent=1, sort_keys=True)
    os.replace(NJ1B_AVOID + ".partial", NJ1B_AVOID)


def nj1b_confirmed() -> bool:
    """True only if the pre-registered NJ1b confirmation (reports/no_news_fade.json) passed."""
    try:
        with open(os.path.join(PROJECT_DIR, "reports", "no_news_fade.json"), encoding="ascii") as f:
            return bool(json.load(f).get("confirmed"))
    except (OSError, ValueError):
        return False


class Filings:
    """NSE announcements since 15:30 on the previous trading day (the NJ1 news window),
    refreshed at most every FILINGS_EVERY seconds."""

    def __init__(self) -> None:
        self.at, self.by_symbol, self.nonroutine, self.nse = 0.0, {}, set(), None
        self.ok = False                                     # False until one read succeeded

    def refresh(self) -> None:
        if time.time() - self.at < FILINGS_EVERY:
            return
        self.at = time.time()
        try:
            from disclosure_radar import NSE, _rows
            from market_data import Session
            from news_analyser import ascii_text
            from news_jumps import ROUTINE
            self.nse = self.nse or NSE()
            d = now_ist().date()
            sess, prev = Session(), d - timedelta(days=1)
            while not sess.is_trading_day(prev):
                prev -= timedelta(days=1)
            since = datetime.combine(prev, CLOSE)
            rows = _rows(self.nse.json("https://www.nseindia.com/api/corporate-announcements?index=equities"
                                       f"&from_date={prev:%d-%m-%Y}&to_date={d:%d-%m-%Y}") or [])
            by: Dict[str, List[str]] = {}
            nonroutine = set()
            for r in rows:
                t = pd.to_datetime(r.get("an_dt"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
                if pd.isna(t) or t.to_pydatetime() < since:
                    continue
                by.setdefault(r.get("symbol"), []).append(
                    ascii_text(f"{(r.get('an_dt') or '')[12:17]} {r.get('desc', '')}: "
                               f"{(r.get('attchmntText') or '')[:140]}"))
                if r.get("desc") not in ROUTINE:
                    nonroutine.add(r.get("symbol"))
            self.by_symbol, self.nonroutine, self.ok = by, nonroutine, True
        except Exception as exc:
            log.warning("NSE filings not read (%s)", type(exc).__name__)
            self.nse = None

    def get(self, symbol: str) -> List[str]:
        self.refresh()
        return self.by_symbol.get(symbol, [])

    def no_news(self, symbol: str) -> bool:
        """NJ1's definition: no non-routine filing in the window. False if filings are unknown."""
        self.refresh()
        return self.ok and symbol not in self.nonroutine


_NAMES: Dict[str, str] = {}


def google_why(symbol: str) -> List[str]:
    """Targeted Google News search for one mover (last 24 h), when the archive has nothing."""
    import feedparser
    import requests
    from datetime import timedelta
    from urllib.parse import quote_plus
    from news_analyser import UA, ascii_text, clean_name, equity_list
    if not _NAMES:
        el = equity_list()
        _NAMES.update(dict(zip(el["symbol"], el["name"].map(clean_name))))
    name = _NAMES.get(symbol, symbol)
    url = (f"https://news.google.com/rss/search?q={quote_plus(chr(34) + name + chr(34) + ' when:1d')}"
           "&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        r = requests.get(url, headers=UA, timeout=20)
        oldest = datetime.utcnow() - timedelta(hours=24)
        out = []
        for e in feedparser.parse(r.content).entries:
            pub = e.get("published_parsed")
            if pub and datetime(*pub[:6]) >= oldest:
                out.append(ascii_text(e.get("title", ""))[:140])
        return out[:2]
    except Exception:
        return []


def headlines(symbol: str) -> List[str]:
    try:
        from news_analyser import load_news
        df = load_news(18.0)
        df = df[df["symbols"].fillna("").str.split(",").apply(lambda x: symbol in x)]
        found = [f"{r.title[:120]} ({r.source}, {r.score:+.2f})" for r in df.sort_values("first_seen").itertuples()][-2:]
    except Exception:
        found = []
    if not found:
        time.sleep(2.0)                                    # polite spacing between searches
        found = google_why(symbol)
    return found


def scan_once(base: pd.DataFrame, tmap: Dict[str, str], reported: Dict[str, float]) -> List[Dict]:
    with open(LATEST, encoding="ascii") as f:
        latest = json.load(f)
    if time.time() - latest.get("written", 0) > 120:
        log.warning("recorder prices are %.0f s old; skipping this cycle", time.time() - latest.get("written", 0))
        return []
    frac = session_fraction(now_ist())
    out = []
    for token, p in latest["prices"].items():
        y = tmap.get(token)
        if y not in base.index or not p.get("ltp"):
            continue
        b = base.loc[y]
        if b["med_value_cr"] < MIN_VALUE_CR or b["prev_close"] <= 0 or b["med_vol"] <= 0:
            continue
        move = (p["ltp"] / b["prev_close"] - 1) * 100
        vr = p.get("volume", 0) / (b["med_vol"] * frac)
        if abs(move) < MIN_MOVE or vr < MIN_VOL_RATIO:
            continue
        level = np.sign(move) * STEP * np.floor(abs(move) / STEP)       # 5, 10, 15 ... (signed)
        if reported.get(y) is not None and abs(level) <= abs(reported[y]):
            continue
        reported[y] = level
        out.append({"symbol": y[:-3], "yahoo": y, "move_pct": round(move, 2), "vol_ratio": round(vr, 1),
                    "ltp": p["ltp"], "value_cr_median": round(float(b["med_value_cr"]), 1)})
    return out


def main() -> None:
    from alerts import send
    from market_data import Session
    if not Session().is_trading_day(now_ist().date()):
        log.info("not a trading day (weekend or NSE holiday); nothing to scan")
        return
    with open(TOKEN_MAP, encoding="ascii") as f:
        tmap = json.load(f)
    base = baseline(sorted(set(tmap.values())))
    log.info("baseline for %d stocks; scanning until %s", len(base), CLOSE.strftime("%H:%M"))
    filings, reported, sent = Filings(), {}, []
    fade_tag = nj1b_confirmed()
    log.info("NJ1b no-news fade tag: %s", "on (confirmed)" if fade_tag else "off (not confirmed)")
    os.makedirs(MOVERS_DIR, exist_ok=True)
    path = os.path.join(MOVERS_DIR, f"movers_{now_ist().date()}.parquet")
    archive: List[Dict] = pd.read_parquet(path).to_dict("records") if os.path.exists(path) else []
    for r in archive:                                       # after a restart, do not re-alert
        reported[r["yahoo"]] = np.sign(r["move_pct"]) * STEP * np.floor(abs(r["move_pct"]) / STEP)
    while now_ist().time() < CLOSE:
        t0 = time.time()
        try:
            if now_ist().time() >= OPEN and os.path.exists(LATEST):
                movers = scan_once(base, tmap, reported)
                for m in movers:
                    m["first_seen"] = now_ist().isoformat(timespec="seconds")
                    m["filings"] = " | ".join(filings.get(m["symbol"])[-2:])
                    m["news"] = " | ".join(headlines(m["symbol"]))
                    # NJ1b: an up-jump with no non-routine filing (headlines do not count, as in the study)
                    m["nj1b_fade"] = bool(fade_tag and m["move_pct"] > 0 and filings.no_news(m["symbol"]))
                    m["ai"] = ""
                fades = [m["yahoo"] for m in movers if m["nj1b_fade"]]
                if fades:
                    record_nj1b(fades)
                    log.info("NJ1b no-buy list: added %s", ", ".join(fades))
                # local AI reading of the filing (or headline) behind the biggest movers, if the model runs
                try:
                    from llm_reader import available, describe, read_item
                    if movers and available():
                        for m in sorted(movers, key=lambda m: -abs(m["move_pct"]))[:3]:
                            text = m["filings"] or m["news"]
                            ev = read_item(f"{m['symbol']}: {text}", timeout=30.0) if text else None
                            m["ai"] = describe(ev) if ev else ""
                except Exception as exc:
                    log.warning("AI reading skipped: %s", type(exc).__name__)
                if movers:
                    archive += movers
                    pd.DataFrame(archive).to_parquet(path + ".partial", index=False)
                    os.replace(path + ".partial", path)
                    sent = [t for t in sent if time.time() - t < 3600]
                    if len(sent) < MAX_ALERTS_PER_HOUR:
                        lines = [f"{m['symbol']} {m['move_pct']:+.1f}% ({m['vol_ratio']:.0f}x vol) "
                                 + (f"- FILING {m['filings'][:150]}" if m["filings"] else
                                    f"- NEWS {m['news'][:150]}" if m["news"] else "- no news found")
                                 + (" [no filing - historically tends to fade (NJ1b)]" if m["nj1b_fade"] else "")
                                 + (f"\n   AI: {m['ai'][:160]}" if m["ai"] else "")
                                 for m in sorted(movers, key=lambda m: -abs(m["move_pct"]))[:6]]
                        if len(movers) > 6:
                            lines.append(f"... and {len(movers) - 6} more")
                        if send(f"market movers ({len(movers)})", "\n".join(lines), priority="default",
                                tags="chart_with_upwards_trend", repeat_minutes=0):
                            sent.append(time.time())
                    log.info("%d new movers: %s", len(movers),
                             ", ".join(f"{m['symbol']} {m['move_pct']:+.1f}%" for m in movers[:10]))
        except Exception as exc:
            log.warning("cycle failed: %s: %s", type(exc).__name__, str(exc)[:200])
        time.sleep(max(5.0, 60.0 - (time.time() - t0)))
    log.info("stopped; %d movers archived today", len(archive))


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(PROJECT_DIR, "logs", "movers.log"), encoding="utf-8")], force=True)
    for noisy in ("radar", "urllib3", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    main()

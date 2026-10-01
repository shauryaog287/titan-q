"""
news_watch.py
=============
Live news watcher for the stocks we hold: every new headline or NSE filing about a held
(or planned) stock is scored with FinBERT and pushed to the phone (alerts.py, ntfy) within
minutes of appearing.

    D:\\trader_env\\Scripts\\python.exe news_watch.py          run until 21:35 IST (scheduled task TitanQ_News)

Cadence (FinBERT stays loaded the whole time):
- NSE corporate announcements of the day: every 3 minutes (the official, fastest source:
  results, orders, resignations, ratings), filtered to our stocks;
- Economic Times, Business Standard and Mint RSS: every 3 minutes;
- Google News for each of our stocks (batched 10 names a query): every 10 minutes, because
  Google throttles more frequent searches.

"Our stocks" = the live paper account's holdings, the forward-test holdings and the 50
planned stocks, re-read every cycle. Everything goes into the same point-in-time archive as
news_analyser.py (data/news/news_<date>.parquet, copied to G:); this process is the only
writer while it runs. Alerts: one per stock per cycle (headlines grouped), high priority for
a strongly negative headline or a sensitive filing, at most 30 an hour, none 22:00-07:00.

Local AI (llm_reader.py, Ollama on this laptop): when the model is running, our stocks' new
items are read before the alert goes out (the alert shows "AI: event type, direction,
materiality, value: summary"; a negative, high-materiality reading makes it high priority),
and the remaining tagged items of the day are read in the time left in each cycle. Without
the model everything works as before.

Observe only: no trade depends on news (see news_analyser.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Set

from titanq_config import PROJECT_DIR, UNIVERSE
from market_data import now_ist

log = logging.getLogger("news_watch")

STOP_AT = dtime(21, 35)
FAST_SECONDS = 180
GOOGLE_SECONDS = 600
MAX_ALERTS_PER_HOUR = 30
LLM_ALERT_BUDGET = 90.0         # seconds the local model may spend on our stocks' new items per cycle
SENSITIVE = re.compile(r"auditor|resign|credit rating|default|fraud|insolvency|pledge|sebi|search|raid|"
                       r"forensic|delay|qualified opinion|downgrade|penalty|show cause", re.I)
LIVE_ACCOUNT = os.path.join(PROJECT_DIR, "paper_account.json")
SHADOW_ACCOUNT = os.path.join(PROJECT_DIR, "paper_account_shadow.json")


def our_stocks() -> Set[str]:
    syms = {s[:-3] for s in UNIVERSE}
    for path in (LIVE_ACCOUNT, SHADOW_ACCOUNT):
        try:
            with open(path, encoding="ascii") as f:
                syms |= {s[:-3] for s, q in json.load(f).get("positions", {}).items() if q}
        except (OSError, ValueError):
            pass
    return syms


def nse_filings(nse, stocks: Set[str]) -> (List[Dict], Dict[str, str]):
    """Today's NSE announcements for our stocks, as news items (title = category + text)."""
    from disclosure_radar import _rows
    from news_analyser import ascii_text
    d = now_ist().date()
    rows = nse.json("https://www.nseindia.com/api/corporate-announcements?index=equities"
                    f"&from_date={d:%d-%m-%Y}&to_date={d:%d-%m-%Y}") or []
    items, fixed = [], {}
    for r in _rows(rows):
        sym = r.get("symbol")
        if sym not in stocks:
            continue
        title = ascii_text(f"[NSE filing] {r.get('desc', '')}: {r.get('attchmntText') or ''}")[:300]
        when = r.get("an_dt") or ""
        try:
            pub = datetime.strptime(when, "%d-%b-%Y %H:%M:%S") - timedelta(hours=5, minutes=30)
            pub_s = pub.isoformat() + "Z"
        except ValueError:
            pub_s = ""
        items.append({"source": "NSE filing", "title": title, "link": r.get("attchmntFile") or "",
                      "summary": "", "published": pub_s})
        fixed[title] = sym
    return items, fixed


class AlertBudget:
    def __init__(self) -> None:
        self.sent: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        self.sent = [t for t in self.sent if now - t < 3600]
        return len(self.sent) < MAX_ALERTS_PER_HOUR

    def use(self) -> None:
        self.sent.append(time.time())


def alert_new(rows: List[Dict], stocks: Set[str], budget: AlertBudget, ai: Optional[Dict[str, Dict]] = None) -> int:
    """ai: {news key: event} from the local model (llm_reader), shown under each headline."""
    from alerts import send
    from news_analyser import STRONG_NEGATIVE
    ai = ai or {}
    hour = now_ist().time()
    if hour >= dtime(22, 0) or hour < dtime(7, 0):
        return 0
    by_sym: Dict[str, List[Dict]] = {}
    for r in rows:
        for s in str(r.get("symbols") or "").split(","):
            if s in stocks:
                by_sym.setdefault(s, []).append(r)
    sent, skipped = 0, 0
    for s, items in sorted(by_sym.items(), key=lambda kv: min(r["score"] for r in kv[1])):
        if not budget.allow():
            skipped += len(items)
            continue
        worst = min(r["score"] for r in items)
        filing = any(r["source"] == "NSE filing" for r in items)
        urgent = worst <= STRONG_NEGATIVE or any(r["source"] == "NSE filing" and SENSITIVE.search(r["title"])
                                                 for r in items) or \
            any(ai.get(r["key"], {}).get("direction") == "negative" and ai[r["key"]].get("materiality") == "high"
                for r in items)
        lines = []
        for r in items[:4]:
            lines.append(f"{'+' if r['score'] > 0.3 else '-' if r['score'] < -0.3 else '='}{r['score']:+.2f} "
                         f"{r['title'][:180]} ({r['source']})")
            if r["key"] in ai:
                from llm_reader import describe
                lines.append(f"   AI: {describe(ai[r['key']])}")
        if len(items) > 4:
            lines.append(f"... and {len(items) - 4} more")
        title = f"{s}: {'NSE filing' if filing else 'news'} ({len(items)})"
        if send(title, "\n".join(lines), priority="high" if urgent else "default",
                tags="warning" if urgent else "newspaper", repeat_minutes=0):
            budget.use()
            sent += 1
    if skipped and budget.allow():
        send("news alerts capped", f"{skipped} more headlines about our stocks this hour were not "
             "pushed (limit reached); they are in the news archive and the trading log.", priority="low",
             repeat_minutes=60)
    return sent


def main() -> None:
    from disclosure_radar import NSE
    import disclosure_radar
    from news_analyser import fetch_items, ingest, make_tagger
    from llm_reader import available as llm_available, read_rows, run_once as run_llm
    disclosure_radar.GAP_SECONDS = 5.0
    stocks = our_stocks()
    tagger = make_tagger(sorted(stocks))
    nse = None
    budget = AlertBudget()
    last_google = 0.0
    first = True                    # the first pass only archives (no flood of earlier items)
    log.info("watching %d stocks until %s IST", len(stocks), STOP_AT.strftime("%H:%M"))
    while now_ist().time() < STOP_AT:
        t0 = time.time()
        try:
            stocks = our_stocks()
            items = fetch_items(tagger, sorted(stocks), rss=True, google=False)
            if time.time() - last_google >= GOOGLE_SECONDS:
                items += fetch_items(tagger, sorted(stocks), rss=False, google=True)
                last_google = time.time()
            fixed: Dict[str, str] = {}
            try:
                nse = nse or NSE()
                f_items, fixed = nse_filings(nse, stocks)
                items += f_items
            except Exception as exc:
                log.warning("NSE filings not read (%s)", type(exc).__name__)
                nse = None                                   # new session next cycle
            rows = ingest(items, tagger, fixed)
            ours = [r for r in rows if set(str(r.get("symbols") or "").split(",")) & stocks]
            llm_ok = llm_available()
            ai = read_rows(ours, budget_s=LLM_ALERT_BUDGET) if llm_ok and ours and not first else {}
            if first:
                first = False
                from alerts import send
                send("news watcher on", f"Watching {len(stocks)} stocks until {STOP_AT.strftime('%H:%M')}. "
                     f"{len(ours)} earlier items about them were archived without alerts; new ones will be "
                     "pushed as they appear.", priority="low", tags="newspaper", repeat_minutes=60)
                n_alerts = 0
            else:
                n_alerts = alert_new(rows, stocks, budget, ai)
            n_read = 0
            if llm_ok:              # the rest of today's tagged items (any stock), in the time left
                spare = FAST_SECONDS - (time.time() - t0) - 30
                n_read = run_llm(max_items=40, budget_s=spare) if spare > 20 else 0
            log.info("cycle: %d items, %d new, %d about our stocks, %d alerts, AI read %d + %d",
                     len(items), len(rows), len(ours), n_alerts, len(ai), n_read)
        except Exception as exc:
            log.warning("cycle failed: %s: %s", type(exc).__name__, str(exc)[:200])
        time.sleep(max(10.0, FAST_SECONDS - (time.time() - t0)))
    log.info("stopped at %s", now_ist().strftime("%H:%M"))


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(PROJECT_DIR, "logs", "news.log"), encoding="utf-8")], force=True)
    for noisy in ("transformers", "radar", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    main()

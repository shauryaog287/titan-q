"""
reddit_collector.py
===================
Retail chatter from Reddit's official API (read-only, free for personal use): posts and
comments of Indian stock-market subreddits, tagged to NSE stocks, toned with FinBERT,
archived point-in-time. Replaces the old Selenium scraper (social_media_scanner.py),
which disguised a browser to scrape pages against Reddit's rules.

    D:\\trader_env\\Scripts\\python.exe reddit_collector.py            collect (scheduled hourly)
    D:\\trader_env\\Scripts\\python.exe reddit_collector.py --report   most-mentioned stocks, last 24 h vs usual

Needs reddit_credentials.json (git-ignored) with the "script" app you create at
https://www.reddit.com/prefs/apps (see reddit_credentials.example.json). Without it the
script logs that and exits; nothing else depends on it.

Files: data/social/reddit_<date>.parquet (first_seen, created, subreddit, kind, id,
text, upvotes, symbols, tone), copied to G:\\My Drive\\Titan-Q_History\\social.

OBSERVE ONLY: a spike in mentions is a documented warning sign (retail attention, then
underperformance), but no trade depends on it until a pre-registered test supports it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from datetime import timedelta
from typing import Dict, List, Sequence

import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR
from market_data import now_ist

log = logging.getLogger("reddit")

CREDENTIALS = os.path.join(PROJECT_DIR, "reddit_credentials.json")
SOCIAL_DIR = os.path.join(DATA_DIR, "social")
DRIVE_DIR = r"G:\My Drive\Titan-Q_History\social"
SUBREDDITS = ("IndianStreetBets", "IndianStockMarket", "DalalStreetTalks")
# Upper-case words that are also NSE symbols but usually just shouting in posts
TICKER_STOPWORDS = {"ALL", "ANY", "BUY", "SELL", "HOLD", "CASH", "HOME", "IDEA", "JUST", "LOVE", "ONE",
                    "PAGE", "RAIN", "NEWS", "BEST", "GOOD", "BIG", "NEW", "NOW", "TOP", "WIN", "YES",
                    "FOR", "THE", "AND", "NOT", "ARE", "YOU", "CAN", "GET", "PUT", "CALL", "LONG",
                    "SHORT", "EXIT", "STOP", "LOSS", "GAIN", "BULL", "BEAR", "MOON", "PUMP", "DUMP",
                    "IPO", "ETF", "NSE", "BSE", "RBI", "GDP", "EPS", "CEO", "USA", "IT", "AI", "EV"}


def credentials() -> Dict[str, str]:
    if not os.path.exists(CREDENTIALS):
        return {}
    with open(CREDENTIALS, encoding="ascii") as f:
        c = json.load(f)
    return c if c.get("client_id") and c.get("client_secret") and "PASTE" not in c["client_id"] else {}


def make_tagger(symbols: Sequence[str]):
    """news_analyser's name tagger plus upper-case tickers, as Reddit users write them."""
    from news_analyser import Tagger
    t = Tagger(symbols)
    for s in symbols:
        if re.fullmatch(r"[A-Z][A-Z0-9&-]{2,}", s) and s not in TICKER_STOPWORDS:
            t.patterns.append((s, re.compile(r"(?<![A-Za-z0-9])\$?" + re.escape(s) + r"(?![A-Za-z0-9])")))
    return t


def fetch(creds: Dict[str, str]) -> List[Dict]:
    import praw
    r = praw.Reddit(client_id=creds["client_id"], client_secret=creds["client_secret"],
                    user_agent=creds.get("user_agent", "windows:titan-q-research:v1 (personal research)"))
    r.read_only = True
    out: List[Dict] = []
    cutoff = (now_ist() - timedelta(hours=26)).timestamp()
    for name in SUBREDDITS:
        try:
            sub = r.subreddit(name)
            for p in sub.new(limit=100):
                if p.created_utc >= cutoff:
                    out.append({"subreddit": name, "kind": "post", "id": p.id, "created": p.created_utc,
                                "text": f"{p.title}. {(p.selftext or '')[:400]}", "upvotes": int(p.score)})
            for c in sub.comments(limit=300):
                if c.created_utc >= cutoff:
                    out.append({"subreddit": name, "kind": "comment", "id": c.id, "created": c.created_utc,
                                "text": (c.body or "")[:400], "upvotes": int(c.score)})
        except Exception as exc:
            log.warning("r/%s: %s", name, type(exc).__name__)
    return out


def day_file(d) -> str:
    return os.path.join(SOCIAL_DIR, f"reddit_{d}.parquet")


def collect() -> int:
    creds = credentials()
    if not creds:
        log.info("no Reddit credentials (reddit_credentials.json); skipped")
        return 0
    from news_analyser import ascii_text, equity_list, finbert_scores
    from universe import load_universe
    uni = load_universe()
    tradable = set(uni.loc[uni["tier"] == "tradable", "symbol"].str.replace("-EQ", "", regex=False))
    tagger = make_tagger(sorted(tradable & set(equity_list()["symbol"])))
    items = fetch(creds)
    now = now_ist()
    os.makedirs(SOCIAL_DIR, exist_ok=True)
    path = day_file(now.date())
    old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    seen = set(old["id"]) if len(old) else set()
    prev = day_file(now.date() - timedelta(days=1))
    if os.path.exists(prev):
        seen |= set(pd.read_parquet(prev)["id"])
    rows = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        text = ascii_text(it["text"])
        if len(text) < 8:
            continue
        rows.append({**it, "text": text, "first_seen": now.isoformat(timespec="seconds"),
                     "created": pd.Timestamp(it["created"], unit="s", tz="UTC").isoformat(),
                     "symbols": ",".join(tagger.tag(text))})
    if rows:
        for r, sc in zip(rows, finbert_scores([r["text"][:300] for r in rows])):
            r["tone"] = sc["p_pos"] - sc["p_neg"]
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
        df.to_parquet(path + ".partial", index=False)
        os.replace(path + ".partial", path)
        if os.path.isdir(os.path.dirname(os.path.dirname(DRIVE_DIR))):
            os.makedirs(DRIVE_DIR, exist_ok=True)
            shutil.copy2(path, os.path.join(DRIVE_DIR, os.path.basename(path)))
    log.info("%d items fetched, %d new (%d mention a stock)", len(items), len(rows),
             sum(1 for r in rows if r["symbols"]))
    return len(rows)


def mentions(hours: float = 24.0, baseline_days: int = 14) -> Dict[str, Dict]:
    """symbol -> {n (last `hours`), per-day baseline, tone} from the stored files."""
    if not os.path.isdir(SOCIAL_DIR):
        return {}
    today = now_ist().date()
    frames = [pd.read_parquet(day_file(today - timedelta(days=k))) for k in range(baseline_days + 1)
              if os.path.exists(day_file(today - timedelta(days=k)))]
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df[df["symbols"].fillna("") != ""].copy()
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df = df.assign(symbol=df["symbols"].str.split(",")).explode("symbol")
    cut = pd.Timestamp(now_ist() - timedelta(hours=hours))
    recent, older = df[df["first_seen"] >= cut], df[df["first_seen"] < cut]
    days_hist = max(1, older["first_seen"].dt.date.nunique())
    base = older.groupby("symbol").size() / days_hist
    out = {}
    for s, g in recent.groupby("symbol"):
        out[s] = {"n": int(len(g)), "baseline_per_day": round(float(base.get(s, 0.0)), 2),
                  "tone": round(float(g["tone"].mean()), 3)}
    return out


def report() -> None:
    m = mentions()
    print(f"Reddit mentions, last 24 h ({len(m)} stocks):")
    for s, v in sorted(m.items(), key=lambda kv: -kv[1]["n"])[:30]:
        print(f"  {s:12s} {v['n']:4d} mentions (usual {v['baseline_per_day']:.1f}/day), tone {v['tone']:+.2f}")


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(PROJECT_DIR, "logs", "reddit.log"), encoding="utf-8")], force=True)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else collect()

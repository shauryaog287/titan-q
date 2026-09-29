"""
news_analyser.py
================
Collects Indian market news, tags each headline to NSE stocks, scores its tone with
FinBERT, and keeps a point-in-time archive for later testing.

    D:\\trader_env\\Scripts\\python.exe news_analyser.py              collect + score (scheduled every 30 min)
    D:\\trader_env\\Scripts\\python.exe news_analyser.py --report     tone of the last 24 h for the stocks we hold/plan

Sources (checked 2026-09-29; Moneycontrol's RSS feeds stopped in 2024 and are not used):
- Economic Times markets and stocks, Business Standard markets, Mint markets (RSS);
- Google News search for the stocks that matter now (the 50 live stocks and the
  forward-test holdings), batched ~10 companies per query to keep requests few.

Each headline is stored once (duplicates across sources are merged) with the time WE
first saw it: data/news/news_<date>.parquet, copied to G:\\My Drive\\Titan-Q_History\\news.
That first-seen time is what any backtest must use, never the publisher's timestamp.

Tone: FinBERT (ProsusAI/finbert, run locally from the Hugging Face cache, offline):
score = P(positive) - P(negative), in [-1, 1].

OBSERVE ONLY. The trading loop logs the news of its stocks but no trade depends on it:
headline tone has not passed a pre-registered test yet (a historical test is planned).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import time
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests

from titanq_config import DATA_DIR, PROJECT_DIR, UNIVERSE
from market_data import IST, now_ist

log = logging.getLogger("news")

NEWS_DIR = os.path.join(DATA_DIR, "news")
DRIVE_DIR = r"G:\My Drive\Titan-Q_History\news"
EQUITY_LIST = os.path.join(NEWS_DIR, "equity_list.csv")
SHADOW_ACCOUNT = os.path.join(PROJECT_DIR, "paper_account_shadow.json")
FEEDS = {
    "ET markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET stocks": "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "Mint": "https://www.livemint.com/rss/markets",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
GNEWS_BATCH = 10
STRONG_NEGATIVE = -0.5
MAX_AGE_HOURS = 48          # Google News returns old articles despite "when:1d"; drop them
NOISE = re.compile(r"share price live|price live today|stock price & chart|live updates:", re.I)

# Short names used in headlines for the 50 live stocks (the full company name is added
# automatically). Keys are NSE symbols.
ALIASES: Dict[str, List[str]] = {
    "RELIANCE": ["Reliance", "RIL"], "TCS": ["TCS"], "HDFCBANK": ["HDFC Bank"], "INFY": ["Infosys"],
    "ICICIBANK": ["ICICI Bank"], "HINDUNILVR": ["HUL", "Hindustan Unilever"], "ITC": ["ITC"],
    "SBIN": ["SBI", "State Bank of India"], "BHARTIARTL": ["Airtel", "Bharti Airtel"],
    "BAJFINANCE": ["Bajaj Finance"], "KOTAKBANK": ["Kotak Mahindra Bank", "Kotak Bank"],
    "LT": ["L&T", "Larsen"], "ASIANPAINT": ["Asian Paints"], "AXISBANK": ["Axis Bank"],
    "HCLTECH": ["HCLTech", "HCL Tech"], "MARUTI": ["Maruti"], "BAJAJFINSV": ["Bajaj Finserv"],
    "TITAN": ["Titan"], "SUNPHARMA": ["Sun Pharma"], "ULTRACEMCO": ["UltraTech"],
    "TATASTEEL": ["Tata Steel"], "WIPRO": ["Wipro"], "NESTLEIND": ["Nestle"], "TECHM": ["Tech Mahindra"],
    "ONGC": ["ONGC"], "POWERGRID": ["Power Grid"], "HINDALCO": ["Hindalco"], "M&M": ["M&M", "Mahindra & Mahindra"],
    "NTPC": ["NTPC"], "INDUSINDBK": ["IndusInd"], "GRASIM": ["Grasim"], "DIVISLAB": ["Divi's", "Divis Lab"],
    "TMPV": ["Tata Motors"], "BAJAJ-AUTO": ["Bajaj Auto"], "JSWSTEEL": ["JSW Steel"], "BRITANNIA": ["Britannia"],
    "ADANIPORTS": ["Adani Ports"], "CIPLA": ["Cipla"], "HDFCLIFE": ["HDFC Life"], "HEROMOTOCO": ["Hero MotoCorp"],
    "COALINDIA": ["Coal India"], "TATACONSUM": ["Tata Consumer"], "APOLLOHOSP": ["Apollo Hospitals"],
    "UPL": ["UPL"], "DRREDDY": ["Dr Reddy", "Dr. Reddy"], "EICHERMOT": ["Eicher", "Royal Enfield"],
    "BPCL": ["BPCL"], "SBILIFE": ["SBI Life"], "SHREECEM": ["Shree Cement"], "TATAPOWER": ["Tata Power"],
}


# ===========================================================================
# TEXT
# ===========================================================================

_PUNCT = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-", 0x2014: "-", 0x2026: "...",
          0x20B9: "Rs. ", 0x00A0: " ", 0x2022: "-", 0x00B7: "-"}


def ascii_text(s: str) -> str:
    """ASCII only (the console is CP1252): typographic punctuation mapped, the rupee sign
    written "Rs.", anything else decomposed or dropped."""
    s = (s or "").translate(_PUNCT)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()


def title_key(title: str) -> str:
    return hashlib.sha1(re.sub(r"[^a-z0-9]", "", title.lower()).encode("ascii")).hexdigest()[:16]


# ===========================================================================
# COMPANY NAMES -> SYMBOLS
# ===========================================================================

def equity_list(max_age_days: float = 7.0) -> pd.DataFrame:
    """NSE's list of listed equities (symbol, company name), cached weekly."""
    os.makedirs(NEWS_DIR, exist_ok=True)
    fresh = os.path.exists(EQUITY_LIST) and time.time() - os.path.getmtime(EQUITY_LIST) < max_age_days * 86400
    if not fresh:
        try:
            s = requests.Session()
            s.headers.update(UA)
            r = s.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", timeout=30)
            r.raise_for_status()
            text = "\n".join(ascii_text(line) for line in r.content.decode("utf-8", "replace").splitlines())
            with open(EQUITY_LIST, "w", encoding="ascii", newline="") as f:
                f.write(text + "\n")
        except Exception as exc:
            log.warning("NSE equity list not refreshed (%s); using the cached copy", type(exc).__name__)
    df = pd.read_csv(EQUITY_LIST)
    df.columns = [c.strip() for c in df.columns]
    return df[["SYMBOL", "NAME OF COMPANY"]].rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "name"})


def clean_name(name: str) -> str:
    n = re.sub(r"\b(limited|ltd\.?|the)\b", "", name, flags=re.I)
    return re.sub(r"\s+", " ", n).strip(" .,-")


class Tagger:
    """Finds the NSE symbols a headline talks about."""

    def __init__(self, symbols: Sequence[str]) -> None:
        names = equity_list()
        names = names[names["symbol"].isin(set(symbols))]
        self.patterns: List[Tuple[str, re.Pattern]] = []
        for sym, name in names[["symbol", "name"]].itertuples(index=False):
            # full company name plus hand-made short names only: matching on a unique first
            # word tagged "IndusInd CEO Rajiv Anand" as Anand Rathi (2026-09-29)
            terms = {clean_name(name)} | set(ALIASES.get(sym, []))
            terms = {t for t in terms if len(t) >= 3}
            # short all-capital terms (TCS, ITC, SBI) must match in capitals; names match any case
            caps = [re.escape(t) for t in terms if t.isupper() and len(t) <= 5]
            words = [re.escape(t) for t in terms if not (t.isupper() and len(t) <= 5)]
            parts = []
            if words:
                parts.append(r"(?i:\b(?:" + "|".join(sorted(words, key=len, reverse=True)) + r")(?:'s)?\b)")
            if caps:
                parts.append(r"\b(?:" + "|".join(caps) + r")\b")
            self.patterns.append((sym, re.compile("|".join(parts))))
        self.names = dict(zip(names["symbol"], names["name"].map(clean_name)))

    def tag(self, text: str) -> List[str]:
        return sorted({sym for sym, p in self.patterns if p.search(text)})


# ===========================================================================
# COLLECT
# ===========================================================================

def focus_symbols() -> List[str]:
    """The stocks that matter now: the live 50 and the forward-test holdings."""
    syms = {s[:-3] for s in UNIVERSE}
    try:
        with open(SHADOW_ACCOUNT, encoding="ascii") as f:
            syms |= {s[:-3] for s, q in json.load(f).get("positions", {}).items() if q}
    except (OSError, ValueError):
        pass
    return sorted(syms)


def fetch_items(tagger: Tagger, focus: Sequence[str]) -> List[Dict]:
    import feedparser
    s = requests.Session()
    s.headers.update(UA)
    items: List[Dict] = []

    oldest = datetime.utcnow() - timedelta(hours=MAX_AGE_HOURS)

    def add(feed_name: str, content: bytes) -> int:
        f = feedparser.parse(content)
        kept = 0
        for e in f.entries:
            title = ascii_text(e.get("title", ""))
            if not title or NOISE.search(title):
                continue
            if feed_name.startswith("Google"):
                title = re.sub(r"\s+-\s+[^-]{2,60}$", "", title)       # drop " - Publisher"
            pub = e.get("published_parsed") or e.get("updated_parsed")
            if pub is None or datetime(*pub[:6]) < oldest:
                continue                                            # undated or old: not news now
            items.append({"source": feed_name, "title": title, "link": e.get("link", ""),
                          "summary": ascii_text(re.sub(r"<[^>]+>", " ", e.get("summary", "")))[:500],
                          "published": datetime(*pub[:6]).isoformat() + "Z"})
            kept += 1
        return kept

    for name, url in FEEDS.items():
        try:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            log.info("%s: %d items", name, add(name, r.content))
        except Exception as exc:
            log.warning("%s: failed (%s)", name, type(exc).__name__)
        time.sleep(1.0)
    for i in range(0, len(focus), GNEWS_BATCH):
        batch = focus[i:i + GNEWS_BATCH]
        q = " OR ".join(f'"{tagger.names.get(sym, sym)}"' for sym in batch)
        url = f"https://news.google.com/rss/search?q={quote_plus(q + ' when:1d')}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            add("Google News", r.content)
        except Exception as exc:
            log.warning("Google News batch %d: failed (%s)", i // GNEWS_BATCH, type(exc).__name__)
        time.sleep(2.0)
    return items


# ===========================================================================
# SCORE
# ===========================================================================

def finbert_scores(texts: List[str]) -> List[Dict[str, float]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # the model is in the local cache
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import pipeline
    clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    out = []
    for i in range(0, len(texts), 32):
        for res in clf(texts[i:i + 32]):
            p = {d["label"].lower(): float(d["score"]) for d in res}
            out.append({"p_pos": p.get("positive", 0.0), "p_neg": p.get("negative", 0.0),
                        "p_neu": p.get("neutral", 0.0)})
    return out


# ===========================================================================
# STORE
# ===========================================================================

def day_file(d) -> str:
    return os.path.join(NEWS_DIR, f"news_{d}.parquet")


def load_news(hours: float = 24.0) -> pd.DataFrame:
    now = now_ist()
    frames = []
    for d in {now.date(), (now - timedelta(hours=hours)).date()}:
        if os.path.exists(day_file(d)):
            frames.append(pd.read_parquet(day_file(d)))
    if not frames:
        return pd.DataFrame(columns=["first_seen", "title", "symbols", "score"])
    df = pd.concat(frames, ignore_index=True)
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    return df[df["first_seen"] >= pd.Timestamp(now - timedelta(hours=hours))]


def collect() -> int:
    os.makedirs(NEWS_DIR, exist_ok=True)
    uni_syms = set(equity_list()["symbol"])
    try:
        from universe import load_universe
        uni = load_universe()
        tradable = set(uni.loc[uni["tier"] == "tradable", "symbol"].str.replace("-EQ", "", regex=False))
    except Exception:
        tradable = set()
    focus = focus_symbols()
    tagger = Tagger(sorted((tradable | set(focus)) & uni_syms))
    items = fetch_items(tagger, focus)
    now = now_ist()
    path = day_file(now.date())
    old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    seen = set(old["key"]) if len(old) else set()
    for d in (now.date() - timedelta(days=1),):            # also skip what yesterday's file has
        if os.path.exists(day_file(d)):
            seen |= set(pd.read_parquet(day_file(d))["key"])
    new: Dict[str, Dict] = {}
    for it in items:
        k = title_key(it["title"])
        if k in seen:
            continue
        if k in new:
            new[k]["sources"] = ",".join(sorted(set(new[k]["sources"].split(",")) | {it["source"]}))
            continue
        syms = tagger.tag(it["title"] + " " + it["summary"])
        new[k] = {"key": k, "first_seen": now.isoformat(timespec="seconds"), "published": it["published"],
                  "source": it["source"], "sources": it["source"], "title": it["title"],
                  "summary": it["summary"], "link": it["link"], "symbols": ",".join(syms)}
    rows = list(new.values())
    if rows:
        scores = finbert_scores([r["title"] for r in rows])
        for r, sc in zip(rows, scores):
            r.update(sc)
            r["score"] = sc["p_pos"] - sc["p_neg"]
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
        df.to_parquet(path + ".partial", index=False)
        os.replace(path + ".partial", path)
    tagged = sum(1 for r in rows if r["symbols"])
    log.info("%d items fetched, %d new headlines (%d tagged to stocks)", len(items), len(rows), tagged)
    if os.path.isdir(os.path.dirname(os.path.dirname(DRIVE_DIR))) and os.path.exists(path):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        shutil.copy2(path, os.path.join(DRIVE_DIR, os.path.basename(path)))
    return len(rows)


# ===========================================================================
# SUMMARY (used by the trading loop in observe mode, and by --report)
# ===========================================================================

def symbol_summary(symbols: Sequence[str], hours: float = 24.0) -> Dict[str, Dict]:
    """symbol -> {n, mean score, most negative headline} for the last `hours`."""
    df = load_news(hours)
    out: Dict[str, Dict] = {}
    if not len(df):
        return out
    df = df[df["symbols"].fillna("") != ""]
    want = set(symbols)
    rows = [(s, r) for r in df.itertuples() for s in str(r.symbols).split(",") if s in want]
    for s in {s for s, _ in rows}:
        mine = [r for sym, r in rows if sym == s]
        worst = min(mine, key=lambda r: r.score)
        out[s] = {"n": len(mine), "mean_score": round(sum(r.score for r in mine) / len(mine), 3),
                  "worst_score": round(float(worst.score), 3), "worst_title": worst.title[:160]}
    return out


def report() -> None:
    summ = symbol_summary(focus_symbols())
    print(f"News tone, last 24 h, stocks we hold or plan ({len(summ)} with news):")
    for s, v in sorted(summ.items(), key=lambda kv: kv[1]["mean_score"]):
        flag = "  <-- strongly negative" if v["worst_score"] <= STRONG_NEGATIVE else ""
        print(f"  {s:12s} n={v['n']:3d} mean {v['mean_score']:+.2f} | worst {v['worst_score']:+.2f}: "
              f"{v['worst_title'][:90]}{flag}")


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "news.log"), encoding="utf-8")],
                        force=True)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else collect()

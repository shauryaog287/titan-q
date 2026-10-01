"""
llm_reader.py
=============
A local AI analyst: reads each new NSE filing and headline about our stocks with a language
model running on this laptop (Ollama + Qwen 2.5 7B, everything on D:) and turns it into a
structured event: what happened, good or bad, how material, the rupee value if stated.

    D:\\trader_env\\Scripts\\python.exe llm_reader.py             read new items (news_watch.py does this every cycle)
    D:\\trader_env\\Scripts\\python.exe llm_reader.py --report    today's events, most material first
    D:\\trader_env\\Scripts\\python.exe llm_reader.py --test      read three sample items

Input: the news archive (data/news/news_<date>.parquet). Output: data/events/events_<date>.parquet
(one row per item: key, symbols, event_type, direction, materiality, value_cr, summary,
confidence, first_seen, read_at). Items already read are skipped, so it can run every few
minutes. Ollama serves on 127.0.0.1:11434 (localhost only; nothing leaves the laptop).

OBSERVE ONLY: these events are information, and the raw material for tests such as "which
event types make small stocks keep rising". No trade depends on them yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from titanq_config import DATA_DIR, PROJECT_DIR
from market_data import now_ist

log = logging.getLogger("llm_reader")

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b"
EVENTS_DIR = os.path.join(DATA_DIR, "events")
EVENT_TYPES = ("order_win", "results", "guidance", "management_change", "auditor_change", "rating_change",
               "fundraise", "acquisition", "divestment", "buyback", "dividend_bonus_split", "regulatory_action",
               "litigation", "pledge", "stake_sale", "stake_buy", "capacity_expansion", "product_launch",
               "market_commentary", "routine", "other")
PROMPT = """You are an equity analyst covering Indian listed companies. Read the item below
(an NSE filing or a news headline) and answer with JSON only, using exactly these keys:
"event_type": one of {types};
"direction": "positive", "negative" or "neutral" for the company's shareholders;
"materiality": "high", "medium" or "low" (how much it could move the share price);
"value_cr": the rupee value in crore if the item states one (a number), else null;
"summary": at most 20 words, plain English, no symbols other than letters, digits and basic punctuation;
"confidence": a number from 0 to 1.
Use "routine" for compliance notices, trading-window closures, meeting intimations and
newspaper copies. Use "market_commentary" for items about the market or many stocks at once.

Item: {item}"""


def ascii_only(s: str) -> str:
    from news_analyser import ascii_text
    return ascii_text(str(s or ""))


def available() -> bool:
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=5).json().get("models", [])
        return any(m.get("name", "").startswith(MODEL) for m in tags)
    except requests.RequestException:
        return False


def read_item(text: str, timeout: float = 120.0) -> Optional[Dict]:
    """One item -> event dict, or None if the model gave no usable JSON."""
    body = {"model": MODEL, "prompt": PROMPT.format(types=", ".join(EVENT_TYPES), item=text[:1500]),
            "format": "json", "stream": False, "options": {"temperature": 0, "num_ctx": 2048}}
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=timeout)
        r.raise_for_status()
        out = json.loads(r.json().get("response", "{}"))
    except (requests.RequestException, ValueError) as exc:
        log.warning("model call failed: %s", type(exc).__name__)
        return None
    et = str(out.get("event_type", "other")).strip().lower()
    di = str(out.get("direction", "neutral")).strip().lower()
    ma = str(out.get("materiality", "low")).strip().lower()
    try:
        val = float(out["value_cr"]) if out.get("value_cr") not in (None, "", "null") else None
    except (TypeError, ValueError):
        val = None
    try:
        conf = float(out.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {"event_type": et if et in EVENT_TYPES else "other",
            "direction": di if di in ("positive", "negative", "neutral") else "neutral",
            "materiality": ma if ma in ("high", "medium", "low") else "low",
            "value_cr": val, "summary": ascii_only(out.get("summary", ""))[:200],
            "confidence": max(0.0, min(1.0, conf))}


def events_file(d) -> str:
    return os.path.join(EVENTS_DIR, f"events_{d}.parquet")


def describe(ev: Dict) -> str:
    """One line for an alert: 'order_win, positive, high, Rs. 1,250 cr: summary'."""
    val = f", Rs. {ev['value_cr']:,.0f} cr" if ev.get("value_cr") is not None and pd.notna(ev["value_cr"]) else ""
    return f"{ev['event_type']}, {ev['direction']}, {ev['materiality']}{val}: {ev['summary']}"


def _save(rows: List[Dict]) -> None:
    path = events_file(now_ist().date())
    done = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    os.makedirs(EVENTS_DIR, exist_ok=True)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
    out = out.drop_duplicates("key", keep="first")
    out.to_parquet(path + ".partial", index=False)
    os.replace(path + ".partial", path)


def read_rows(rows: List[Dict], budget_s: float = 90.0) -> Dict[str, Dict]:
    """Read news rows (dicts with key, title, summary, symbols, source, first_seen) now, within
    a time budget; saves them to today's events file and returns {key: event}."""
    t0, out, save = time.time(), {}, []
    for r in rows:
        if time.time() - t0 > budget_s:
            break
        ev = read_item(f"{r['title']}. {r.get('summary') or ''}",
                       timeout=min(120.0, max(10.0, budget_s - (time.time() - t0))))
        if ev is None:
            continue
        out[r["key"]] = ev
        save.append({"key": r["key"], "symbols": r.get("symbols", ""), "source": r.get("source", ""),
                     "title": r["title"], "first_seen": r.get("first_seen", ""),
                     "read_at": now_ist().isoformat(timespec="seconds"), **ev})
    if save:
        _save(save)
    return out


def run_once(max_items: int = 200, budget_s: Optional[float] = None) -> int:
    """Read the not-yet-read items of today's news archive that name a stock (all stocks,
    oldest first), stopping after budget_s seconds if given."""
    from news_analyser import day_file
    if not available():
        log.warning("Ollama or %s not available; nothing read", MODEL)
        return 0
    d = now_ist().date()
    if not os.path.exists(day_file(d)):
        return 0
    news = pd.read_parquet(day_file(d))
    news = news[news["symbols"].fillna("") != ""]
    path = events_file(d)
    done = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    seen = set(done["key"]) if len(done) else set()
    todo = news[~news["key"].isin(seen)].sort_values("first_seen").head(max_items)
    got = read_rows(todo.to_dict("records"), budget_s=budget_s or 1e9)
    log.info("read %d new items (%d waiting)", len(got), max(0, len(news) - len(seen) - len(got)))
    return len(got)


def report() -> None:
    path = events_file(now_ist().date())
    if not os.path.exists(path):
        print("no events read today")
        return
    e = pd.read_parquet(path)
    e = e[~e["event_type"].isin(["routine", "market_commentary"])]
    order = {"high": 0, "medium": 1, "low": 2}
    e = e.assign(m=e["materiality"].map(order))
    # many outlets carry the same story: one line per stock and event type, with the count,
    # the most material reading and the majority direction
    rows = []
    for (sym, et), g in e.groupby(["symbols", "event_type"]):
        g = g.sort_values(["m", "confidence"], ascending=[True, False])
        top = g.iloc[0]
        rows.append({"symbols": sym, "event_type": et, "n": len(g), "m": top["m"],
                     "materiality": top["materiality"], "direction": g["direction"].mode().iloc[0],
                     "value_cr": g["value_cr"].dropna().max() if g["value_cr"].notna().any() else None,
                     "summary": top["summary"]})
    out = pd.DataFrame(rows).sort_values(["m", "n"], ascending=[True, False]) if rows else pd.DataFrame()
    print(f"{len(e)} company items today in {len(out)} events (routine and market commentary left out):")
    for r in out.head(25).itertuples():
        val = f" Rs. {r.value_cr:,.0f} cr" if r.value_cr is not None and pd.notna(r.value_cr) else ""
        print(f"  {r.symbols:14s} {r.materiality:6s} {r.direction:8s} {r.event_type:20s}{val} "
              f"({r.n} item{'s' if r.n > 1 else ''}) | {r.summary}")


def test() -> None:
    samples = [
        "[NSE filing] Bagging/Receiving of orders/contracts: The company has received an order worth Rs. 1,250 crore "
        "from the Ministry of Defence for supply of radars.",
        "[NSE filing] Resignation of Statutory Auditors: The statutory auditors have resigned citing non-availability "
        "of information and explanations sought from the management.",
        "[NSE filing] Trading Window: Closure of trading window from October 1, 2026 till results.",
    ]
    for s in samples:
        t0 = time.time()
        ev = read_item(s)
        print(f"{time.time() - t0:5.1f}s", ev, "|", s[:70])


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(PROJECT_DIR, "logs", "llm_reader.log"), encoding="utf-8")], force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    test() if a.test else report() if a.report else run_once()

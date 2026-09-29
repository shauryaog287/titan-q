"""
main_trading_loop.py
====================
Titan-Q orchestrator: paper trading on NSE large caps.

    D:\\trader_env\\Scripts\\python.exe main_trading_loop.py                 live (default)
    D:\\trader_env\\Scripts\\python.exe main_trading_loop.py --mode replay   replay a past session
        --date YYYY-MM-DD   which archived session to replay (default: the latest)
        --speed N           replay speed in 1-minute bars per second (default 2)

LIVE MODE
    One process per trading day: the scheduled task starts it at 09:10 IST, it
    trades 09:15-15:30, and it exits 15 minutes after the close (archiving the
    day's 1-minute bars). Started on a weekend or after the close, it exits at
    once. A lock file stops a second live instance from running alongside the
    first. At start-up it refreshes daily data and builds the daily context
    (regime, selection, sizing, tail risk) from bars dated BEFORE the session.
    During the session it polls real 1-minute bars every minute, marks the
    paper account to market, scores VPIN toxicity, and rebalances toward the
    target portfolio through the execution engine. The paper account persists
    in paper_account.json; delete that file to start again from the initial
    capital.

REPLAY MODE
    Replays an archived session bar by bar on a throwaway account, with the
    context built from data before that session. Use it to watch the system
    trade outside market hours. It never touches the live paper account.

WHAT IS NOT USED, AND WHY
    The Phase 18 transformer and the Phase 16 spoofing autoencoder need
    order-book depth, and the Phase 14 Hawkes filter needs trade-by-trade
    timestamps. The data source (Yahoo Finance) provides neither, so they are
    reported as unavailable instead of being fed random numbers. The PPO policy
    and the neural SDE are used only if they beat simple baselines out of
    sample (train_models.py, models/manifest.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from titanq_config import CONFIG, INDEX_SYMBOL, LOGS_DIR, PROJECT_DIR, STATE_FILE, UNIVERSE
from market_data import ALL_SYMBOLS, IST, DailyStore, LiveBarFeed, ReplayFeed, Session, now_ist
from titanq_brain import (DailyContext, Rebalancer, ToxicityMonitor, build_daily_context,
                          save_context)
from execution_engine import EngineConfig, EngineState, ExecutionEngine, OrderIntent
from paper_broker import LIVE_ACCOUNT_FILE, PaperTradingBroker

log = logging.getLogger("titanq")


def setup_logging(mode: str) -> str:
    """Console and a dated file, so a session can be diagnosed after its window closes."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    suffix = "_replay" if mode == "replay" else ""
    path = os.path.join(LOGS_DIR, datetime.now().strftime(f"titanq_%Y-%m-%d{suffix}.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # it logs every missing ticker
    return path


LOCK_FILE = os.path.join(PROJECT_DIR, "titanq_live.lock")
SHUTDOWN_AFTER_CLOSE = timedelta(minutes=15)
# Graceful restart: create this file (any content = the reason) and the live loop finishes
# its queued orders, saves, and exits with RESTART_EXIT_CODE; run_live.py then starts it
# again on the current code. Used to apply a fix mid-session without killing a process.
RESTART_FILE = os.path.join(PROJECT_DIR, "restart_trading_loop.request")
RESTART_EXIT_CODE = 75
CONTEXT_RETRY = timedelta(minutes=10)      # rebuild an invalid context this often in session
_exit_code = 0


def acquire_live_lock():
    """Exclusive OS-level lock; released automatically when the process exits.
    Returns the open file (keep a reference) or None if another live instance
    holds it."""
    f = open(LOCK_FILE, "a+")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()} started {datetime.now(IST).isoformat(timespec='seconds')}\n")
    f.flush()
    return f


def start_forward_test() -> None:
    """Launch the after-close forward-test update (reports/forward_test/) as its own
    process, so this session can exit while it downloads the day's closes."""
    import subprocess
    import sys
    try:
        out = open(os.path.join(LOGS_DIR, datetime.now().strftime("forward_test_%Y-%m-%d.log")), "a")
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen([sys.executable, os.path.join(PROJECT_DIR, "forward_test.py")], cwd=PROJECT_DIR,
                         stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
        log.info("forward-test update started (log: logs\\forward_test_%s.log)", datetime.now().strftime("%Y-%m-%d"))
    except Exception as exc:
        log.warning("could not start the forward-test update: %s", exc)


def write_json_atomic(path: str, obj: Dict) -> None:
    """Write-then-rename, so the dashboard never reads a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="ascii") as f:
        json.dump(obj, f, default=float)
    os.replace(tmp, path)


def radar_overlay(ctx: DailyContext) -> Dict[str, List[str]]:
    """Public-disclosure red flags (disclosure_radar.py) for the stocks in today's plan.
    OBSERVE mode: logged and shown on the dashboard, never used to change a trade. No flag
    has passed a pre-registered test yet (INS1 insider buys failed its test; INS2 pledge
    invocations did not qualify); a flag gets a say only after one does."""
    try:
        from disclosure_radar import latest_flags
        day, flags = latest_flags()
    except Exception as exc:
        log.warning("[RADAR] not available: %s", exc)
        return {}
    if day is None:
        log.info("[RADAR] no recent disclosure data collected (disclosure_radar.py runs 19:45 weekdays)")
        return {}
    hits = {s: flags[s[:-3]] for s in ctx.target_weights if s[:-3] in flags}
    log.info("[RADAR] disclosures of %s: %d of %d stocks in the plan flagged (observe mode, no effect "
             "on trades)", day, len(hits), len(ctx.target_weights))
    for s, reasons in sorted(hits.items()):
        log.info("[RADAR] %s: %s", s, "; ".join(reasons)[:300])
    return hits


NEWS_REFRESH = timedelta(minutes=30)
STALE_ALERT = timedelta(minutes=15)


def alert(mode: str, title: str, message: str, priority: str = "default", tags: str = "",
          repeat_minutes: float = 30.0) -> None:
    """Phone alert (alerts.py, ntfy) for the live session only; never raises."""
    if mode != "live":
        return
    try:
        from alerts import send
        send(title, message, priority=priority, tags=tags, repeat_minutes=repeat_minutes)
    except Exception:
        pass


def day_open_equity(day: str, equity: float) -> float:
    """The paper equity at the first trading step of `day`, kept in a file so a restart
    mid-session cannot reset the daily loss limit."""
    path = os.path.join(LOGS_DIR, f"day_open_equity_{day}.json")
    if os.path.exists(path):
        with open(path, encoding="ascii") as f:
            return float(json.load(f)["equity"])
    with open(path, "w", encoding="ascii") as f:
        json.dump({"equity": equity}, f)
    return equity


def news_overlay(symbols) -> Dict[str, Dict]:
    """Headline tone (news_analyser.py, FinBERT) of the last 24 h for the given stocks.
    OBSERVE mode like the radar: logged and exported, never used to change a trade, until
    news tone passes a pre-registered test."""
    try:
        from news_analyser import STRONG_NEGATIVE, symbol_summary
        summ = symbol_summary([s[:-3] for s in symbols])
    except Exception as exc:
        log.warning("[NEWS] not available: %s", exc)
        return {}
    neg = {s: v for s, v in summ.items() if v["worst_score"] <= STRONG_NEGATIVE}
    log.info("[NEWS] last 24 h: %d of %d stocks in the plan have news, %d with a strongly negative "
             "headline (observe mode, no effect on trades)", len(summ), len(symbols), len(neg))
    for s, v in sorted(neg.items(), key=lambda kv: kv[1]["agg_score"]):
        log.info("[NEWS] %s aggregated %+.2f (confidence %.2f, %d stories); worst %+.2f: %s", s,
                 v["agg_score"], v["confidence"], v["stories"], v["worst_score"], v["worst_title"][:130])
    return summ


def attention_overlay(symbols) -> Dict[str, Dict]:
    """Retail attention for the plan's stocks: Google Trends search spikes and Reddit
    mentions. OBSERVE mode, like news and the radar: logged and exported only."""
    out: Dict[str, Dict] = {}
    names = [s[:-3] for s in symbols]
    try:
        from trends_collector import SPIKE_RATIO, attention
        att = attention()
        spikes = {s: v for s, v in att.items() if s in names and v["ratio"] >= SPIKE_RATIO}
        log.info("[ATTENTION] Google Trends: %d of %d stocks with data, %d search spikes (>= x%.1f; observe "
                 "mode)", len([s for s in names if s in att]), len(names), len(spikes), SPIKE_RATIO)
        for s, v in sorted(spikes.items(), key=lambda kv: -kv[1]["ratio"]):
            log.info("[ATTENTION] %s searches x%.2f vs usual", s, v["ratio"])
        for s in names:
            if s in att:
                out.setdefault(s, {})["search_ratio"] = att[s]["ratio"]
    except Exception as exc:
        log.warning("[ATTENTION] Trends not available: %s", exc)
    try:
        from reddit_collector import mentions
        m = mentions()
        for s in names:
            if s in m:
                out.setdefault(s, {}).update({"reddit_mentions": m[s]["n"], "reddit_tone": m[s]["tone"]})
        if m:
            top = sorted(((s, v) for s, v in m.items() if s in names), key=lambda kv: -kv[1]["n"])[:5]
            log.info("[ATTENTION] Reddit, last 24 h: %s", ", ".join(f"{s} {v['n']}" for s, v in top) or "none")
    except Exception as exc:
        log.warning("[ATTENTION] Reddit not available: %s", exc)
    return out


ANGEL_SHADOW_EVERY = timedelta(minutes=5)


def angel_shadow(symbols, yahoo_price, now_ts: float) -> Optional[Dict]:
    """Compare the Angel recorder's latest prices (data/ticks/latest_prices.json) with the
    Yahoo feed the loop trades on. SHADOW only: nothing trades on Angel prices until a few
    sessions show they agree and are fresher."""
    from angel_feed import LATEST_FILE, TOKEN_MAP_FILE
    if not (os.path.exists(LATEST_FILE) and os.path.exists(TOKEN_MAP_FILE)):
        return None
    with open(TOKEN_MAP_FILE, encoding="ascii") as f:
        by_yahoo = {y: t for t, y in json.load(f).items()}
    with open(LATEST_FILE, encoding="ascii") as f:
        latest = json.load(f)
    diffs, ages, missing = [], [], 0
    for s in symbols:
        a = latest["prices"].get(by_yahoo.get(s, ""))
        y = yahoo_price(s)
        if not a or not a.get("ltp") or not y:
            missing += 1
            continue
        diffs.append((abs(a["ltp"] / y - 1) * 1e4, s))
        ages.append(now_ts - (a.get("exch_ts") or a["recv_ts"]))
    if not diffs:
        return {"compared": 0, "missing": missing, "file_age_s": round(now_ts - latest["written"], 1)}
    d = sorted(diffs)
    return {"compared": len(d), "missing": missing, "median_diff_bps": round(d[len(d) // 2][0], 1),
            "max_diff_bps": round(d[-1][0], 1), "max_diff_symbol": d[-1][1],
            "median_angel_age_s": round(float(np.median(ages)), 1),
            "file_age_s": round(now_ts - latest["written"], 1)}


class Orchestrator:
    def __init__(self, mode: str, feed, broker: PaperTradingBroker, engine: ExecutionEngine,
                 store: DailyStore, session: Session) -> None:
        self.mode = mode
        self.feed = feed
        self.broker = broker
        self.engine = engine
        self.store = store
        self.session = session
        self.ctx: Optional[DailyContext] = None
        self.daily: Dict = {}
        self.rebalancer = Rebalancer(CONFIG, session)
        self.tox = ToxicityMonitor(CONFIG)
        self.toxicity: Dict[str, Optional[float]] = {}
        self.vpin: Dict[str, Optional[float]] = {}
        self.status = ""
        self.notes: List[str] = []
        self._seen_notes: set = set()
        self._ctx_built_at: Optional[datetime] = None
        self._started = datetime.now().timestamp()
        self._open_equity: Optional[float] = None
        self.halted = False
        self._stale_since: Optional[datetime] = None

    # ---- daily context ------------------------------------------------------

    async def build_context(self) -> None:
        log.info("[CONTEXT] building for session %s ...", self.feed.session_date)
        self.daily = await asyncio.to_thread(self.store.load, ALL_SYMBOLS, self.mode == "live")
        await asyncio.to_thread(self.prepare_account)
        self.ctx = await asyncio.to_thread(build_daily_context, self.daily, self.feed.session_date,
                                           self.broker.equity(), holdings=sorted(self.broker.positions))
        save_context(self.ctx)
        self._ctx_built_at = self.feed.now()
        self._seen_notes.clear()
        log.info("[CONTEXT] %s", self.ctx.summary())
        for name, st in self.ctx.components.items():
            log.info("[COMPONENT] %-20s %s", name, st)
        for r in self.ctx.reasons:
            log.warning("[CONTEXT] %s", r)
        self.radar = radar_overlay(self.ctx)
        self.news = news_overlay(list(self.ctx.target_weights))
        self._news_at = self.feed.now()
        self.attention = attention_overlay(list(self.ctx.target_weights))
        if self.context_degraded():
            alert(self.mode, "plan not valid", "Today's plan cannot trade: " + "; ".join(self.ctx.reasons)[:300]
                  + ". It is rebuilt every 10 minutes.", "high", "warning", repeat_minutes=60)
        else:
            alert(self.mode, "plan ready", f"{self.ctx.regime}, exposure {100 * self.ctx.exposure:.0f}% in "
                  f"{len(self.ctx.target_weights)} stocks; equity Rs. {self.broker.equity():,.0f}.", "low",
                  "chart_with_upwards_trend", repeat_minutes=240)

    radar: Dict[str, List[str]] = {}
    news: Dict[str, Dict] = {}
    attention: Dict[str, Dict] = {}
    shadow: Dict = {}
    _shadow_at: Optional[datetime] = None
    _news_at: Optional[datetime] = None

    def _adv(self, symbol: str) -> Optional[float]:
        if self.ctx and symbol in self.ctx.adv_shares:
            return self.ctx.adv_shares[symbol]
        df = self.daily.get(symbol)
        if df is None or not len(df):
            return None
        return float(df[df.index < str(self.feed.session_date)]["Volume"].iloc[-20:].median())

    # ---- per cycle ------------------------------------------------------------

    def _symbols(self) -> List[str]:
        return sorted(set(self.ctx.target_weights if self.ctx else {}) | set(self.broker.positions))

    def update_toxicity(self) -> None:
        for s in self._symbols():
            self.toxicity[s], self.vpin[s] = self.tox.score(
                s, self.feed.bars(s), self._adv(s), self.feed.session_date)

    def set_status(self, status: str) -> None:
        if status != self.status:
            log.info("[STATUS] %s", status)
            self.status = status

    def step(self, now: datetime) -> None:
        if not self.session.is_open(now):
            self.set_status("market closed; next open %s" %
                            self.session.next_open(now).strftime("%a %d %b %H:%M IST"))
            return
        if self.feed.is_stale():
            self.set_status("waiting: no new 1-minute bar (feed stale or exchange holiday)")
            self._stale_since = self._stale_since or now
            if now - self._stale_since >= STALE_ALERT:
                alert(self.mode, "price feed stale", f"No new 1-minute bars since "
                      f"{self._stale_since.strftime('%H:%M')} (network or VPN down?). Trading waits.",
                      "high", "warning")
            return
        self._stale_since = None
        if self.engine.state is not EngineState.LIVE:
            self.set_status(f"waiting: execution engine is {self.engine.state.name}")
            return
        if not self.ctx.tradeable:
            self.set_status("NOT TRADING: " + "; ".join(self.ctx.reasons))
            return
        if len(self.engine.intents):
            return                                  # let queued orders execute first
        eq = self.broker.equity()
        if self._open_equity is None:
            self._open_equity = day_open_equity(str(now.date()), eq) if self.mode == "live" else eq
        if not self.halted and eq < self._open_equity * (1 - CONFIG.daily_loss_halt):
            self.halted = True
            log.error("[RISK] daily loss limit: equity Rs. %s is %.1f%% below today's open Rs. %s; "
                      "no new orders today", f"{eq:,.2f}", 100 * (1 - eq / self._open_equity),
                      f"{self._open_equity:,.2f}")
            alert(self.mode, "daily loss limit hit", f"Paper equity Rs. {eq:,.0f} is "
                  f"{100 * (1 - eq / self._open_equity):.1f}% below the open. No new orders today.",
                  "urgent", "rotating_light", repeat_minutes=600)
        if self.halted:
            self.set_status(f"HALTED: daily loss limit ({100 * CONFIG.daily_loss_halt:.0f}%); no new orders today")
            return
        self.set_status("trading")

        positions = dict(self.broker.positions)
        prices = {s: self.feed.last_price(s) for s in self._symbols()}
        orders, notes = self.rebalancer.plan(
            self.ctx.target_weights, positions, prices, self.broker.capital,
            self.broker.equity(), now, self.toxicity)
        for n in notes:
            if n not in self._seen_notes:
                self._seen_notes.add(n)
                log.info("[REBALANCE] %s", n)
        self.notes = (notes + self.notes)[:10]
        for o in orders:
            log.info("[REBALANCE] %s %d %s @ ~Rs. %.2f (%s)", o.side, o.qty, o.symbol, o.price, o.reason)
            self.engine.intents.put(OrderIntent(symbol=o.symbol, side=o.side, quantity=o.qty,
                                                order_type="MARKET", reduce_only=o.reduce_only,
                                                decision_price=o.price))

    def export(self, now: datetime) -> None:
        snap = self.broker.snapshot()
        ctx = self.ctx
        tox_vals = [v for v in self.toxicity.values() if v is not None]
        state = {
            # the contract index.html reads
            "current_pnl": snap["pnl"],
            "toxicity_score": max(tox_vals) if tox_vals else 0.0,
            # the dashboard shows up to 8 assets: send the 8 largest weights
            "quantum_assets": [{"ticker": s, "w": w} for s, w in sorted(
                (ctx.basket_weights.items() if ctx else []), key=lambda kv: -kv[1])[:8]],
            "recent_trades": [{"id": t["id"], "side": t["side"], "ticker": t["ticker"],
                               "price": t["price"], "qty": t["qty"], "ts": t["ts"]}
                              for t in self.broker.recent_trades()],
            # extra detail for anyone reading the JSON
            "mode": self.mode, "status": self.status,
            "market_time": now.isoformat(timespec="seconds"),
            "session_open": self.session.is_open(now),
            "engine_state": self.engine.state.name,
            "equity": snap["equity"], "cash": snap["cash"],
            "initial_capital": self.broker.initial_capital,
            "realised_pnl": snap["realised_pnl"], "fees_paid": snap["fees_paid"],
            "positions": snap["positions"],
            "exposure_target": ctx.exposure if ctx else None,
            "regime": ctx.regime if ctx else None,
            "target_weights": ctx.target_weights if ctx else {},
            "toxicity_by_symbol": self.toxicity, "vpin_by_symbol": self.vpin,
            "components": ctx.components if ctx else {},
            "halted_daily_loss_limit": self.halted,
            "radar_flags_observe_only": self.radar,
            "news_observe_only": self.news,
            "attention_observe_only": self.attention,
            "angel_shadow": self.shadow,
            "notes": self.notes,
        }
        try:
            write_json_atomic(STATE_FILE, state)
        except Exception as exc:
            log.warning("dashboard export failed: %s", exc)

    def prepare_account(self) -> None:
        """Before planning: book splits/bonuses/dividends on held stocks (live only), and set
        today's circuit limits from yesterday's closes and NSE's price bands."""
        from corporate_actions import apply_to_account, circuit_limits, load_bands
        if self.mode == "live" and self.broker.positions:
            for m in apply_to_account(self.broker):
                log.info("[ACCOUNT] %s", m)
        try:
            day = pd.Timestamp(self.feed.session_date)
            prev = {s: float(df["Close"][df.index < day].dropna().iloc[-1])
                    for s, df in self.daily.items() if len(df) and (df.index < day).any()}
            limits = circuit_limits(prev, load_bands())
            self.broker.limits_fn = limits.get
            log.info("[ACCOUNT] circuit limits set for %d of %d stocks (the rest have no price band)",
                     len(limits), len(prev))
        except Exception as exc:
            log.warning("[ACCOUNT] circuit limits not set: %s", exc)

    def context_degraded(self) -> bool:
        """An invalid plan (bad data, a failed model load) is retried, not kept all day."""
        return self.ctx is None or not self.ctx.tradeable or self.ctx.kelly_binding == "invalid_input"

    def restart_requested(self) -> Optional[str]:
        """Reason text if a restart request newer than this process exists."""
        if self.mode != "live" or not os.path.exists(RESTART_FILE):
            return None
        if os.path.getmtime(RESTART_FILE) < self._started:
            os.remove(RESTART_FILE)                  # left over from before this process
            return None
        with open(RESTART_FILE, encoding="ascii", errors="replace") as f:
            reason = f.read().strip()[:200] or "no reason given"
        os.remove(RESTART_FILE)
        return reason

    async def run(self) -> None:
        global _exit_code
        await self.build_context()
        self.update_toxicity()
        while True:
            try:
                new_bar = await self.feed.poll()
                now = self.feed.now()
                if (self.mode == "live" and self.session.is_open(now)
                        and self.ctx.asof != str(now.date())):
                    await self.build_context()      # a new session began while running
                elif (self.mode == "live" and self.session.is_open(now) and self.context_degraded()
                        and now - self._ctx_built_at >= CONTEXT_RETRY):
                    log.warning("[CONTEXT] plan is not valid; rebuilding with fresh data")
                    await self.build_context()
                if (self.mode == "live" and self.session.is_open(now) and self._news_at is not None
                        and now - self._news_at >= NEWS_REFRESH):
                    self.news = await asyncio.to_thread(news_overlay, list(self.ctx.target_weights))
                    self._news_at = now
                if (self.mode == "live" and self.session.is_open(now)
                        and (self._shadow_at is None or now - self._shadow_at >= ANGEL_SHADOW_EVERY)):
                    self._shadow_at = now
                    try:
                        self.shadow = angel_shadow(list(self.ctx.target_weights), self.feed.last_price,
                                                   now.timestamp()) or {}
                        if self.shadow:
                            log.info("[ANGEL SHADOW] %s", self.shadow)
                    except Exception as exc:
                        log.warning("[ANGEL SHADOW] failed: %s", exc)
                if not len(self.engine.intents):
                    reason = self.restart_requested()
                    if reason:
                        log.warning("[CONTROL] restart requested (%s): saving and exiting so "
                                    "run_live.py restarts on the current code", reason)
                        self.finish(f"RESTART REQUESTED ({reason})")
                        _exit_code = RESTART_EXIT_CODE
                        self.engine._running = False
                        return
                if new_bar:
                    self.update_toxicity()
                self.step(now)
                self.export(now)
                if self.mode == "replay" and self.feed.finished and not len(self.engine.intents):
                    await asyncio.sleep(1.0)        # let the last fills land
                    self.finish(f"REPLAY OF {self.feed.session_date} COMPLETE")
                    self.engine._running = False     # stops engine.run(), which then logs out
                    return
                if self.mode == "live" and now >= (self.session.bounds(now.date())[1]
                                                   + SHUTDOWN_AFTER_CLOSE):
                    await asyncio.to_thread(self.feed.store.archive, self.feed._bars)
                    self.finish(f"SESSION {now.date()} CLOSED")
                    snap = self.broker.snapshot()
                    today = [t for t in self.broker.trade_history if t.get("time", "").startswith(str(now.date()))]
                    day_pnl = snap["equity"] - (self._open_equity or snap["equity"])
                    alert(self.mode, "session closed", f"Equity Rs. {snap['equity']:,.0f} (today Rs. "
                          f"{day_pnl:+,.0f}; since start Rs. {snap['pnl']:+,.0f}), {len(today)} trades today, "
                          f"{len(snap['positions'])} holdings.", "default", "bar_chart", repeat_minutes=600)
                    start_forward_test()
                    self.engine._running = False
                    return
            except Exception:
                log.exception("orchestrator cycle error")
            await asyncio.sleep(CONFIG.cycle_seconds if self.mode == "live" else 0.25)

    def finish(self, title: str) -> None:
        snap = self.broker.snapshot()
        log.info("=" * 60)
        log.info(title)
        log.info(" trades: %d | fees: Rs. %s | equity: Rs. %s | PnL: Rs. %s",
                 len(self.broker.trade_history), f"{snap['fees_paid']:,.2f}",
                 f"{snap['equity']:,.2f}", f"{snap['pnl']:,.2f}")
        for p in snap["positions"]:
            log.info(" holding %-14s %5d @ avg Rs. %.2f, last Rs. %s", p["ticker"], p["qty"],
                     p["avg_cost"], p["price"])
        log.info("=" * 60)


async def main(args) -> None:
    session = Session()
    lock = None
    if args.mode == "live":
        now = now_ist()
        close = session.bounds(now.date())[1]
        if now.weekday() >= 5 or now >= close + SHUTDOWN_AFTER_CLOSE:
            log.info("Market closed for today (%s). Next session opens %s.",
                     now.strftime("%a %d %b %H:%M IST"),
                     session.next_open(now).strftime("%a %d %b %H:%M IST"))
            log.info("Archiving recent 1-minute bars, then exiting. To watch the system trade "
                     "now, run: main_trading_loop.py --mode replay")
            await LiveBarFeed(session=session).start()
            return
        lock = acquire_live_lock()
        if lock is None:
            log.error("Another live Titan-Q instance is already running (lock %s). Exiting.", LOCK_FILE)
            return
    if args.mode == "replay":
        feed = ReplayFeed(session_date=date.fromisoformat(args.date) if args.date else None,
                          bars_per_second=args.speed)
        state_file = None                     # throwaway account
    else:
        feed = LiveBarFeed(session=session)
        state_file = LIVE_ACCOUNT_FILE
    broker = PaperTradingBroker(CONFIG.initial_capital, price_fn=feed.last_price,
                                clock_fn=feed.now, state_file=state_file)
    log.info("=" * 60)
    log.info("TITAN-Q  mode=%s  session=%s  account=%s", args.mode, feed.session_date,
             state_file or "in-memory")
    log.info("=" * 60)
    await feed.start()

    engine = ExecutionEngine(broker=broker,
                             config=EngineConfig(algo_id="TITAN-Q-PAPER", warmup_bars_required=0))
    broker.fill_callback = engine.on_fill
    engine.local_positions.update(broker.positions)   # a restarted account is not a discrepancy
    orch = Orchestrator(args.mode, feed, broker, engine, DailyStore(), session)
    await asyncio.gather(engine.run(), orch.run())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Titan-Q paper trading orchestrator")
    ap.add_argument("--mode", choices=("live", "replay"), default="live")
    ap.add_argument("--date", default=None, help="replay: session date YYYY-MM-DD")
    ap.add_argument("--speed", type=float, default=CONFIG.replay_bars_per_second,
                    help="replay: 1-minute bars per second")
    args = ap.parse_args()
    log_path = setup_logging(args.mode)
    log.info("logging to %s", log_path)
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("Shutting down...")
    sys.exit(_exit_code)

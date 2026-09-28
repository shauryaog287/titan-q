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
import tempfile
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

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

    # ---- daily context ------------------------------------------------------

    async def build_context(self) -> None:
        log.info("[CONTEXT] building for session %s ...", self.feed.session_date)
        self.daily = await asyncio.to_thread(self.store.load, ALL_SYMBOLS, self.mode == "live")
        self.ctx = await asyncio.to_thread(build_daily_context, self.daily, self.feed.session_date,
                                           self.broker.equity(), holdings=sorted(self.broker.positions))
        save_context(self.ctx)
        self._seen_notes.clear()
        log.info("[CONTEXT] %s", self.ctx.summary())
        for name, st in self.ctx.components.items():
            log.info("[COMPONENT] %-20s %s", name, st)
        for r in self.ctx.reasons:
            log.warning("[CONTEXT] %s", r)

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
            return
        if self.engine.state is not EngineState.LIVE:
            self.set_status(f"waiting: execution engine is {self.engine.state.name}")
            return
        if not self.ctx.tradeable:
            self.set_status("NOT TRADING: " + "; ".join(self.ctx.reasons))
            return
        if len(self.engine.intents):
            return                                  # let queued orders execute first
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
            "notes": self.notes,
        }
        try:
            write_json_atomic(STATE_FILE, state)
        except Exception as exc:
            log.warning("dashboard export failed: %s", exc)

    async def run(self) -> None:
        await self.build_context()
        self.update_toxicity()
        while True:
            try:
                new_bar = await self.feed.poll()
                now = self.feed.now()
                if (self.mode == "live" and self.session.is_open(now)
                        and self.ctx.asof != str(now.date())):
                    await self.build_context()      # a new session began while running
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

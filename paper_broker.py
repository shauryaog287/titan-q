"""
paper_broker.py
===============
Paper trading broker. Executes the engine's orders with fake money at prices
from the market data feed, charges the Indian equity-delivery cost stack, and
keeps a ledger that survives restarts.

    fills       immediately, at the feed's last price plus/minus slippage
                (buys pay up, sells receive less)
    costs       STT on both sides, exchange and SEBI fees, stamp duty on buys,
                GST on brokerage + exchange + SEBI (rates in titanq_config)
    rejections  raise PaperOrderRejected, so the engine records REJECTED
                (no price, not enough cash, selling shares not held)
    ledger      cash, positions, average cost, realised PnL, fees, trade
                history. In live mode it is saved to paper_account.json after
                every fill and reloaded at start-up; delete that file to reset
                the account to the initial capital.

It implements the full BrokerClient protocol of execution_engine.py, and
reports every fill to the engine through `fill_callback` (engine.on_fill), so
the engine's position ledger always matches this one.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from titanq_config import CONFIG, Config, PROJECT_DIR
from market_data import IST

log = logging.getLogger("paper_broker")

LIVE_ACCOUNT_FILE = os.path.join(PROJECT_DIR, "paper_account.json")


class PaperOrderRejected(Exception):
    """The paper broker refused the order; the engine records it as REJECTED."""


def _atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="ascii") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class PaperTradingBroker:
    def __init__(self, initial_capital: float = CONFIG.initial_capital,
                 price_fn: Optional[Callable[[str], Optional[float]]] = None,
                 clock_fn: Optional[Callable[[], datetime]] = None,
                 cfg: Config = CONFIG, state_file: Optional[str] = None) -> None:
        self.cfg = cfg
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)          # cash (name kept for older callers)
        self.positions: Dict[str, int] = {}
        self.avg_cost: Dict[str, float] = {}
        self.realised_pnl = 0.0
        self.fees_paid = 0.0
        self.trade_history: List[dict] = []
        self._orders: Dict[str, dict] = {}
        self._seq = 0
        self.price_fn = price_fn
        self.clock_fn = clock_fn or (lambda: datetime.now(IST))
        # Set to ExecutionEngine.on_fill so the engine's position ledger sees our fills.
        # Without it the engine's reconciler finds a mismatch and halts all trading.
        self.fill_callback: Optional[Callable[[str, int, float], None]] = None
        # symbol -> (lower, upper) circuit limits for today, or None (no band). A buy at the
        # upper limit or a sell at the lower one cannot fill on NSE, so it is rejected here.
        self.limits_fn: Optional[Callable[[str], Optional[Tuple[float, float]]]] = None
        self.ca_applied: List[str] = []               # corporate actions already booked
        self.dividends_received = 0.0
        self.state_file = state_file
        if state_file and os.path.exists(state_file):
            self._load()

    # ---- persistence ---------------------------------------------------------

    def _load(self) -> None:
        with open(self.state_file, "r", encoding="ascii") as f:
            s = json.load(f)
        self.initial_capital = float(s["initial_capital"])
        self.capital = float(s["cash"])
        self.positions = {k: int(v) for k, v in s["positions"].items() if int(v)}
        self.avg_cost = {k: float(v) for k, v in s["avg_cost"].items() if k in self.positions}
        self.realised_pnl = float(s.get("realised_pnl", 0.0))
        self.fees_paid = float(s.get("fees_paid", 0.0))
        self.trade_history = list(s.get("trades", []))
        self._seq = int(s.get("seq", len(self.trade_history)))
        self.ca_applied = list(s.get("ca_applied", []))
        self.dividends_received = float(s.get("dividends_received", 0.0))
        log.info("paper account loaded from %s: cash Rs. %s, %d positions",
                 self.state_file, f"{self.capital:,.2f}", len(self.positions))

    def save(self) -> None:
        if not self.state_file:
            return
        _atomic_write_json(self.state_file, {
            "initial_capital": self.initial_capital, "cash": self.capital,
            "positions": self.positions, "avg_cost": self.avg_cost,
            "realised_pnl": self.realised_pnl, "fees_paid": self.fees_paid,
            "trades": self.trade_history[-1000:], "seq": self._seq,
            "ca_applied": self.ca_applied[-2000:], "dividends_received": self.dividends_received,
            "updated": datetime.now(IST).isoformat(timespec="seconds"),
        })

    # ---- corporate actions -------------------------------------------------------

    def held_since(self, symbol: str) -> Optional[str]:
        """Date (YYYY-MM-DD) of the buy that opened the current position, from the ledger."""
        if symbol not in self.positions:
            return None
        q, since = 0, None
        for t in self.trade_history:
            if t["ticker"] != symbol:
                continue
            if q == 0 and t["side"] == "BUY":
                since = t["time"][:10]
            q += t["qty"] if t["side"] == "BUY" else -t["qty"]
        return since

    def apply_corporate_action(self, symbol: str, ex_date: str, kind: str, value: float) -> Optional[str]:
        """Book a split/bonus (kind "split", value = new shares per old share, as Yahoo
        reports it: 5.0 for a 1:5 split, 2.0 for a 1:1 bonus) or a cash dividend (kind
        "dividend", value = Rs. per share) on a position held before the ex-date. Each
        action is booked once. Returns a description, or None if it does not apply."""
        key = f"{symbol}|{ex_date}|{kind}|{value:g}"
        since = self.held_since(symbol)
        if key in self.ca_applied or since is None or not since < ex_date or value <= 0:
            return None
        q = self.positions[symbol]
        if kind == "split":
            new_q = int(math.floor(q * value + 1e-9))
            frac = q * value - new_q                       # cash in lieu of fractional shares
            cost = q * self.avg_cost.get(symbol, 0.0)
            px = self._price(symbol) or (self.avg_cost.get(symbol, 0.0) / value)
            self.positions[symbol] = new_q
            self.avg_cost[symbol] = cost / (q * value) if q * value > 0 else 0.0
            self.capital += frac * px
            msg = f"{symbol}: split/bonus x{value:g} ex {ex_date}: {q} -> {new_q} shares"
        elif kind == "dividend":
            cash = q * value
            self.capital += cash
            self.dividends_received += cash
            msg = f"{symbol}: dividend Rs. {value:g}/share ex {ex_date} on {q} shares = Rs. {cash:,.2f}"
        else:
            return None
        self.ca_applied.append(key)
        log.info("[CORPORATE ACTION] %s", msg)
        self.save()
        return msg

    # ---- costs and valuation ------------------------------------------------------

    def fees(self, side: str, value: float) -> float:
        c = self.cfg
        brokerage = value * c.brokerage_rate
        exchange = value * c.exchange_rate
        sebi = value * c.sebi_rate
        stt = value * c.stt_rate
        stamp = value * c.stamp_rate_buy if side == "BUY" else 0.0
        gst = c.gst_rate * (brokerage + exchange + sebi)
        return brokerage + exchange + sebi + stt + stamp + gst

    def _price(self, symbol: str) -> Optional[float]:
        if self.price_fn is None:
            return None
        try:
            p = self.price_fn(symbol)
        except Exception:
            return None
        return float(p) if p and p > 0 else None

    def equity(self) -> float:
        return self.capital + sum(q * (self._price(s) or self.avg_cost.get(s, 0.0))
                                  for s, q in self.positions.items())

    def snapshot(self) -> Dict:
        rows = []
        for s, q in sorted(self.positions.items()):
            px = self._price(s)
            avg = self.avg_cost.get(s, 0.0)
            rows.append({"ticker": s, "qty": q, "avg_cost": round(avg, 2),
                         "price": round(px, 2) if px else None,
                         "value": round(q * (px or avg), 2),
                         "unrealised_pnl": round(q * ((px or avg) - avg), 2)})
        eq = self.equity()
        return {"cash": round(self.capital, 2), "equity": round(eq, 2),
                "pnl": round(eq - self.initial_capital, 2),
                "realised_pnl": round(self.realised_pnl, 2), "fees_paid": round(self.fees_paid, 2),
                "positions": rows}

    def recent_trades(self, n: int = CONFIG.recent_trades_shown) -> List[dict]:
        return self.trade_history[-n:]

    # ---- BrokerClient protocol -----------------------------------------------------

    async def authenticate(self) -> str:
        log.info("=========================================")
        log.info(" PAPER TRADING ENGINE ONLINE")
        log.info(" Cash: Rs. %s | positions: %d | account: %s", f"{self.capital:,.2f}",
                 len(self.positions), self.state_file or "in-memory (replay)")
        log.info("=========================================")
        return "PAPER_TOKEN"

    async def logout(self) -> None:
        self.save()

    async def get_quote(self, symbol: str) -> dict:
        p = self._price(symbol)
        return {"last": p if p is not None else 0.0}

    async def get_positions(self) -> Dict[str, int]:
        return {s: q for s, q in self.positions.items() if q}

    async def get_working_orders(self) -> Dict[str, dict]:
        return {}                         # paper fills are immediate

    async def cancel_order(self, broker_order_id: str) -> None:
        return None                       # nothing is ever left working

    async def get_order_by_client_id(self, client_order_id: str) -> Optional[dict]:
        return self._orders.get(client_order_id)

    async def place_order(self, intent, algo_id: str) -> str:
        symbol, side, qty = intent.symbol, intent.side, int(intent.quantity)
        if qty <= 0:
            raise PaperOrderRejected(f"{symbol}: quantity {qty}")
        px = self._price(symbol)
        if px is None:
            raise PaperOrderRejected(f"{symbol}: no price from the feed")
        lim = self.limits_fn(symbol) if self.limits_fn is not None else None
        if lim:
            lo, hi = lim
            if side == "BUY" and px >= hi * (1 - 1e-4):
                raise PaperOrderRejected(f"{symbol}: at the upper circuit (Rs. {hi:,.2f}); no sellers")
            if side == "SELL" and px <= lo * (1 + 1e-4):
                raise PaperOrderRejected(f"{symbol}: at the lower circuit (Rs. {lo:,.2f}); no buyers")
        slip = self.cfg.slippage_bps / 1e4
        fill = px * (1 + slip) if side == "BUY" else px * (1 - slip)
        value = fill * qty
        fee = self.fees(side, value)

        if side == "BUY":
            if self.capital < value + fee:
                raise PaperOrderRejected(
                    f"{symbol}: insufficient funds, need Rs. {value + fee:,.2f}, have Rs. {self.capital:,.2f}")
            held = self.positions.get(symbol, 0)
            self.avg_cost[symbol] = (held * self.avg_cost.get(symbol, 0.0) + value) / (held + qty)
            self.positions[symbol] = held + qty
            self.capital -= value + fee
        elif side == "SELL":
            held = self.positions.get(symbol, 0)
            if held < qty:
                raise PaperOrderRejected(f"{symbol}: selling {qty}, holding {held}")
            self.realised_pnl += (fill - self.avg_cost.get(symbol, fill)) * qty
            self.positions[symbol] = held - qty
            if self.positions[symbol] == 0:
                del self.positions[symbol]
                self.avg_cost.pop(symbol, None)
            self.capital += value - fee
        else:
            raise PaperOrderRejected(f"bad side {side}")

        self.fees_paid += fee
        self._seq += 1
        order_id = f"PAPER_{self._seq:06d}"
        ts = self.clock_fn()
        self.trade_history.append({"id": order_id, "client_order_id": intent.client_order_id,
                                   "ts": ts.timestamp(), "time": ts.isoformat(timespec="seconds"),
                                   "side": side, "ticker": symbol, "qty": qty,
                                   "price": round(fill, 2), "fees": round(fee, 2)})
        self._orders[intent.client_order_id] = {"order_id": order_id, "status": "COMPLETE",
                                                "filled_quantity": qty, "average_price": fill}
        log.info("[PAPER TRADE] %s %d %s @ Rs. %.2f | Fees: Rs. %.2f | Cash: Rs. %s",
                 side, qty, symbol, fill, fee, f"{self.capital:,.2f}")
        self.save()
        if self.fill_callback is not None:
            self.fill_callback(intent.client_order_id, qty, fill)
        return order_id


# ===========================================================================
# SELF-TESTS
# ===========================================================================

def _test_ledger_costs_and_persistence() -> None:
    import asyncio
    from types import SimpleNamespace

    prices = {"AAA.NS": 1000.0}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "acct.json")
        fills = []
        b = PaperTradingBroker(100_000.0, price_fn=prices.get, state_file=path)
        b.fill_callback = lambda cid, q, p: fills.append((cid, q, p))

        def intent(side, q, cid):
            return SimpleNamespace(symbol="AAA.NS", side=side, quantity=q, client_order_id=cid)

        asyncio.run(b.place_order(intent("BUY", 50, "c1"), "T"))
        buy_px = 1000.0 * (1 + CONFIG.slippage_bps / 1e4)
        buy_fee = b.fees("BUY", 50 * buy_px)
        assert abs(b.capital - (100_000 - 50 * buy_px - buy_fee)) < 1e-9
        assert b.positions == {"AAA.NS": 50} and fills[0][:2] == ("c1", 50)
        # delivery costs: STT 0.1% + stamp 0.015% + small fees ~ 0.118% of value
        assert 0.0011 < buy_fee / (50 * buy_px) < 0.0013, buy_fee / (50 * buy_px)

        prices["AAA.NS"] = 1100.0
        asyncio.run(b.place_order(intent("SELL", 20, "c2"), "T"))
        sell_px = 1100.0 * (1 - CONFIG.slippage_bps / 1e4)
        assert abs(b.realised_pnl - 20 * (sell_px - buy_px)) < 1e-9
        for bad, msg in ((intent("SELL", 31, "c3"), "holding"), (intent("BUY", 10_000, "c4"), "funds")):
            try:
                asyncio.run(b.place_order(bad, "T"))
                raise AssertionError("expected a rejection")
            except PaperOrderRejected as exc:
                assert msg in str(exc)
        prices.pop("AAA.NS")
        try:
            asyncio.run(b.place_order(intent("BUY", 1, "c5"), "T"))
            raise AssertionError("expected a rejection")
        except PaperOrderRejected as exc:
            assert "no price" in str(exc)

        b2 = PaperTradingBroker(999.0, price_fn=lambda s: 1100.0, state_file=path)
        assert b2.positions == {"AAA.NS": 30} and abs(b2.capital - b.capital) < 1e-9
        assert b2.initial_capital == 100_000.0 and len(b2.trade_history) == 2
        assert asyncio.run(b2.get_positions()) == {"AAA.NS": 30}
        assert abs(b2.equity() - (b2.capital + 30 * 1100.0)) < 1e-9
    print("[ok] paper broker: slippage both ways, delivery costs ~0.12% on buys, realised PnL, "
          "rejections raise, fills reported to the engine, ledger survives a restart")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _test_ledger_costs_and_persistence()

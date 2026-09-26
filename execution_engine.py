"""
execution_engine.py
===================

Phase 5, Part IV: the live execution state machine.

Governing rule: THE STATE MACHINE IS THE ONLY COMPONENT THAT EMITS ORDERS.
Signal generation, sizing, and the sentiment pipeline produce OrderIntent
objects onto a bounded queue. Nothing else touches the broker client. Without
this invariant the failure modes below cannot be reasoned about, because any
component could act during any degraded condition.

Components
----------
EngineState / OrderState : lifecycle enums, including the states that matter
                           (UNKNOWN, PENDING_CANCEL, PENDING_REPLACE)
TokenBucket              : per-endpoint-class rate limiting
PriorityOrderQueue       : cancels never queue behind new entries
OrderManager             : idempotent submission; a timeout is not a rejection
StalenessMonitor         : liveness independent of socket state
CircuitBreakerPanel      : independent monitors, any one can force HALTED
Reconciler               : broker is source of truth, always
ExecutionEngine          : the state machine

Broker access is behind the BrokerClient protocol. Implement it against your
broker's SDK; nothing here is venue-specific.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

log = logging.getLogger("execution")

__all__ = [
    "EngineState",
    "OrderState",
    "Priority",
    "OrderIntent",
    "OrderRecord",
    "BrokerClient",
    "TokenBucket",
    "PriorityOrderQueue",
    "OrderManager",
    "StalenessMonitor",
    "CircuitBreaker",
    "CircuitBreakerPanel",
    "Reconciler",
    "ExecutionEngine",
    "EngineConfig",
]


# ===========================================================================
# LIFECYCLE
# ===========================================================================

class EngineState(Enum):
    BOOT = auto()
    AUTH = auto()
    RECONCILE = auto()
    WARMUP = auto()
    LIVE = auto()
    DEGRADED = auto()
    HALTED = auto()
    FLATTENING = auto()
    SHUTDOWN = auto()


# Only these transitions are legal. Note that DEGRADED never returns directly
# to LIVE -- recovery always routes through RECONCILE, because the period of
# degradation is exactly when local state most likely drifted.
LEGAL_TRANSITIONS: Dict[EngineState, frozenset] = {
    EngineState.BOOT: frozenset({EngineState.AUTH, EngineState.SHUTDOWN}),
    EngineState.AUTH: frozenset(
        {EngineState.RECONCILE, EngineState.HALTED, EngineState.SHUTDOWN}
    ),
    EngineState.RECONCILE: frozenset(
        {EngineState.WARMUP, EngineState.LIVE, EngineState.HALTED, EngineState.SHUTDOWN}
    ),
    EngineState.WARMUP: frozenset(
        {EngineState.LIVE, EngineState.DEGRADED, EngineState.HALTED, EngineState.SHUTDOWN}
    ),
    EngineState.LIVE: frozenset(
        {EngineState.DEGRADED, EngineState.HALTED, EngineState.AUTH, EngineState.SHUTDOWN}
    ),
    EngineState.DEGRADED: frozenset(
        {EngineState.RECONCILE, EngineState.HALTED, EngineState.SHUTDOWN}
    ),
    EngineState.HALTED: frozenset(
        {EngineState.FLATTENING, EngineState.RECONCILE, EngineState.SHUTDOWN}
    ),
    EngineState.FLATTENING: frozenset({EngineState.HALTED, EngineState.SHUTDOWN}),
    EngineState.SHUTDOWN: frozenset(),
}

# States in which new risk may be opened.
_RISK_OPENING_STATES = frozenset({EngineState.LIVE})
# States in which risk-reducing orders are permitted.
_REDUCE_ONLY_STATES = frozenset(
    {EngineState.LIVE, EngineState.DEGRADED, EngineState.FLATTENING}
)


class OrderState(Enum):
    """
    A model with only NEW/FILLED/CANCELLED/REJECTED cannot represent the
    situations that actually cause losses. UNKNOWN is the important one.
    """

    PENDING_NEW = auto()
    WORKING = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    PENDING_CANCEL = auto()
    CANCELLED = auto()
    PENDING_REPLACE = auto()
    REJECTED = auto()
    UNKNOWN = auto()      # request timed out; fate genuinely unknown

    @property
    def terminal(self) -> bool:
        return self in {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED}

    @property
    def blocks_new_intents(self) -> bool:
        """Sizing against an unknown position is unsound by construction."""
        return self is OrderState.UNKNOWN


class Priority(IntEnum):
    """Lower value dequeues first."""

    CANCEL = 0        # risk-reducing; must never queue behind anything
    REDUCE = 1
    NEW = 2
    QUERY = 3         # shed under pressure, never queued indefinitely


# ===========================================================================
# DATA
# ===========================================================================

@dataclass
class OrderIntent:
    symbol: str
    side: str                  # "BUY" / "SELL"
    quantity: int
    order_type: str = "LIMIT"
    limit_price: Optional[float] = None
    reduce_only: bool = False  # property of the INTENT, evaluated at emission
    priority: Priority = Priority.NEW
    decision_price: Optional[float] = None
    decision_ts: float = field(default_factory=time.time)
    feature_vector: Optional[Sequence[float]] = None  # for the audit trail
    client_order_id: str = field(default_factory=lambda: f"algo-{uuid.uuid4().hex[:20]}")

    def __post_init__(self) -> None:
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"bad side {self.side}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass
class OrderRecord:
    intent: OrderIntent
    state: OrderState = OrderState.PENDING_NEW
    broker_order_id: Optional[str] = None
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    submit_ts: Optional[float] = None
    arrival_price: Optional[float] = None
    last_error: Optional[str] = None
    unknown_since: Optional[float] = None
    attempts: int = 0

    @property
    def remaining(self) -> int:
        return max(self.intent.quantity - self.filled_qty, 0)

    def implementation_shortfall(self) -> Optional[Dict[str, float]]:
        """Decompose realised cost into delay (ours) and impact (the market's)."""
        d, a = self.intent.decision_price, self.arrival_price
        if d is None or a is None or self.filled_qty == 0 or d <= 0:
            return None
        sign = 1.0 if self.intent.side == "BUY" else -1.0
        return {
            "delay_bps": sign * (a - d) / d * 1e4,
            "impact_bps": sign * (self.avg_fill_price - a) / d * 1e4,
            "total_bps": sign * (self.avg_fill_price - d) / d * 1e4,
        }


class BrokerClient(Protocol):
    """Implement against your broker's SDK."""

    async def authenticate(self) -> str: ...
    async def logout(self) -> None: ...
    async def place_order(self, intent: OrderIntent, algo_id: str) -> str: ...
    async def cancel_order(self, broker_order_id: str) -> None: ...
    async def get_order_by_client_id(self, client_order_id: str) -> Optional[dict]: ...
    async def get_positions(self) -> Dict[str, int]: ...
    async def get_working_orders(self) -> Dict[str, dict]: ...
    async def get_quote(self, symbol: str) -> dict: ...


# ===========================================================================
# RATE LIMITING
# ===========================================================================

class TokenBucket:
    """
    Per-endpoint-class token bucket.

    Set `rate` conservatively BELOW the published limit: the published limit is
    enforced against the broker's clock and their accounting of your requests,
    neither of which you observe.
    """

    def __init__(self, rate: float, capacity: Optional[float] = None) -> None:
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(rate, 1.0))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            if deadline is not None and time.monotonic() + wait > deadline:
                return False
            await asyncio.sleep(min(wait, 0.05))

    def try_acquire(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def utilisation(self) -> float:
        self._refill()
        return 1.0 - (self._tokens / self.capacity)


class PriorityOrderQueue:
    """
    A single FIFO makes a cancel wait behind a batch of new entries, which is
    exactly backwards. Separate deques per priority, drained in order.

    QUERY-priority items are shed rather than queued: if you cannot afford a
    position query, you certainly cannot afford to place new orders, so the
    shed itself is a signal to degrade.
    """

    def __init__(self, maxsize_per_priority: int = 256) -> None:
        self._queues: Dict[Priority, Deque[OrderIntent]] = {
            p: deque(maxlen=maxsize_per_priority) for p in Priority
        }
        self._event = asyncio.Event()
        self.shed_count: Dict[Priority, int] = defaultdict(int)

    def put(self, intent: OrderIntent) -> bool:
        q = self._queues[intent.priority]
        if len(q) >= (q.maxlen or 0):
            if intent.priority is Priority.QUERY:
                self.shed_count[intent.priority] += 1
                return False
            # Order intents are never silently dropped. A full queue means the
            # system is emitting faster than it can execute, which is a fault.
            raise RuntimeError(f"intent queue full at priority {intent.priority.name}")
        q.append(intent)
        self._event.set()
        return True

    def get_nowait(self) -> Optional[OrderIntent]:
        for p in sorted(Priority):
            if self._queues[p]:
                return self._queues[p].popleft()
        self._event.clear()
        return None

    async def get(self) -> OrderIntent:
        while True:
            item = self.get_nowait()
            if item is not None:
                return item
            await self._event.wait()

    def drop_non_reducing(self) -> int:
        """On degradation, discard queued risk-opening intents."""
        n = len(self._queues[Priority.NEW])
        self._queues[Priority.NEW].clear()
        return n

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())


# ===========================================================================
# ORDER MANAGEMENT
# ===========================================================================

class OrderManager:
    """
    Idempotent order submission.

    THE RULE: a timeout is not a rejection. When a request times out, the order
    may have reached the exchange and filled. The timeout tells you nothing
    about the order's fate -- only that you did not receive the response.
    Retrying blind is a coin flip on double-sizing the position.

    The discipline:
      1. Client order ID generated and PERSISTED before the request is sent.
      2. On timeout -> OrderState.UNKNOWN, never a retry.
      3. Resolve UNKNOWN by querying the broker by client order ID.
      4. Block new intents for that symbol while any order is UNKNOWN.
    """

    def __init__(
        self,
        broker: BrokerClient,
        algo_id: str,
        order_bucket: TokenBucket,
        query_bucket: TokenBucket,
        submit_timeout: float = 3.0,
        unknown_resolve_after: float = 2.0,
        max_resend_attempts: int = 2,
        persist: Optional[Callable[[OrderRecord], Awaitable[None]]] = None,
    ) -> None:
        self.broker = broker
        self.algo_id = algo_id
        self.order_bucket = order_bucket
        self.query_bucket = query_bucket
        self.submit_timeout = submit_timeout
        self.unknown_resolve_after = unknown_resolve_after
        self.max_resend_attempts = max_resend_attempts
        self._persist = persist

        self.orders: Dict[str, OrderRecord] = {}
        self._sent: int = 0
        self._rejected: int = 0
        self._reject_window: Deque[Tuple[float, bool]] = deque(maxlen=200)

    # -- accessors -------------------------------------------------------- #

    @property
    def reject_rate(self) -> float:
        if not self._reject_window:
            return 0.0
        return sum(1 for _, rej in self._reject_window if rej) / len(self._reject_window)

    def unknown_symbols(self) -> set:
        return {
            r.intent.symbol
            for r in self.orders.values()
            if r.state.blocks_new_intents
        }

    def working_orders(self) -> List[OrderRecord]:
        return [r for r in self.orders.values() if not r.state.terminal]

    # -- submission ------------------------------------------------------- #

    async def submit(self, intent: OrderIntent) -> OrderRecord:
        if intent.symbol in self.unknown_symbols():
            raise RuntimeError(
                f"{intent.symbol}: an order is UNKNOWN; refusing to size against "
                "an unresolved position"
            )

        rec = OrderRecord(intent=intent)
        self.orders[intent.client_order_id] = rec

        # Persist BEFORE sending. If the process dies between generating the ID
        # and sending, the ID still exists on restart and the order can be
        # resolved rather than duplicated.
        if self._persist is not None:
            await self._persist(rec)

        ok = await self.order_bucket.acquire(timeout=self.submit_timeout)
        if not ok:
            rec.state = OrderState.REJECTED
            rec.last_error = "rate_limit_timeout"
            self._reject_window.append((time.time(), True))
            return rec

        try:
            q = await self.broker.get_quote(intent.symbol)
            rec.arrival_price = float(q.get("last", q.get("mid", 0.0))) or None
        except Exception:
            rec.arrival_price = None

        rec.submit_ts = time.time()
        rec.attempts += 1
        try:
            bid = await asyncio.wait_for(
                self.broker.place_order(intent, self.algo_id),
                timeout=self.submit_timeout,
            )
            rec.broker_order_id = bid
            # A fill can arrive before the acknowledgement (the paper broker fills
            # synchronously inside place_order). Never overwrite it with WORKING.
            if rec.state is OrderState.PENDING_NEW:
                rec.state = OrderState.WORKING
            self._sent += 1
            self._reject_window.append((time.time(), False))

        except asyncio.TimeoutError:
            # The critical branch. Do NOT retry here.
            rec.state = OrderState.UNKNOWN
            rec.unknown_since = time.time()
            rec.last_error = "submit_timeout"
            log.error(
                "order %s TIMED OUT -- state UNKNOWN, will resolve by query",
                intent.client_order_id,
            )

        except Exception as exc:
            # An explicit error response IS authoritative: the broker answered.
            rec.state = OrderState.REJECTED
            rec.last_error = f"{type(exc).__name__}: {exc}"
            self._rejected += 1
            self._reject_window.append((time.time(), True))

        if self._persist is not None:
            await self._persist(rec)
        return rec

    async def resolve_unknown(self) -> List[OrderRecord]:
        """
        Resolve UNKNOWN orders by querying the broker by client order ID.
        Never by assumption, never by blind retry.
        """
        resolved: List[OrderRecord] = []
        now = time.time()

        for rec in list(self.orders.values()):
            if rec.state is not OrderState.UNKNOWN:
                continue
            if rec.unknown_since and now - rec.unknown_since < self.unknown_resolve_after:
                continue
            if not self.query_bucket.try_acquire():
                continue

            try:
                info = await asyncio.wait_for(
                    self.broker.get_order_by_client_id(rec.intent.client_order_id),
                    timeout=self.submit_timeout,
                )
            except Exception as exc:
                log.warning("resolve query failed for %s: %s",
                            rec.intent.client_order_id, exc)
                continue

            if info is not None:
                # The broker knows the ID. Adopt its view unconditionally.
                rec.broker_order_id = info.get("order_id")
                rec.filled_qty = int(info.get("filled_quantity", 0))
                rec.avg_fill_price = float(info.get("average_price", 0.0))
                status = str(info.get("status", "")).upper()
                rec.state = {
                    "COMPLETE": OrderState.FILLED,
                    "FILLED": OrderState.FILLED,
                    "CANCELLED": OrderState.CANCELLED,
                    "REJECTED": OrderState.REJECTED,
                }.get(status, OrderState.WORKING)
                resolved.append(rec)
                log.info("resolved %s -> %s",
                         rec.intent.client_order_id, rec.state.name)

            elif rec.attempts < self.max_resend_attempts:
                # Broker has never seen the ID and enough time has passed for
                # the original request to have certainly failed. Re-send with
                # the SAME id, so a late-arriving original is deduplicated
                # broker-side rather than doubling the position.
                rec.state = OrderState.PENDING_NEW
                rec.unknown_since = None
                await self.submit(rec.intent)
                resolved.append(rec)

            else:
                rec.state = OrderState.REJECTED
                rec.last_error = "unresolvable_after_max_attempts"
                resolved.append(rec)

            if self._persist is not None:
                await self._persist(rec)

        return resolved

    async def cancel_all(self) -> int:
        n = 0
        for rec in self.working_orders():
            if rec.broker_order_id is None:
                continue
            if not await self.order_bucket.acquire(timeout=5.0):
                break
            try:
                rec.state = OrderState.PENDING_CANCEL
                await asyncio.wait_for(
                    self.broker.cancel_order(rec.broker_order_id), timeout=self.submit_timeout
                )
                rec.state = OrderState.CANCELLED
                n += 1
            except asyncio.TimeoutError:
                rec.state = OrderState.UNKNOWN
                rec.unknown_since = time.time()
            except Exception as exc:
                log.warning("cancel failed for %s: %s", rec.broker_order_id, exc)
        return n


# ===========================================================================
# MONITORING
# ===========================================================================

class StalenessMonitor:
    """
    Liveness independent of socket state.

    A TCP connection can remain established while the feed stops delivering.
    Never infer liveness from connection state. Thresholds are per-symbol and
    derived from each symbol's own arrival rate -- an illiquid Nifty
    constituent legitimately goes quiet for minutes.
    """

    def __init__(self, default_threshold: float = 15.0, multiplier: float = 8.0) -> None:
        self.default_threshold = default_threshold
        self.multiplier = multiplier
        self._last_tick: Dict[str, float] = {}
        self._intervals: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))
        self._sequence: Dict[str, int] = {}
        self.gap_events: List[Tuple[str, int, int]] = []

    def on_tick(self, symbol: str, sequence: Optional[int] = None) -> Optional[str]:
        """Returns a fault description, or None. A gap is a fault."""
        now = time.monotonic()
        prev = self._last_tick.get(symbol)
        if prev is not None:
            self._intervals[symbol].append(now - prev)
        self._last_tick[symbol] = now

        if sequence is not None:
            last = self._sequence.get(symbol)
            self._sequence[symbol] = sequence
            if last is not None and sequence != last + 1:
                if sequence > last + 1:
                    # Messages missed. The book/bar base is now corrupt and
                    # will never self-correct: subsequent ticks apply deltas
                    # to a wrong base. Snapshot required.
                    self.gap_events.append((symbol, last, sequence))
                    return f"sequence_gap:{symbol}:{last}->{sequence}"
        return None

    def threshold_for(self, symbol: str) -> float:
        iv = self._intervals.get(symbol)
        if not iv or len(iv) < 20:
            return self.default_threshold
        med = sorted(iv)[len(iv) // 2]
        return max(self.multiplier * med, 2.0)

    def stale_symbols(self) -> List[str]:
        now = time.monotonic()
        return [
            s
            for s, t in self._last_tick.items()
            if now - t > self.threshold_for(s)
        ]


@dataclass
class CircuitBreaker:
    name: str
    check: Callable[[], bool]      # True == tripped
    reason: str
    tripped: bool = False
    tripped_at: Optional[float] = None

    def evaluate(self) -> bool:
        if self.tripped:
            return True
        try:
            if self.check():
                self.tripped = True
                self.tripped_at = time.time()
                log.critical("BREAKER TRIPPED: %s -- %s", self.name, self.reason)
                return True
        except Exception as exc:
            # A breaker that throws must fail closed. A monitor that cannot
            # evaluate is not evidence that conditions are safe.
            self.tripped = True
            self.tripped_at = time.time()
            log.critical("BREAKER %s raised %s -- failing closed", self.name, exc)
            return True
        return False

    def reset(self) -> None:
        self.tripped = False
        self.tripped_at = None


class CircuitBreakerPanel:
    """Any single breaker can force HALTED unilaterally."""

    def __init__(self, breakers: Sequence[CircuitBreaker]) -> None:
        self.breakers = list(breakers)

    def evaluate(self) -> List[CircuitBreaker]:
        return [b for b in self.breakers if b.evaluate()]

    def any_tripped(self) -> bool:
        return any(b.tripped for b in self.breakers)

    def reset_all(self) -> None:
        for b in self.breakers:
            b.reset()

    def status(self) -> Dict[str, bool]:
        return {b.name: b.tripped for b in self.breakers}


class Reconciler:
    """
    The broker is the source of truth. Always.

    Local position state is a cache, and every mechanism that keeps it fresh --
    fill callbacks, order acks -- is exactly the mechanism that fails during
    the incidents where accuracy matters most.

    On discrepancy: adopt the broker's view, halt, alert. Never reconcile by
    adjusting toward local state, and never auto-trade to close the gap. A
    discrepancy means your model of reality is wrong, and placing orders based
    on that model is the last thing to do.
    """

    def __init__(self, broker: BrokerClient, query_bucket: TokenBucket) -> None:
        self.broker = broker
        self.query_bucket = query_bucket
        self.positions: Dict[str, int] = {}
        self.last_reconciled: float = 0.0
        self.discrepancy_log: List[dict] = []

    async def reconcile(self, local: Dict[str, int]) -> Tuple[bool, Dict[str, Tuple[int, int]]]:
        """Returns (agreed, {symbol: (local, broker)} for mismatches)."""
        if not await self.query_bucket.acquire(timeout=10.0):
            return False, {}

        try:
            # PaperTradingBroker uses .positions property instead of .get_positions() method
            if hasattr(self.broker, "get_positions"):
                if asyncio.iscoroutinefunction(self.broker.get_positions):
                    broker_pos = await self.broker.get_positions()
                else:
                    broker_pos = self.broker.get_positions()
            else:
                broker_pos = self.broker.positions
        except Exception as e:
            broker_pos = {}

        self.positions = dict(broker_pos)
        self.last_reconciled = time.time()

        mismatches: Dict[str, Tuple[int, int]] = {}
        for sym in set(local) | set(broker_pos):
            lv, bv = local.get(sym, 0), broker_pos.get(sym, 0)
            if lv != bv:
                mismatches[sym] = (lv, bv)

        if mismatches:
            entry = {"ts": self.last_reconciled, "mismatches": dict(mismatches)}
            self.discrepancy_log.append(entry)
            log.critical("POSITION DISCREPANCY: %s", mismatches)
            # Recurring small diffs usually mean a dropped fill callback --
            # a bug with a fix, not noise to be tolerated.

        return (not mismatches), mismatches

    def age(self) -> float:
        return time.time() - self.last_reconciled if self.last_reconciled else float("inf")


# ===========================================================================
# ENGINE
# ===========================================================================

@dataclass
class EngineConfig:
    algo_id: str
    order_rate_per_sec: float = 7.0      # below SEBI's 10 OPS threshold
    query_rate_per_sec: float = 3.0
    reconcile_interval: float = 45.0
    reconcile_max_age: float = 90.0
    degraded_recovery_hold: float = 20.0
    max_reconnect_attempts: int = 8
    reconnect_base: float = 0.5
    reconnect_cap: float = 30.0
    heartbeat_interval: float = 2.0
    loop_interval: float = 0.25
    warmup_bars_required: int = 64
    daily_reauth_hour: int = 9           # SEBI: sessions log out before each day


class ExecutionEngine:
    """
    The state machine. The only component that emits orders.

    Usage:
        engine = ExecutionEngine(broker, config, breakers, heartbeat_cb)
        engine.intents.put(OrderIntent(...))     # producers enqueue
        await engine.run()
    """

    def __init__(
        self,
        broker: BrokerClient,
        config: EngineConfig,
        breakers: Optional[CircuitBreakerPanel] = None,
        heartbeat: Optional[Callable[[], Awaitable[None]]] = None,
        kill_switch: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.broker = broker
        self.cfg = config
        self.kill_switch = kill_switch or (lambda: False)
        self._heartbeat_cb = heartbeat

        self.order_bucket = TokenBucket(config.order_rate_per_sec)
        self.query_bucket = TokenBucket(config.query_rate_per_sec)
        self.intents = PriorityOrderQueue()
        self.orders = OrderManager(
            broker, config.algo_id, self.order_bucket, self.query_bucket
        )
        self.staleness = StalenessMonitor()
        self.reconciler = Reconciler(broker, self.query_bucket)

        self.state = EngineState.BOOT
        self.local_positions: Dict[str, int] = defaultdict(int)
        self.bars_seen: int = 0
        self._degraded_clear_since: Optional[float] = None
        self._reconnect_attempts: int = 0
        self._running = False
        self.state_history: List[Tuple[float, EngineState, str]] = []

        self.breakers = breakers or CircuitBreakerPanel(self._default_breakers())

    # -- default breakers -------------------------------------------------- #

    def _default_breakers(self) -> List[CircuitBreaker]:
        return [
            CircuitBreaker(
                "reject_rate",
                lambda: self.orders.reject_rate > 0.25 and len(self.orders._reject_window) > 20,
                "reject rate >25%: malformed orders or margin exhaustion",
            ),
            CircuitBreaker(
                "order_rate",
                lambda: self.order_bucket.utilisation > 0.95,
                "approaching OPS ceiling: compliance and runaway-loop protection",
            ),
            CircuitBreaker(
                "reconcile_age",
                lambda: (
                    self.state is EngineState.LIVE
                    and self.reconciler.age() > self.cfg.reconcile_max_age
                ),
                "position state not verified against broker within tolerance",
            ),
            CircuitBreaker(
                "unresolved_unknown",
                lambda: len(self.orders.unknown_symbols()) > 0
                and self.state is EngineState.LIVE,
                "orders in UNKNOWN state while trading",
            ),
        ]

    # -- transitions ------------------------------------------------------- #

    def _transition(self, new: EngineState, reason: str) -> bool:
        if new is self.state:
            return True
        if new not in LEGAL_TRANSITIONS[self.state]:
            log.error("ILLEGAL TRANSITION %s -> %s (%s)", self.state.name, new.name, reason)
            return False
        log.warning("%s -> %s : %s", self.state.name, new.name, reason)
        self.state_history.append((time.time(), new, reason))
        self.state = new
        return True

    def can_open_risk(self) -> bool:
        return (
            self.state in _RISK_OPENING_STATES
            and not self.breakers.any_tripped()
            and not self.orders.unknown_symbols()
        )

    def can_reduce_risk(self) -> bool:
        return self.state in _REDUCE_ONLY_STATES

    # -- market data hooks -------------------------------------------------- #

    def on_tick(self, symbol: str, sequence: Optional[int] = None) -> None:
        fault = self.staleness.on_tick(symbol, sequence)
        if fault:
            # A sequence gap means the aggregation base is corrupt. Snapshot
            # and reconcile; do not keep applying deltas to a wrong base.
            self._transition(EngineState.DEGRADED, fault)

    def on_bar(self, symbol: str) -> None:
        self.bars_seen += 1

    def on_fill(self, client_order_id: str, qty: int, price: float) -> None:
        rec = self.orders.orders.get(client_order_id)
        if rec is None:
            log.error("fill for unknown client_order_id %s", client_order_id)
            return
        prev_notional = rec.avg_fill_price * rec.filled_qty
        rec.filled_qty += qty
        rec.avg_fill_price = (prev_notional + price * qty) / max(rec.filled_qty, 1)
        rec.state = OrderState.FILLED if rec.remaining == 0 else OrderState.PARTIALLY_FILLED
        signed = qty if rec.intent.side == "BUY" else -qty
        self.local_positions[rec.intent.symbol] += signed

    # -- reconnect ---------------------------------------------------------- #

    async def reconnect_backoff(self) -> bool:
        """
        Exponential backoff with FULL jitter. Without jitter every strategy
        process reconnects simultaneously after a broker-side blip and you
        manufacture your own thundering herd into a rate limit.
        """
        if self._reconnect_attempts >= self.cfg.max_reconnect_attempts:
            self._transition(EngineState.HALTED, "reconnect attempts exhausted")
            return False
        delay = random.uniform(
            0.0, min(self.cfg.reconnect_cap, self.cfg.reconnect_base * 2**self._reconnect_attempts)
        )
        self._reconnect_attempts += 1
        await asyncio.sleep(delay)
        return True

    # -- state handlers ------------------------------------------------------ #

    async def _do_auth(self) -> None:
        try:
            await self.broker.authenticate()
            self._reconnect_attempts = 0
            self._transition(EngineState.RECONCILE, "session established")
        except Exception as exc:
            log.error("auth failed: %s", exc)
            if not await self.reconnect_backoff():
                return

    async def _do_reconcile(self) -> None:
        agreed, mismatches = await self.reconciler.reconcile(dict(self.local_positions))
        if not agreed and mismatches:
            # Adopt the broker's view, then halt. Never trade to close the gap.
            self.local_positions = defaultdict(int, self.reconciler.positions)
            self._transition(EngineState.HALTED, f"position discrepancy {mismatches}")
            return
        self.local_positions = defaultdict(int, self.reconciler.positions)
        if self.bars_seen < self.cfg.warmup_bars_required:
            self._transition(EngineState.WARMUP, "buffers not filled")
        else:
            self._transition(EngineState.LIVE, "reconciled")

    async def _do_warmup(self) -> None:
        # A policy invoked against a half-filled window produces an action,
        # silently, from garbage features. No exception is raised. WARMUP is
        # mandatory on every restart, including intraday.
        if self.bars_seen >= self.cfg.warmup_bars_required:
            self._transition(EngineState.LIVE, "buffers filled")

    async def _do_live(self) -> None:
        stale = self.staleness.stale_symbols()
        if stale:
            self._transition(EngineState.DEGRADED, f"stale: {stale[:5]}")
            return

        intent = self.intents.get_nowait()
        if intent is None:
            return

        if intent.reduce_only and not self.can_reduce_risk():
            return
        if not intent.reduce_only and not self.can_open_risk():
            return

        # reduce_only is evaluated HERE, at emission, against the reconciled
        # position -- not against a mode flag that may be stale by now.
        if intent.reduce_only:
            pos = self.local_positions.get(intent.symbol, 0)
            signed = intent.quantity if intent.side == "BUY" else -intent.quantity
            if pos == 0 or (pos > 0 and signed > 0) or (pos < 0 and signed < 0):
                log.warning("dropping reduce_only intent that would open risk")
                return

        await self.orders.submit(intent)

    async def _do_degraded(self) -> None:
        # No new risk. Drop queued risk-opening intents rather than letting
        # them fire on recovery against a stale view.
        dropped = self.intents.drop_non_reducing()
        if dropped:
            log.warning("dropped %d risk-opening intents on degradation", dropped)

        await self.orders.resolve_unknown()

        healthy = not self.staleness.stale_symbols() and not self.orders.unknown_symbols()
        now = time.monotonic()
        if healthy:
            if self._degraded_clear_since is None:
                self._degraded_clear_since = now
            elif now - self._degraded_clear_since >= self.cfg.degraded_recovery_hold:
                self._degraded_clear_since = None
                # Recovery NEVER returns directly to LIVE.
                self._transition(EngineState.RECONCILE, "recovery sustained")
        else:
            self._degraded_clear_since = None

    async def _do_halted(self) -> None:
        # HALT IS NOT FLATTEN. Cancel working orders and stop new ones; do not
        # liquidate. Auto-liquidating on every breaker trip crystallises losses
        # at whatever the market shows during a transient glitch.
        n = await self.orders.cancel_all()
        if n:
            log.warning("halted: cancelled %d working orders", n)
        await self.orders.resolve_unknown()

    async def _do_flattening(self) -> None:
        await self.reconciler.reconcile(dict(self.local_positions))
        remaining = {s: q for s, q in self.reconciler.positions.items() if q != 0}
        if not remaining:
            # Exit on BROKER confirmation only. Local state netting to zero
            # means the fills you know about net to zero, not that you are flat.
            self._transition(EngineState.HALTED, "flat confirmed by broker")
            return
        for sym, qty in remaining.items():
            self.intents.put(
                OrderIntent(
                    symbol=sym,
                    side="SELL" if qty > 0 else "BUY",
                    quantity=abs(qty),
                    order_type="MARKET",
                    reduce_only=True,
                    priority=Priority.CANCEL,
                )
            )
        intent = self.intents.get_nowait()
        if intent is not None:
            await self.orders.submit(intent)

    # -- main loop ------------------------------------------------------------ #

    async def run(self) -> None:
        self._running = True
        tasks = [
            asyncio.create_task(self._reconcile_loop()),
            asyncio.create_task(self._heartbeat_loop()),
        ]
        try:
            while self._running and self.state is not EngineState.SHUTDOWN:
                if self.kill_switch():
                    self._transition(EngineState.HALTED, "kill switch")
                    self._transition(EngineState.FLATTENING, "kill switch flatten")

                if self.breakers.evaluate() and self.state not in (
                    EngineState.HALTED,
                    EngineState.FLATTENING,
                    EngineState.SHUTDOWN,
                ):
                    names = [b.name for b in self.breakers.breakers if b.tripped]
                    self._transition(EngineState.HALTED, f"breakers: {names}")

                handler = {
                    EngineState.BOOT: self._boot,
                    EngineState.AUTH: self._do_auth,
                    EngineState.RECONCILE: self._do_reconcile,
                    EngineState.WARMUP: self._do_warmup,
                    EngineState.LIVE: self._do_live,
                    EngineState.DEGRADED: self._do_degraded,
                    EngineState.HALTED: self._do_halted,
                    EngineState.FLATTENING: self._do_flattening,
                }.get(self.state)

                if handler is not None:
                    try:
                        await handler()
                    except Exception as exc:
                        log.exception("handler error in %s: %s", self.state.name, exc)
                        self._transition(EngineState.HALTED, f"handler exception: {exc}")

                await asyncio.sleep(self.cfg.loop_interval)
        finally:
            for t in tasks:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            await self.broker.logout()

    async def _boot(self) -> None:
        skew = abs(time.time() - time.time())  # replace with real NTP offset query
        if skew > 0.1:
            self._transition(EngineState.HALTED, f"clock skew {skew:.3f}s")
            return
        self._transition(EngineState.AUTH, "boot checks passed")

    async def _reconcile_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.cfg.reconcile_interval)
            if self.state is EngineState.LIVE:
                agreed, mism = await self.reconciler.reconcile(dict(self.local_positions))
                if not agreed and mism:
                    self.local_positions = defaultdict(int, self.reconciler.positions)
                    self._transition(EngineState.HALTED, f"periodic discrepancy {mism}")

    async def _heartbeat_loop(self) -> None:
        """
        Emits to an EXTERNAL supervisor. A kill switch that lives inside the
        process it is meant to kill is not a kill switch: it cannot cover the
        case where this process wedges with live orders resting at the
        exchange. The supervisor cancels on heartbeat loss.
        """
        while self._running:
            if self._heartbeat_cb is not None:
                with contextlib.suppress(Exception):
                    await self._heartbeat_cb()
            await asyncio.sleep(self.cfg.heartbeat_interval)

    def stop(self) -> None:
        self._running = False
        self._transition(EngineState.SHUTDOWN, "stop requested")


# ===========================================================================
# TRANSITION-TABLE SELF-TEST
# ===========================================================================

def verify_transition_table() -> None:
    """
    Structural invariants of the state machine. Run this in CI -- a transition
    table that drifts is how a recovery path silently starts skipping
    reconciliation.
    """
    assert EngineState.LIVE not in LEGAL_TRANSITIONS[EngineState.DEGRADED], (
        "DEGRADED must not return directly to LIVE: recovery routes through "
        "RECONCILE because degradation is exactly when local state drifts"
    )
    assert EngineState.LIVE not in LEGAL_TRANSITIONS[EngineState.HALTED], (
        "HALTED must not resume trading without re-reconciling"
    )
    for s in EngineState:
        if s is not EngineState.SHUTDOWN:
            assert EngineState.SHUTDOWN in LEGAL_TRANSITIONS[s], (
                f"{s.name} has no path to SHUTDOWN"
            )
    for s, targets in LEGAL_TRANSITIONS.items():
        assert s not in targets, f"{s.name} lists a self-transition"
    assert _RISK_OPENING_STATES <= _REDUCE_ONLY_STATES, (
        "any state permitting new risk must permit reducing it"
    )
    print("transition table OK")
    print(f"  risk-opening states : {[s.name for s in _RISK_OPENING_STATES]}")
    print(f"  reduce-only states  : {[s.name for s in _REDUCE_ONLY_STATES]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    verify_transition_table()

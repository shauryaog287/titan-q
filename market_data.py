"""
market_data.py
==============
Market data for the live system from one source (Yahoo Finance via yfinance),
with local caches so a restart does not re-download everything.

    DailyStore      daily OHLCV, 12 years, cached under data/daily/
    IntradayStore   1-minute bars. Every completed session is archived under
                    data/intraday/YYYY-MM-DD.parquet, so intraday history
                    accumulates over time (Yahoo only serves the last ~5 days).
    Session         NSE cash-market session times in IST
    LiveBarFeed     live 1-minute bars, fetched in a worker thread so the asyncio
                    event loop (and with it the execution engine) never blocks
    ReplayFeed      replays an archived session at a chosen speed, so the whole
                    system can run outside market hours on real data

Both feeds expose the same interface: start(), poll(), now(), session_date,
last_price(sym), bars(sym), is_stale(), finished.

WHAT THIS DATA CAN AND CANNOT SUPPORT
Yahoo provides bars, not the order book and not individual trades. Models that
need depth (the Phase 18 transformer, the Phase 16 spoofing autoencoder) or
trade-by-trade timestamps (the Phase 14 Hawkes filter) cannot be fed from here,
and the live loop reports them as unavailable instead of feeding them noise.
VPIN (the other half of Phase 14) works on bars via bulk volume
classification, so it is supported.

NO LOOK-AHEAD
Anything that decides trades for session D must only see daily bars dated
before D. `close_panel(..., before=D)` enforces that, and ReplayFeed.bars()
never returns a bar later than the replay clock.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from titanq_config import CONFIG, DATA_DIR, INDEX_SYMBOL, UNIVERSE

log = logging.getLogger("market_data")

# IST has no daylight saving, so a fixed offset is exact and needs no tz database.
IST = timezone(timedelta(hours=5, minutes=30), "IST")
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
ALL_SYMBOLS: Tuple[str, ...] = tuple(UNIVERSE) + (INDEX_SYMBOL,)


def now_ist() -> datetime:
    return datetime.now(IST)


def _safe_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", symbol)


# ===========================================================================
# 1. SESSION CALENDAR
# ===========================================================================

HOLIDAYS_FILE = os.path.join(DATA_DIR, "reference", "nse_holidays.json")


def nse_holidays() -> set:
    """NSE cash-market trading holidays (NSE's holiday-master, cached in
    data/reference/nse_holidays.json; refreshed by alerts.py --preflight when stale).
    Missing file -> no holidays known (weekends still apply)."""
    try:
        import json as _json
        with open(HOLIDAYS_FILE, encoding="ascii") as f:
            rows = _json.load(f)
        return {datetime.strptime(r["tradingDate"], "%d-%b-%Y").date() for r in rows if r.get("tradingDate")}
    except (OSError, ValueError, KeyError):
        return set()


class Session:
    """
    NSE cash-market hours in IST, Monday to Friday, minus NSE's trading holidays
    (data/reference/nse_holidays.json). Before 2026-10-01 holidays were unknown, so on a
    holiday the system waited all day for bars that never came.
    """

    def __init__(self, open_hm: Tuple[int, int] = CONFIG.session_open,
                 close_hm: Tuple[int, int] = CONFIG.session_close) -> None:
        self.open_t = dtime(*open_hm)
        self.close_t = dtime(*close_hm)
        self.holidays = nse_holidays()

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def bounds(self, d: date) -> Tuple[datetime, datetime]:
        return (datetime.combine(d, self.open_t, IST), datetime.combine(d, self.close_t, IST))

    def is_open(self, ts: datetime) -> bool:
        ts = ts.astimezone(IST)
        if not self.is_trading_day(ts.date()):
            return False
        o, c = self.bounds(ts.date())
        return o <= ts < c

    def minutes_to_close(self, ts: datetime) -> float:
        ts = ts.astimezone(IST)
        return (self.bounds(ts.date())[1] - ts).total_seconds() / 60.0

    def next_open(self, ts: datetime) -> datetime:
        ts = ts.astimezone(IST)
        d = ts.date()
        for _ in range(15):
            o, _c = self.bounds(d)
            if self.is_trading_day(d) and ts < o:
                return o
            d = d + timedelta(days=1)
            ts = datetime.combine(d, dtime(0, 0), IST)
        return self.bounds(d)[0]


# ===========================================================================
# 2. DOWNLOADS
# ===========================================================================

def _normalise(raw: pd.DataFrame, symbols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """yfinance frame (MultiIndex by ticker, or flat for one ticker) -> {sym: OHLCV}."""
    out: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        tickers = set(raw.columns.get_level_values(0))
        for s in symbols:
            if s not in tickers:
                continue
            df = raw[s]
            cols = [c for c in FIELDS if c in df.columns]
            df = df[cols].dropna(subset=["Close"])
            if len(df):
                out[s] = df.astype(float)
    else:
        cols = [c for c in FIELDS if c in raw.columns]
        df = raw[cols].dropna(subset=["Close"])
        if len(df) and len(symbols) == 1:
            out[symbols[0]] = df.astype(float)
    return out


def yf_download(symbols: Sequence[str], retries: int = 3, **kw) -> Dict[str, pd.DataFrame]:
    """Blocking. Call through asyncio.to_thread from async code."""
    import yfinance as yf

    symbols = list(symbols)
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            raw = yf.download(symbols, group_by="ticker", progress=False, threads=True, **kw)
            return _normalise(raw, symbols)
        except Exception as exc:  # network, rate limit, Yahoo format change
            last_exc = exc
            log.warning("yfinance download failed (attempt %d/%d): %s", attempt + 1, retries, exc)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance download failed after {retries} attempts: {last_exc}")


# ===========================================================================
# 3. DAILY STORE
# ===========================================================================

class DailyStore:
    """Daily OHLCV per symbol, cached as parquet and refreshed at most every
    `max_age_hours`. Index: tz-naive dates."""

    def __init__(self, root: str = os.path.join(DATA_DIR, "daily"),
                 years: int = CONFIG.daily_history_years) -> None:
        self.root = root
        self.years = int(years)
        os.makedirs(self.root, exist_ok=True)

    def path(self, symbol: str) -> str:
        return os.path.join(self.root, _safe_name(symbol) + ".parquet")

    def _is_stale(self, symbol: str, max_age_hours: float) -> bool:
        p = self.path(symbol)
        return (not os.path.exists(p)) or (time.time() - os.path.getmtime(p) > 3600 * max_age_hours)

    def load(self, symbols: Iterable[str] = ALL_SYMBOLS, refresh: bool = True,
             max_age_hours: float = 12.0, batch: int = 150) -> Dict[str, pd.DataFrame]:
        symbols = list(symbols)
        stale = [s for s in symbols if refresh and self._is_stale(s, max_age_hours)]
        got = 0
        for i in range(0, len(stale), batch):        # batches keep Yahoo's rate limit happy
            chunk = stale[i:i + batch]
            try:
                fresh = yf_download(chunk, period=f"{self.years}y", interval="1d", auto_adjust=True)
                for s, df in fresh.items():
                    df = df.copy()
                    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    df.to_parquet(self.path(s))
                got += len(fresh)
            except Exception as exc:
                log.warning("daily refresh failed for a batch of %d (%s); using cached files",
                            len(chunk), exc)
            if len(stale) > batch:
                time.sleep(1.0)
        if stale:
            log.info("daily data refreshed for %d/%d symbols", got, len(stale))
        out: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            p = self.path(s)
            if os.path.exists(p):
                out[s] = pd.read_parquet(p)
        missing = [s for s in symbols if s not in out]
        if missing:
            log.warning("no daily data for %s", missing)
        return out


def close_panel(daily: Dict[str, pd.DataFrame], symbols: Sequence[str],
                before: Optional[date] = None, field: str = "Close") -> pd.DataFrame:
    """Aligned panel of one field. `before` drops every row dated on or after
    that day, which is how look-ahead is prevented."""
    cols = {s: daily[s][field] for s in symbols if s in daily}
    panel = pd.DataFrame(cols).sort_index()
    if before is not None:
        panel = panel[panel.index < pd.Timestamp(before)]
    return panel


# ===========================================================================
# 4. INTRADAY STORE
# ===========================================================================

class IntradayStore:
    """1-minute bars. Completed sessions are archived one parquet per date in
    long format (ts in UTC, symbol, OHLCV)."""

    def __init__(self, root: str = os.path.join(DATA_DIR, "intraday"),
                 session: Optional[Session] = None) -> None:
        self.root = root
        self.session = session or Session()
        os.makedirs(self.root, exist_ok=True)

    def path(self, d: date) -> str:
        return os.path.join(self.root, d.isoformat() + ".parquet")

    @staticmethod
    def to_ist(df: pd.DataFrame) -> pd.DataFrame:
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        out = df.copy()
        out.index = idx.tz_convert(IST)
        return out

    def fetch_recent(self, symbols: Iterable[str] = ALL_SYMBOLS,
                     period: str = CONFIG.intraday_period) -> Dict[str, pd.DataFrame]:
        """Blocking download of recent 1-minute bars; archives completed sessions."""
        bars = yf_download(list(symbols), period=period, interval="1m",
                           auto_adjust=False, prepost=False)
        bars = {s: self.to_ist(df) for s, df in bars.items()}
        self.archive(bars)
        return bars

    def archive(self, bars: Dict[str, pd.DataFrame], now: Optional[datetime] = None) -> List[date]:
        now = (now or now_ist()).astimezone(IST)
        frames = []
        for s, df in bars.items():
            if len(df):
                f = df[FIELDS].copy()
                f["symbol"] = s
                f["ts"] = pd.DatetimeIndex(f.index).tz_convert("UTC")
                frames.append(f.reset_index(drop=True))
        if not frames:
            return []
        long = pd.concat(frames, ignore_index=True)
        local_day = long["ts"].dt.tz_convert(IST).dt.date
        written = []
        for d, part in long.groupby(local_day):
            complete = d < now.date() or now >= self.session.bounds(d)[1] + timedelta(minutes=15)
            if not complete:
                continue
            p = self.path(d)
            if os.path.exists(p) and len(pd.read_parquet(p)) >= len(part):
                continue
            part.to_parquet(p, index=False)
            written.append(d)
        if written:
            log.info("archived intraday sessions: %s", ", ".join(str(d) for d in written))
        return written

    def sessions(self) -> List[date]:
        out = []
        for name in os.listdir(self.root):
            m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.parquet", name)
            if m:
                out.append(date.fromisoformat(m.group(1)))
        return sorted(out)

    def load_session(self, d: date) -> Dict[str, pd.DataFrame]:
        long = pd.read_parquet(self.path(d))
        out: Dict[str, pd.DataFrame] = {}
        for s, part in long.groupby("symbol"):
            df = part.set_index(pd.DatetimeIndex(part["ts"]).tz_convert(IST))[FIELDS]
            out[str(s)] = df.sort_index()
        return out


# ===========================================================================
# 5. FEEDS
# ===========================================================================

def _merge(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    df = new if old is None or not len(old) else pd.concat([old, new])
    return df[~df.index.duplicated(keep="last")].sort_index()


class LiveBarFeed:
    """
    Live 1-minute bars for the whole universe. Every download runs in a worker
    thread (asyncio.to_thread): a slow Yahoo response delays the next price
    update, never the event loop.
    """

    mode = "live"

    def __init__(self, symbols: Sequence[str] = ALL_SYMBOLS,
                 store: Optional[IntradayStore] = None,
                 poll_seconds: float = CONFIG.live_poll_seconds,
                 session: Optional[Session] = None) -> None:
        self.symbols = list(symbols)
        self.session = session or Session()
        self.store = store or IntradayStore(session=self.session)
        self.poll_seconds = float(poll_seconds)
        self._bars: Dict[str, pd.DataFrame] = {}
        self._last_poll = 0.0
        self.last_bar_time: Optional[datetime] = None
        self.errors = 0
        self.finished = False

    async def start(self) -> None:
        try:
            self._bars = await asyncio.to_thread(self.store.fetch_recent, self.symbols)
        except Exception as exc:
            self.errors += 1
            log.error("initial intraday fetch failed: %s", exc)
            self._bars = {}
        self._last_poll = time.monotonic()
        self._update_last()

    async def poll(self) -> bool:
        """Fetch new bars if poll_seconds have passed. True if a newer bar arrived."""
        if time.monotonic() - self._last_poll < self.poll_seconds:
            return False
        self._last_poll = time.monotonic()
        before = self.last_bar_time
        try:
            fresh = await asyncio.to_thread(
                yf_download, self.symbols, period="1d", interval="1m",
                auto_adjust=False, prepost=False)
        except Exception as exc:
            self.errors += 1
            log.warning("live bar poll failed: %s", exc)
            return False
        for s, df in fresh.items():
            self._bars[s] = _merge(self._bars.get(s), IntradayStore.to_ist(df))
        self._update_last()
        if self.session.minutes_to_close(self.now()) < -15:
            await asyncio.to_thread(self.store.archive, self._bars)
        return self.last_bar_time is not None and self.last_bar_time != before

    def _update_last(self) -> None:
        times = [df.index[-1] for df in self._bars.values() if len(df)]
        self.last_bar_time = max(times).to_pydatetime() if times else None

    def now(self) -> datetime:
        return now_ist()

    @property
    def session_date(self) -> date:
        return self.now().date()

    def bars(self, symbol: str) -> pd.DataFrame:
        return self._bars.get(symbol, pd.DataFrame(columns=FIELDS))

    def last_price(self, symbol: str) -> Optional[float]:
        df = self._bars.get(symbol)
        if df is None or not len(df):
            return None
        return float(df["Close"].iloc[-1])

    def is_stale(self) -> bool:
        now = self.now()
        if not self.session.is_open(now):
            return False
        if self.last_bar_time is None:
            return True
        return (now - self.last_bar_time).total_seconds() > CONFIG.stale_after_seconds


class ReplayFeed:
    """
    Replays an archived session bar by bar. The replay clock is the timestamp
    of the current bar; bars() and last_price() never see past it. Up to
    `history_sessions` earlier sessions are prepended, as a live feed would
    have them, so intraday statistics have their usual reference window.
    """

    mode = "replay"

    def __init__(self, session_date: Optional[date] = None,
                 symbols: Sequence[str] = ALL_SYMBOLS,
                 store: Optional[IntradayStore] = None,
                 bars_per_second: float = CONFIG.replay_bars_per_second,
                 history_sessions: int = 4) -> None:
        self.symbols = list(symbols)
        self.store = store or IntradayStore()
        available = self.store.sessions()
        if not available:
            log.info("no archived sessions yet; downloading recent 1-minute bars")
            self.store.fetch_recent(self.symbols)
            available = self.store.sessions()
        if not available:
            raise RuntimeError("no intraday sessions available to replay")
        if session_date is None:
            session_date = available[-1]
        if session_date not in available:
            raise ValueError(f"session {session_date} not archived; have {available[-5:]}")
        self.session_date: date = session_date
        self.speed = float(bars_per_second)

        self._day = {s: df for s, df in self.store.load_session(session_date).items()
                     if s in self.symbols}
        earlier = [d for d in available if d < session_date][-history_sessions:]
        hist: Dict[str, List[pd.DataFrame]] = {}
        for d in earlier:
            for s, df in self.store.load_session(d).items():
                if s in self.symbols:
                    hist.setdefault(s, []).append(df)
        self._hist = {s: pd.concat(v).sort_index() for s, v in hist.items()}

        clock = sorted(set().union(*[set(df.index) for df in self._day.values()]))
        if not clock:
            raise RuntimeError(f"session {session_date} has no bars")
        self._clock: List[pd.Timestamp] = clock
        self._i = 0
        self._t0: Optional[float] = None
        self.finished = False

    async def start(self) -> None:
        self._t0 = time.monotonic()

    async def poll(self) -> bool:
        if self._t0 is None:
            await self.start()
        target = min(int((time.monotonic() - self._t0) * self.speed), len(self._clock) - 1)
        moved = target > self._i
        self._i = max(self._i, target)
        if self._i >= len(self._clock) - 1:
            self.finished = True
        return moved

    def step(self, n: int = 1) -> None:
        """Advance the clock by n bars (used by tests and fast replays)."""
        self._i = min(self._i + n, len(self._clock) - 1)
        self.finished = self._i >= len(self._clock) - 1

    def now(self) -> datetime:
        return self._clock[self._i].to_pydatetime()

    @property
    def progress(self) -> float:
        return self._i / max(len(self._clock) - 1, 1)

    def bars(self, symbol: str) -> pd.DataFrame:
        now = self._clock[self._i]
        day = self._day.get(symbol)
        cur = day[day.index <= now] if day is not None else None
        hist = self._hist.get(symbol)
        parts = [p for p in (hist, cur) if p is not None and len(p)]
        return pd.concat(parts) if parts else pd.DataFrame(columns=FIELDS)

    def last_price(self, symbol: str) -> Optional[float]:
        day = self._day.get(symbol)
        if day is None:
            return None
        cur = day[day.index <= self._clock[self._i]]
        if len(cur):
            return float(cur["Close"].iloc[-1])
        hist = self._hist.get(symbol)
        return float(hist["Close"].iloc[-1]) if hist is not None and len(hist) else None

    def is_stale(self) -> bool:
        return False


# ===========================================================================
# SELF-TESTS (offline)
# ===========================================================================

def _synthetic_session(d: date, symbols: Sequence[str], seed: int = 0) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    o, _c = Session().bounds(d)
    idx = pd.date_range(o, periods=375, freq="1min")
    out = {}
    for s in symbols:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, idx.size)))
        out[s] = pd.DataFrame({"Open": px, "High": px * 1.001, "Low": px * 0.999,
                               "Close": px, "Volume": rng.integers(100, 1000, idx.size).astype(float)},
                              index=idx)
    return out


def _test_session_gate() -> None:
    s = Session()
    fri = date(2026, 9, 25)
    assert s.is_open(datetime(2026, 9, 25, 9, 15, tzinfo=IST))
    assert s.is_open(datetime(2026, 9, 25, 15, 29, 59, tzinfo=IST))
    assert not s.is_open(datetime(2026, 9, 25, 15, 30, tzinfo=IST))
    assert not s.is_open(datetime(2026, 9, 25, 9, 14, 59, tzinfo=IST))
    assert not s.is_open(datetime(2026, 9, 26, 11, 0, tzinfo=IST))       # Saturday
    assert not s.is_open(datetime(2026, 9, 25, 23, 0, tzinfo=IST))       # old kill-switch hole
    assert s.next_open(datetime(2026, 9, 25, 16, 0, tzinfo=IST)) == \
        datetime(2026, 9, 28, 9, 15, tzinfo=IST)                          # Friday -> Monday
    # a UTC timestamp is converted before the check
    assert s.is_open(datetime(2026, 9, 25, 4, 0, tzinfo=timezone.utc))    # 09:30 IST
    print("[ok] session gate: 09:15-15:30 IST Mon-Fri only; 23:00 and weekends closed; "
          "Friday evening -> next open Monday 09:15")


def _test_archive_and_replay_no_lookahead(tmp: str) -> None:
    syms = ["AAA.NS", "BBB.NS"]
    store = IntradayStore(root=tmp)
    days = [date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
    for k, d in enumerate(days):
        store.archive(_synthetic_session(d, syms, seed=k), now=datetime(2026, 9, 26, 12, 0, tzinfo=IST))
    assert store.sessions() == days
    rf = ReplayFeed(session_date=days[-1], symbols=syms, store=store, history_sessions=2)
    assert rf.now().hour == 9 and rf.now().minute == 15
    for _ in range(40):
        rf.step(7)
        now = pd.Timestamp(rf.now())
        for s in syms:
            b = rf.bars(s)
            assert b.index.max() <= now, "replay leaked a future bar"
            assert abs(rf.last_price(s) - float(b["Close"].iloc[-1])) < 1e-12
    assert len(rf.bars("AAA.NS")) == 2 * 375 + (rf._i + 1)   # two history sessions + today so far
    rf.step(10_000)
    assert rf.finished
    # an incomplete session (today, before the close) is not archived
    today = date(2026, 9, 25)
    wrote = store.archive(_synthetic_session(today, syms), now=datetime(2026, 9, 25, 12, 0, tzinfo=IST))
    assert wrote == []
    print("[ok] archive writes completed sessions only; replay never returns a bar later than "
          "its clock, and prepends earlier sessions as history")


def _test_close_panel_before() -> None:
    idx = pd.date_range("2026-09-21", periods=5, freq="D")
    daily = {"A": pd.DataFrame({"Close": np.arange(5.0)}, index=idx)}
    p = close_panel(daily, ["A"], before=date(2026, 9, 24))
    assert p.index.max() == pd.Timestamp("2026-09-23")
    print("[ok] close_panel(before=D) drops day D and later (no look-ahead into the traded day)")


def _test_normalise_formats() -> None:
    idx = pd.date_range("2026-09-21", periods=3, freq="D")
    cols = pd.MultiIndex.from_product([["X.NS", "Y.NS"], FIELDS])
    raw = pd.DataFrame(np.ones((3, 10)), index=idx, columns=cols)
    raw[("Y.NS", "Close")] = np.nan                              # a failed ticker
    out = _normalise(raw, ["X.NS", "Y.NS"])
    assert list(out) == ["X.NS"] and list(out["X.NS"].columns) == FIELDS
    flat = pd.DataFrame(np.ones((3, 5)), index=idx, columns=FIELDS)
    assert list(_normalise(flat, ["Z.NS"])) == ["Z.NS"]
    print("[ok] yfinance frames normalised; tickers with no data are dropped, not zero-filled")


if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _test_session_gate()
    _test_close_panel_before()
    _test_normalise_formats()
    with tempfile.TemporaryDirectory() as tmp:
        _test_archive_and_replay_no_lookahead(tmp)

    if "--fetch" in sys.argv:
        t0 = time.time()
        daily = DailyStore().load(ALL_SYMBOLS)
        print(f"daily: {len(daily)} symbols, e.g. {INDEX_SYMBOL} "
              f"{daily[INDEX_SYMBOL].index[0].date()} .. {daily[INDEX_SYMBOL].index[-1].date()} "
              f"({time.time() - t0:.1f}s)")
        t0 = time.time()
        bars = IntradayStore().fetch_recent(ALL_SYMBOLS)
        print(f"intraday: {len(bars)} symbols; archived sessions: "
              f"{[str(d) for d in IntradayStore().sessions()]} ({time.time() - t0:.1f}s)")

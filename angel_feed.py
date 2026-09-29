"""
angel_feed.py
=============
Read-only market data from Angel One SmartAPI: whole-NSE coverage, live ticks with
5-level order-book depth, recorded to disk every session.

    python angel_feed.py                 offline self-tests (no login needed)
    python angel_feed.py --check         log in and print a few live quotes
    python angel_feed.py --record        record today's session for the whole NSE cash
                                         market (09:10-15:35 IST, or until Ctrl+C)

THIS MODULE CANNOT PLACE ORDERS. ReadOnlySmartConnect blocks every order, GTT,
position-conversion and fund method at the class level. All trading stays in
paper_broker.py with fake money.

CREDENTIALS
    angel_credentials.json in the project folder (git-ignored), or the environment
    variables ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_MPIN, ANGEL_TOTP_SECRET.
    See angel_credentials.example.json. Never commit them; never paste them into a chat.

WHAT IS RECORDED
    SnapQuote packets: last price, last traded quantity and time, average price,
    day volume, total buy/sell quantity, OHLC, and the best 5 bid and ask levels
    (price, quantity, number of orders). Prices arrive in paise and are stored in
    rupees. The SDK's buy/sell labels for the 5 levels can be swapped, so sides are
    assigned by price: the side whose best price is lower is the bid.

    Files: data/ticks/YYYY-MM-DD/HHMMSS_<connection>.parquet (zstd), flushed every
    minute. copy_session_to_archive() packs a finished day into one file per day
    on the Google Drive archive.

SECURITY NOTE
    SmartWebSocketV2.connect() in smartapi-python 1.5.5 turns TLS certificate
    checks OFF (cert_reqs=CERT_NONE) while sending the session token in the
    headers. VerifiedWebSocket re-implements connect() with certificate checks ON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, PROJECT_DIR
from market_data import IST, Session, now_ist

log = logging.getLogger("angel_feed")

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
ANGEL_DIR = os.path.join(DATA_DIR, "angel")
TICK_DIR = os.path.join(DATA_DIR, "ticks")
CREDENTIALS_FILE = os.path.join(PROJECT_DIR, "angel_credentials.json")
ARCHIVE_DIR = r"G:\My Drive\Titan-Q_Ticks"          # Google Drive (5 TB); real data only

NSE_CM = 1                    # exchangeType for NSE cash market
SNAP_QUOTE = 3                # LTP + quote + best 5 levels
TOKENS_PER_CONNECTION = 1000  # Angel's per-connection subscription limit
MAX_CONNECTIONS = 3           # Angel allows a few concurrent websocket connections


# ===========================================================================
# 1. INSTRUMENTS
# ===========================================================================

def load_scrip_master(max_age_hours: float = 20.0) -> List[dict]:
    """Angel's public instrument list (~140,000 rows, ~33 MB), cached daily."""
    import requests

    os.makedirs(ANGEL_DIR, exist_ok=True)
    path = os.path.join(ANGEL_DIR, "scrip_master.json")
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > 3600 * max_age_hours:
        r = requests.get(SCRIP_MASTER_URL, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        log.info("scrip master refreshed (%d bytes)", len(r.content))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nse_equities(scrip: Sequence[dict]) -> pd.DataFrame:
    """Every NSE cash-market equity (series EQ): symbol, token, name, tick size in
    rupees, and the matching Yahoo ticker for daily history."""
    rows = [{"symbol": d["symbol"], "token": str(d["token"]), "name": d["name"],
             "tick_size": float(d.get("tick_size") or 5.0) / 100.0,
             "yahoo": d["name"] + ".NS"}
            for d in scrip if d.get("exch_seg") == "NSE" and str(d.get("symbol", "")).endswith("-EQ")]
    return pd.DataFrame(rows).drop_duplicates("token").sort_values("symbol").reset_index(drop=True)


# ===========================================================================
# 2. CREDENTIALS AND READ-ONLY LOGIN
# ===========================================================================

def load_credentials() -> Dict[str, str]:
    keys = ("api_key", "client_code", "mpin", "totp_secret")
    creds = {k: os.environ.get("ANGEL_" + k.upper(), "") for k in keys}
    if not all(creds.values()) and os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            creds.update({k: v for k, v in json.load(f).items() if k in keys and v})
    missing = [k for k in keys if not creds.get(k)]
    if missing:
        raise RuntimeError(f"Angel One credentials missing: {missing}. Fill in {CREDENTIALS_FILE} "
                           f"(see angel_credentials.example.json).")
    return creds


_BLOCKED = ("placeOrder", "placeOrderFullResponse", "modifyOrder", "cancelOrder",
            "convertPosition", "gttCreateRule", "gttModifyRule", "gttCancelRule",
            "verifyDis", "generateTPIN")


def _make_read_only_class():
    from SmartApi.smartConnect import SmartConnect

    def _refuse(name):
        def method(self, *a, **k):
            raise PermissionError(f"Titan-Q is read-only on Angel One: {name}() is blocked. "
                                  f"All trading is paper trading (paper_broker.py).")
        return method

    return type("ReadOnlySmartConnect", (SmartConnect,), {n: _refuse(n) for n in _BLOCKED})


def login(creds: Optional[Dict[str, str]] = None):
    """Returns (api, auth_token, feed_token, client_code). The api object cannot trade."""
    import pyotp

    creds = creds or load_credentials()
    api = _make_read_only_class()(api_key=creds["api_key"])
    otp = pyotp.TOTP(creds["totp_secret"]).now()
    res = api.generateSession(creds["client_code"], creds["mpin"], otp)
    if not res or not res.get("status"):
        raise RuntimeError(f"Angel One login failed: {res.get('message') if res else res}")
    return api, res["data"]["jwtToken"], api.getfeedToken(), creds["client_code"]


# ===========================================================================
# 3. PACKET NORMALISATION
# ===========================================================================

def _levels(side: Sequence[dict]) -> List[Tuple[float, float, float]]:
    out = []
    for lv in side or []:
        p = float(lv.get("price", 0)) / 100.0
        q = float(lv.get("quantity", 0))
        n = float(lv.get("no of orders", lv.get("num_of_orders", 0)))
        if p > 0:
            out.append((p, q, n))
    return out


def normalise_snapquote(msg: dict, recv_ts: float) -> Optional[dict]:
    """SDK SnapQuote dict -> one flat row in rupees, bids and asks assigned by price."""
    if msg.get("subscription_mode") != SNAP_QUOTE:
        return None
    a = _levels(msg.get("best_5_buy_data"))
    b = _levels(msg.get("best_5_sell_data"))
    # the side with the lower prices is the bid side, whatever the SDK called it
    if a and b and max(p for p, _, _ in a) > min(p for p, _, _ in b):
        a, b = b, a
    elif (a and not b and a[0][0] > msg.get("last_traded_price", 0) / 100.0) or (b and not a and b[0][0] < msg.get("last_traded_price", 0) / 100.0):
        a, b = b, a
    bids = sorted(a, key=lambda x: -x[0])[:5]
    asks = sorted(b, key=lambda x: x[0])[:5]
    row = {
        "recv_ts": recv_ts,
        "exch_ts": msg.get("exchange_timestamp", 0) / 1000.0,
        "token": str(msg.get("token")),
        "seq": int(msg.get("sequence_number", 0)),
        "ltp": msg.get("last_traded_price", 0) / 100.0,
        "ltq": float(msg.get("last_traded_quantity", 0)),
        "ltt": msg.get("last_traded_timestamp", 0),
        "atp": msg.get("average_traded_price", 0) / 100.0,
        "volume": float(msg.get("volume_trade_for_the_day", 0)),
        "total_buy_qty": float(msg.get("total_buy_quantity", 0)),
        "total_sell_qty": float(msg.get("total_sell_quantity", 0)),
        "open": msg.get("open_price_of_the_day", 0) / 100.0,
        "high": msg.get("high_price_of_the_day", 0) / 100.0,
        "low": msg.get("low_price_of_the_day", 0) / 100.0,
        "prev_close": msg.get("closed_price", 0) / 100.0,
    }
    for i in range(5):
        for side, lv in (("bid", bids), ("ask", asks)):
            p, q, n = lv[i] if i < len(lv) else (np.nan, np.nan, np.nan)
            row[f"{side}_px{i + 1}"], row[f"{side}_qty{i + 1}"], row[f"{side}_n{i + 1}"] = p, q, n
    return row


# ===========================================================================
# 4. RECORDER
# ===========================================================================

class TickRecorder:
    """Thread-safe buffer of normalised rows, flushed to parquet every
    `flush_seconds`, plus the latest row per token for live use."""

    def __init__(self, root: str = TICK_DIR, flush_seconds: float = 60.0) -> None:
        self.root = root
        self.flush_seconds = float(flush_seconds)
        self._lock = threading.Lock()
        self._buf: Dict[str, List[dict]] = {}
        self.latest: Dict[str, dict] = {}
        self.rows_total = 0
        self.files_written = 0
        self.bytes_written = 0
        self._last_flush = time.time()

    def add(self, conn: str, row: dict) -> None:
        with self._lock:
            self._buf.setdefault(conn, []).append(row)
            self.latest[row["token"]] = row
            self.rows_total += 1

    def maybe_flush(self, force: bool = False) -> None:
        if not force and time.time() - self._last_flush < self.flush_seconds:
            return
        with self._lock:
            bufs, self._buf = self._buf, {}
            self._last_flush = time.time()
        stamp = datetime.now(IST)
        day_dir = os.path.join(self.root, stamp.strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        for conn, rows in bufs.items():
            if not rows:
                continue
            path = os.path.join(day_dir, f"{stamp.strftime('%H%M%S')}_{conn}.parquet")
            pd.DataFrame(rows).to_parquet(path, compression="zstd", index=False)
            self.files_written += 1
            self.bytes_written += os.path.getsize(path)


LATEST_FILE = os.path.join(TICK_DIR, "latest_prices.json")
TOKEN_MAP_FILE = os.path.join(TICK_DIR, "token_map.json")


def write_latest(rec: "TickRecorder", path: str = LATEST_FILE) -> None:
    """Latest price per token, for other processes (the trading loop's shadow check).
    Written atomically so a reader never sees half a file."""
    with rec._lock:
        snap = {t: {"ltp": r["ltp"], "exch_ts": r["exch_ts"], "recv_ts": r["recv_ts"], "volume": r["volume"]}
                for t, r in rec.latest.items()}
    tmp = path + ".partial"
    with open(tmp, "w", encoding="ascii") as f:
        json.dump({"written": time.time(), "prices": snap}, f)
    os.replace(tmp, path)


def load_session_ticks(day: str, root: str = TICK_DIR) -> pd.DataFrame:
    """All recorded rows for one day (YYYY-MM-DD), sorted by receive time."""
    d = os.path.join(root, day)
    parts = [pd.read_parquet(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".parquet")]
    return pd.concat(parts, ignore_index=True).sort_values("recv_ts") if parts else pd.DataFrame()


def copy_session_to_archive(day: str, root: str = TICK_DIR, archive: str = ARCHIVE_DIR) -> Optional[str]:
    """Pack one recorded day into a single zstd parquet file on the Google Drive
    archive (one file per day, not thousands of small ones)."""
    df = load_session_ticks(day, root)
    if df.empty or not os.path.isdir(os.path.dirname(archive)):
        return None
    os.makedirs(archive, exist_ok=True)
    out = os.path.join(archive, f"ticks_{day}.parquet")
    df.to_parquet(out, compression="zstd", index=False)
    return out


# ===========================================================================
# 5. WEBSOCKET STREAM
# ===========================================================================

def _verified_ws_class():
    import ssl
    import certifi
    import websocket
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2

    class VerifiedWebSocket(SmartWebSocketV2):
        """SmartWebSocketV2 with TLS certificate verification switched back on."""

        def connect(self):
            headers = {"Authorization": self.auth_token, "x-api-key": self.api_key,
                       "x-client-code": self.client_code, "x-feed-token": self.feed_token}
            self.wsapp = websocket.WebSocketApp(
                self.ROOT_URI, header=headers, on_open=self._on_open, on_error=self._on_error,
                on_close=self._on_close, on_data=self._on_data, on_ping=self._on_ping,
                on_pong=self._on_pong)
            self.wsapp.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": certifi.where()},
                                   ping_interval=self.HEART_BEAT_INTERVAL)

        def _on_close(self, wsapp, *args):
            # websocket-client >= 1.x passes (ws, close_status_code, close_msg); the SDK's
            # _on_close accepts only (ws) and raised a TypeError on every shutdown
            self.on_close(wsapp)

    return VerifiedWebSocket


class MarketStream:
    """Splits the token list across up to MAX_CONNECTIONS websocket connections
    (TOKENS_PER_CONNECTION each), subscribes in SnapQuote mode, and feeds every
    packet to the recorder.

    Reconnection is ours, not the SDK's. On 2026-09-29 two connections dropped at
    14:32 and the third at 15:04; the SDK's own retries reconnected while the old
    sessions still counted, Angel answered "429 Connection Limit Exceeded", and its
    retry counter (never reset during the day) then gave up for good. Now each
    connection runs in a supervisor thread: connect (blocks until the connection
    ends), wait (30 s, doubling to 5 min; reset after 5 min of good data), and
    connect again with a fresh socket. `check_stalls` closes a connection that has
    gone silent, so it too is reconnected."""

    RECONNECT_FIRST, RECONNECT_MAX, HEALTHY_AFTER = 30.0, 300.0, 300.0

    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str,
                 tokens: Sequence[str], recorder: TickRecorder, ws_class=None) -> None:
        self.recorder = recorder
        self.creds = (auth_token, api_key, client_code, feed_token)
        cap = TOKENS_PER_CONNECTION * MAX_CONNECTIONS
        tokens = list(dict.fromkeys(tokens))
        if len(tokens) > cap:
            log.warning("%d tokens requested, capacity %d: keeping the first %d", len(tokens), cap, cap)
            tokens = tokens[:cap]
        chunks = [tokens[i:i + TOKENS_PER_CONNECTION] for i in range(0, len(tokens), TOKENS_PER_CONNECTION)]
        self.WS = ws_class or _verified_ws_class()
        self.sockets = [(f"c{k}", chunk) for k, chunk in enumerate(chunks)]
        self.current: Dict[str, object] = {}
        self.last_data: Dict[str, float] = {}
        self.reconnects: Dict[str, int] = {name: 0 for name, _ in self.sockets}
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _make(self, name: str, chunk: List[str]):
        # max_retry_attempt=0: on an error the SDK just closes, and the supervisor decides
        ws = self.WS(*self.creds, max_retry_attempt=0)
        ws.on_open = self._on_open(ws, name, chunk)
        ws.on_data = self._on_data(name)
        ws.on_error = self._on_error(name)
        ws.on_close = lambda wsapp, n=name: log.warning("connection %s closed", n)
        return ws

    def _on_open(self, ws, name, chunk):
        def f(wsapp):
            for i in range(0, len(chunk), 250):          # modest request sizes
                ws.subscribe(f"{name}s{i // 250}"[:10], SNAP_QUOTE,
                             [{"exchangeType": NSE_CM, "tokens": chunk[i:i + 250]}])
            log.info("connection %s subscribed to %d tokens", name, len(chunk))
        return f

    def _on_data(self, name):
        def f(wsapp, msg):
            self.last_data[name] = time.time()
            row = normalise_snapquote(msg, time.time())
            if row is not None:
                self.recorder.add(name, row)
        return f

    def _on_error(self, name):
        def f(*args):
            self.errors.append(f"{name}: {args}")
            log.error("connection %s error: %s", name, str(args)[:300])
        return f

    def _supervise(self, k: int, name: str, chunk: List[str]) -> None:
        self._stop.wait(5.0 * k)                           # stagger the first connects
        backoff = self.RECONNECT_FIRST
        while not self._stop.is_set():
            ws = self._make(name, chunk)
            with self._lock:
                self.current[name] = ws
            t0 = time.time()
            self.last_data[name] = t0
            try:
                ws.connect()                                # blocks until the connection ends
            except Exception as exc:
                log.error("connection %s failed: %s", name, str(exc)[:200])
            if self._stop.is_set():
                return
            up = time.time() - t0
            if up >= self.HEALTHY_AFTER:
                backoff = self.RECONNECT_FIRST
            self.reconnects[name] += 1
            log.warning("connection %s ended after %.0f s; reconnecting in %.0f s (reconnect %d)",
                        name, up, backoff, self.reconnects[name])
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.RECONNECT_MAX)

    def check_stalls(self, max_silence: float = 180.0) -> None:
        """Close any connection with no data for max_silence seconds (call in session)."""
        now = time.time()
        with self._lock:
            for name, ws in self.current.items():
                if now - self.last_data.get(name, now) > max_silence:
                    log.warning("connection %s silent for %.0f s; closing it to reconnect",
                                name, now - self.last_data[name])
                    self.last_data[name] = now
                    try:
                        ws.close_connection()
                    except Exception:
                        pass

    def start(self) -> None:
        for k, (name, chunk) in enumerate(self.sockets):
            threading.Thread(target=self._supervise, args=(k, name, chunk), name=f"angel-{name}",
                             daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for ws in self.current.values():
                try:
                    ws.close_connection()
                except Exception:
                    pass


# ===========================================================================
# 6. COMMANDS
# ===========================================================================

def cmd_check() -> None:
    api, jwt, feed, client = login()
    print("login ok; client", client, "; feed token", "present" if feed else "MISSING")
    eq = nse_equities(load_scrip_master())
    probe = eq[eq["name"].isin(["RELIANCE", "TCS", "HDFCBANK"])]
    res = api.getMarketData("FULL", {"NSE": probe["token"].tolist()})
    for q in (res or {}).get("data", {}).get("fetched", []):
        print(f"  {q.get('tradingSymbol'):<14} ltp Rs. {q.get('ltp')}  vol {q.get('tradeVolume')}")
    try:
        api.placeOrder({})
    except PermissionError as exc:
        print("read-only guard ok:", str(exc)[:60], "...")


def cmd_record(until: Optional[str] = None) -> None:
    session = Session()
    now = now_ist()
    if not until and (now.weekday() >= 5 or now >= session.bounds(now.date())[1]):
        log.info("market closed today (%s); nothing to record", now.strftime("%a %d %b %H:%M IST"))
        return
    try:
        creds = load_credentials()
    except RuntimeError as exc:
        log.error("%s Recording skipped.", exc)
        return
    eq = nse_equities(load_scrip_master())
    tokens = eq["token"].tolist()
    api, jwt, feed, client = login(creds)
    rec = TickRecorder()
    os.makedirs(TICK_DIR, exist_ok=True)
    with open(TOKEN_MAP_FILE, "w", encoding="ascii") as f:                 # token -> Yahoo ticker
        json.dump(dict(zip(eq["token"], eq["yahoo"])), f)
    stream = MarketStream(jwt, creds["api_key"], client, feed, tokens, rec)
    stream.start()
    end = (datetime.combine(now_ist().date(), datetime.strptime(until, "%H:%M").time(), IST)
           if until else session.bounds(now_ist().date())[1] + timedelta(minutes=5))
    log.info("recording %d NSE equities on %d connections until %s", len(tokens),
             len(stream.sockets), end.strftime("%H:%M IST"))
    last_report, last_rows = time.time(), 0
    last_growth, rows_seen = time.time(), 0
    last_latest = 0.0
    if os.name == "nt":                     # keep a woken laptop awake until the recording ends
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    try:
        while now_ist() < end:
            time.sleep(2.0)
            rec.maybe_flush()
            if time.time() - last_latest >= 10:
                last_latest = time.time()
                try:
                    write_latest(rec)
                except Exception as exc:
                    log.warning("latest prices not written: %s", exc)
            if session.is_open(now_ist()):
                stream.check_stalls()
                if rec.rows_total != rows_seen:
                    last_growth, rows_seen = time.time(), rec.rows_total
                elif time.time() - last_growth > 600:
                    try:
                        from alerts import send
                        send("tick recorder silent", f"No ticks for {(time.time() - last_growth) / 60:.0f} "
                             "minutes during the session (network/VPN down?). It keeps reconnecting.",
                             priority="high", tags="warning", repeat_minutes=30)
                    except Exception:
                        pass
            if time.time() - last_report >= 60:
                rate = (rec.rows_total - last_rows) / (time.time() - last_report)
                log.info("rows %d (%.0f/s) | tokens seen %d | files %d | %.1f MB written",
                         rec.rows_total, rate, len(rec.latest), rec.files_written,
                         rec.bytes_written / 1e6)
                last_report, last_rows = time.time(), rec.rows_total
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        stream.stop()
        rec.maybe_flush(force=True)
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)      # allow sleep again
        day = now_ist().strftime("%Y-%m-%d")
        out = copy_session_to_archive(day)
        log.info("session %s: %d rows, %d files, %.1f MB; archive: %s", day, rec.rows_total,
                 rec.files_written, rec.bytes_written / 1e6, out or "Google Drive not available")


# ===========================================================================
# SELF-TESTS (offline)
# ===========================================================================

def _test_normalise_assigns_sides_by_price() -> None:
    lv = lambda ps: [{"flag": 0, "quantity": 100 + i, "price": int(round(p * 100)), "no of orders": 3}
                     for i, p in enumerate(ps)]
    base = {"subscription_mode": 3, "token": "2885", "sequence_number": 7,
            "exchange_timestamp": 1790400000000, "last_traded_price": 125050,
            "last_traded_quantity": 10, "average_traded_price": 124900,
            "volume_trade_for_the_day": 12345, "total_buy_quantity": 5e4, "total_sell_quantity": 4e4,
            "open_price_of_the_day": 124000, "high_price_of_the_day": 126000,
            "low_price_of_the_day": 123500, "closed_price": 124200, "last_traded_timestamp": 1790399999}
    bids, asks = [1250.4, 1250.3, 1250.2, 1250.1, 1250.0], [1250.6, 1250.7, 1250.8, 1250.9, 1251.0]
    for swapped in (False, True):
        m = dict(base)
        m["best_5_buy_data"], m["best_5_sell_data"] = (lv(asks), lv(bids)) if swapped else (lv(bids), lv(asks))
        r = normalise_snapquote(m, 1.0)
        assert r["bid_px1"] == 1250.4 and r["ask_px1"] == 1250.6, (swapped, r["bid_px1"], r["ask_px1"])
        assert r["bid_px1"] < r["ask_px1"] and r["ltp"] == 1250.5 and r["exch_ts"] == 1790400000.0
    assert normalise_snapquote({"subscription_mode": 1}, 0.0) is None
    print("[ok] SnapQuote normalised to rupees; bid/ask assigned by price even when the SDK "
          "swaps the labels")


def _test_read_only_guard() -> None:
    cls = _make_read_only_class()
    api = cls(api_key="dummy")
    for name in _BLOCKED:
        try:
            getattr(api, name)({})
            raise AssertionError(f"{name} was not blocked")
        except PermissionError:
            pass
    try:
        api.placeOrder({"tradingsymbol": "SBIN-EQ", "quantity": 1})
        raise AssertionError("placeOrder was not blocked")
    except PermissionError:
        pass
    print(f"[ok] read-only guard: {len(_BLOCKED)} order/GTT/position methods raise PermissionError")


def _test_recorder_roundtrip(tmp: str) -> None:
    rec = TickRecorder(root=tmp, flush_seconds=0)
    for i in range(50):
        rec.add("c0", {"recv_ts": 1.0 + i, "token": str(i % 5), "ltp": 100.0 + i, "bid_px1": 99.0})
    rec.maybe_flush(force=True)
    day = datetime.now(IST).strftime("%Y-%m-%d")
    df = load_session_ticks(day, tmp)
    assert len(df) == 50 and len(rec.latest) == 5 and rec.files_written == 1
    out = copy_session_to_archive(day, tmp, os.path.join(tmp, "archive"))
    assert out and len(pd.read_parquet(out)) == 50
    print("[ok] recorder: rows flushed to zstd parquet, read back, packed into one archive file")


def _test_stream_reconnects() -> None:
    """Dropped connections come back with a fresh socket after a wait; a silent one is
    closed by check_stalls; stop() ends every supervisor."""
    made: List["Fake"] = []

    class Fake:
        def __init__(self, *creds, max_retry_attempt=1):
            assert max_retry_attempt == 0                   # SDK retries must be off
            self.closed = threading.Event()
            self.drop = len(made) < 2                       # the first two drop at once
            made.append(self)

        def connect(self):
            self.on_open(None)
            self.on_data(None, {})
            if not self.drop:
                self.closed.wait(5.0)                       # a healthy connection stays up

        def subscribe(self, *args):
            pass

        def close_connection(self):
            self.closed.set()

    class Quick(MarketStream):
        RECONNECT_FIRST, RECONNECT_MAX = 0.05, 0.2

    s = Quick("jwt", "key", "client", "feed", [str(i) for i in range(10)], TickRecorder(root="."), ws_class=Fake)
    s.start()
    deadline = time.time() + 3.0
    while len(made) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert len(made) == 3 and s.reconnects["c0"] == 2, (len(made), s.reconnects)
    time.sleep(0.05)
    s.last_data["c0"] = time.time() - 999                   # silent for too long
    s.check_stalls(max_silence=180.0)
    deadline = time.time() + 3.0
    while len(made) < 4 and time.time() < deadline:
        time.sleep(0.01)
    assert made[2].closed.is_set() and len(made) == 4, len(made)
    s.stop()
    assert all(f.closed.is_set() for f in made[3:])
    print("[ok] stream: dropped connections reconnect after a wait with a fresh socket (SDK retries "
          "off); a silent connection is closed and reconnected; stop() ends all")


if __name__ == "__main__":
    import sys
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="log in and print a few live quotes")
    ap.add_argument("--record", action="store_true", help="record today's session")
    ap.add_argument("--until", default=None, help="record until HH:MM IST (default: close + 5 min)")
    a = ap.parse_args()
    handlers = [logging.StreamHandler()]
    if a.record:
        logs = os.path.join(PROJECT_DIR, "logs")
        os.makedirs(logs, exist_ok=True)
        handlers.append(logging.FileHandler(
            os.path.join(logs, datetime.now().strftime("angel_record_%Y-%m-%d.log")), encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=handlers, force=True)
    if a.check:
        cmd_check()
    elif a.record:
        cmd_record(a.until)
    else:
        _test_normalise_assigns_sides_by_price()
        _test_read_only_guard()
        with tempfile.TemporaryDirectory() as tmp:
            _test_recorder_roundtrip(tmp)
        _test_stream_reconnects()

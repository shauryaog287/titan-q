"""
alerts.py
=========
Push alerts to the user's phone through ntfy (https://ntfy.sh): free, no account. The
user installs the ntfy app and subscribes to the private topic in alerts_config.json
(git-ignored; a long random name, so only someone who knows it can read the alerts).

    D:\\trader_env\\Scripts\\python.exe alerts.py --test          send a test alert
    D:\\trader_env\\Scripts\\python.exe alerts.py --preflight     pre-open health check (scheduled 08:50)

Messages carry status only (running/stopped, feeds, paper P&L), never credentials.
The same alert title is not repeated within `repeat_minutes`, so a stuck condition
does not flood the phone. Sending never raises: an alert failure must not stop trading.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import socket
import ssl
import time
from typing import List, Optional

import requests

from titanq_config import LOGS_DIR, PROJECT_DIR

log = logging.getLogger("alerts")

CONFIG_FILE = os.path.join(PROJECT_DIR, "alerts_config.json")
STATE_FILE = os.path.join(LOGS_DIR, "alerts_state.json")
SERVER = "https://ntfy.sh"


def config() -> dict:
    """The alert settings; creates a private random topic on first use."""
    if not os.path.exists(CONFIG_FILE):
        cfg = {"ntfy_topic": "titanq-" + secrets.token_hex(12), "enabled": True}
        with open(CONFIG_FILE, "w", encoding="ascii") as f:
            json.dump(cfg, f, indent=1)
    with open(CONFIG_FILE, encoding="ascii") as f:
        return json.load(f)


def _recent(title: str, repeat_minutes: float) -> bool:
    try:
        with open(STATE_FILE, encoding="ascii") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    now = time.time()
    if now - st.get(title, 0) < 60 * repeat_minutes:
        return True
    st[title] = now
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="ascii") as f:
        json.dump({k: v for k, v in st.items() if now - v < 86400}, f)
    return False


def send(title: str, message: str, priority: str = "default", tags: str = "",
         repeat_minutes: float = 30.0) -> bool:
    """priority: min, low, default, high, urgent. Returns True if delivered."""
    try:
        cfg = config()
        if not cfg.get("enabled", True) or _recent(title, repeat_minutes):
            return False
        a = lambda s: s.encode("ascii", "replace").decode("ascii")
        r = requests.post(f"{SERVER}/{cfg['ntfy_topic']}", data=a(message).encode("ascii"),
                          headers={"Title": a("Titan-Q: " + title), "Priority": priority, "Tags": tags},
                          timeout=10)
        ok = r.status_code == 200
        log.info("alert %s: %s (%s)", "sent" if ok else f"failed HTTP {r.status_code}", title, priority)
        return ok
    except Exception as exc:
        log.warning("alert not sent (%s): %s", type(exc).__name__, title)
        return False


# ===========================================================================
# PRE-OPEN HEALTH CHECK
# ===========================================================================

HOSTS = {"Yahoo (prices)": "query1.finance.yahoo.com", "Angel One (recorder)": "apiconnect.angelone.in",
         "NSE (disclosures)": "www.nseindia.com"}


def tls_ok(host: str) -> Optional[str]:
    """None if a verified HTTPS connection works, else the reason."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with socket.create_connection((host, 443), timeout=15) as s, ctx.wrap_socket(s, server_hostname=host):
            return None
    except ssl.SSLCertVerificationError:
        return "certificate check failed (network intercepting HTTPS; is the VPN on?)"
    except Exception as exc:
        return f"{type(exc).__name__}"


def on_battery() -> Optional[int]:
    """Battery percent if running on battery (Windows power status), else None."""
    if os.name != "nt":
        return None
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    st = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return None
    return int(st.BatteryLifePercent) if st.ACLineStatus == 0 else None


def refresh_holidays(max_age_days: float = 25.0) -> None:
    """Keep NSE's trading-holiday list (data/reference/nse_holidays.json) fresh."""
    from market_data import HOLIDAYS_FILE
    if os.path.exists(HOLIDAYS_FILE) and time.time() - os.path.getmtime(HOLIDAYS_FILE) < max_age_days * 86400:
        return
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                          "Accept": "application/json, text/plain, */*",
                          "Referer": "https://www.nseindia.com/resources/exchange-communication-holidays"})
        try:
            s.get("https://www.nseindia.com/", timeout=30)
        except requests.RequestException:
            pass
        cm = s.get("https://www.nseindia.com/api/holiday-master?type=trading", timeout=60).json().get("CM", [])
        if cm:
            os.makedirs(os.path.dirname(HOLIDAYS_FILE), exist_ok=True)
            with open(HOLIDAYS_FILE, "w", encoding="ascii") as f:
                json.dump(cm, f, ensure_ascii=True)
            log.info("NSE holiday list refreshed (%d dates)", len(cm))
    except Exception as exc:
        log.warning("NSE holiday list not refreshed (%s)", type(exc).__name__)


def preflight() -> List[str]:
    from market_data import Session, now_ist
    refresh_holidays()
    today = now_ist().date()
    if not Session().is_trading_day(today):
        send("market holiday", f"{today:%a %d %b}: NSE is closed today; no session, no trading.",
             priority="low", tags="palm_tree", repeat_minutes=600)
        return []
    problems = [f"{name}: {why}" for name, host in HOSTS.items() for why in [tls_ok(host)] if why]
    power = on_battery()
    if power is not None:
        problems.append(f"laptop on battery ({power}%): plug in the charger")
    free_gb = shutil.disk_usage(PROJECT_DIR).free / 1e9
    if free_gb < 10:
        problems.append(f"only {free_gb:.1f} GB free on D:")
    if problems:
        send("pre-open check FAILED", "\n".join(problems), priority="urgent", tags="warning", repeat_minutes=60)
    else:
        send("pre-open check ok", "Network, power and disk look fine for today's session.", priority="low",
             tags="white_check_mark", repeat_minutes=60)
    return problems


if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.path.join(LOGS_DIR, "alerts.log"), encoding="utf-8")], force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    if a.preflight:
        p = preflight()
        print("preflight:", "ok" if not p else "; ".join(p))
    else:
        print("topic:", config()["ntfy_topic"])
        print("sent" if send("test alert", "If you can read this, Titan-Q alerts reach your phone.",
                             repeat_minutes=0) else "not sent")

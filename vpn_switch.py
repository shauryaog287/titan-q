"""
vpn_switch.py
=============
Turns X-VPN on or off depending on which Wi-Fi the laptop is on, so the trading system
always has a connection that does not intercept HTTPS.

    D:\\trader_env\\Scripts\\pythonw.exe vpn_switch.py     (scheduled task TitanQ_VPN_Switch: every
                                                         2 minutes and whenever a network connects)

- On "Mayo Students-M" (school Wi-Fi; it intercepts HTTPS to Yahoo, Angel, NSE and GitHub,
  and is switched off from about 22:45 until early morning): start X-VPN if it is not
  running, then check after 90 s that Yahoo/Angel/NSE pass a certificate-verified
  connection; if not, send a phone alert (X-VPN may need its "connect on launch" setting,
  or a manual Connect).
- On "Anjali - RES" (no interception): close X-VPN if it is running.
- Any other network: leave X-VPN as it is.

X-VPN is a Microsoft Store app; it is started through its app ID and stopped by ending its
processes (xvpn.exe and its own SystrayComponent.exe). Log: logs\\vpn_switch.log.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHOOL_SSID = "Mayo Students-M"
HOME_SSID = "Anjali - RES"
XVPN_APP = r"shell:AppsFolder\29645FreeConnectedLimited.X-VPN-FreeUnlimitedVPNPr_qjvpctbgym0d0!App"
CHECK_HOSTS = ("query1.finance.yahoo.com", "apiconnect.angelone.in", "www.nseindia.com")
NO_WINDOW = 0x08000000

log = logging.getLogger("vpn_switch")


def run(cmd) -> str:
    out = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    return out.stdout.decode("utf-8", "replace").encode("ascii", "replace").decode("ascii")


def current_ssid() -> str:
    for line in run(["netsh", "wlan", "show", "interfaces"]).splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2 and parts[0].strip() == "SSID":
            return parts[1].strip()
    return ""


def xvpn_running() -> bool:
    return "xvpn.exe" in run(["tasklist", "/FI", "IMAGENAME eq xvpn.exe", "/NH"]).lower()


def start_xvpn() -> None:
    subprocess.Popen(["explorer.exe", XVPN_APP], creationflags=NO_WINDOW)


def stop_xvpn() -> None:
    run(["taskkill", "/IM", "xvpn.exe", "/F"])
    # SystrayComponent.exe is a generic name: end only X-VPN's own copy
    run(["powershell", "-NoProfile", "-Command",
         "Get-Process SystrayComponent -ErrorAction SilentlyContinue | "
         "Where-Object { $_.Path -like '*X-VPN*' } | Stop-Process -Force"])


def blocked_hosts() -> list:
    import socket
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    bad = []
    for h in CHECK_HOSTS:
        try:
            with socket.create_connection((h, 443), timeout=15) as s, ctx.wrap_socket(s, server_hostname=h):
                pass
        except Exception as exc:
            bad.append(f"{h} ({type(exc).__name__})")
    return bad


def alert(title: str, message: str) -> None:
    try:
        import sys
        sys.path.insert(0, PROJECT_DIR)
        from alerts import send
        send(title, message, priority="high", tags="warning", repeat_minutes=60)
    except Exception:
        pass


def main() -> None:
    ssid = current_ssid()
    running = xvpn_running()
    if ssid == SCHOOL_SSID:
        if not running:
            log.info("on %s without X-VPN: starting it", ssid)
            start_xvpn()
            time.sleep(90)
            bad = blocked_hosts()
            if bad:
                log.warning("X-VPN started but still blocked: %s", ", ".join(bad))
                alert("VPN not connected", "On school Wi-Fi X-VPN was opened but finance sites are still "
                      "blocked (" + ", ".join(bad) + "). Open X-VPN and press Connect, and turn on its "
                      "auto-connect setting.")
            else:
                log.info("X-VPN up: Yahoo, Angel and NSE pass")
    elif ssid == HOME_SSID:
        if running:
            log.info("on %s: closing X-VPN (not needed here)", ssid)
            stop_xvpn()
    # any other network or no Wi-Fi: leave X-VPN as it is


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "vpn_switch.log"),
                                                      encoding="utf-8")], force=True)
    try:
        main()
    except Exception as exc:
        log.warning("vpn_switch failed: %s: %s", type(exc).__name__, exc)

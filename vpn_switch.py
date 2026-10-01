"""
vpn_switch.py
=============
Turns X-VPN on or off depending on which Wi-Fi the laptop is on, so the trading system
always has a connection that does not intercept HTTPS.

    D:\\trader_env\\Scripts\\pythonw.exe vpn_switch.py     (scheduled task TitanQ_VPN_Switch: every
                                                         2 minutes and whenever a network connects)

- On "Mayo Students-M" (school Wi-Fi; it intercepts HTTPS to Yahoo, Angel, NSE and GitHub,
  and carries no traffic at all from about 22:45 until early morning):
  - if Yahoo/Angel/NSE pass a certificate-verified connection, nothing to do;
  - if there is no internet at all (the night shutdown), wait;
  - otherwise start X-VPN if needed, press its Connect button (xvpn_connect.ps1, Windows
    UI Automation; X-VPN has no command line), wait 45 s and check again; one retry, then
    a phone alert.
- On "Anjali - RES" (no interception): close X-VPN if it is running.
- Any other network: leave X-VPN as it is.

X-VPN is a Microsoft Store app; it is started through its app ID and stopped by ending its
processes (xvpn.exe and its own SystrayComponent.exe). Tested 2026-10-01: pressing Connect
moved the exit to Singapore within ~40 s. Log: logs\\vpn_switch.log.
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


def internet_up() -> bool:
    """Any internet at all (the school Wi-Fi carries none from about 22:45 to the morning)."""
    import socket
    try:
        socket.create_connection(("www.google.com", 443), timeout=10).close()
        return True
    except OSError:
        return False


def press_connect() -> str:
    """X-VPN has no command line: press its Connect button via UI Automation."""
    return run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                os.path.join(PROJECT_DIR, "xvpn_connect.ps1")]).strip()


def main() -> None:
    ssid = current_ssid()
    running = xvpn_running()
    if ssid == SCHOOL_SSID:
        bad = blocked_hosts()
        if not bad:
            return                                    # VPN already working (or not needed)
        if not internet_up():
            return                                    # network flow off (night): nothing to do
        log.info("on %s, finance sites blocked (%s): connecting X-VPN", ssid, ", ".join(bad))
        for attempt in (1, 2):
            if not xvpn_running():
                start_xvpn()
                time.sleep(15)
            log.info("X-VPN: %s", press_connect() or "no answer from the UI helper")
            time.sleep(45)                            # the tunnel takes ~30-45 s to come up
            bad = blocked_hosts()
            if not bad:
                log.info("X-VPN connected: Yahoo, Angel and NSE pass (attempt %d)", attempt)
                return
        log.warning("X-VPN still not working: %s", ", ".join(bad))
        alert("VPN not connected", "On school Wi-Fi, X-VPN was opened and Connect pressed twice, but "
              "finance sites are still blocked (" + ", ".join(bad) + "). Check X-VPN.")
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

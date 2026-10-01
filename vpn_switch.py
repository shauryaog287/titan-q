"""
vpn_switch.py
=============
Turns the VPN on or off depending on which Wi-Fi the laptop is on, so the trading system
always has a connection that does not intercept HTTPS.

    D:\\trader_env\\Scripts\\pythonw.exe vpn_switch.py     (scheduled task TitanQ_VPN_Switch: every
                                                         2 minutes and whenever a network connects)

- On "Mayo Students-M" (school Wi-Fi; it intercepts HTTPS to Yahoo, Angel, NSE and GitHub,
  and carries no traffic at all from about 22:45 until early morning):
  - if Yahoo/Angel/NSE pass a certificate-verified connection, nothing to do;
  - if there is no internet at all (the night shutdown), wait;
  - otherwise Proton VPN first (primary since 2026-10-01): close X-VPN, open Proton, press
    its Connect button (proton_connect.ps1), wait until it says "Protected" (up to 75 s)
    and check again;
  - if Proton does not get through, X-VPN as before (xvpn_connect.ps1, two attempts) after
    disconnecting Proton; then a phone alert.
  Only one VPN runs at a time.
- On "Anjali - RES" (no interception): disconnect Proton and close X-VPN if they are on.
- Any other network: leave both as they are.

Both apps have no command line, so their buttons are pressed through Windows UI Automation.
- Proton VPN 5.1.5 (C:\\Program Files\\Proton\\VPN): fixed element ids. Tested on 2026-10-01
  on the school Wi-Fi: it was "Protected" in 32 s and all three sites passed. Its kill switch
  is off, so disconnecting never cuts the internet.
- X-VPN (a Microsoft Store app): started through its app ID and stopped by ending its
  processes. It hung on its loading screen 14:40-15:41 on 2026-10-01.
- Cloudflare WARP was tried on 2026-10-01 and cannot connect on the school Wi-Fi.

Log: logs\\vpn_switch.log.
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
PROTON_APP = r"C:\Program Files\Proton\VPN\ProtonVPN.Launcher.exe"
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


def proton(action: str) -> str:
    """Proton VPN's button via UI Automation: 'connect', 'disconnect' or 'status'."""
    return run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                os.path.join(PROJECT_DIR, "proton_connect.ps1"), action]).strip()


def proton_running() -> bool:
    return "protonvpn.client.exe" in run(["tasklist", "/FI", "IMAGENAME eq ProtonVPN.Client.exe", "/NH"]).lower()


def open_proton() -> None:
    """Starts Proton VPN, or brings its window back if it sits in the tray (single instance)."""
    if os.path.exists(PROTON_APP):
        subprocess.Popen([PROTON_APP], creationflags=NO_WINDOW)


def try_proton() -> bool:
    """Connect with Proton VPN; True if the finance sites pass afterwards."""
    if not os.path.exists(PROTON_APP):
        return False
    if xvpn_running():
        stop_xvpn()                                   # one VPN at a time
        time.sleep(5)
    open_proton()
    time.sleep(8)
    result = proton("connect") or "no answer from the UI helper"
    log.info("Proton: %s", result)
    if not (result.startswith("CLICKED") or result.startswith("ALREADY")):
        return False
    t0 = time.time()
    while time.time() - t0 < 75:
        st = proton("status")
        if st.startswith("STATUS Protected"):
            break
        time.sleep(5)
    time.sleep(5)
    bad = blocked_hosts()
    if not bad:
        log.info("Proton connected (%s): Yahoo, Angel and NSE pass", proton("status"))
        return True
    log.warning("Proton did not get through (%s): %s", proton("status"), ", ".join(bad))
    proton("disconnect")
    time.sleep(5)
    return False


def main() -> None:
    ssid = current_ssid()
    running = xvpn_running()
    if ssid == SCHOOL_SSID:
        bad = blocked_hosts()
        if not bad:
            return                                    # VPN already working (or not needed)
        if not internet_up():
            return                                    # network flow off (night): nothing to do
        log.info("on %s, finance sites blocked (%s): connecting Proton VPN", ssid, ", ".join(bad))
        if try_proton():
            return
        log.info("falling back to X-VPN")
        for attempt in (1, 2):
            if not xvpn_running():
                start_xvpn()
                time.sleep(15)
            result = press_connect() or "no answer from the UI helper"
            log.info("X-VPN: %s", result)
            if result.startswith("LOADING") or result.startswith("NOWINDOW"):
                # stuck on the loading screen (X in the middle): X-VPN cannot reach its
                # servers. Restart the app for the next attempt instead of waiting on it.
                stop_xvpn()
                time.sleep(5)
                continue
            time.sleep(45)                            # the tunnel takes ~30-45 s to come up
            bad = blocked_hosts()
            if not bad:
                log.info("X-VPN connected: Yahoo, Angel and NSE pass (attempt %d)", attempt)
                return
        bad = blocked_hosts()
        if not bad:
            return
        log.warning("no VPN got through: %s", ", ".join(bad))
        alert("VPN not connected", "On school Wi-Fi, neither Proton VPN nor X-VPN could connect. "
              "Finance sites still blocked: " + ", ".join(bad) + ". It retries every 2 minutes.")
    elif ssid == HOME_SSID:
        if running:
            log.info("on %s: closing X-VPN (not needed here)", ssid)
            stop_xvpn()
        if proton_running():
            st = proton("status")
            if st.startswith("STATUS Protected") or st.startswith("STATUS Connecting"):
                log.info("on %s: disconnecting Proton VPN (not needed here): %s", ssid, proton("disconnect"))
    # any other network or no Wi-Fi: leave both as they are


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "vpn_switch.log"),
                                                      encoding="utf-8")], force=True)
    try:
        main()
    except Exception as exc:
        log.warning("vpn_switch failed: %s: %s", type(exc).__name__, exc)

# Overnight status

## 2026-09-28, 20:15 IST

### Running now (unattended, survives this terminal closing)

- **Angel 1-minute history download** (`angel_history.py`, pid 28340):
  - 9,146 of 67,122 chunks done at 20:08. By 20:20 the speed had dropped to about 70 a
    minute, apparently near Angel's hourly cap, so the finish is expected between 07:00 and
    09:00. If it is not done by 08:55, it pauses for the session and finishes after 15:50,
    and M1 and VPIN then run tomorrow evening instead of tomorrow morning;
  - 98 chunks (1.3%) hit Angel's "exceeding access rate" limit; they are retried
    automatically;
  - it pauses itself at 08:55 on weekdays and resumes at 15:50.
- **`after_download.ps1`**: waits for the download, then:
  - makes 2 retry passes for failed chunks;
  - runs M1 on development data only and commits the report;
  - backs up to `G:\My Drive\Titan-Q_History` after market hours.
- **`after_m1.ps1`**: after M1, runs the VPIN gate test on development data only and commits
  the report.
- Progress for both chains: `logs\after_download.log`. No test period is revealed by either.

### Done tonight

- **Google Drive backup:** `backup_to_drive.py` packs about 700 symbols into one file each,
  plus one file for all daily data, instead of about 70,000 small files. It also runs every
  Saturday at 10:00 (task `TitanQ_Drive_Backup`). It never deletes anything.
- **Bug fix, `start_titan_q.bat`:** it ended with `pause`, so the scheduled task never
  finished and the next day's 09:10 start would have been skipped (the task is set to
  "do not start a new instance"). It now ends by itself. The file also had Unix line
  endings; converted to Windows ones.
- **Forward test fallback:** the daily forward-test update now also runs at 16:30 on
  weekdays (task `TitanQ_Forward_Test`), in case the live loop does not reach the close.
  A lock prevents double recording; a second run on the same day does nothing (tested).
- **NIFTY open data check:** on 125 days (Mar-Sep 2026), Yahoo's daily open equals
  Angel's 09:15 candle exactly, so the open is not a data error. In this window
  open-to-close averaged +0.3 bps/day, not -12.6. Full check to follow once the download
  reaches 2020.
- **M1 pre-registration, amendment 1:** made before any result. The 20-day indicator
  warm-up would have silently dropped 2021 Q1 from the development period; now fixed and
  documented.
- **VPIN gate test pre-registered** (`reports/preregistration_vpin_gate.md`) before any
  result. It asks whether delaying a buy while toxicity is high gets a better price
  relative to NIFTY. The research code was checked to give exactly the live score (84 of
  84 comparisons identical).
- **Live start for 09:10 tomorrow:** the replay test of the current config passed (46
  fills, 0 errors, VPIN in observe mode), and the scheduled task is ready.

### For you

- Keep the laptop on and awake; leave this terminal open if you want me to check at
  06:37 (download and M1) and 09:23 (live start).
- Still pending from earlier:
  - change the Angel MPIN and reset TOTP, then update `angel_credentials.json` yourself;
  - optionally delete `G:\My Drive\NSE_Data_Lake` (the old synthetic files).

## 2026-09-28, 22:50 IST: update

- **Network:** the Wi-Fi "Mayo Students-M" intercepts HTTPS to Yahoo Finance, Angel One and
  NSE, and blocks GitHub. It started intermittently at 20:57. All downloads are stalled until
  the laptop is on another network (phone hotspot or home Wi-Fi). Tomorrow's 09:10 paper
  trading needs that too. Never install the network's certificate, and never click through
  certificate warnings.
- **Downloader:** rewritten to stay under Angel's ~5,000 requests an hour and to wait out
  network failures. The running copy is the older version. Stop it once on a good network
  (`Stop-Process -Id 34424,22576 -Force`), and it will be restarted with the new code.
- **M1 and VPIN:** they will not run until the research set is at least 99.5% complete.
- **INS1 (insider buys):** passed development (+2.28%) but failed its test (-1.42%). Not used.
- **INS2 (pledge invocations):** did not qualify in development (52 events, t -0.94).
- **Disclosure radar (`disclosure_radar.py`):** built and scheduled weekdays at 19:45. Its
  first full run is pending a working network.
- **Next studies:** delivery % and retail-frenzy reversal (DEL1), commodity-linked stocks,
  sector momentum (PSU and defence), intraday patterns, F&O participant positioning, order wins.
- **One commit is not pushed yet** (GitHub is blocked on this network).

## 2026-09-29, 07:30 IST: morning check

- **Minute download (research set):** 3,380 of 22,731 chunks done, 28 failed (those get
  re-swept). It stalled overnight because of the Wi-Fi interception, from about 22:34 until
  the VPN came on (exit in Zurich). It is running again at about 69 a minute.
  - It pauses itself 08:55-15:50 (the quiet-hours check runs before every request).
  - Expected: about 9,600 done by 08:55, the research set complete around 19:00 tonight.
- **M1 and VPIN:** not run yet, correctly; no marker, so the data is incomplete.
  `after_download.ps1` (pid 27496) waits for the marker, then runs M1 and VPIN on
  development data only and commits both. `after_m1.ps1` is superseded. No test period has
  been revealed.
- **During market hours:** nothing heavy runs. The downloader sleeps, and the research
  chain waits for the marker.
- **Pushed:** the commits that were held back overnight are on GitHub.

## 2026-09-29, 09:30 IST: live-start check

- **Scheduled start:** worked. The task ran at 09:10 with result 0, which also confirms the
  `pause` fix. The trading loop, the Angel recorder and the dashboard are all running.
- **Angel recorder:** logged in through the VPN and receiving the whole market: 2,702
  stocks on 3 connections, about 370 rows a second, about 1.7 MB a minute on disk.
- **Dashboard:** serves on http://127.0.0.1:5000.
- **Bug, fixed in code (commit 9e62d09):**
  - The day's plan had 0% exposure ("Kelly f nan"). Yahoo has no 2026-09-28 bar for 7
    of the 50 stocks (HCLTECH, ULTRACEMCO, NESTLEIND, CIPLA, APOLLOHOSP, UPL, SBILIFE).
  - The sizing used the last row's close, which was NaN, and `max(0, nan)` silently
    returned 0.
  - It now uses the last known close, and a NaN sizing is logged as an explicit reason
    instead of hiding.
  - Rebuilt for today, the plan is 100% invested in all 50 (inverse volatility).
  - The running loop still has this morning's plan: it must be restarted to trade today.

## 2026-09-29, 15:15 IST: what happened today

- **Paper trading:** the loop kept this morning's 0% plan, because it was not restarted after
  the fix. So there were no trades today. The fix (commit 9e62d09) applies automatically
  from tomorrow's 09:10 start.
- **Recorder:** it recorded the whole market from 09:10 to 14:32, 10.7 million rows and
  about 740 MB.
  - At 14:32 two of its three connections dropped, and the third at 15:04.
  - The Angel library's own retries hit "429 Connection Limit Exceeded" and gave up.
  - Result: about a third of the market was recorded from 14:32, and nothing from 15:04.
- **Recorder fix (angel_feed.py):**
  - Our own supervisor reconnects each connection with a fresh socket after a wait
    (30 s, doubling to 5 min). The library's retries are off.
  - A watchdog closes and reconnects any connection that is silent for 3 minutes.
  - An offline self-test covers both.
  - The fix applies from tomorrow.

## 2026-09-29, 15:55 IST: post-close check

- **Live session:** closed normally at 15:45 with 0 trades (this morning's 0% plan, fixed
  since), today's bars archived, and the forward-test update started.
- **Replay of today with the fixed code:** Kelly 1.000, exposure 100% in 50 stocks, 46 trades
  (4 stocks cost more per share than their target), fees Rs. 503.26, PnL Rs. -780.09 on the
  day, 0 errors. The radar overlay logged 2 flags (M&M and SHREECEM, credit rating
  announcements) in observe mode.
- **The VPN dropped at about 15:35:** the exit is the school network again (Mumbai), so
  Yahoo, Angel, NSE and GitHub are intercepted.
  - The minute download resumed at 15:50, but every request fails. The running (old)
    version gives up on those chunks; later sweeps retry them.
  - Local commits wait to be pushed.

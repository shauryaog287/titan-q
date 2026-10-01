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

## 2026-09-29, 22:25 IST: evening

- **Network:** fine on the phone hotspot (Airtel, India). There was a brief DNS drop at
  22:23.
- **Forward test:** 2026-09-29 recorded at 16:41 with complete prices. Composite top-30
  Rs. 505,843.09 (+1.2% since the start), equal-weight market 99.56.
- **Disclosure radar:** ran at 19:55 and archived to G:.
- **DEL1 (delivery %), development 2016-2020:** does not qualify; the test stays sealed.
  - DEL1a abnormal delivery: IC t 2.89, but decile excess -0.2%/yr.
  - DEL1b retail-frenzy reversal: 955 events, 10-day excess -0.12%, t +0.02.
- **FO1 (F&O positioning):** the download is at 300/2647 days. Early days failed while the
  network was down; a second pass fills them. The development run follows when it is
  complete.
- **Cloud run (CM1 commodity-linked, SM1 industry momentum):** rules and instructions are
  pushed (`reports/cloud_task_2026-09-29.md`). It could not be launched from this session
  (the CLI needs an interactive terminal; the remote agent could not see git). The user
  starts it from claude.ai/code or with `claude --cloud`.
- **Laptop sleep:** the 08:50 preflight and the 09:10 trading task can now wake it on mains
  power, and the session keeps it awake until the close.

## 2026-09-30, 09:25 IST: live-start check

- **The fixes worked:**
  - the task started at 09:10 under the supervisor;
  - the plan is Kelly 1.000 and 100% in 50 stocks, although Yahoo refused every refresh (0/51;
    the cache was used);
  - the observe overlays and the "plan ready" alert all worked.
- **But the laptop is on school Wi-Fi without a working VPN** (exit: Jio, Mumbai), so Yahoo,
  Angel, NSE and GitHub are intercepted.
  - The 08:50 preflight caught it and sent an urgent alert.
  - The loop has been waiting since 09:15 ("feed stale"). It trades by itself as soon as
    prices get through.
- **Recorder:** it crashed at login on the intercepted network, and nothing restarts it.
  - Fixed: the login now retries every minute until the close, with an alert, and a failed
    scrip-master refresh falls back to the cached copy.
  - Restarted by hand at 09:23; it is waiting for the network.
- **Pending:** commits wait to be pushed (GitHub is blocked on this network).

**2026-09-30, 10:07 IST.** Still blocked: the exit is the school network (Jio, Mumbai), and
Yahoo, Angel, NSE and GitHub are intercepted. The loop has waited since 09:15 with 0 trades;
"price feed stale" alerts went out at 09:30 and 10:00. The recorder retries its login every
minute. Both resume by themselves once a clean connection (a working VPN or the hotspot) is
available. The next check is at 11:07.

**2026-09-30, 11:07 IST.** Unchanged: the school network (Jio, Mumbai) still intercepts
Yahoo, Angel, NSE and GitHub. The loop is waiting (0 trades; stale alerts every 30 minutes)
and the recorder retries its login every minute. The next check is at 12:07.

**2026-09-30, 12:07 IST.** Unchanged: the school network still intercepts the finance sites.
0 trades, the loop is waiting, and the recorder keeps retrying. The next check is at 13:07.

**2026-09-30, 13:07 IST.** Unchanged: the school network still intercepts the finance sites.
0 trades, the loop is waiting, and the recorder keeps retrying. The next check is at 14:07.

**2026-09-30, 14:07 IST.** Unchanged: the school network still intercepts the finance sites.
0 trades, the loop is waiting, and the recorder keeps retrying. The last check of the session
is at 15:07.

**2026-09-30, the day.** There was no paper trading and no tick recording today. The laptop
spent the whole session on the school Wi-Fi without a working VPN. The exit was Jio, Mumbai,
at every check from 07:30 to 15:07; the network intercepted HTTPS to Yahoo, Angel, NSE and
GitHub.
- **What worked:** the software did what it was built to do.
  - The 09:10 start ran under the supervisor, and the plan was valid (Kelly 1.000, 100% in
    50 stocks) from cached data.
  - The 08:50 preflight raised an urgent alert. The loop waited and sent a stale-feed alert
    every 30 minutes.
  - The recorder, fixed at 09:23, retried its login every minute instead of dying.
- **Nothing was lost:** the paper account is untouched (Rs. 500,000, no positions).
- **Consequences for tonight:** the forward-test update, news, radar and Trends will fail the
  same way until the laptop has a clean connection. Six or more commits wait to be pushed.
- **Lesson:** the trading PC needs a connection that is not intercepted, either a VPN that
  covers the whole machine and reconnects on its own, or the hotspot. Alternatively, a small
  cloud server in Mumbai.

## 2026-10-01, 10:10 IST

- **Session start:** the laptop did not wake by itself at 08:50 or 09:10. The battery was at
  22%, so it had run unplugged overnight, and wake timers are off on battery. The scheduled
  tasks ran at 09:58, when it was woken.
- **VPN:** on (exit Singapore). Yahoo, Angel, NSE and GitHub all pass.
- **Trading:** the plan is Kelly 1.000 and 100% in 50 stocks. Daily data refreshed 23/51; the
  rest came from the cache. Trading started at 10:00 with 47 paper buys; cash left Rs. 68,410.
- **Recorder:**
  - Logged in at 09:59 after one failed try, and records 2,700 stocks.
  - At 10:02 all 3 connections dropped while the network switched to the VPN. The new
    supervisor reconnected them by 10:05, back to about 590 rows/s, with no crash.
- **Forward test:** 2026-09-30 recorded (Rs. 509,049.16; equal-weight 99.58).
- **Pushed:** all pending commits.
- **Background downloads:** FO participant 2,100/2,647 days, with 627 failed during outages
  to be re-fetched; Angel minute research set 13,459/22,731.

## 2026-10-01, evening (18:30-21:00 IST)

- **Research results** (all pre-registered; development decisions applied mechanically):

  | Study | Question | Development result | Decision |
  |---|---|---|---|
  | ORD1 | Do order-win announcements drift up? | 86 events 2017-2020, 20-day net -3.01% (t -2.04) | Not adopted; test sealed |
  | CM1 | Do oil and gold stocks drift after overnight crude or gold moves? | 5,571 trades, +0.008% per trade (t 0.68) | Not adopted; test sealed |
  | SM1 | Do the strongest industries keep winning? | Top 3 industries +2.3% a year over the market (t 0.37) | Not adopted; test sealed |
  | **NJ1b** | Do jumps with no news fade? | Confirmed on unseen 2021-2026 data: 13,847 jumps, 5-day excess -0.26% (t -3.99) | **Confirmed** |

  - NJ1b is the first finding to pass a sealed test. It is an avoid rule: the mover scanner
    tags these jumps, and the trading loop will not buy such a stock on the jump day or for
    the next 5 trading days.
- **Data check:** Yahoo's NSE opens match Angel's first 1-minute opens. On about 5-10% of
  days the open equals the previous close, and that is real, not a data error.
- **Local AI:**
  - Ollama is installed to `D:\Ollama` (2.8 GB), with models in `D:\OllamaModels` and none on
    C:. It runs on the RTX 3050 (CUDA). The qwen2.5:7b download is in progress.
  - News alerts and mover alerts get an "AI:" line once the model is ready. Everything works
    without it.
  - C: free space fell from 22.5 to 14.8 GB. The cause is Windows' page file growing to
    20 GB under memory load, not Ollama; it shrinks after a restart.
- **Network:**
  - The laptop went into standby 18:56-19:54 (taken off the charger) and again 20:05-20:21.
  - On Mayo, X-VPN was stuck on its loading screen from 14:40 to 15:41, so the price feed
    was down for the last hour of the session. It also dropped briefly at 19:54, 20:00 and
    20:21. The switcher reconnected it each time.
  - A second VPN app as a fallback (for example Cloudflare WARP) would cover the loading-
    screen case; that is the user's decision.
- **Minute history:** the downloader is working, not stuck. It wrote 4,564 files between
  15:50 and 19:00. Its progress line only looked frozen because it waits on the oldest chunk.

**2026-10-01, 22:20 IST: paused for a shutdown until Monday 2026-10-05.**
- **Stopped cleanly:** the EV1 labelling run (600 of 41,437 done), the fundamentals collector,
  the M1/VPIN chain and the dashboard. All of them resume where they left off.
- **The minute download had crashed at about 21:20** on a failed login during the VPN
  switches. That is now fixed, and `TitanQ_Angel_History` restarts it and the M1/VPIN chain
  at 16:05 each day. It never starts at boot.
- **Monday:**
  - the usual 07:30 / 08:50 / 09:10 / 09:12 jobs run with Proton VPN as the primary VPN, the
    NJ1b no-buy rule and AI lines in alerts;
  - the labelling and fundamentals jobs catch up at boot but wait until 15:45;
  - the weekly report missed on Saturday runs at boot.

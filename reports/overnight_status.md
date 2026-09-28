# Overnight status

## 2026-09-28, 20:15 IST

### Running now (unattended, survives this terminal closing)

- **Angel 1-minute history download** (`angel_history.py`, pid 28340):
  - 9,146 of 67,122 chunks done, about 99 a minute, expected to finish around 06:00;
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

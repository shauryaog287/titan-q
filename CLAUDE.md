# Titan-Q (D:\Quant_Project)

Paper-trading system for NSE large caps (Python, Windows, asyncio).

Version 2 dates from 2026-09-26. In v2 every model is either fed real market data or
reported as unavailable; none gets random numbers. Architecture, mathematics and history
are in `Titan-Q_Whitepaper.md`: Section 10 describes v2, and Sections 1-9 describe v1 and
why it was rebuilt.

## Hard rules

- **Pure ASCII in every file written**: source, comments, strings, log messages, HTML,
  markdown, JSON. The console is CP1252 and crashes on non-ASCII output. Currency is always
  "Rs." (for example "Rs. 1,23,456.78"). Do not use the rupee glyph, and do not use
  "\u20b9" escapes either: the escape keeps the source ASCII but prints the same glyph at
  runtime. No emoji, curly quotes, em dashes, arrows or Greek letters.
- After writing a file, check that it has zero bytes above 127:
  `$b=[IO.File]::ReadAllBytes('path'); @($b | ? { $_ -gt 127 }).Count`
- All project files live on D:. The one exception is the dashboard HTML at
  `C:\Users\jiten\titan-q-dashboard\index.html`; do not move it without asking.
- Never quote `historical_backtest.py` numbers (Sharpe 11.46 etc.). It is a smoke test with
  random PnL. The real backtest is `backtest_daily.py`.
- Models are switched on only when they beat simple baselines out of sample. Do not
  switch one on without evidence.

## How to run

The interpreter is always `D:\trader_env\Scripts\python.exe`, run from `D:\Quant_Project`.

| What | Command |
|---|---|
| Live paper trading | `python main_trading_loop.py`. `start_titan_q.bat` runs it, and the scheduled task `TitanQ_Live_Trading` starts that at 09:10 IST daily. One process per trading day; it exits 15 minutes after the close, or at once on weekends and after hours. A lock file stops a second instance. |
| Watch it trade any time | `python main_trading_loop.py --mode replay [--date YYYY-MM-DD] [--speed N]` replays an archived session on a throwaway account. |
| Retrain models (about weekly) | `python train_models.py` writes `models/` plus `manifest.json`, which records what is enabled and why. |
| Walk-forward backtest | `python backtest_daily.py [--start YYYY-MM-DD] [--end ...]` writes `reports/`. |
| Refresh data only | `python market_data.py --fetch` |
| Self-tests | `python market_data.py`, `python titanq_brain.py`, `python paper_broker.py`, plus each phase module. |
| Dashboard | `python dashboard_server.py`, then http://localhost:5000 (localhost only). |

- Reset the live paper account by deleting `paper_account.json`.
- Logs are written to `logs\titanq_YYYY-MM-DD.log`, and each day's decision context to
  `logs\context_YYYY-MM-DD.json`.
- Git is at `D:\Git\cmd\git.exe`. Commit before and after changes.
- Cloud research runs (training, backtests while the PC is off) are described in `CLOUD.md`.
  Paths come from the location of `titanq_config.py` (or `TITANQ_HOME`), so the code also
  runs on Linux.

## Scheduled and unattended jobs (Windows Task Scheduler)

| Task | When | What |
|---|---|---|
| TitanQ_Live_Trading | 09:10 daily | `start_titan_q.bat`: trading loop (under `run_live.py`), Angel recorder, dashboard. The .bat must end (no `pause`), or the next day's start is skipped. |
| TitanQ_Forward_Test | 16:30 Mon-Fri | `forward_test.py` fallback (the live loop also starts it after the close). Idempotent, locked. |
| TitanQ_News | starts 07:30 (watchdog every 30 min) | `news_watch.py` runs until 21:35: NSE filings and ET, Business Standard and Mint RSS every 3 min, Google News every 10 min for our ~79 stocks; FinBERT tone; phone alert per stock for each new item (high for strongly negative or sensitive filings; max 30/h); archive `data/news/` copied to G:. Observe only. |
| TitanQ_Preflight | 08:50 Mon-Fri | `alerts.py --preflight`: HTTPS to Yahoo, Angel and NSE (VPN check), mains power, disk space. Phone alert either way. |
| TitanQ_Trends | 20:40 daily | `trends_collector.py`: Google Trends searches for "<company> share price" (held/planned stocks); daily 90-day snapshot to D: and G:. Observe only. |
| TitanQ_Reddit | every 20 min, 07:05-00:05 | `reddit_collector.py`: official Reddit API if `reddit_credentials.json` exists (new apps need Reddit's approval), else ONE public RSS request per run, rotating over 3 subreddits; mentions and tone per stock. No disguised scraping. Observe only. Replaces the Selenium scraper `social_media_scanner.py`. |
| TitanQ_VPN_Switch | every 2 min, at logon, on each network connect | `vpn_switch.py` (pythonw, no window). On `Mayo Students-M`, if Yahoo/Angel/NSE are intercepted and the internet is up, it starts X-VPN and presses its Connect button via UI Automation (`xvpn_connect.ps1`), verifies after 45 s, retries once, then alerts. If there is no internet (the night shutdown) it waits. On `Anjali - RES` it closes X-VPN. |
| TitanQ_Movers | 09:12 Mon-Fri (watchdog every 30 min) | `mover_scanner.py`: every minute over about 2,700 stocks (Angel latest prices), it flags moves of 5% or more on 3x or more volume (in steps of 5%), attaches today's NSE filing or news (a targeted Google News search when the archive has nothing), sends phone alerts (max 20/h) and archives to `data/movers/`. Information only, pending NJ1/NJ2. |
| TitanQ_Drive_Backup | 10:00 Saturday | `backup_to_drive.py`: minute history, one file per symbol; daily data in one file; Yahoo sessions -> `G:\My Drive\Titan-Q_History`. Never deletes. |

`after_download.ps1` is a one-off detached chain, logged to `logs\after_download.log`.
Once `angel_history.py` writes `data\intraday_hist_priority_done.json` (the research set is
complete), it:
- runs the development halves of M1 (`intraday_research.py`) and of the VPIN gate test
  (`vpin_research.py`), and commits their reports;
- after the whole download ends, runs one retry pass for failed chunks;
- backs up to Drive after market hours.

It never reveals a test period.

Angel's history API has a cap of roughly 5,000 requests an hour. Above it, every request is
refused for about 50 minutes. `angel_history.py` therefore runs at 1.2 requests a second
and pauses all workers together whenever it gets a rate-limit answer. Never raise the rate.

## Unattended operation (the user is away during market hours)

- `run_live.py` supervises the live loop and restarts it until 15 minutes after the close.
  A crash is restarted after 60 s. Exit code 75 is a requested restart and waits 5 s. It
  allows at most 12 restarts a day.
- **To apply a code fix mid-session:** fix and commit, then write a one-line reason to
  `restart_trading_loop.request` in the project folder. The loop lets queued orders finish,
  saves, and exits 75, and the supervisor starts it again on the new code. Do not kill the
  process.
- An invalid daily plan (not tradeable, or a NaN Kelly) is rebuilt with fresh data every 10
  minutes in session.
- The Angel recorder reconnects by itself: 30 s to 5 min backoff, and a watchdog for
  connections silent for 3 minutes.
- Phone alerts go through ntfy (`alerts.py`; the topic is in the git-ignored `alerts_config.json`). They cover:
  - a crash or give-up;
  - an invalid plan;
  - a stale feed (15 min);
  - a silent recorder (10 min);
  - the daily loss limit;
  - the end-of-day summary;
  - forward-test results.
- Daily loss limit: `daily_loss_halt` = 3% below the day's opening equity (kept in a file,
  so it survives restarts). After that, no new orders are sent that day.
- The school Wi-Fi intercepts HTTPS to finance sites, so a VPN must stay on. Never switch off
  certificate checks.

## Market calendar

`market_data.Session` knows NSE trading holidays from NSE's holiday-master (`data/reference/nse_holidays.json`,
refreshed by the 08:50 preflight when it is more than 25 days old). On a holiday:
- the loop, the recorder, the supervisor and the mover scanner exit at once;
- the preflight sends a "market holiday" note instead of checking the network;
- the forward test finds no new close.

Before 2026-10-01, holidays were unknown and the system would have waited all day and sent
stale-feed alerts.

## Data and research (2026-09-28)

- Angel One SmartAPI is read-only in code (`angel_feed.py`, `ReadOnlySmartConnect` blocks
  order methods). Credentials are in the git-ignored `angel_credentials.json`; never print
  or commit them.
- `angel_history.py` downloads 1-minute candles (2020 onward) for about 700 tradable stocks
  plus NIFTY and BANKNIFTY. They go to `data/intraday_hist/`, and the download pauses
  08:55-15:50 IST on weekdays.
- Pre-registered studies are in `reports/preregistration_*.md`. Results so far:
  - no single whole-market signal qualified;
  - the composite is in a forward paper test (`forward_test.py`, `reports/forward_test/`);
  - M2 and M3 did not qualify;
  - FO1 (retail crowding in NIFTY futures) passed development (+1.06%/week, t 2.18) but
    failed its test (-0.08%/week, t -0.31). DEL1 (delivery %) failed development. Not used.
  - INS1 (insider open-market buys, SEBI PIT disclosures from NSE, `insider_research.py`)
    passed development (2016-2020: +2.28% net excess over 60 days, t 2.23) but failed its
    test (2021-2026: -1.42%, hit rate 41%). It is not used;
  - M1 and the VPIN gate test are pending the download.
- Insider trades: NSE's PIT API (Reg 7) has returned nothing since about 2026-05.`disclosure_radar.py` therefore also collects SAST Reg 29 substantial-holding filings
  (buyers and sellers crossing 5%, promoters flagged; working as of 2026-10-01) and flags
  open-market ones (observe only).
- NIFTY opens: Yahoo's daily open equals Angel's 09:15 candle open (checked on 125 days in
  2026). The open-to-close drift question awaits the full history.

## Architecture (v2)

| File | Role |
|---|---|
| titanq_config.py | Every parameter, path and the 50-stock universe. |
| market_data.py | Yahoo daily bars (12 years, cached in data/daily) and 1-minute bars (every session archived in data/intraday). Provides LiveBarFeed (downloads in a worker thread, so it never blocks), ReplayFeed, and the IST session calendar. |
| train_models.py | Trains HMM regime (Phase 11) and neural SDE (21); rebuilds PPO normalisation and runs the out-of-sample evaluation (Phase 5). |
| titanq_brain.py | DailyContext: regime multiplier, market drift, momentum alpha, Ledoit-Wolf covariance, exact K-of-N selection with incumbent hysteresis (20), HRP (12), Kelly on the basket (4), EVT expected-shortfall trim (23), TE diagnostics (24). Also ToxicityMonitor (VPIN, 14) and the Rebalancer. |
| main_trading_loop.py | Orchestrator: feed -> context -> rebalancer -> ExecutionEngine -> broker. Also writes the dashboard state. |
| execution_engine.py | Order state machine; the only component that sends orders. |
| paper_broker.py | Fills at feed price plus/minus slippage, Indian delivery cost stack, full BrokerClient protocol, fills reported via `fill_callback`, ledger persisted in `paper_account.json` (live only). |
| backtest_daily.py | Walk-forward backtest of the same decision code (fills at the open, full costs). |

Each day before the open, the context is built from data before that day and saved. During
the session the loop does five things:
1. polls 1-minute bars every minute;
2. marks the account to market;
3. scores VPIN toxicity;
4. plans trades toward target weights, subject to a no-trade band, a minimum size, a
   cooldown, a cash check and no buys in the last 10 minutes;
5. queues market orders to the engine.

## Strategy (v3, decided 2026-09-27)

Set in `titanq_config.py` per `reports/decision_2026-09-27.md` (whitepaper 10.5-10.6):
hold every eligible stock (`selection_mode="all"`), inverse-volatility weights, no alpha tilt
(`alpha_ic=0`), quarter Kelly with an informative market prior (10%/yr, sd 3%), capped at
100%. Regimes other than CRASH use multiplier 1.0; CRASH -> 0, a novel regime x0.5 and the
EVT trim x0.5 are catastrophe limits only. The v3 backtest (2019-2026) is CAGR 5.9%,
Sharpe 0.59, against 7.7% for NIFTY at the same 70% average exposure; the whole gap is 2020,
where quarter Kelly on a 252-day variance kept exposure near 24% through the rebound.
The concentrated K-of-N mode (`"optimize"`) and HRP weighting remain available, not default.

v3.1 (2026-09-28): Kelly now divides by an EWMA variance of the basket with a 21-day half-life
(`sizing_vol_halflife_days=21`) instead of the 252-day window. It was adopted through the
pre-registered protocol in `reports/preregistration_sizing_vol.md`:
- development 2016-2020: Sharpe 0.91 vs 0.65, max drawdown 16.3% vs 23.9%;
- test 2021-2026: Sharpe 0.91 vs 0.85, a small gain, with about 45% more trades.

The decisions are in `reports/decision_sizing_vol_{dev,test}.md`.

v3.2 (always invested, no brakes) was the user's choice for one session on 2026-10-01; that
evening the user asked for "growth with a little bit of brakes", and v3.1 is the default again:
`sizing_mode="kelly"`, CRASH 0.0, UNKNOWN 0.5, novel 0.5 and `es_trim` 0.5. `sizing_mode="full"` with all
multipliers at 1.0 reproduces v3.2.

## Model status (manifest of 2026-09-26)

- HMM regime: ON, as a catastrophe limit (see above). Four volatility regimes, stable fit.
  Labels are decided on raw means (a standardised-means bug was fixed).
- Neural SDE: OFF. It lost to a constant GBM on held-out data, so GBM drift and
  volatility are used.
- PPO policy: OFF. Out of sample (2021-04 to 2026-09) a long/flat version scored Sortino
  -1.48 against 0.87 for buy-and-hold. Its original normalisation statistics were never
  saved and had to be rebuilt; its action space is Sell=0/Buy=1.
- TFT (18), spoofing autoencoder (16) and Hawkes (14, trade-arrival part): UNAVAILABLE.
  They need order-book depth or tick data, which Yahoo does not provide. Enabling them
  needs a broker feed such as the Zerodha Kite websocket (kiteconnect is not installed).
- VPIN (14): OBSERVE mode (`vpin_gate="observe"`) since 2026-09-28. It is computed and
  logged but does not block buys. It has never been validated in a backtest, and in the
  2026-09-25 replay it held back 10 of 50 stocks all session. Switch it to `"block"` only
  after a pre-registered test on the Angel minute history shows it helps.
- Not used in v2 decisions: copula (13), signatures (17), PINN pricers, HJB quoter (22),
  and the Phase 25 transpiler (no g++).

## Known limitations and gotchas

- The universe is a fixed list of about 2023 NIFTY 50 names, with TMPV.NS in place of
  TATAMOTORS.NS. It is not today's index.
- No stock-selection signal showed an edge (`reports/alpha_research.md`), so v3 has no
  alpha tilt; the momentum code stays for the "optimize" mode. Read
  `reports/backtest_daily_report.md` for what the strategy actually did, and for the biases:
  survivorship, NIFTY as a price index, cash earning 0%.
- Yahoo 1-minute bars can lag a minute or two. There is no order book and no tick data.
- The regime model's labels are volatility regimes. Over 2015-2026 even the most volatile
  regime had a positive mean return, so "CRASH" rarely appears.
- Legacy scripts (trading_bot.py, local_ai_trader.py, nse_*_feed.py, *_dashboard.py,
  etc.) are not part of v2.
- `torch.set_default_dtype` is no longer changed at import. NeuralSDE is float64 by
  itself.

## Working conventions

- Read a module's docstring and self-tests before changing it; they state its
  assumptions and limits.
- Keep every decision point-in-time. Anything that trades day D may only see daily data
  dated before D (`close_panel(..., before=D)`).
- Re-run `backtest_daily.py` after changing the decision layer, and compare with the
  previous report.
- The user wants Claude to run commands, installs and setup steps itself rather than
  handing them commands to type. Destructive actions still need confirmation first.

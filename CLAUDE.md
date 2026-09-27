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

## Model status (manifest of 2026-09-26)

- HMM regime: ON. Four volatility regimes, stable fit. Labels are decided on raw means
  (a standardised-means bug was fixed).
- Neural SDE: OFF. It lost to a constant GBM on held-out data, so GBM drift and
  volatility are used.
- PPO policy: OFF. Out of sample (2021-04 to 2026-09) a long/flat version scored Sortino
  -1.48 against 0.87 for buy-and-hold. Its original normalisation statistics were never
  saved and had to be rebuilt; its action space is Sell=0/Buy=1.
- TFT (18), spoofing autoencoder (16) and Hawkes (14, trade-arrival part): UNAVAILABLE.
  They need order-book depth or tick data, which Yahoo does not provide. Enabling them
  needs a broker feed such as the Zerodha Kite websocket (kiteconnect is not installed).
- VPIN (14): ON, intraday, from 1-minute bars.
- Not used in v2 decisions: copula (13), signatures (17), PINN pricers, HJB quoter (22),
  and the Phase 25 transpiler (no g++).

## Known limitations and gotchas

- The universe is a fixed list of about 2023 NIFTY 50 names, with TMPV.NS in place of
  TATAMOTORS.NS. It is not today's index.
- The momentum alpha (IC 0.04) is a documented baseline, not a proven edge. Read
  `reports/backtest_daily_report.md` for what it actually did, and for the biases:
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

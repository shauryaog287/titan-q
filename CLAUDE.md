# Titan-Q (D:\Quant_Project)

Research and paper-trading system for NSE equities: Python, Windows, asyncio.
The full architecture, the mathematics and the per-phase integration status are in
`Titan-Q_Whitepaper.md`. Section 7 is the status table and Section 8 the roadmap;
keep both current when integration changes.

## Hard rules

- **Pure ASCII in every file written**: source, comments, strings, log messages, HTML,
  markdown. The console is CP1252 and crashes on non-ASCII output. Currency is always
  "Rs." (for example "Rs. 1,23,456.78"). Do not use the rupee glyph, and do not use
  "\u20b9" escapes either: the escape keeps the source ASCII but prints the same glyph at
  runtime. Also avoid emoji, curly quotes, em dashes, arrows and Greek letters.
- After writing a file, check that it has zero bytes above 127:
  `$b=[IO.File]::ReadAllBytes('path'); @($b | ? { $_ -gt 127 }).Count`
- All project files live on D:. The one exception is the dashboard HTML at
  `C:\Users\jiten\titan-q-dashboard\index.html`, which `dashboard_server.py` serves.
  Do not move it without asking.
- Never quote `historical_backtest.py` results (Sharpe 11.46, win rate 53-59%) as
  performance. The script is a smoke test: each trade's PnL is a random draw (line 117).
  See whitepaper Section 6.

## Environment

- Python: always use `D:\trader_env\Scripts\python.exe`. Shell: PowerShell 5.1.
- Most phase modules have self-tests: run `D:\trader_env\Scripts\python.exe <module>.py`
  from `D:\Quant_Project`. Some take minutes.
- `start_titan_q.bat` starts three windows: `nse_whole_market_feed.py`,
  `main_trading_loop.py` and `dashboard_server.py`. The scheduled task
  `TitanQ_Live_Trading` runs it daily at 09:10 IST.
- The dashboard is at http://localhost:5000 (stdlib `http.server`, not Flask).
  `index.html` loads React 18 from a CDN. In AUTO mode it falls back to a built-in
  simulator when the backend is down, so check the mode before trusting a screen.
- Missing tools: `g++` is not on PATH, which blocks the Phase 25 transpiler tests, and git
  is not installed yet. `.gitignore` is already prepared.
- Baseline snapshot taken before switching from agy:
  `D:\Quant_Project_backups\Quant_Project_2026-09-26_baseline.zip`

## Architecture map

Processes:
- the feed, which writes `live_state.json` and the data lake; the loop does NOT read
  either output;
- `main_trading_loop.py`, which runs `ExecutionEngine.run()` alongside
  `live_trading_orchestrator()` via `asyncio.gather` and writes `dashboard_state.json`;
- `dashboard_server.py`.

Orchestrator cycle (about every 5 s): Hawkes/VPIN toxicity -> spoof autoencoder -> HMM
regime -> signatures -> TFT -> transfer entropy -> PPO direction -> annealer (4 of 50) ->
neural SDE + Kelly -> HRP -> EVT trim -> OrderIntents -> dashboard export.

| Phase | File |
|---|---|
| 4 Kelly sizer + CostModel | fusion_and_sizing.py |
| 5 execution state machine | execution_engine.py |
| 11 HMM regime | regime_hmm_detector.py |
| 12 HRP | portfolio_hrp_allocator.py |
| 13 copula stat-arb | stat_arb_copula.py |
| 14 Hawkes / VPIN toxicity | hft_hawkes_toxicity.py |
| 16 spoof autoencoder | adversarial_spoof_detector.py |
| 17 rough-path signatures | rough_volatility_signatures.py |
| 18 TFT LOB forecaster | transformer_lob_forecaster.py |
| 20 QUBO / swap annealer | quantum_portfolio_annealer.py |
| 21 neural SDE | neural_sde_discovery.py |
| 22 HJB market maker | hjb_market_maker.py |
| 23 EVT tail risk | extreme_value_theory_risk.py |
| 24 transfer entropy | transfer_entropy_causality.py |
| 25 Python-to-C++ transpiler | alpha_cpp_transpiler.py |
| paper broker | paper_broker.py |
| synthetic GBM feed | nse_whole_market_feed.py |

## Current state (as of 2026-09-26)

The phase libraries are complete and self-tested, but the live loop mostly does not use
them yet. In `main_trading_loop.py`:
- **Direction is random.** The PPO checkpoint `checkpoints/ppo_sortino_18000000_steps.zip`
  does not exist, and its observation is random noise anyway. A 2 MB
  `master_ai_brain_sortino.zip` sits at the project root; it may be the trained policy.
- **Four shields are placeholders.** Regime, spoof flag, EVT ES and VPIN are random draws;
  the HMM is a `MockRegimeFilter`, and CRASH logs but does not liquidate.
- **The learned models are untrained.** The TFT and neural SDE load no weights and get
  synthetic inputs. The SDE output drives Kelly to f = 0 on every cycle, so every order
  is 1 share.
- **The allocation layer runs on noise.** The annealer and HRP work on a random mu/Sigma
  regenerated each cycle.
- **Execution is simplified.** The HJB quoter is not wired in, and the paper broker fills
  at the yfinance last price.

## Known bugs and gotchas

- `neural_sde_discovery.py` sets the torch default dtype to float64 at import, so every
  module built afterwards, the TFT included, is float64. Workaround in place: inputs are
  cast to float64 in `main_trading_loop.py`.
- The kill switch covers only 22:40-22:59 (`now.hour == 22`), so trading resumes at 23:00.
  The NSE session is 09:15-15:30 IST. The overnight yfinance SSL errors come from the
  router's block page while the internet is off.
- In the dashboard export, `current_pnl` is broker cash, not equity. `toxicity_score` is
  always 0.0 because `tox_values` is never defined, and `recent_trades` is always [].
- `paper_broker.py`:
  - implements 3 of the 8 `BrokerClient` methods, and fills never reach
    `engine.on_fill`, so position mismatches in reconciliation are expected;
  - is long-only;
  - still has the rupee glyph on lines 24, 46 and 60. The same problem is in
    `historical_backtest.py` (69, 147, 148), and `nse_whole_market_feed.py` line 88
    uses "\u20b9".
- The yfinance calls inside async methods are blocking and stall the shared event loop.
- Annealer indices map to a hard-coded NIFTY_50 list of real .NS tickers. An earlier fake
  ticker, BASKET-TECH-4, caused Yahoo 404s.
- `dashboard_server.py` binds 0.0.0.0 with `Access-Control-Allow-Origin: *`, and its 404
  branch sends a 200 status first.
- `historical_backtest.py` runs 2,000 ticks, not 10,000. `DATA_LAKE_PATH` has a stray
  space, and no data is ever read.

## Working conventions

- Read a module's docstring and self-tests before changing it; they state its
  assumptions and limits.
- Prefer replacing a placeholder in the loop with the real module call over adding new
  modules.
- Until git is installed, copy a file to `D:\Quant_Project_backups` before a non-trivial
  edit.

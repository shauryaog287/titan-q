"""
titanq_config.py
================
Every tunable parameter of the live Titan-Q system, in one place.

Paths are absolute so the system behaves the same whatever the working
directory is (the Windows scheduled task starts in C:\\Windows\\System32).

Numbers marked APPROX are reasonable defaults, not facts about your account:
check them against your broker's contract note before relying on them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The folder this file lives in (D:\Quant_Project on the trading PC; the repo
# checkout on a cloud machine). TITANQ_HOME overrides it.
PROJECT_DIR = os.environ.get("TITANQ_HOME", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
STATE_FILE = os.path.join(PROJECT_DIR, "dashboard_state.json")
MANIFEST_FILE = os.path.join(MODELS_DIR, "manifest.json")

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

INDEX_SYMBOL = "^NSEI"

# The 50 large caps the system trades. This is the list the project has always
# used (a NIFTY 50 composition from around 2023), with TATAMOTORS.NS replaced by
# TMPV.NS after the Tata Motors demerger. It is NOT guaranteed to match today's
# NIFTY 50; update it if you want the current index members.
UNIVERSE: Tuple[str, ...] = (
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "HCLTECH.NS",
    "MARUTI.NS", "BAJAJFINSV.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
    "TATASTEEL.NS", "WIPRO.NS", "NESTLEIND.NS", "TECHM.NS", "ONGC.NS",
    "POWERGRID.NS", "HINDALCO.NS", "M&M.NS", "NTPC.NS", "INDUSINDBK.NS",
    "GRASIM.NS", "DIVISLAB.NS", "TMPV.NS", "BAJAJ-AUTO.NS", "JSWSTEEL.NS",
    "BRITANNIA.NS", "ADANIPORTS.NS", "CIPLA.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "COALINDIA.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "UPL.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "BPCL.NS", "SBILIFE.NS", "SHREECEM.NS", "TATAPOWER.NS",
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # ---- account ---------------------------------------------------------
    initial_capital: float = 500_000.0

    # ---- data --------------------------------------------------------------
    daily_history_years: int = 12          # daily bars kept for training/estimation
    intraday_period: str = "5d"            # 1-minute bars fetched at start-up
    live_poll_seconds: float = 60.0        # how often live 1-minute bars are fetched
    stale_after_seconds: float = 240.0     # no new bar for this long in session -> stale

    # ---- NSE session (IST) ----------------------------------------------
    session_open: Tuple[int, int] = (9, 15)
    session_close: Tuple[int, int] = (15, 30)
    no_new_risk_last_minutes: int = 10     # no buys in the final minutes of the session

    # ---- loop ----------------------------------------------------------------
    cycle_seconds: float = 5.0             # orchestrator cycle in live mode
    replay_bars_per_second: float = 2.0    # replay speed (375 bars ~ 3 minutes)

    # ---- selection (Phase 20) and weights (Phase 12) -------------------------
    # Decided on 2015-2020 data before looking at 2021-2026 (reports/decision_2026-09-27.md):
    # no selection signal showed an edge, so the default holds every eligible stock,
    # weighted by inverse volatility. "optimize" is the original concentrated mode:
    # the best-Sharpe n_select subset via exact enumeration / annealing.
    selection_mode: str = "all"            # "all" | "optimize"
    weighting: str = "inverse_vol"         # "inverse_vol" | "hrp"
    n_select: int = 4                      # assets held in "optimize" mode
    selection_switch_margin: float = 0.20  # keep the current holdings unless the new best
                                           # subset's expected Sharpe beats theirs by 20%
                                           # (without this, daily re-optimisation churned
                                           # ~3000% of equity a year in the backtest)
    cov_lookback_days: int = 252           # covariance estimation window
    momentum_lookback_days: int = 252      # 12-1 momentum: 12 months...
    momentum_skip_days: int = 21           # ...skipping the most recent month
    alpha_ic: float = 0.0                  # information coefficient of the momentum score
                                           # (monthly). 0 = no alpha tilt: momentum showed
                                           # no edge in development (IC t 0.52) and top-4
                                           # momentum lost 8%/yr vs equal weight in test.

    # ---- sizing (Phase 4) ------------------------------------------------------
    kelly_fraction: float = 0.25           # quarter Kelly
    kelly_prior_edge_vol: float = 0.03     # prior sd of the true annual return
    prior_market_return: float = 0.10      # prior mean of the market's annual return
                                           # (NIFTY 1995-2020 compounded ~11%; set from
                                           # pre-2021 history). The v2 baseline used 0,
                                           # which kept the system ~85% in cash.
    max_exposure: float = 1.0              # never more than 100% of equity invested
    sizing_vol_halflife_days: int = 21     # Kelly divides by an EWMA variance of the basket's
                                           # daily returns with this half-life (0 = the
                                           # cov_lookback_days window, as v3 did). Adopted
                                           # 2026-09-28 by the pre-registered protocol
                                           # (reports/preregistration_sizing_vol.md): dev
                                           # 2016-2020 Sharpe 0.91 vs 0.65; test 2021-2026
                                           # 0.91 vs 0.85, at ~45% more trades.

    # ---- regime (Phase 11): exposure multiplier per regime ---------------------
    # Routine regimes do not cut exposure: timing overlays lowered returns without
    # improving Sharpe in development. CRASH and novel regimes remain hard limits.
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "BULL": 1.0, "SIDEWAYS": 1.0, "HIGH_VOL": 1.0, "BEAR": 1.0,
        "CRASH": 0.0, "UNKNOWN": 0.5,
    })
    novel_regime_multiplier: float = 0.5   # observation unlike every trained regime

    # ---- PPO timing (Phase 5 policy) ------------------------------------------
    # "auto": use the policy only if train_models.py found it beat buy-and-hold
    # out of sample. "on"/"off" force it.
    ppo_timing: str = "auto"
    ppo_risk_off_multiplier: float = 0.0   # exposure when the policy says Sell

    # ---- tail risk (Phase 23) ------------------------------------------------------
    es_level: float = 0.99                 # expected shortfall confidence (1-day)
    # Halve exposure when the basket's 1-day ES exceeds this multiple of the NIFTY
    # index's own 1-day ES. (The original spec said "12%" with no level or horizon;
    # a 99% 1-day ES of a large-cap basket is typically 3-6%, so 12% never fired.)
    es_relative_threshold: float = 1.5
    es_trim: float = 0.5

    # ---- intraday toxicity (Phase 14, VPIN part) ---------------------------------
    vpin_buckets_per_day: int = 50
    vpin_window_buckets: int = 50
    toxicity_threshold: float = 0.90       # percentile vs the stock's own recent days

    # ---- transfer entropy diagnostics (Phase 24) -----------------------------------
    te_bins: int = 6
    te_surrogates: int = 100

    # ---- rebalancing -----------------------------------------------------------
    min_trade_value: float = 2_000.0       # ignore trades smaller than this (Rs.)
    rebalance_band: float = 0.20           # trade only if a position is >20% off target
    rebalance_cooldown_minutes: float = 15.0

    # ---- paper execution costs (APPROX, NSE equity delivery, 2025) --------------
    brokerage_rate: float = 0.0            # Zerodha equity delivery brokerage is zero
    stt_rate: float = 0.001                # 0.1% on buy and on sell (delivery)
    exchange_rate: float = 0.0000297       # NSE transaction charge
    sebi_rate: float = 0.000001            # Rs. 10 per crore
    stamp_rate_buy: float = 0.00015        # 0.015% on buys
    gst_rate: float = 0.18                 # on brokerage + exchange + SEBI fees
    slippage_bps: float = 5.0              # paper fill = last price +/- 5 bps

    # ---- dashboard ------------------------------------------------------------------
    recent_trades_shown: int = 15


CONFIG = Config()

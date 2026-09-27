"""
backtest_daily.py
=================
Walk-forward daily backtest of the LIVE decision layer.

    D:\\trader_env\\Scripts\\python.exe backtest_daily.py                  2019-01-01 .. latest
    D:\\trader_env\\Scripts\\python.exe backtest_daily.py --start 2021-01-01 --end 2024-12-31

For every trading day D in the test window:
  1. the daily context is built by titanq_brain.build_daily_context, the SAME
     function the live system calls, from daily bars dated before D;
  2. the regime comes from a Gaussian HMM trained only on data before
     1 January of D's year (refitted each year on an expanding window) and
     filtered causally up to the day before D;
  3. the Rebalancer (same code as live) turns the target weights into orders,
     which fill at D's OPEN through the PaperTradingBroker (same cost stack,
     same slippage);
  4. the account is marked to market at D's close.

Benchmarks: NIFTY 50 price index buy-and-hold; equal-weight buy-and-hold of
the same 50-stock universe; and NIFTY scaled to the strategy's average
exposure (the same average market risk, held passively).

KNOWN BIASES -- read before believing any number
  * Survivorship: the universe is today's list of 50 large caps applied to
    the past. Names that dropped out of the index are missing. This flatters
    anything that buys from the list; the equal-weight benchmark shares the
    bias, NIFTY does not.
  * Dividends: stock prices are dividend-adjusted (total return) while ^NSEI
    is a price index, so NIFTY buy-and-hold is understated by roughly its
    dividend yield (about 1.2-1.5% a year).
  * Cash earns nothing here; in reality idle cash in a liquid fund earns
    about 6% a year in India. The strategy holds a lot of cash.
  * VPIN toxicity is not applied: there is no 1-minute history before the
    archive started.
  * The PPO policy and the neural SDE are off, as in live. That decision was
    made on data up to 2026, but switching models OFF cannot flatter the
    result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time as _time
from datetime import date, datetime, time
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

from titanq_config import CONFIG, INDEX_SYMBOL, PROJECT_DIR, UNIVERSE
from market_data import ALL_SYMBOLS, IST, DailyStore
from titanq_brain import Rebalancer, build_daily_context, load_manifest
from paper_broker import PaperOrderRejected, PaperTradingBroker

log = logging.getLogger("backtest")

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
TRADING_DAYS = 252


# ===========================================================================
# WALK-FORWARD REGIME
# ===========================================================================

class WalkForwardRegime:
    """One HMM per test year, trained on data before that year; its causal
    filter state is recorded for every day so lookups are O(1)."""

    def __init__(self, index_df: pd.DataFrame, years: List[int]) -> None:
        from regime_hmm_detector import MarketRegimeFilter
        from train_models import fit_regime

        self.states: Dict[int, pd.Series] = {}
        self.models: Dict[int, List[str]] = {}
        for y in years:
            t0 = _time.time()
            train = index_df[index_df.index < pd.Timestamp(f"{y}-01-01")]
            model, pipe, _X, _rows, _fit, _scores = fit_regime(train, select_restarts=4, fit_restarts=6)
            F, rows = pipe.build(index_df["Close"].to_numpy(), None)   # causal features
            Xall = pipe.transform(F[:, :2])                              # train-only scaler
            filt = MarketRegimeFilter(model)
            recs = [filt.update(x, store=False) for x in Xall]
            self.states[y] = pd.Series([(r.regime.name, float(r.confidence), bool(r.novel)) for r in recs],
                                       index=index_df.index[rows])
            self.models[y] = model.label_names()
            log.info("regime model for %d: trained to %s, K=%d %s (%.0fs)", y,
                     train.index[-1].date(), model.n_states, model.label_names(), _time.time() - t0)

    def fn_for(self, d: pd.Timestamp):
        ser = self.states[d.year]
        prior = ser[ser.index < d]
        state = prior.iloc[-1]
        return lambda _index_df: state


# ===========================================================================
# METRICS
# ===========================================================================

def metrics(equity: pd.Series) -> Dict[str, float]:
    r = equity.pct_change().dropna()
    years = len(r) / TRADING_DAYS
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = r.std(ddof=1) * math.sqrt(TRADING_DAYS)
    down = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * math.sqrt(TRADING_DAYS)
    dd = 1 - equity / equity.cummax()
    return {
        "cagr": float(cagr), "ann_vol": float(vol),
        "sharpe": float(r.mean() * TRADING_DAYS / vol) if vol > 0 else 0.0,
        "sortino": float(r.mean() * TRADING_DAYS / down) if down > 0 else 0.0,
        "max_drawdown": float(dd.max()),
        "calmar": float(cagr / dd.max()) if dd.max() > 0 else float("nan"),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
    }


def yearly_returns(equity: pd.Series) -> Dict[str, float]:
    out = {}
    for y, part in equity.groupby(equity.index.year):
        prev = equity[equity.index < pd.Timestamp(f"{y}-01-01")]
        base = prev.iloc[-1] if len(prev) else part.iloc[0]
        out[str(y)] = float(part.iloc[-1] / base - 1)
    return out


# ===========================================================================
# BACKTEST
# ===========================================================================

def run(start: date, end: date, capital: float = CONFIG.initial_capital) -> Dict:
    t_start = _time.time()
    daily = DailyStore().load(ALL_SYMBOLS, refresh=False)
    idx = daily[INDEX_SYMBOL]
    days = idx.index[(idx.index >= pd.Timestamp(start)) & (idx.index <= pd.Timestamp(end))]
    if len(days) < 20:
        raise ValueError("test window too short")
    opens = pd.DataFrame({s: daily[s]["Open"] for s in UNIVERSE if s in daily}).reindex(days)
    closes = pd.DataFrame({s: daily[s]["Close"] for s in UNIVERSE if s in daily}).reindex(days)

    regime = WalkForwardRegime(idx, sorted({d.year for d in days}))
    manifest = load_manifest()

    px_today: Dict[str, float] = {}
    broker = PaperTradingBroker(capital, price_fn=lambda s: px_today.get(s))
    rb = Rebalancer(CONFIG)
    loop = asyncio.new_event_loop()
    rows = []
    trade_value = 0.0
    rejected = 0
    ctx = None
    equity_prev = capital
    for i, d in enumerate(days):
        ctx = build_daily_context(daily, d.date(), equity_prev, manifest=manifest,
                                  with_te=False, regime_fn=regime.fn_for(d),
                                  holdings=sorted(broker.positions))
        # trade at the open
        o = opens.loc[d]
        px_today.clear()
        px_today.update({s: float(v) for s, v in o.items() if np.isfinite(v)})
        eq_open = broker.capital + sum(q * px_today.get(s, 0.0) for s, q in broker.positions.items())
        now = datetime.combine(d.date(), time(9, 15), IST)
        orders, _notes = rb.plan(ctx.target_weights, dict(broker.positions), px_today,
                                 broker.capital, eq_open, now, {})
        for k, od in enumerate(orders):
            try:
                loop.run_until_complete(broker.place_order(SimpleNamespace(
                    symbol=od.symbol, side=od.side, quantity=od.qty,
                    client_order_id=f"bt-{d.date()}-{k}"), "BACKTEST"))
                trade_value += od.qty * px_today[od.symbol]
            except PaperOrderRejected:
                rejected += 1
        # mark to the close
        c = closes.loc[d]
        invested = sum(q * float(c.get(s, np.nan)) if np.isfinite(c.get(s, np.nan))
                       else q * px_today.get(s, 0.0) for s, q in broker.positions.items())
        eq = broker.capital + invested
        rows.append({"date": d, "equity": eq, "cash": broker.capital, "invested": invested / eq,
                     "target_exposure": ctx.exposure, "regime": ctx.regime,
                     "holdings": ";".join(sorted(broker.positions)), "fees": broker.fees_paid})
        equity_prev = eq
        if i % 100 == 0:
            log.info("%s  equity Rs. %s  invested %4.1f%%  regime %-9s  %s", d.date(), f"{eq:,.0f}",
                     100 * invested / eq, ctx.regime, ", ".join(ctx.selection))
    loop.close()

    bt = pd.DataFrame(rows).set_index("date")
    eq = bt["equity"]
    nifty = idx["Close"].reindex(days)
    nifty_eq = capital * nifty / nifty.iloc[0]
    first = closes.iloc[0].dropna()
    ew = capital * (closes[first.index].div(first).mean(axis=1))
    avg_exp = float(bt["invested"].mean())
    matched = capital * (1 + avg_exp * nifty.pct_change().fillna(0.0)).cumprod()
    years = len(days) / TRADING_DAYS

    report = {
        "window": [str(days[0].date()), str(days[-1].date())], "trading_days": int(len(days)),
        "initial_capital": capital,
        "strategy": metrics(eq),
        "benchmarks": {
            "nifty_price_index_buy_and_hold": metrics(nifty_eq),
            "equal_weight_universe_buy_and_hold": metrics(ew),
            "nifty_at_strategy_average_exposure": metrics(matched),
        },
        "yearly": {"strategy": yearly_returns(eq), "nifty": yearly_returns(nifty_eq),
                   "equal_weight": yearly_returns(ew)},
        "average_invested": avg_exp,
        "days_fully_in_cash": int((bt["invested"] < 0.01).sum()),
        "trades": len(broker.trade_history), "rejected_orders": rejected,
        "fees_paid": float(broker.fees_paid),
        "annual_turnover": float(trade_value / eq.mean() / years),
        "regime_days": bt["regime"].value_counts().to_dict(),
        "regime_models": {str(k): v for k, v in regime.models.items()},
        "settings": {k: getattr(CONFIG, k) for k in (
            "n_select", "kelly_fraction", "kelly_prior_edge_vol", "alpha_ic", "cov_lookback_days",
            "momentum_lookback_days", "rebalance_band", "min_trade_value", "es_relative_threshold")},
        "biases": ["survivorship (today's universe applied to the past)",
                   "NIFTY is a price index; stocks include dividends",
                   "cash earns 0%", "no VPIN toxicity (no historical 1-minute data)"],
        "seconds": round(_time.time() - t_start, 1),
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    bt.assign(nifty_equity=nifty_eq, equal_weight_equity=ew,
              nifty_matched_equity=matched).to_csv(os.path.join(REPORT_DIR, "backtest_daily_equity.csv"))
    with open(os.path.join(REPORT_DIR, "backtest_daily_report.json"), "w", encoding="ascii") as f:
        json.dump(report, f, indent=2, default=float)
    write_markdown(report)
    plot(bt, nifty_eq, ew, matched)
    return report


def write_markdown(r: Dict) -> None:
    def row(name, m):
        return (f"| {name} | {100 * m['cagr']:.1f}% | {100 * m['ann_vol']:.1f}% | {m['sharpe']:.2f} | "
                f"{m['sortino']:.2f} | {100 * m['max_drawdown']:.1f}% | {100 * m['total_return']:.1f}% |")

    lines = [
        "# Titan-Q walk-forward daily backtest",
        "",
        f"Window {r['window'][0]} to {r['window'][1]} ({r['trading_days']} trading days), "
        f"starting capital Rs. {r['initial_capital']:,.0f}. Generated by backtest_daily.py; "
        "every decision uses only data from before the day it trades, and orders fill at the "
        "day's open with the full Indian delivery cost stack.",
        "",
        "| | CAGR | Volatility | Sharpe | Sortino | Max drawdown | Total return |",
        "|---|---|---|---|---|---|---|",
        row("**Titan-Q strategy**", r["strategy"]),
        row("NIFTY 50 buy-and-hold (price index)", r["benchmarks"]["nifty_price_index_buy_and_hold"]),
        row("Equal-weight universe buy-and-hold", r["benchmarks"]["equal_weight_universe_buy_and_hold"]),
        row("NIFTY at the strategy's average exposure", r["benchmarks"]["nifty_at_strategy_average_exposure"]),
        "",
        f"Average invested: {100 * r['average_invested']:.1f}% of equity; days fully in cash: "
        f"{r['days_fully_in_cash']}; trades: {r['trades']} ({r['rejected_orders']} rejected); fees: "
        f"Rs. {r['fees_paid']:,.0f}; annual turnover: {100 * r['annual_turnover']:.0f}% of equity.",
        "",
        "| Year | Strategy | NIFTY | Equal weight |",
        "|---|---|---|---|",
    ]
    for y in r["yearly"]["strategy"]:
        lines.append(f"| {y} | {100 * r['yearly']['strategy'][y]:.1f}% | "
                     f"{100 * r['yearly']['nifty'].get(y, float('nan')):.1f}% | "
                     f"{100 * r['yearly']['equal_weight'].get(y, float('nan')):.1f}% |")
    lines += ["", "Known biases: " + "; ".join(r["biases"]) + ".", ""]
    with open(os.path.join(REPORT_DIR, "backtest_daily_report.md"), "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def plot(bt: pd.DataFrame, nifty_eq, ew, matched) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(bt.index, bt["equity"], label="Titan-Q", lw=2)
        ax[0].plot(nifty_eq.index, nifty_eq, label="NIFTY buy-and-hold", lw=1)
        ax[0].plot(ew.index, ew, label="Equal-weight universe", lw=1)
        ax[0].plot(matched.index, matched, label="NIFTY at strategy's avg exposure", lw=1, ls="--")
        ax[0].set_ylabel("Equity (Rs.)")
        ax[0].legend()
        ax[0].set_title("Titan-Q walk-forward backtest (fills at the open, full costs)")
        ax[1].fill_between(bt.index, 100 * bt["invested"], step="post", alpha=0.5)
        ax[1].set_ylabel("Invested %")
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "backtest_daily.png"), dpi=110)
        plt.close(fig)
    except Exception as exc:
        log.warning("plot skipped: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("paper_broker").setLevel(logging.WARNING)      # one line per trade otherwise
    ap = argparse.ArgumentParser(description="Walk-forward daily backtest of the live decision layer")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    rep = run(date.fromisoformat(a.start), date.fromisoformat(a.end) if a.end else date.today())
    with open(os.path.join(REPORT_DIR, "backtest_daily_report.md"), encoding="ascii") as f:
        print(f.read())
    print(f"completed in {rep['seconds']:.0f}s")

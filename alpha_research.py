"""
alpha_research.py
=================
Out-of-sample research for Titan-Q's stock selection and market exposure.

    D:\\trader_env\\Scripts\\python.exe alpha_research.py      -> reports/alpha_research.md

PROTOCOL (fixed before any result was seen)
    Development period   formation dates 2015-09 .. 2020-12
    Test period          formation dates 2021-01 .. 2026-08  (looked at once)

    Seven documented cross-sectional signals, nothing else:
        MOM_12_1   12-month momentum skipping the last month (Jegadeesh-Titman 1993)
        MOM_6_1    6-month momentum skipping the last month
        REV_1M     1-month reversal (Jegadeesh 1990; Lehmann 1990)
        LOWVOL     minus 1-year daily volatility (Ang et al. 2006; Blitz-van Vliet 2007)
        LOWBETA    minus 1-year beta to NIFTY (Frazzini-Pedersen 2014)
        HI52       price / 52-week high (George-Hwang 2004)
        TREND200   price / 200-day average - 1

    Monthly: signals at the close of the last trading day of month t; the
    position is entered at the NEXT day's close (one-day implementation lag)
    and held to the day after the next month-end. Every rebalance pays
    STT + stamp + fees + slippage on the weight traded.

    A signal is only worth using if it is positive in development AND keeps
    its sign in the test period. Seven signals times two portfolio sizes is
    fourteen comparisons, so one |t| > 2 is expected by chance.

    Exposure overlays on NIFTY (daily, same two periods): buy-and-hold, a
    200-day trend filter, volatility targeting, and both combined.

BIASES
    The 50-stock list is today's, so the universe is survivorship-biased; the
    comparisons are therefore made AGAINST the equal-weight universe, which
    shares the bias. NIFTY is a price index; cash earns nothing.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from titanq_config import CONFIG, INDEX_SYMBOL, PROJECT_DIR, UNIVERSE
from market_data import ALL_SYMBOLS, DailyStore

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2015-09-01", "2020-12-31")
TEST = ("2021-01-01", "2026-08-31")
COST_PER_UNIT_TRADED = (CONFIG.stt_rate + CONFIG.exchange_rate + CONFIG.sebi_rate
                        + CONFIG.stamp_rate_buy / 2 + CONFIG.slippage_bps / 1e4)
SIGNALS = ("MOM_12_1", "MOM_6_1", "REV_1M", "LOWVOL", "LOWBETA", "HI52", "TREND200")
TOP_K = (4, 8)


def load_panel() -> pd.DataFrame:
    daily = DailyStore().load(ALL_SYMBOLS, refresh=False)
    return pd.DataFrame({s: daily[s]["Close"] for s in ALL_SYMBOLS if s in daily}).sort_index()


def signals_at(px: pd.DataFrame, i: int) -> pd.DataFrame:
    """All seven signals from data up to and including row i."""
    win = px.iloc[i - 252:i + 1]
    stocks = [s for s in UNIVERSE if s in win and win[s].notna().all()]
    p = win[stocks]
    r = np.log(p).diff().iloc[1:]
    ri = np.log(win[INDEX_SYMBOL]).diff().iloc[1:]
    out = pd.DataFrame(index=stocks)
    out["MOM_12_1"] = np.log(p.iloc[-22] / p.iloc[0])
    out["MOM_6_1"] = np.log(p.iloc[-22] / p.iloc[-127])
    out["REV_1M"] = -np.log(p.iloc[-1] / p.iloc[-22])
    out["LOWVOL"] = -r.std()
    out["LOWBETA"] = -r.apply(lambda c: np.cov(c, ri)[0, 1] / ri.var())
    out["HI52"] = p.iloc[-1] / p.max()
    out["TREND200"] = p.iloc[-1] / p.iloc[-200:].mean() - 1
    return out


def monthly_study(px: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    idx = px.index
    month_end_rows = [i for i in range(len(idx) - 1) if idx[i].month != idx[i + 1].month]
    ics, rets = [], []
    held: Dict[str, pd.Series] = {}
    for a, b in zip(month_end_rows[:-1], month_end_rows[1:]):
        if a < 253:
            continue
        sig = signals_at(px, a)
        enter, exit_ = a + 1, min(b + 1, len(idx) - 1)
        fwd = np.log(px[sig.index].iloc[exit_] / px[sig.index].iloc[enter]).dropna()
        sig = sig.loc[fwd.index]
        t = idx[a]
        row_ic = {"date": t}
        row_ret = {"date": t, "EW": float(np.expm1(fwd).mean())}
        for name in SIGNALS:
            rho = stats.spearmanr(sig[name], fwd).correlation
            row_ic[name] = rho
            ranked = sig[name].sort_values(ascending=False)
            for k in TOP_K:
                key = f"{name}_top{k}"
                w = pd.Series(1.0 / k, index=ranked.index[:k])
                prev = held.get(key, pd.Series(dtype=float))
                turnover = w.subtract(prev, fill_value=0.0).abs().sum()
                held[key] = w
                row_ret[key] = float((np.expm1(fwd[w.index]) * w).sum() - turnover * COST_PER_UNIT_TRADED)
        ics.append(row_ic)
        rets.append(row_ret)
    return {"ic": pd.DataFrame(ics).set_index("date"), "ret": pd.DataFrame(rets).set_index("date")}


def summarise(study: Dict[str, pd.DataFrame], period) -> List[Dict]:
    ic = study["ic"].loc[period[0]:period[1]]
    ret = study["ret"].loc[period[0]:period[1]]
    rows = []
    for name in SIGNALS:
        s = ic[name].dropna()
        row = {"signal": name, "months": int(s.size), "mean_ic": float(s.mean()),
               "ic_t": float(s.mean() / s.std(ddof=1) * math.sqrt(s.size)),
               "ic_hit": float((s > 0).mean())}
        for k in TOP_K:
            ex = ret[f"{name}_top{k}"] - ret["EW"]
            row[f"top{k}_excess_ann"] = float(ex.mean() * 12)
            row[f"top{k}_excess_t"] = float(ex.mean() / ex.std(ddof=1) * math.sqrt(ex.size))
            row[f"top{k}_sharpe"] = float(ret[f"{name}_top{k}"].mean() / ret[f"{name}_top{k}"].std(ddof=1)
                                          * math.sqrt(12))
        rows.append(row)
    ew = ret["EW"]
    rows.append({"signal": "EW universe", "months": int(ew.size),
                 "ew_ann_return": float(ew.mean() * 12),
                 "ew_sharpe": float(ew.mean() / ew.std(ddof=1) * math.sqrt(12))})
    return rows


# ---- exposure overlays on NIFTY ------------------------------------------------

def perf(r: pd.Series) -> Dict[str, float]:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std(ddof=1) * math.sqrt(252)
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(vol),
            "sharpe": float(r.mean() * 252 / vol) if vol > 0 else 0.0,
            "max_dd": float((1 - eq / eq.cummax()).max())}


def overlay_study(px: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    p = px[INDEX_SYMBOL].dropna()
    r = p.pct_change()
    sma = p.rolling(200).mean()
    vol60 = r.rolling(60).std() * math.sqrt(252)
    # every exposure is decided on data up to t-1 and applied to the return of day t
    trend = (p > sma).astype(float).shift(1)
    volt = (0.15 / vol60).clip(upper=1.0).shift(1)
    rules = {"buy_and_hold": pd.Series(1.0, index=p.index), "trend_200d": trend,
             "vol_target_15": volt, "trend_x_vol_target": trend * volt}
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, e in rules.items():
        e = e.reindex(p.index)
        net = e * r - e.diff().abs() * COST_PER_UNIT_TRADED
        out[name] = {}
        for label, (a, b) in (("development", DEV), ("test", TEST)):
            seg = net.loc[a:b].dropna()
            out[name][label] = perf(seg) | {"avg_exposure": float(e.loc[a:b].mean())}
    return out


def write_report(dev: List[Dict], test: List[Dict], overlays: Dict, construction: Dict) -> str:
    L = ["# Titan-Q alpha research", "",
         f"Development formation dates {DEV[0]}..{DEV[1]}; test {TEST[0]}..{TEST[1]}. "
         "Monthly rebalancing, one-day implementation lag, costs "
         f"{10000 * COST_PER_UNIT_TRADED:.1f} bps per unit of weight traded. Excess returns are "
         "against the equal-weight universe (which shares its survivorship bias).", ""]
    for label, rows in (("Development (2015-2020)", dev), ("Test (2021-2026)", test)):
        L += [f"## {label}", "",
              "| Signal | Mean rank IC | IC t | IC>0 | Top-4 excess/yr | t | Top-8 excess/yr | t |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows[:-1]:
            L.append(f"| {r['signal']} | {r['mean_ic']:+.3f} | {r['ic_t']:+.2f} | {100 * r['ic_hit']:.0f}% | "
                     f"{100 * r['top4_excess_ann']:+.1f}% | {r['top4_excess_t']:+.2f} | "
                     f"{100 * r['top8_excess_ann']:+.1f}% | {r['top8_excess_t']:+.2f} |")
        ew = rows[-1]
        L += ["", f"Equal-weight universe: {100 * ew['ew_ann_return']:.1f}% a year, Sharpe "
                  f"{ew['ew_sharpe']:.2f} ({ew['months']} months).", ""]
    L += ["## Exposure overlays on NIFTY (daily, after costs)", "",
          "| Rule | Dev CAGR | Dev Sharpe | Dev max DD | Test CAGR | Test Sharpe | Test max DD | Avg exposure (test) |",
          "|---|---|---|---|---|---|---|---|"]
    for name, d in overlays.items():
        a, b = d["development"], d["test"]
        L.append(f"| {name} | {100 * a['cagr']:.1f}% | {a['sharpe']:.2f} | {100 * a['max_dd']:.1f}% | "
                 f"{100 * b['cagr']:.1f}% | {b['sharpe']:.2f} | {100 * b['max_dd']:.1f}% | "
                 f"{100 * b['avg_exposure']:.0f}% |")
    L += ["", "## Weighting the whole universe (monthly, fully invested, after costs)", "",
          "| Weights | Dev CAGR | Dev Sharpe | Dev max DD | Test CAGR | Test Sharpe | Test max DD |",
          "|---|---|---|---|---|---|---|"]
    for name, d in construction.items():
        a, b = d["development"], d["test"]
        L.append(f"| {name} | {100 * a['cagr']:.1f}% | {a['sharpe']:.2f} | {100 * a['max_dd']:.1f}% | "
                 f"{100 * b['cagr']:.1f}% | {b['sharpe']:.2f} | {100 * b['max_dd']:.1f}% |")
    return "\n".join(L) + "\n"


def construction_study(px: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    With no selection signal, how should the whole universe be weighted?
    Monthly, same timing and costs as monthly_study: equal weight, inverse
    volatility, and HRP on a Ledoit-Wolf covariance -- all fully invested.
    """
    from sklearn.covariance import LedoitWolf
    from portfolio_hrp_allocator import HRPAllocator

    idx = px.index
    ends = [i for i in range(len(idx) - 1) if idx[i].month != idx[i + 1].month]
    rows, held = [], {}
    for a, b in zip(ends[:-1], ends[1:]):
        if a < 253:
            continue
        win = px.iloc[a - 252:a + 1]
        stocks = [s for s in UNIVERSE if s in win and win[s].notna().all()]
        r = np.log(win[stocks]).diff().iloc[1:]
        cov = LedoitWolf().fit(r.to_numpy()).covariance_
        vol = np.sqrt(np.diag(cov))
        enter, exit_ = a + 1, min(b + 1, len(idx) - 1)
        fwd = np.expm1(np.log(px[stocks].iloc[exit_] / px[stocks].iloc[enter])).fillna(0.0)
        weights = {
            "equal_weight": pd.Series(1.0 / len(stocks), index=stocks),
            "inverse_vol": pd.Series((1 / vol) / (1 / vol).sum(), index=stocks),
            "hrp": pd.Series(HRPAllocator(method="single", bisection="dendrogram")
                             .allocate(cov).weights, index=stocks),
        }
        row = {"date": idx[a]}
        for k, w in weights.items():
            turnover = w.subtract(held.get(k, pd.Series(dtype=float)), fill_value=0.0).abs().sum()
            held[k] = w
            row[k] = float((fwd[w.index] * w).sum() - turnover * COST_PER_UNIT_TRADED)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for k in df.columns:
        out[k] = {}
        for label, (a, b) in (("development", DEV), ("test", TEST)):
            s = df[k].loc[a:b]
            eq = (1 + s).cumprod()
            out[k][label] = {"cagr": float(eq.iloc[-1] ** (12 / len(s)) - 1),
                             "vol": float(s.std(ddof=1) * math.sqrt(12)),
                             "sharpe": float(s.mean() / s.std(ddof=1) * math.sqrt(12)),
                             "max_dd": float((1 - eq / eq.cummax()).max()), "avg_exposure": 1.0}
    return out


def blind(overlays: Dict) -> Dict:
    nan = {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"),
           "max_dd": float("nan"), "avg_exposure": float("nan")}
    return {k: {"development": v["development"], "test": nan} for k, v in overlays.items()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal-test", action="store_true",
                    help="show the held-out 2021-2026 results (do this once, after deciding)")
    reveal = ap.parse_args().reveal_test
    px = load_panel()
    st = monthly_study(px)
    dev, test = summarise(st, DEV), summarise(st, TEST)
    ov = overlay_study(px)
    cs = construction_study(px)
    if not reveal:
        test = [{**r, **{k: float("nan") for k in r if k not in ("signal", "months")}} for r in test]
        ov = blind(ov)
        cs = blind(cs)
    os.makedirs(REPORT_DIR, exist_ok=True)
    md = write_report(dev, test, ov, cs)
    with open(os.path.join(REPORT_DIR, "alpha_research.md"), "w", encoding="ascii") as f:
        f.write(md)
    with open(os.path.join(REPORT_DIR, "alpha_research.json"), "w", encoding="ascii") as f:
        json.dump({"development": dev, "test": test, "overlays": ov, "construction": cs,
                   "test_revealed": reveal, "cost_per_unit_traded": COST_PER_UNIT_TRADED},
                  f, indent=2)
    print(md)

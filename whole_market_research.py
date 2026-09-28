"""
whole_market_research.py
========================
Implements reports/preregistration_whole_market.md exactly: the same seven signals
across the whole liquid NSE market (point-in-time universe), with rank ICs, excess
returns after costs and hit rates ("accuracy").

    D:\\trader_env\\Scripts\\python.exe whole_market_research.py               development only
    D:\\trader_env\\Scripts\\python.exe whole_market_research.py --reveal-test once, after the
                                                                              decision is committed

Needs data/universe/nse_universe.parquet and the daily cache (run universe.py first).
Output: reports/whole_market_research.md and .json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from titanq_config import INDEX_SYMBOL, PROJECT_DIR
from market_data import DailyStore
from universe import MIN_HISTORY_DAYS, MIN_PRICE, TRADABLE_VALUE, load_universe

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2015-09-01", "2020-12-31")
TEST = ("2021-01-01", "2026-08-31")
COST = 0.0025                        # 25 bps per unit of weight traded (pre-registered)
SIGNALS = ("MOM_12_1", "MOM_6_1", "REV_1M", "LOWVOL", "LOWBETA", "HI52", "TREND200")
TOP_N = 30
IC_T_MIN, EXCESS_T_MIN = 2.7, 2.0    # pre-registered thresholds


def load_panels():
    uni = load_universe()
    names = uni.loc[uni["has_daily"], "yahoo"].tolist()
    daily = DailyStore().load(names + [INDEX_SYMBOL], refresh=False)
    px = pd.DataFrame({s: d["Close"] for s, d in daily.items()}).sort_index()
    vol = pd.DataFrame({s: d["Volume"] for s, d in daily.items() if s != INDEX_SYMBOL}).reindex(px.index)
    return px, vol


def eligible_at(px: pd.DataFrame, value: pd.DataFrame, a: int) -> List[str]:
    """Point in time: uses rows strictly before row a for liquidity, price and history,
    and requires a complete 253-row window for the signals."""
    past_value = value.iloc[max(0, a - 60):a]
    med = past_value.median()
    hist = px.iloc[:a].notna().sum()
    last = px.iloc[a - 1]
    win_ok = px.iloc[a - 252:a + 1].notna().all()
    ok = (med >= TRADABLE_VALUE) & (last >= MIN_PRICE) & (hist >= MIN_HISTORY_DAYS) & win_ok
    return [s for s in ok.index[ok.to_numpy()] if s != INDEX_SYMBOL]


def signals_at(px: pd.DataFrame, a: int, stocks: List[str]) -> pd.DataFrame:
    win = px.iloc[a - 252:a + 1]
    p = win[stocks]
    r = np.log(p).diff().iloc[1:]
    ri = np.log(win[INDEX_SYMBOL]).diff().iloc[1:].to_numpy()
    rc = r.to_numpy() - r.to_numpy().mean(0)
    ric = ri - ri.mean()
    out = pd.DataFrame(index=stocks)
    out["MOM_12_1"] = np.log(p.iloc[-22] / p.iloc[0])
    out["MOM_6_1"] = np.log(p.iloc[-22] / p.iloc[-127])
    out["REV_1M"] = -np.log(p.iloc[-1] / p.iloc[-22])
    out["LOWVOL"] = -r.std()
    out["LOWBETA"] = -(rc * ric[:, None]).sum(0) / (ric ** 2).sum()
    out["HI52"] = p.iloc[-1] / p.max()
    out["TREND200"] = p.iloc[-1] / p.iloc[-200:].mean() - 1
    return out


def monthly_study(px: pd.DataFrame, vol: pd.DataFrame, signal_names=SIGNALS,
                  composite: List[str] = ()) -> pd.DataFrame:
    value = px.drop(columns=[INDEX_SYMBOL]) * vol
    idx = px.index
    ends = [i for i in range(len(idx) - 1) if idx[i].month != idx[i + 1].month]
    held: Dict[str, pd.Series] = {}
    rows = []
    for a, b in zip(ends[:-1], ends[1:]):
        if a < 505:
            continue
        stocks = eligible_at(px, value, a)
        if len(stocks) < 50:
            continue
        sig = signals_at(px, a, stocks)
        if composite:
            z = (sig[list(composite)] - sig[list(composite)].mean()) / sig[list(composite)].std()
            sig["COMPOSITE"] = z.mean(axis=1)
        enter, exit_ = a + 1, min(b + 1, len(idx) - 1)
        fwd = (px[stocks].iloc[exit_] / px[stocks].iloc[enter] - 1).dropna()
        sig = sig.loc[fwd.index]
        med = float(fwd.median())
        row = {"date": idx[a], "n": len(fwd), "EW": float(fwd.mean())}
        names = list(signal_names) + (["COMPOSITE"] if composite else [])
        for name in names:
            s = sig[name].dropna()
            f = fwd.loc[s.index]
            row[f"{name}_ic"] = stats.spearmanr(s, f).correlation
            ranked = s.sort_values(ascending=False)
            for label, k in (("dec", max(len(ranked) // 10, 1)), ("top30", TOP_N)):
                picks = ranked.index[:k]
                w = pd.Series(1.0 / len(picks), index=picks)
                key = f"{name}_{label}"
                turnover = w.subtract(held.get(key, pd.Series(dtype=float)), fill_value=0.0).abs().sum()
                held[key] = w
                row[f"{key}_ret"] = float((f[picks] * w).sum() - turnover * COST)
                row[f"{key}_hit_rel"] = float((f[picks] > med).mean())
                row[f"{key}_hit_abs"] = float((f[picks] - COST > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def summarise(df: pd.DataFrame, period, names) -> List[Dict]:
    d = df.loc[period[0]:period[1]]
    out = []
    for name in names:
        ic = d[f"{name}_ic"].dropna()
        r = {"signal": name, "months": int(len(ic)), "avg_stocks": float(d["n"].mean()),
             "mean_ic": float(ic.mean()), "ic_t": float(ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)))}
        for label in ("dec", "top30"):
            ex = d[f"{name}_{label}_ret"] - d["EW"]
            r[f"{label}_excess_ann"] = float(ex.mean() * 12)
            r[f"{label}_excess_t"] = float(ex.mean() / ex.std(ddof=1) * math.sqrt(len(ex)))
            r[f"{label}_hit_rel"] = float(d[f"{name}_{label}_hit_rel"].mean())
            r[f"{label}_hit_abs"] = float(d[f"{name}_{label}_hit_abs"].mean())
            r[f"{label}_month_hit"] = float((ex > 0).mean())
        r["qualifies"] = bool(r["ic_t"] >= IC_T_MIN and r["dec_excess_ann"] > 0
                              and r["dec_excess_t"] >= EXCESS_T_MIN and r["dec_hit_rel"] > 0.5)
        out.append(r)
    ew = d["EW"]
    out.append({"signal": "EW universe", "months": int(len(ew)), "ann_return": float(ew.mean() * 12),
                "sharpe": float(ew.mean() / ew.std(ddof=1) * math.sqrt(12))})
    return out


def table(rows: List[Dict], title: str) -> List[str]:
    L = [f"## {title}", "",
         "| Signal | Mean IC | IC t | Decile excess/yr | t | Decile hit (vs median) | Decile hit (>0 net) | "
         "Months decile beat EW | Top-30 excess/yr | Qualifies |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows[:-1]:
        L.append(f"| {r['signal']} | {r['mean_ic']:+.3f} | {r['ic_t']:+.2f} | "
                 f"{100 * r['dec_excess_ann']:+.1f}% | {r['dec_excess_t']:+.2f} | "
                 f"{100 * r['dec_hit_rel']:.1f}% | {100 * r['dec_hit_abs']:.1f}% | "
                 f"{100 * r['dec_month_hit']:.0f}% | {100 * r['top30_excess_ann']:+.1f}% | "
                 f"{'YES' if r['qualifies'] else 'no'} |")
    ew = rows[-1]
    L += ["", f"Equal-weight universe: {100 * ew['ann_return']:.1f}% a year, Sharpe {ew['sharpe']:.2f}, "
              f"{ew['months']} months, about {rows[0]['avg_stocks']:.0f} eligible stocks per month.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal-test", action="store_true")
    a = ap.parse_args()
    px, vol = load_panels()
    df = monthly_study(px, vol)
    dev = summarise(df, DEV, SIGNALS)
    qualified = [r["signal"] for r in dev[:-1] if r["qualifies"]]
    candidate = ("COMPOSITE of " + ", ".join(qualified)) if len(qualified) >= 2 else \
                (qualified[0] if qualified else "none")
    L = ["# Titan-Q whole-market signal research", "",
         "Implements reports/preregistration_whole_market.md. Point-in-time liquid NSE universe "
         "(median 60-day traded value >= Rs. 10 crore, price >= Rs. 20, >= 2 years of history), "
         "monthly rebalancing, one-day lag, 25 bps per unit traded. Hit rate (vs median) = share of "
         "picked stocks that beat the universe median next month.", ""]
    L += table(dev, "Development (formation dates 2015-09 to 2020-12)")
    L += [f"**Pre-registered decision on development: candidate = {candidate}.**", ""]
    result = {"development": dev, "candidate": candidate, "qualified": qualified, "test_revealed": a.reveal_test}
    if a.reveal_test:
        names = list(SIGNALS)
        df_test = df
        if len(qualified) >= 2:
            df_test = monthly_study(px, vol, composite=qualified)
            names = names + ["COMPOSITE"]
        test = summarise(df_test, TEST, names)
        L += table(test, "Test (formation dates 2021-01 to 2026-08), revealed once")
        result["test"] = test
    with open(os.path.join(REPORT_DIR, "whole_market_research.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "whole_market_research.json"), "w", encoding="ascii") as f:
        json.dump(result, f, indent=2, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()

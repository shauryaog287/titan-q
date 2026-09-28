"""
ml_research.py
==============
Implements reports/preregistration_ml_models.md: walk-forward ML models.

    D:\\trader_env\\Scripts\\python.exe ml_research.py m3               development only
    D:\\trader_env\\Scripts\\python.exe ml_research.py m2               development only
    D:\\trader_env\\Scripts\\python.exe ml_research.py m3 --reveal-test once, after the decision is committed

M3  NIFTY open-to-close direction on day D from information available before 09:15
    IST on D (global markets' last completed sessions, D-1 India state).
M2  Monthly whole-market ranker: gradient boosting on ranked stock features plus
    market context, predicting next month's cross-sectional return rank.

Every prediction comes from a model fitted only on data whose outcome was known
before the prediction date. Hyperparameters are the pre-registered defaults.
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

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2016-01-01", "2020-12-31")
TEST_M3 = ("2021-01-01", "2026-09-30")
TEST_M2 = ("2021-01-01", "2026-08-31")


# ===========================================================================
# M3: NIFTY intraday direction from pre-open information
# ===========================================================================

US = ("^GSPC", "^IXIC", "^DJI")
LAGGED_RET = ("^GSPC", "^IXIC", "^DJI", "BZ=F", "GC=F", "SI=F", "USDINR=X", "DX-Y.NYB",
              "^N225", "^HSI", "000001.SS", "^KS11", "^FTSE", "^GDAXI")
LAGGED_CHG = ("^VIX", "^TNX")
COST_BPS_PER_SIDE = 2.0


def _last_before(series: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """For each India day D, the value on the latest date STRICTLY before D."""
    s = series.dropna().sort_index()
    pos = np.searchsorted(s.index.values, days.values, side="left") - 1
    out = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(out, index=days)


def m3_dataset() -> pd.DataFrame:
    from titanq_config import MACRO_SYMBOLS
    d = DailyStore().load(list(MACRO_SYMBOLS) + [INDEX_SYMBOL], refresh=False)
    nifty = d[INDEX_SYMBOL][["Open", "Close"]].dropna()
    nifty = nifty[(nifty["Open"] > 0) & (nifty["Close"] > 0)]
    days = nifty.index
    X = pd.DataFrame(index=days)
    for s in LAGGED_RET:                              # last completed session's return
        r = np.log(d[s]["Close"].dropna()).diff()
        X[f"r_{s}"] = _last_before(r, days)
    for s in LAGGED_CHG:
        c = d[s]["Close"].dropna().diff()
        X[f"d_{s}"] = _last_before(c, days)
    c = np.log(nifty["Close"])
    oc = np.log(nifty["Close"] / nifty["Open"])
    X["nifty_r1"] = c.diff().shift(1)
    X["nifty_r5"] = (c - c.shift(5)).shift(1)
    X["nifty_r20"] = (c - c.shift(20)).shift(1)
    X["nifty_vol20"] = c.diff().rolling(20).std().shift(1)
    X["nifty_oc_prev"] = oc.shift(1)
    vix = d["^INDIAVIX"]["Close"]
    X["indiavix"] = _last_before(vix, days)
    X["indiavix_chg"] = _last_before(vix.diff(), days)
    for s, name in (("^NSEBANK", "bank_rel"), ("^NSEMDCP50", "mid_rel")):
        rs = np.log(d[s]["Close"].dropna()).diff()
        X[name] = _last_before(rs, days) - X["nifty_r1"]
    X["y_ret"] = oc
    X["y"] = (oc > 0).astype(int)
    return X.dropna()


def m3_walk_forward(df: pd.DataFrame, refit_every: int = 21, min_train: int = 500) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in df.columns if c not in ("y", "y_ret")]
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    out = pd.DataFrame(index=df.index, columns=["p_logit", "p_gbm"], dtype=float)
    models = None
    for i in range(min_train, len(df)):
        if models is None or (i - min_train) % refit_every == 0:
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                min_samples_leaf=50, random_state=0)
            lr.fit(X[:i], y[:i])                    # days before i only: outcomes all known
            gb.fit(X[:i], y[:i])
            models = (lr, gb)
        out.iloc[i, 0] = models[0].predict_proba(X[i:i + 1])[0, 1]
        out.iloc[i, 1] = models[1].predict_proba(X[i:i + 1])[0, 1]
    return out.dropna()


def m3_evaluate(df: pd.DataFrame, preds: pd.DataFrame, period) -> Dict[str, Dict[str, float]]:
    p = preds.loc[period[0]:period[1]]
    r = df.loc[p.index, "y_ret"]
    cost = 2 * COST_BPS_PER_SIDE / 1e4
    res = {}
    for col in ("p_logit", "p_gbm"):
        pos = np.where(p[col] > 0.55, 1.0, np.where(p[col] < 0.45, -1.0, 0.0))
        traded = pos != 0
        net = pos * r - traded * cost
        acc = float(((pos * r) > 0)[traded].mean()) if traded.any() else float("nan")
        sd = net.std(ddof=1)
        res[col] = {"days": int(len(net)), "traded_days": int(traded.sum()), "accuracy_traded": acc,
                    "accuracy_all_days": float(((p[col] > 0.5).astype(int) == (r > 0).astype(int)).mean()),
                    "mean_net_bps": float(net.mean() * 1e4),
                    "t": float(net.mean() / sd * math.sqrt(len(net))) if sd > 0 else 0.0,
                    "sharpe": float(net.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0}
    al = r - cost
    res["always_long"] = {"days": int(len(al)), "accuracy_all_days": float((r > 0).mean()),
                          "mean_net_bps": float(al.mean() * 1e4),
                          "sharpe": float(al.mean() / al.std(ddof=1) * math.sqrt(252))}
    return res


def m3_qualifies(e: Dict) -> bool:
    return (e["accuracy_traded"] > 0.52 and e["mean_net_bps"] > 0 and e["t"] >= 2.0)


def run_m3(reveal: bool) -> None:
    df = m3_dataset()
    preds = m3_walk_forward(df)
    dev = m3_evaluate(df, preds, DEV)
    al_sharpe = dev["always_long"]["sharpe"]
    ok = {m: m3_qualifies(dev[m]) and dev[m]["sharpe"] > al_sharpe for m in ("p_logit", "p_gbm")}
    passed = [m for m, v in ok.items() if v]
    choice = max(passed, key=lambda m: dev[m]["sharpe"]) if passed else None
    L = ["# M3: NIFTY open-to-close direction from pre-open information", "",
         "Implements reports/preregistration_ml_models.md (M3). Walk-forward, refit every 21 days; "
         "long if P(up) > 0.55, short if < 0.45, 2 bps per side.", "",
         f"Features: {len([c for c in df.columns if c not in ('y', 'y_ret')])}; first prediction "
         f"{preds.index[0].date()}.", "", "## Development", ""] + _m3_table(dev)
    L += ["", f"**Decision: {'candidate = ' + choice if choice else 'no model qualifies; test stays sealed'}.**", ""]
    out = {"development": dev, "qualifies": ok, "candidate": choice, "test_revealed": bool(reveal and choice)}
    if reveal and choice:
        test = m3_evaluate(df, preds, TEST_M3)
        L += ["## Test (revealed once)", ""] + _m3_table(test)
        out["test"] = test
    _write("ml_m3", L, out)


def _m3_table(e: Dict) -> List[str]:
    L = ["| Model | Days | Traded | Accuracy (traded) | Accuracy (all days) | Mean net/day | t | Sharpe |",
         "|---|---|---|---|---|---|---|---|"]
    for m in ("p_logit", "p_gbm"):
        x = e[m]
        L.append(f"| {m} | {x['days']} | {x['traded_days']} | {100 * x['accuracy_traded']:.1f}% | "
                 f"{100 * x['accuracy_all_days']:.1f}% | {x['mean_net_bps']:+.1f} bps | {x['t']:+.2f} | "
                 f"{x['sharpe']:.2f} |")
    a = e["always_long"]
    L.append(f"| always long | {a['days']} | {a['days']} | - | {100 * a['accuracy_all_days']:.1f}% (up days) | "
             f"{a['mean_net_bps']:+.1f} bps | - | {a['sharpe']:.2f} |")
    return L


# ===========================================================================
# M2: monthly whole-market ML ranker
# ===========================================================================

M2_FEATURES = ("MOM_12_1", "MOM_6_1", "REV_1M", "LOWVOL", "LOWBETA", "HI52", "TREND200",
               "VOL_1M", "VOL_TREND", "SIZE", "DIST_52W_LOW")


def m2_dataset():
    from whole_market_research import eligible_at, load_panels, signals_at
    px, vol = load_panels()
    vix = DailyStore().load(["^INDIAVIX"], refresh=False)["^INDIAVIX"]["Close"]
    value = px.drop(columns=[INDEX_SYMBOL]) * vol
    idx = px.index
    ends = [i for i in range(len(idx) - 1) if idx[i].month != idx[i + 1].month]
    rows = []
    for a, b in zip(ends[:-1], ends[1:]):
        if a < 505:
            continue
        stocks = eligible_at(px, value, a)
        if len(stocks) < 50:
            continue
        enter, exit_ = a + 1, min(b + 1, len(idx) - 1)
        fwd = (px[stocks].iloc[exit_] / px[stocks].iloc[enter] - 1).dropna()
        if len(fwd) < 50:
            continue
        stocks = list(fwd.index)
        f = signals_at(px, a, stocks)
        p = px[stocks].iloc[a - 252:a + 1]
        v = vol[stocks].iloc[a - 252:a + 1]
        f["VOL_1M"] = np.log(p).diff().iloc[-21:].std()
        f["VOL_TREND"] = v.iloc[-20:].mean() / v.mean()
        f["SIZE"] = np.log(value[stocks].iloc[a - 60:a].median())
        f["DIST_52W_LOW"] = p.iloc[-1] / p.min() - 1
        f = f[list(M2_FEATURES)].rank(pct=True)
        n = px[INDEX_SYMBOL].ffill()
        f["mkt_r1m"] = float(np.log(n.iloc[a] / n.iloc[a - 21]))
        vv = vix[vix.index <= idx[a]]
        f["vix"] = float(vv.iloc[-1]) if len(vv) else np.nan
        f["vix_chg1m"] = float(vv.iloc[-1] - vv.iloc[-22]) if len(vv) > 22 else np.nan
        f["label"] = fwd.rank(pct=True)
        f["fwd"] = fwd
        f["date"] = idx[a]
        rows.append(f.reset_index().rename(columns={"index": "stock"}))
    return pd.concat(rows, ignore_index=True)


def m2_walk_forward(data: pd.DataFrame, min_months: int = 24) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor

    feats = list(M2_FEATURES) + ["mkt_r1m", "vix", "vix_chg1m"]
    months = sorted(data["date"].unique())
    preds = []
    for k, m in enumerate(months):
        train_months = months[:max(k - 1, 0)]     # month k-1's label ends after month k's formation
        if len(train_months) < min_months:
            continue
        tr = data[data["date"].isin(train_months)]
        model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        model.fit(tr[feats], tr["label"])
        cur = data[data["date"] == m].copy()
        cur["score"] = model.predict(cur[feats])
        z = cur[list(M2_FEATURES[:7])]
        cur["composite"] = ((z - z.mean()) / z.std()).mean(axis=1)
        preds.append(cur)
    return pd.concat(preds, ignore_index=True)


def m2_evaluate(p: pd.DataFrame, period, cost: float = 0.0025) -> Dict[str, Dict[str, float]]:
    p = p[(p["date"] >= pd.Timestamp(period[0])) & (p["date"] <= pd.Timestamp(period[1]))]
    res = {}
    for col in ("score", "composite"):
        ics, ex, hit_rel, hit_abs = [], [], [], []
        held = pd.Series(dtype=float)
        for m, g in p.groupby("date"):
            ics.append(stats.spearmanr(g[col], g["fwd"]).correlation)
            top = g.nlargest(max(len(g) // 10, 1), col)
            w = pd.Series(1.0 / len(top), index=top["stock"].to_numpy())
            turnover = w.subtract(held, fill_value=0.0).abs().sum()
            held = w
            ret = float((top["fwd"].to_numpy() * w.to_numpy()).sum() - turnover * cost)
            ex.append(ret - float(g["fwd"].mean()))
            med = float(g["fwd"].median())
            hit_rel.append(float((top["fwd"] > med).mean()))
            hit_abs.append(float((top["fwd"] - cost > 0).mean()))
        ics, ex = np.array(ics), np.array(ex)
        res[col] = {"months": int(len(ics)), "mean_ic": float(ics.mean()),
                    "ic_t": float(ics.mean() / ics.std(ddof=1) * math.sqrt(len(ics))),
                    "dec_excess_ann": float(ex.mean() * 12),
                    "dec_excess_t": float(ex.mean() / ex.std(ddof=1) * math.sqrt(len(ex))),
                    "hit_rel": float(np.mean(hit_rel)), "hit_abs": float(np.mean(hit_abs)),
                    "month_hit": float((ex > 0).mean())}
    return res


def run_m2(reveal: bool) -> None:
    data = m2_dataset()
    preds = m2_walk_forward(data)
    dev = m2_evaluate(preds, DEV)
    s, c = dev["score"], dev["composite"]
    passed = (s["ic_t"] >= 2.7 and s["dec_excess_ann"] > 0 and s["dec_excess_t"] >= 2.0
              and s["hit_rel"] > 0.5 and s["mean_ic"] > c["mean_ic"])
    L = ["# M2: whole-market ML ranker (gradient boosting)", "",
         "Implements reports/preregistration_ml_models.md (M2). Walk-forward monthly; compared with the "
         "seven-signal composite on the same months.", "", "## Development", ""] + _m2_table(dev)
    L += ["", f"**Decision: {'PASSED -- candidate' if passed else 'did not qualify; test stays sealed'}.**", ""]
    out = {"development": dev, "passed": passed, "test_revealed": bool(reveal and passed)}
    if reveal and passed:
        test = m2_evaluate(preds, TEST_M2)
        L += ["## Test (revealed once)", ""] + _m2_table(test)
        out["test"] = test
    _write("ml_m2", L, out)


def _m2_table(e: Dict) -> List[str]:
    L = ["| Model | Months | Mean IC | IC t | Decile excess/yr | t | Hit (vs median) | Hit (>0 net) | Months beat EW |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m, name in (("score", "ML ranker"), ("composite", "7-signal composite")):
        x = e[m]
        L.append(f"| {name} | {x['months']} | {x['mean_ic']:+.3f} | {x['ic_t']:+.2f} | "
                 f"{100 * x['dec_excess_ann']:+.1f}% | {x['dec_excess_t']:+.2f} | {100 * x['hit_rel']:.1f}% | "
                 f"{100 * x['hit_abs']:.1f}% | {100 * x['month_hit']:.0f}% |")
    return L


def _write(name: str, lines: List[str], obj: Dict) -> None:
    with open(os.path.join(REPORT_DIR, f"{name}.md"), "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORT_DIR, f"{name}.json"), "w", encoding="ascii") as f:
        json.dump(obj, f, indent=2, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=("m2", "m3"))
    ap.add_argument("--reveal-test", action="store_true")
    a = ap.parse_args()
    (run_m2 if a.model == "m2" else run_m3)(a.reveal_test)

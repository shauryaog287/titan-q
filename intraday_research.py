"""
intraday_research.py
====================
Implements reports/preregistration_m1_intraday.md: M1, 30-minute direction of
liquid NSE stocks from 5-minute bars, walk-forward, after intraday costs.

    D:\\trader_env\\Scripts\\python.exe intraday_research.py               development only
    D:\\trader_env\\Scripts\\python.exe intraday_research.py --reveal-test once, after the decision is committed
    D:\\trader_env\\Scripts\\python.exe intraday_research.py --smoke       feature pipeline check only (no metrics)

Needs the Angel 1-minute history (angel_history.py) and the daily cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import time as dtime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR
from market_data import DailyStore

HIST_DIR = os.path.join(DATA_DIR, "intraday_hist")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
CACHE = os.path.join(DATA_DIR, "m1_dataset.parquet")
DEV = ("2021-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-30")
TOP_N = 100
COST = 0.0010                                  # 10 bps round trip (pre-registered)
DECISION_ENDS = [dtime(9, 45)] + [dtime(h, m) for h in range(10, 15) for m in (15, 45)]
FEATURES = ["r5", "r15", "r30", "r60", "r_open", "gap", "rel_vol", "rv30_rel", "vwap_dev",
            "n_r5", "n_r15", "n_r30", "n_open", "b_r5", "b_r15", "b_r30", "b_open",
            "rel30", "rel_open", "tod", "prev_ret", "prev_range"]


# ===========================================================================
# BARS AND FEATURES
# ===========================================================================

def load_minutes(symbol: str) -> Optional[pd.DataFrame]:
    d = os.path.join(HIST_DIR, symbol)
    if not os.path.isdir(d):
        return None
    parts = [pd.read_parquet(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".parquet")]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    m = pd.concat(parts, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    m["ts"] = pd.to_datetime(m["ts"])
    return m


def five_minute_bars(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["bar"] = m["ts"].dt.floor("5min")
    m["value"] = m["close"] * m["volume"]
    b = m.groupby("bar").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                             close=("close", "last"), volume=("volume", "sum"), value=("value", "sum"))
    b["date"] = b.index.date
    b["end"] = (b.index + pd.Timedelta(minutes=5)).time
    return b


def bar_features(b: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Within-day return features (and, for stocks, volume/VWAP/volatility features)."""
    lc = np.log(b["close"])
    g = lc.groupby(b["date"])
    f = pd.DataFrame(index=b.index)
    f[prefix + "r5"] = lc - g.shift(1)
    f[prefix + "r15"] = lc - g.shift(3)
    f[prefix + "r30"] = lc - g.shift(6)
    day_open = b.groupby("date")["open"].transform("first")
    f[prefix + ("r_open" if not prefix else "open")] = lc - np.log(day_open)
    if prefix:
        return f
    f["r60"] = lc - g.shift(12)
    f["fwd30"] = g.shift(-6) - lc                              # label: t -> t + 30 min
    daily = b.groupby("date").agg(close=("close", "last"), high=("high", "max"), low=("low", "min"),
                                  open=("open", "first"))
    dret = np.log(daily["close"]).diff()
    prev = pd.DataFrame({"prev_ret": dret.shift(0), "prev_close": daily["close"],
                         "prev_range": (daily["high"] - daily["low"]) / daily["close"],
                         "vol20": dret.rolling(20).std()}).shift(1)   # all from D-1 and before
    f = f.join(prev.reindex(b["date"].to_numpy()).set_index(b.index))
    f["gap"] = np.log(day_open / f["prev_close"])
    vol15 = b["volume"].groupby(b["date"]).transform(lambda v: v.rolling(3).sum())
    tod = b.index.hour * 60 + b.index.minute
    piv = pd.DataFrame({"date": b["date"], "tod": tod, "v": vol15.to_numpy()})
    avg = (piv.pivot_table(index="date", columns="tod", values="v")
           .rolling(20, min_periods=10).mean().shift(1))           # previous 20 days only
    f["rel_vol"] = vol15.to_numpy() / avg.stack().reindex(pd.MultiIndex.from_arrays([piv["date"], piv["tod"]])).to_numpy()
    r5 = f["r5"]
    rv30 = r5.groupby(b["date"]).transform(lambda x: x.rolling(6).std())
    f["rv30_rel"] = rv30 / f["vol20"]
    vwap = b["value"].groupby(b["date"]).cumsum() / b["volume"].groupby(b["date"]).cumsum()
    f["vwap_dev"] = b["close"] / vwap - 1
    f["tod"] = tod - (9 * 60 + 15)
    return f.drop(columns=["prev_close", "vol20"])


def market_features() -> pd.DataFrame:
    n = bar_features(five_minute_bars(load_minutes("NIFTY50_IDX")), "n_")
    bk = bar_features(five_minute_bars(load_minutes("BANKNIFTY_IDX")), "b_")
    return n.join(bk, how="outer")


def stock_dataset(symbol: str, mkt: pd.DataFrame) -> Optional[pd.DataFrame]:
    m = load_minutes(symbol)
    if m is None or len(m) < 5000:
        return None
    b = five_minute_bars(m)
    f = bar_features(b).join(mkt, how="left")
    f["rel30"] = f["r30"] - f["n_r30"]
    f["rel_open"] = f["r_open"] - f["n_open"]
    f["stock"] = symbol
    f["date"] = b["date"]
    keep = np.isin(b["end"].to_numpy(), np.array(DECISION_ENDS, dtype=object))
    f = f[keep]
    return f.dropna(subset=FEATURES + ["fwd30"])


def monthly_top(n: int = TOP_N) -> Dict[pd.Period, List[str]]:
    """Point in time: at each month start, the n most liquid names by median 60-day
    traded value from daily bars dated before that month."""
    names = [d for d in os.listdir(HIST_DIR) if not d.endswith("_IDX")]
    daily = DailyStore().load([s + ".NS" for s in names], refresh=False)
    val = pd.DataFrame({s[:-3]: d["Close"] * d["Volume"] for s, d in daily.items()}).sort_index()
    out = {}
    for p in pd.period_range("2020-01", pd.Timestamp.today().strftime("%Y-%m"), freq="M"):
        past = val[val.index < p.start_time].iloc[-60:]
        out[p] = past.median().dropna().nlargest(n).index.tolist()
    return out


def build_dataset() -> pd.DataFrame:
    mkt = market_features()
    tops = monthly_top()
    members = set().union(*tops.values())
    parts = []
    for s in sorted(members):
        df = stock_dataset(s, mkt)
        if df is None:
            continue
        per = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
        ok = np.array([s in tops.get(p, ()) for p in per])
        parts.append(df[ok])
    data = pd.concat(parts)
    data[FEATURES + ["fwd30"]] = data[FEATURES + ["fwd30"]].astype("float32")
    return data


# ===========================================================================
# WALK-FORWARD AND EVALUATION
# ===========================================================================

def walk_forward(data: pd.DataFrame, start: str, end: str, min_months: int = 12) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dates = pd.to_datetime(data["date"])
    first = dates.min()
    out = []
    for q in pd.period_range(start, end, freq="Q"):
        train = data[dates < q.start_time]
        if (q.start_time - first).days < 30 * min_months or not len(train):
            continue
        test = data[(dates >= q.start_time) & (dates <= q.end_time)]
        if not len(test):
            continue
        y = (train["fwd30"] > 0).astype(int)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                            min_samples_leaf=500, l2_regularization=1.0, random_state=0)
        lr.fit(train[FEATURES], y)
        gb.fit(train[FEATURES], y)
        t = test[["stock", "date", "fwd30"]].copy()
        t["p_logit"] = lr.predict_proba(test[FEATURES])[:, 1]
        t["p_gbm"] = gb.predict_proba(test[FEATURES])[:, 1]
        out.append(t)
        print(f"  {q}: trained on {len(train):,} rows, predicted {len(test):,}", flush=True)
    return pd.concat(out)


def evaluate(p: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    res = {}
    for col in ("p_logit", "p_gbm"):
        pos = np.where(p[col] > 0.55, 1.0, np.where(p[col] < 0.45, -1.0, 0.0))
        traded = pos != 0
        gross = pos * p["fwd30"].to_numpy()
        net = gross - COST * traded
        tr = pd.DataFrame({"date": p["date"].to_numpy()[traded], "net": net[traded]})
        daily = tr.groupby("date")["net"].mean()
        sd = daily.std(ddof=1)
        res[col] = {
            "predictions": int(len(p)), "trades": int(traded.sum()),
            "accuracy_all": float(((p[col] > 0.5) == (p["fwd30"] > 0)).mean()),
            "accuracy_traded": float((gross[traded] > 0).mean()) if traded.any() else float("nan"),
            "mean_net_bps": float(net[traded].mean() * 1e4) if traded.any() else float("nan"),
            "mean_gross_bps": float(gross[traded].mean() * 1e4) if traded.any() else float("nan"),
            "days": int(len(daily)),
            "day_t": float(daily.mean() / sd * math.sqrt(len(daily))) if len(daily) > 1 and sd > 0 else 0.0,
            "sharpe_daily": float(daily.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
        }
    return res


def qualifies(e: Dict) -> bool:
    return e["accuracy_traded"] > 0.52 and e["mean_net_bps"] > 0 and e["day_t"] >= 2.0


def table(e: Dict) -> List[str]:
    L = ["| Model | Predictions | Trades | Accuracy (all) | Accuracy (traded) | Gross/trade | Net/trade | Day-level t | Sharpe |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in ("p_logit", "p_gbm"):
        x = e[m]
        L.append(f"| {m} | {x['predictions']:,} | {x['trades']:,} | {100 * x['accuracy_all']:.2f}% | "
                 f"{100 * x['accuracy_traded']:.2f}% | {x['mean_gross_bps']:+.1f} bps | {x['mean_net_bps']:+.1f} bps | "
                 f"{x['day_t']:+.2f} | {x['sharpe_daily']:.2f} |")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="build features for 3 stocks; print shapes only")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached dataset")
    a = ap.parse_args()
    if a.smoke:
        mkt = market_features()
        for s in ("RELIANCE", "HDFCBANK", "ICICIBANK"):
            df = stock_dataset(s, mkt)
            print(s, None if df is None else (df.shape, str(df['date'].min()), str(df['date'].max()),
                                              int(df[FEATURES].isna().sum().sum())))
        print("features:", len(FEATURES), "| decision times:", [t.strftime("%H:%M") for t in DECISION_ENDS])
        return
    if a.rebuild or not os.path.exists(CACHE):
        data = build_dataset()
        data.to_parquet(CACHE, index=True)
    data = pd.read_parquet(CACHE)
    print(f"dataset: {len(data):,} rows, {data['stock'].nunique()} stocks, "
          f"{data['date'].min()} .. {data['date'].max()}")
    preds = walk_forward(data, DEV[0], DEV[1])
    dev = evaluate(preds)
    passed = [m for m in ("p_logit", "p_gbm") if qualifies(dev[m])]
    choice = max(passed, key=lambda m: dev[m]["day_t"]) if passed else None
    L = ["# M1: 30-minute direction of liquid NSE stocks (intraday)", "",
         "Implements reports/preregistration_m1_intraday.md. 5-minute bars from Angel 1-minute history; "
         "decisions every 30 min 09:45-14:45; long if P(up) > 0.55, short if < 0.45; 10 bps round trip.", "",
         f"Dataset: {len(data):,} decision rows, {data['stock'].nunique()} stocks.", "",
         "## Development (predictions 2021-01 to 2023-12)", ""] + table(dev)
    L += ["", f"**Decision: {'candidate = ' + choice if choice else 'no model qualifies; test stays sealed'}.**", ""]
    out = {"development": dev, "candidate": choice, "test_revealed": bool(a.reveal_test and choice)}
    if a.reveal_test and choice:
        tp = walk_forward(data, TEST[0], TEST[1])
        test = evaluate(tp)
        L += ["## Test (revealed once)", ""] + table(test)
        out["test"] = test
    with open(os.path.join(REPORT_DIR, "m1_intraday.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "m1_intraday.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()

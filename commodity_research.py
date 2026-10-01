"""
commodity_research.py
=====================
Implements reports/preregistration_commodity_linked.md (CM1): after a big crude or gold move
made while NSE is shut, do commodity-linked stocks keep drifting the same way during the next
NSE session (open to close, net of NIFTY)?

    D:\\trader_env\\Scripts\\python.exe commodity_research.py      development only

There is deliberately no --reveal-test path yet: it is added only if the development
decision qualifies and has been committed.

Output: reports/commodity_linked.md and .json
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from titanq_config import INDEX_SYMBOL, PROJECT_DIR

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2016-01-01", "2020-12-31")
TRIGGER, COST = 0.015, 0.0010
MIN_MEAN, MIN_T, MIN_TRADES = 0.0010, 2.0, 100

# (group, commodity, exposure, stocks): fixed by business line in the pre-registration
GROUPS: Tuple[Tuple[str, str, int, Tuple[str, ...]], ...] = (
    ("crude producers", "BZ=F", +1, ("ONGC.NS", "OIL.NS")),
    ("crude users", "BZ=F", -1, ("BPCL.NS", "HINDPETRO.NS", "IOC.NS", "ASIANPAINT.NS", "BERGEPAINT.NS",
                                 "MRF.NS", "APOLLOTYRE.NS", "CEATLTD.NS", "INDIGO.NS")),
    ("gold lenders", "GC=F", +1, ("MUTHOOTFIN.NS", "MANAPPURAM.NS")),
)
JEWELLERS = ("jewellers (exploratory)", "GC=F", +1, ("TITAN.NS", "KALYANKJIL.NS"))


def load() -> Dict[str, pd.DataFrame]:
    from market_data import yf_download
    names = ["BZ=F", "GC=F", INDEX_SYMBOL] + [s for g in GROUPS + (JEWELLERS,) for s in g[3]]
    data = yf_download(names, start="2015-01-01", interval="1d", auto_adjust=False)
    missing = [n for n in names if n not in data or data[n].empty]
    if missing:
        raise SystemExit(f"Yahoo returned nothing for {missing}; is Yahoo reachable?")
    return data


def commodity_moves(com: pd.DataFrame, nse_days: pd.DatetimeIndex) -> pd.Series:
    """Commodity return per NSE day N. Every commodity date D maps to the first NSE day
    strictly after D. When several dates map to one N (an NSE holiday on a commodity
    trading day), their returns are compounded into one move (decided before any result)."""
    c = com["Close"].dropna()
    c = c[c > 0]
    r = (c / c.shift(1) - 1).dropna()
    pos = np.searchsorted(nse_days.values, r.index.values, side="right")
    keep = pos < len(nse_days)
    n = pd.Series(nse_days[pos[keep]], index=r.index[keep])
    return (1 + r[keep]).groupby(n.values).prod() - 1


def nse_calendar(data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """NSE trading days: the index's days plus any day a loaded NSE stock traded (Yahoo misses
    a few ^NSEI days; without them a commodity move would map one session too late)."""
    days = data[INDEX_SYMBOL].dropna(subset=["Close"]).index
    for s, d in data.items():
        if s.endswith(".NS"):
            days = days.union(d.dropna(subset=["Close"]).index)
    return days.sort_values()


def trades(data: Dict[str, pd.DataFrame], group, trigger: float) -> pd.DataFrame:
    name, com, expo, stocks = group
    idx = data[INDEX_SYMBOL]
    idx = idx[(idx["Open"] > 0) & (idx["Close"] > 0)]
    mv = commodity_moves(data[com], nse_calendar(data))
    nifty_oc = idx["Close"] / idx["Open"] - 1
    out = []
    for s in stocks:
        d = data[s]
        ok = (d["Open"] > 0) & (d["Close"] > 0)
        oc = (d["Close"] / d["Open"] - 1)[ok]
        prev_close = d["Close"].where(d["Close"] > 0).shift(1)
        gap = (d["Open"] / prev_close - 1)[ok]
        for day, m in mv.items():
            if day not in oc.index or day not in nifty_oc.index:
                continue
            row = {"group": name, "stock": s, "day": day, "move": float(m),
                   "gap": float(gap.get(day, np.nan)), "traded": abs(m) >= trigger}
            sign = float(np.sign(m)) * expo
            row["net"] = sign * (oc[day] - nifty_oc[day]) - COST
            out.append(row)
    return pd.DataFrame(out)


def summary(t: pd.DataFrame) -> Dict:
    t = t[t["traded"]]
    if not len(t):
        return {"trades": 0}
    by = t.groupby("day")["net"].mean()
    sd = by.std(ddof=1)
    return {"trades": int(len(t)), "days": int(len(by)), "mean": float(t["net"].mean()),
            "t": float(by.mean() / sd * math.sqrt(len(by))) if len(by) > 1 and sd > 0 else 0.0,
            "hit": float((t["net"] > 0).mean())}


def gap_slope(t: pd.DataFrame, expo: int) -> Dict:
    """Pooled OLS of the stock's close(N-1) -> open(N) gap on exposure x commodity move,
    over every development day (triggered or not). 1.0 would mean fully priced at the open
    one for one; the size of a full reaction is unknown, so read it with the drift."""
    g = t.dropna(subset=["gap"])
    if len(g) < 30:
        return {"n": int(len(g))}
    x = expo * g["move"].to_numpy()
    y = g["gap"].to_numpy()
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    resid = y - y.mean() - b * xc
    se = float(math.sqrt((resid ** 2).sum() / (len(g) - 2) / (xc ** 2).sum()))
    return {"n": int(len(g)), "slope": b, "se": se}


def in_dev(t: pd.DataFrame) -> pd.DataFrame:
    return t[(t["day"] >= pd.Timestamp(DEV[0])) & (t["day"] <= pd.Timestamp(DEV[1]))] if len(t) else t


def main() -> None:
    data = load()
    res: Dict[str, Dict] = {}
    per_group: List[pd.DataFrame] = []
    per_group_1: List[pd.DataFrame] = []
    slopes: Dict[str, Dict] = {}
    for g in GROUPS:
        t = in_dev(trades(data, g, TRIGGER))
        t1 = in_dev(trades(data, g, 0.010))
        per_group.append(t)
        per_group_1.append(t1)
        res[f"{g[0]}, 1.5%"] = summary(t)
        res[f"{g[0]}, 1.0%"] = summary(t1)
        slopes[g[0]] = gap_slope(t, g[2])
    allt = pd.concat(per_group, ignore_index=True)
    main_res = summary(allt)
    res_all_1 = summary(pd.concat(per_group_1, ignore_index=True))
    tj = in_dev(trades(data, JEWELLERS, TRIGGER))
    res_j = summary(tj)
    slopes[JEWELLERS[0]] = gap_slope(tj, JEWELLERS[2])
    ok = bool(main_res["trades"] >= MIN_TRADES and main_res["mean"] >= MIN_MEAN and main_res["t"] >= MIN_T)

    def line(name, s):
        if not s.get("trades"):
            return f"| {name} | 0 | - | - | - | - |"
        return (f"| {name} | {s['trades']:,} | {s['days']:,} | {100 * s['mean']:+.3f}% | "
                f"{s['t']:+.2f} | {100 * s['hit']:.1f}% |")
    L = ["# CM1: overnight crude and gold moves and commodity-linked stocks", "",
         "Implements reports/preregistration_commodity_linked.md. Yahoo daily, auto_adjust=False. A commodity "
         "date D maps to the first NSE day N after it. Trade on N if the commodity moved >= 1.5% on D: each "
         "group stock from N's open to N's close in the expected direction, minus NIFTY's open-to-close in the "
         "same direction, minus 0.10% cost.", "",
         "## Development (days N from 2016-01 to 2020-12)", "",
         "| Group and trigger | Trades | Days | Mean net per trade | Day-clustered t | Hit rate |",
         "|---|---|---|---|---|---|",
         line("**Primary: all three groups, 1.5% trigger**", main_res),
         line("All three groups, 1.0% trigger", res_all_1)]
    L += [line(k, v) for k, v in res.items()]
    L += [line("Jewellers vs gold (exploratory, long on gold up), 1.5%", res_j), "",
          "## Gap capture (development, every day, not only triggered ones)", "",
          "Slope of the stock's close-to-open gap on N against exposure x commodity move (pooled OLS).", "",
          "| Group | Stock-days | Slope | Std. error |", "|---|---|---|---|"]
    for k, v in slopes.items():
        L.append(f"| {k} | {v['n']:,} | {v['slope']:+.3f} | {v['se']:.3f} |" if "slope" in v else f"| {k} | {v['n']} | - | - |")
    L += ["", f"Rule: mean net >= +{100 * MIN_MEAN:.2f}% per trade, day-clustered t >= {MIN_T:.1f}, "
          f">= {MIN_TRADES} trades.", "",
          "Note: when an NSE holiday falls on a commodity trading day, the commodity's returns on the dates "
          "that map to the same NSE day are compounded into one move (the pre-registration did not cover this; "
          "decided before any result).", "",
          f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**", ""]
    out = {"development": {"primary": main_res, "all_1.0pct": res_all_1, **res, "jewellers": res_j},
           "gap_slopes": slopes, "qualifies": ok}
    with open(os.path.join(REPORT_DIR, "commodity_linked.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "commodity_linked.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

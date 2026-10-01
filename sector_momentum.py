"""
sector_momentum.py
==================
Implements reports/preregistration_sector_momentum.md (SM1): do the strongest industries
(12-1 month return of their eligible members) keep beating the equal-weight market?

    D:\\trader_env\\Scripts\\python.exe sector_momentum.py --fetch-industries   Yahoo industry labels (resumable)
    D:\\trader_env\\Scripts\\python.exe sector_momentum.py                      development only

There is deliberately no --reveal-test path yet: it is added only if the development
decision qualifies and has been committed.

Needs the universe and daily cache (universe.py). Output: reports/sector_momentum.md and .json,
reports/industry_map.csv (a copy of data/industries.csv).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from titanq_config import DATA_DIR, INDEX_SYMBOL, PROJECT_DIR
from whole_market_research import COST, eligible_at, load_panels

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
IND_FILE = os.path.join(DATA_DIR, "industries.csv")
DEV = ("2016-01-01", "2020-12-31")
LABEL_RANGE = ("2016-01-01", "2026-08-31")       # labels only; no test-period returns are computed
TOP_K, MIN_MEMBERS = 3, 5
MIN_EXCESS, MIN_T, MIN_HIT = 0.03, 2.0, 0.55


def month_ends(idx: pd.DatetimeIndex) -> List[int]:
    return [i for i in range(len(idx) - 1) if idx[i].month != idx[i + 1].month]


def label_tickers(px: pd.DataFrame, vol: pd.DataFrame) -> List[str]:
    value = px.drop(columns=[INDEX_SYMBOL]) * vol
    idx = px.index
    out = set()
    for a in month_ends(idx):
        if a < 252 or not (pd.Timestamp(LABEL_RANGE[0]) <= idx[a] <= pd.Timestamp(LABEL_RANGE[1])):
            continue
        out.update(eligible_at(px, value, a))
    return sorted(out)


def fetch_industries() -> None:
    import yfinance as yf
    px, vol = load_panels()
    tickers = label_tickers(px, vol)
    done = pd.read_csv(IND_FILE, keep_default_na=False) if os.path.exists(IND_FILE) else \
        pd.DataFrame(columns=["ticker", "industry", "sector", "status"])
    have = set(done.loc[done["status"] != "failed", "ticker"])
    done = done[done["status"] != "failed"]
    todo = [t for t in tickers if t not in have]
    print(f"{len(tickers)} tickers eligible at some month end {LABEL_RANGE[0][:7]} to {LABEL_RANGE[1][:7]}; "
          f"{len(todo)} to fetch", flush=True)
    rows = done.to_dict("records")
    for i, t in enumerate(todo):
        rec = {"ticker": t, "industry": "", "sector": "", "status": "failed"}
        for attempt in range(3):
            try:
                info = yf.Ticker(t).info or {}
                rec.update(industry=str(info.get("industry") or ""), sector=str(info.get("sector") or ""),
                           status="ok" if info.get("industry") else "none")
                break
            except Exception as exc:                     # rate limit, network, odd ticker
                print(f"  {t}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                time.sleep(5.0 * (attempt + 1))
        rows.append(rec)
        time.sleep(1.0)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            _save(rows)
            print(f"  {i + 1}/{len(todo)}", flush=True)
    _save(rows)
    df = pd.DataFrame(rows)
    print(df["status"].value_counts().to_string(), flush=True)


def _save(rows: List[Dict]) -> None:
    df = pd.DataFrame(rows, columns=["ticker", "industry", "sector", "status"])
    # Yahoo labels are ASCII in practice; force it so the CP1252 rule cannot break
    for c in ("industry", "sector"):
        df[c] = df[c].astype(str).str.encode("ascii", "replace").str.decode("ascii")
    df.to_csv(IND_FILE + ".partial", index=False, encoding="ascii")
    os.replace(IND_FILE + ".partial", IND_FILE)


def study(px: pd.DataFrame, vol: pd.DataFrame, ind: pd.Series, sector: pd.Series, lo: str, hi: str) -> pd.DataFrame:
    """One row per month end a in [lo, hi]: EW benchmark and each variant's net return,
    entering at row a+1 and exiting at the next month end's a+1 (as monthly_study)."""
    value = px.drop(columns=[INDEX_SYMBOL]) * vol
    idx = px.index
    ends = month_ends(idx)
    held: Dict[str, pd.Series] = {}
    rows = []
    for a, b in zip(ends[:-1], ends[1:]):
        if a < 252 or not (pd.Timestamp(lo) <= idx[a] <= pd.Timestamp(hi)):
            continue
        stocks = eligible_at(px, value, a)
        if len(stocks) < 50:
            continue
        enter, exit_ = a + 1, min(b + 1, len(idx) - 1)
        fwd = (px[stocks].iloc[exit_] / px[stocks].iloc[enter] - 1).dropna()
        if len(fwd) < 50:
            continue
        members = pd.Series({s: ind.get(s, "") for s in fwd.index})
        members = members[members != ""]
        row = {"date": idx[a], "n": len(fwd), "n_labelled": len(members), "EW": float(fwd.mean())}
        for look, start in (("12_1", a - 252), ("6_1", a - 126)):
            mom = px[members.index].iloc[a - 21] / px[members.index].iloc[start] - 1
            g = pd.DataFrame({"ind": members, "mom": mom}).dropna()
            counts = g.groupby("ind")["mom"].size()
            score = g.groupby("ind")["mom"].mean()[counts >= MIN_MEMBERS].sort_values(ascending=False)
            row[f"n_industries_{look}"] = int(len(score))
            variants = {"top3": score.index[:TOP_K]} if look == "6_1" else \
                {"top3": score.index[:TOP_K], "top1": score.index[:1], "top5": score.index[:5],
                 "bottom3": score.index[-TOP_K:]}
            for name, inds in variants.items():
                key = f"{name}_{look}"
                picks = members.index[members.isin(inds)]
                if not len(picks):
                    continue
                w = pd.Series(1.0 / len(picks), index=picks)
                turnover = w.subtract(held.get(key, pd.Series(dtype=float)), fill_value=0.0).abs().sum()
                held[key] = w
                row[f"{key}_ret"] = float((fwd[picks] * w).sum() - turnover * COST)
                if key == "top3_12_1":
                    row["top3_industries"] = " | ".join(inds)
                    secs = {sector.get(s, "") for s in picks}
                    row["has_defense"] = "Aerospace & Defense" in set(inds)
                    row["has_utilities"] = "Utilities" in secs or any(str(i).startswith("Utilities") for i in inds)
                    row["has_banks"] = any(str(i).startswith("Banks") for i in inds)
                    row["has_energy"] = "Energy" in secs
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def summarise(d: pd.DataFrame, key: str) -> Dict:
    ex = (d[f"{key}_ret"] - d["EW"]).dropna()
    if len(ex) < 2:
        return {"months": int(len(ex))}
    return {"months": int(len(ex)), "excess_ann": float(ex.mean() * 12),
            "t": float(ex.mean() / ex.std(ddof=1) * math.sqrt(len(ex))),
            "month_hit": float((ex > 0).mean()),
            "port_ann": float(d.loc[ex.index, f"{key}_ret"].mean() * 12)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-industries", action="store_true")
    a = ap.parse_args()
    if a.fetch_industries:
        fetch_industries()
        return
    if not os.path.exists(IND_FILE):
        raise SystemExit("run --fetch-industries first")
    lab = pd.read_csv(IND_FILE, keep_default_na=False)
    ind = lab.set_index("ticker")["industry"]
    sector = lab.set_index("ticker")["sector"]
    px, vol = load_panels()
    d = study(px, vol, ind, sector, *DEV)
    main_res = summarise(d, "top3_12_1")
    ok = bool(main_res.get("months", 0) >= 2 and main_res["excess_ann"] >= MIN_EXCESS
              and main_res["t"] >= MIN_T and main_res["month_hit"] >= MIN_HIT)
    res = {"primary_top3_12_1": main_res}
    for key in ("top1_12_1", "top5_12_1", "top3_6_1", "bottom3_12_1"):
        res[key] = summarise(d, key)
    share = {k: float(d[k].mean()) for k in ("has_defense", "has_utilities", "has_banks", "has_energy")}
    ew = d["EW"]

    def line(name, s):
        if s.get("months", 0) < 2:
            return f"| {name} | {s.get('months', 0)} | - | - | - | - |"
        return (f"| {name} | {s['months']} | {100 * s['port_ann']:+.1f}% | {100 * s['excess_ann']:+.1f}% | "
                f"{s['t']:+.2f} | {100 * s['month_hit']:.0f}% |")
    labelled = (lab["status"] == "ok").sum()
    L = ["# SM1: industry momentum on NSE", "",
         "Implements reports/preregistration_sector_momentum.md. Each month end: industry score = equal-weight "
         "12-1 month return of its eligible members (industries with >= 5 members); hold the eligible stocks of "
         "the top 3 industries equally from the next day to the next month end; benchmark = the equal-weight "
         "eligible universe; 25 bps per unit of turnover.", "",
         f"Industry labels: Yahoo `industry` for {len(lab):,} tickers ({labelled:,} labelled). These are today's "
         "labels. Unlabelled stocks are dropped from the portfolios (they stay in the benchmark).", "",
         "## Development (month ends 2016-01 to 2020-12)", "",
         "| Portfolio | Months | Return/yr (net) | Excess/yr vs EW | Month t | Months beating EW |",
         "|---|---|---|---|---|---|",
         line("**Primary: top 3 industries, 12-1**", main_res),
         line("Top 1 industry, 12-1", res["top1_12_1"]),
         line("Top 5 industries, 12-1", res["top5_12_1"]),
         line("Top 3 industries, 6-1", res["top3_6_1"]),
         line("Bottom 3 industries, 12-1", res["bottom3_12_1"]), "",
         f"Equal-weight eligible universe: {100 * ew.mean() * 12:.1f}% a year over {len(ew)} months; about "
         f"{d['n'].mean():.0f} eligible stocks ({d['n_labelled'].mean():.0f} labelled) and "
         f"{d['n_industries_12_1'].mean():.0f} ranked industries per month.", "",
         "Share of months with these among the top 3 (12-1): "
         f"Aerospace & Defense {100 * share['has_defense']:.0f}%, utilities {100 * share['has_utilities']:.0f}%, "
         f"banks {100 * share['has_banks']:.0f}%, energy {100 * share['has_energy']:.0f}%.", "",
         f"Rule: excess >= +{100 * MIN_EXCESS:.1f}% a year, month t >= {MIN_T:.1f}, beats EW in >= "
         f"{100 * MIN_HIT:.0f}% of months.", "",
         f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the test stays sealed'}.**", ""]
    out = {"development": res, "share_top3": share, "ew_ann": float(ew.mean() * 12), "months": int(len(d)),
           "qualifies": ok, "top3_by_month": {str(k.date()): v for k, v in d["top3_industries"].items()}}
    with open(os.path.join(REPORT_DIR, "sector_momentum.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "sector_momentum.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    lab[["ticker", "industry"]].to_csv(os.path.join(REPORT_DIR, "industry_map.csv"), index=False, encoding="ascii")
    print("\n".join(L))


if __name__ == "__main__":
    main()

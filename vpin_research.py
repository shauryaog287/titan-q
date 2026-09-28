"""
vpin_research.py
================
Implements reports/preregistration_vpin_gate.md: does holding back a buy while the
stock's VPIN toxicity score is above the threshold get a better price (net of the
market's move over the delay) than buying at once?

    D:\\trader_env\\Scripts\\python.exe vpin_research.py               development only
    D:\\trader_env\\Scripts\\python.exe vpin_research.py --reveal-test once, after the decision is committed
    D:\\trader_env\\Scripts\\python.exe vpin_research.py --check       fast path vs ToxicityMonitor.score (no metrics)

The score is computed once per stock-day on the same 5-session window the live feed
holds, then read at each minute; --check confirms that equals calling the live
ToxicityMonitor.score on the bars available at that minute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date, datetime, time as dtime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from titanq_config import CONFIG, PROJECT_DIR, UNIVERSE
from market_data import IST, DailyStore
from intraday_research import load_minutes

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEV = ("2021-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-30")
DECISIONS = [dtime(9, 20), dtime(9, 30)] + [dtime(h, 30) for h in range(10, 15)]
LAST_BUY = dtime(15, 19)
NEXT_SESSION_BUY = dtime(9, 16)
PRIOR_SESSIONS = 4                    # the live feed holds 5 sessions: 4 earlier + today
MIN_IMPROVEMENT_BPS = 3.0
MIN_DAY_T = 2.0
NS = 1_000_000_000
MINUTE = 60 * NS


def _clock(d: date, t: dtime) -> int:
    """Epoch nanoseconds of an IST wall-clock time."""
    return pd.Timestamp(datetime.combine(d, t), tz=IST).value


class PriceSeries:
    """Price at clock time t = close of the last 1-minute bar completed by t."""

    def __init__(self, m: pd.DataFrame) -> None:
        self.start = m["ts"].to_numpy().astype("datetime64[ns]").astype(np.int64) if m["ts"].dt.tz is None \
            else m["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)
        self.close = m["close"].to_numpy(dtype=float)

    def at(self, clock_ns: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.start, np.asarray(clock_ns) - MINUTE, side="right") - 1
        out = np.where(i >= 0, self.close[np.clip(i, 0, None)], np.nan)
        return out


def stock_frames(symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    m = load_minutes(symbol[:-3])
    if m is None:
        return None
    daily = DailyStore().load([symbol], refresh=False).get(symbol)
    if daily is None or not len(daily):
        return None
    vol = daily["Volume"]
    vol.index = pd.to_datetime(vol.index).tz_localize(None) if getattr(vol.index, "tz", None) is not None \
        else pd.to_datetime(vol.index)
    return m, vol


def window_bars(m: pd.DataFrame, sessions: List[date], i: int) -> pd.DataFrame:
    """Bars of sessions[i - PRIOR_SESSIONS .. i], in the live feed's format."""
    lo = sessions[i - PRIOR_SESSIONS]
    w = m[(m["date"] >= lo) & (m["date"] <= sessions[i])]
    return pd.DataFrame({"Close": w["close"].to_numpy(dtype=float), "Volume": w["volume"].to_numpy(dtype=float)},
                        index=pd.DatetimeIndex(w["ts"]))


def adv_before(vol: pd.Series, d: date) -> Optional[float]:
    v = vol[vol.index < pd.Timestamp(d)].iloc[-20:]
    return float(v.median()) if len(v) else None


def day_scores(bars: pd.DataFrame, adv: float, d: date) -> Optional[Tuple[np.ndarray, np.ndarray, object]]:
    """(t_end ns of finite-VPIN buckets, their VPIN, fitted ToxicityScore) or None."""
    from hft_hawkes_toxicity import ToxicityScore, VPINEstimator
    b = bars[bars["Volume"] > 0]
    if len(b) < 50 or not adv or adv <= 0:
        return None
    try:
        est = VPINEstimator(bucket_volume=adv / CONFIG.vpin_buckets_per_day,
                            n_buckets=CONFIG.vpin_window_buckets, classifier="bvc", sigma_mode="rolling")
        res = est.compute(b.index.asi8 / 1e9, b["Close"].to_numpy(), b["Volume"].to_numpy())
    except Exception:
        return None
    vpin = res["vpin"]
    t_end = np.round(res["t_end"]).astype(np.int64) * NS      # bar starts are whole seconds
    fin = np.isfinite(vpin)
    start = _clock(d, dtime(0, 0))
    ref = vpin[(t_end < start) & fin]
    if ref.size < 20:
        return None
    tox = ToxicityScore(combine="geometric", weight_hawkes=0.0, alert_threshold=CONFIG.toxicity_threshold)
    tox.fit_reference(np.zeros(ref.size), ref)
    return t_end[fin], vpin[fin], tox


def score_at(ds: Tuple[np.ndarray, np.ndarray, object], clock_ns: np.ndarray) -> np.ndarray:
    """Score at each clock time from buckets closed by a bar that completed by then."""
    t_end, vpin, tox = ds
    i = np.searchsorted(t_end, np.asarray(clock_ns) - MINUTE, side="right") - 1
    v = np.where(i >= 0, vpin[np.clip(i, 0, None)], np.nan)
    s = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.any():
        s[ok] = tox.score(np.zeros(int(ok.sum())), v[ok])
    return s


def events_for_stock(symbol: str, nifty: PriceSeries, lo: str, hi: str) -> Tuple[List[Dict], List[Dict]]:
    fr = stock_frames(symbol)
    if fr is None:
        return [], []
    m, vol = fr
    m = m.copy()
    m["date"] = m["ts"].dt.date
    px = PriceSeries(m)
    sessions = sorted(m["date"].unique())
    thr = CONFIG.toxicity_threshold
    ev, dec = [], []
    for i in range(PRIOR_SESSIONS, len(sessions)):
        d = sessions[i]
        if not (pd.Timestamp(lo).date() <= d <= pd.Timestamp(hi).date()):
            continue
        ds = day_scores(window_bars(m, sessions, i), adv_before(vol, d), d)
        if ds is None:
            continue
        grid = np.arange(_clock(d, dtime(9, 16)), _clock(d, LAST_BUY) + 1, MINUTE)
        sc = score_at(ds, grid)
        for t in DECISIONS:
            c0 = _clock(d, t)
            s0 = score_at(ds, np.array([c0]))[0]
            p0, p30 = px.at(np.array([c0, c0 + 30 * MINUTE]))
            flagged = bool(np.isfinite(s0) and s0 > thr)
            dec.append({"date": d, "stock": symbol, "flagged": flagged,
                        "absfwd30_bps": abs(math.log(p30 / p0)) * 1e4 if p0 > 0 and p30 > 0 else np.nan})
            if not flagged:
                continue
            later = (grid > c0) & ~(sc > thr)            # unflagged or not computable
            if later.any():
                c1, nxt = int(grid[np.argmax(later)]), False
            elif i + 1 < len(sessions):
                c1, nxt = _clock(sessions[i + 1], NEXT_SESSION_BUY), True
            else:
                continue
            p1 = px.at(np.array([c1]))[0]
            n0, n1 = nifty.at(np.array([c0, c1]))
            if not all(np.isfinite([p0, p1, n0, n1])):
                continue
            raw = math.log(p0 / p1)
            ev.append({"date": d, "stock": symbol, "time": t.strftime("%H:%M"),
                       "imp_bps": (raw - math.log(n0 / n1)) * 1e4, "raw_bps": raw * 1e4,
                       "delay_min": (c1 - c0) / MINUTE, "next_session": nxt})
    return ev, dec


def run(lo: str, hi: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nm = load_minutes("NIFTY50_IDX")
    nifty = PriceSeries(nm)
    ev, dec = [], []
    for k, s in enumerate(UNIVERSE):
        e, d = events_for_stock(s, nifty, lo, hi)
        ev += e
        dec += d
        print(f"  {k + 1}/{len(UNIVERSE)} {s}: {len(d)} decisions, {len(e)} flagged events", flush=True)
    return pd.DataFrame(ev), pd.DataFrame(dec)


def evaluate(ev: pd.DataFrame, dec: pd.DataFrame) -> Dict[str, float]:
    out = {"decisions": int(len(dec)), "flagged": int(dec["flagged"].sum()) if len(dec) else 0}
    out["share_flagged"] = out["flagged"] / out["decisions"] if out["decisions"] else float("nan")
    out["events"] = int(len(ev))
    if not len(ev):
        return {**out, "mean_imp_bps": float("nan"), "day_t": 0.0, "days": 0}
    daily = ev.groupby("date")["imp_bps"].mean()
    sd = daily.std(ddof=1)
    out.update({
        "mean_imp_bps": float(ev["imp_bps"].mean()),
        "days": int(len(daily)),
        "day_t": float(daily.mean() / sd * math.sqrt(len(daily))) if len(daily) > 1 and sd > 0 else 0.0,
        "mean_raw_bps": float(ev["raw_bps"].mean()),
        "median_delay_min": float(ev["delay_min"].median()),
        "share_next_session": float(ev["next_session"].mean()),
        "absfwd30_flagged_bps": float(dec.loc[dec["flagged"], "absfwd30_bps"].mean()),
        "absfwd30_unflagged_bps": float(dec.loc[~dec["flagged"], "absfwd30_bps"].mean()),
    })
    return out


def qualifies(e: Dict) -> bool:
    return e["events"] > 0 and e["mean_imp_bps"] >= MIN_IMPROVEMENT_BPS and e["day_t"] >= MIN_DAY_T


def table(e: Dict) -> List[str]:
    return [
        "| Measure | Value |", "|---|---|",
        f"| Decisions (stock x session x time) | {e['decisions']:,} |",
        f"| Flagged (score > {CONFIG.toxicity_threshold:.2f}) | {e['flagged']:,} ({100 * e['share_flagged']:.1f}%) |",
        f"| Flagged events priced | {e['events']:,} over {e.get('days', 0):,} sessions |",
        f"| Mean improvement, market-adjusted | {e['mean_imp_bps']:+.2f} bps |",
        f"| Day-level t | {e['day_t']:+.2f} |",
        f"| Mean improvement, raw | {e.get('mean_raw_bps', float('nan')):+.2f} bps |",
        f"| Median delay | {e.get('median_delay_min', float('nan')):.0f} min |",
        f"| Pushed to next session | {100 * e.get('share_next_session', float('nan')):.1f}% |",
        f"| Mean abs 30-min move after flagged / unflagged | {e.get('absfwd30_flagged_bps', float('nan')):.1f} / "
        f"{e.get('absfwd30_unflagged_bps', float('nan')):.1f} bps |",
    ]


def check(n: int = 12) -> None:
    """Fast path vs titanq_brain.ToxicityMonitor.score on sampled stock-days. No metrics."""
    from titanq_brain import ToxicityMonitor
    rng = np.random.default_rng(0)
    worst, done, finite = 0.0, 0, 0
    for s in rng.choice(list(UNIVERSE), size=n, replace=False):
        s = str(s)
        fr = stock_frames(s)
        if fr is None:
            continue
        m, vol = fr
        m = m.copy()
        m["date"] = m["ts"].dt.date
        sessions = sorted(m["date"].unique())
        if len(sessions) <= PRIOR_SESSIONS:
            continue
        i = int(rng.integers(PRIOR_SESSIONS, len(sessions)))
        d = sessions[i]
        bars, adv = window_bars(m, sessions, i), adv_before(vol, d)
        ds = day_scores(bars, adv, d)
        if ds is None:
            continue
        for t in DECISIONS:
            c = _clock(d, t)
            fast = score_at(ds, np.array([c]))[0]
            known = bars[bars.index.asi8 <= c - MINUTE]
            live, _ = ToxicityMonitor(CONFIG).score(s, known, adv, d)
            live = np.nan if live is None else live
            if np.isnan(fast) != np.isnan(live):
                raise SystemExit(f"MISMATCH {s} {d} {t}: fast {fast} live {live}")
            if np.isfinite(fast):
                worst = max(worst, abs(fast - live))
                finite += 1
            done += 1
    print(f"check: {done} stock-day-times compared ({finite} with a score), max |fast - live| = {worst:.2e}")
    if worst > 1e-9 or finite < done // 2:
        raise SystemExit("fast path differs from the live score; do not use it")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal-test", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check()
    if a.check:
        return
    ev, dec = run(*DEV)
    dev = evaluate(ev, dec)
    ok = qualifies(dev)
    L = ["# VPIN buy gate: does holding back toxic buys get better prices?", "",
         "Implements reports/preregistration_vpin_gate.md. Angel 1-minute history, the 50 live stocks, "
         f"decisions at {', '.join(t.strftime('%H:%M') for t in DECISIONS)}; a flagged buy waits for the first "
         "unflagged minute (by 15:19, else 09:16 next session).", "",
         "## Development (sessions 2021-01 to 2023-12)", ""] + table(dev)
    L += ["", f"Rule: mean improvement >= +{MIN_IMPROVEMENT_BPS:.0f} bps and day-level t >= {MIN_DAY_T:.1f}.", "",
          f"**Decision: {'qualifies; reveal the test once' if ok else 'does not qualify; the gate stays in observe mode and the test stays sealed'}.**", ""]
    out = {"development": dev, "qualifies": ok, "test_revealed": bool(a.reveal_test and ok)}
    if a.reveal_test and ok:
        tev, tdec = run(*TEST)
        test = evaluate(tev, tdec)
        passed = test["events"] > 0 and test["mean_imp_bps"] > 0
        L += ["## Test (revealed once)", ""] + table(test)
        L += ["", f"**Test: {'passed; switch vpin_gate to block' if passed else 'failed; the gate stays in observe mode'}.**", ""]
        out["test"], out["test_passed"] = test, passed
    with open(os.path.join(REPORT_DIR, "vpin_gate.md"), "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "vpin_gate.json"), "w", encoding="ascii") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

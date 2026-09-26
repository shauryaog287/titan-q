"""
train_models.py
===============
Trains and evaluates Titan-Q's learned components on real NSE daily data and
writes them, with a manifest, to models/. Run it now and then about weekly:

    D:\\trader_env\\Scripts\\python.exe D:\\Quant_Project\\train_models.py

    models/regime_hmm.pkl     Phase 11  Gaussian HMM on NIFTY daily features
    models/neural_sde.pt      Phase 21  neural SDE on NIFTY daily log price
    models/ppo_obs_rms.npz    Phase 5   PPO policy: rebuilt observation normalisation
    models/manifest.json                spans, metrics, decisions, file hashes

Training data ends the day before --asof (default: today), so a model used to
trade day D has never seen day D.

THE PPO POLICY
master_ai_brain_sortino.zip was trained by local_ai_trader.py on NIFTY daily
closes from 2015 (gym-anytrading, window 30, actions Sell=0 / Buy=1) behind a
VecNormalize wrapper whose statistics were never saved. Without them the
policy sees un-normalised prices and its actions are meaningless. The
statistics are rebuilt here by replaying the training frames through the
policy (price features are exact; the five reward-state features are
approximate). The policy is then scored OUT OF SAMPLE (2021 onward, data it
never trained on) as a long/flat timing rule against buy-and-hold. The
manifest records whether it won; the live system uses it only if it did,
unless titanq_config.ppo_timing forces it.

MODELS THAT CANNOT BE TRAINED HERE
The Phase 18 transformer and Phase 16 spoofing autoencoder need order-book
depth; the Phase 14 Hawkes filter needs trade-by-trade timestamps. Yahoo
provides neither. They are recorded in the manifest as unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import time
import warnings
from datetime import date, datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from titanq_config import CONFIG, INDEX_SYMBOL, MANIFEST_FILE, MODELS_DIR, PROJECT_DIR
from market_data import DailyStore, close_panel

log = logging.getLogger("train_models")

PPO_ZIP = os.path.join(PROJECT_DIR, "master_ai_brain_sortino.zip")
HMM_FILE = os.path.join(MODELS_DIR, "regime_hmm.pkl")
SDE_FILE = os.path.join(MODELS_DIR, "neural_sde.pt")
PPO_RMS_FILE = os.path.join(MODELS_DIR, "ppo_obs_rms.npz")

TRADING_DAYS = 252
LONG_FLAT_SWITCH_COST = 0.0012     # one-way cost of moving fully in or out (STT etc.)

# The exact environment settings local_ai_trader.py trained with.
PPO_ENV_KW = dict(window_size=30, reward_mode="differential", eta=0.02, trade_fee=0.002,
                  reward_scale=0.05, lambda_dd=1.0, l_floor=2.5e-5, augment_observation=True)
PPO_TRAIN_FRAME = (30, 1500)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perf(returns: np.ndarray) -> Dict[str, float]:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return {}
    eq = np.cumsum(r)
    dd = np.maximum.accumulate(eq) - eq
    down = r[r < 0]
    dd_sd = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if r.size else float("nan")
    return {
        "days": int(r.size),
        "ann_return": float(r.mean() * TRADING_DAYS),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0,
        "sortino": float(r.mean() / dd_sd * math.sqrt(TRADING_DAYS)) if dd_sd > 0 else 0.0,
        "total_log_return": float(eq[-1]),
        "max_drawdown_log": float(dd.max()),
        "down_days": int(down.size),
    }


# ===========================================================================
# PHASE 11: REGIME HMM
# ===========================================================================

def train_regime(index_df: pd.DataFrame) -> Dict:
    from regime_hmm_detector import (FeaturePipeline, MarketRegimeFilter, RegimeModel,
                                     select_n_states)

    # Features: daily return and log realised variance. NIFTY volume is left
    # out: with it, EM landed in different optima from different starts (a
    # log-likelihood spread of ~1,000 across restarts, measured 2026-09-26)
    # and produced a one-day "regime"; without it every restart agrees.
    prices = index_df["Close"].to_numpy()
    use_volume = False
    pipe = FeaturePipeline(rv_window=21, volume_window=63)
    F, rows = pipe.build(prices, None)
    F = F[:, :2]                          # the volume column is all zeros without volume
    X = pipe.fit_transform(F)
    k, scores = select_n_states(X, candidates=(2, 3, 4), n_restarts=8, min_dwell=5.0)
    # refit, and fall back to the next-best admissible K if the refit breaks the
    # dwell guard (a regime shorter than a week is not a regime)
    order = sorted((kk for kk, v in scores.items() if v["admissible"]),
                   key=lambda kk: scores[kk]["bic"]) or [k]
    for k in order:
        model = RegimeModel(n_states=k)
        fit = model.fit(X, n_restarts=12, pipeline=pipe)
        if model.analytics().expected_duration().min() >= 5.0:
            break

    filt = MarketRegimeFilter(model)
    for x in X:
        st = filt.update(x, store=False)
    an = model.analytics()
    dis = model.filtered_vs_smoothed_disagreement(X)

    with open(HMM_FILE, "wb") as f:
        pickle.dump({"model": model, "pipeline": pipe, "use_volume": bool(use_volume),
                     "n_features": int(X.shape[1])}, f)
    info = {
        "file": os.path.basename(HMM_FILE),
        "n_states": int(k),
        "labels": model.label_names(),
        "use_volume": bool(use_volume),
        "features": ["return", "log_rv"] + (["rel_log_volume"] if use_volume else []),
        "train_start": str(index_df.index[rows[0]].date()),
        "train_end": str(index_df.index[-1].date()),
        "n_obs": int(X.shape[0]),
        "log_likelihood": float(fit.log_likelihood),
        "bic": float(fit.bic),
        "restart_spread": float(fit.restart_spread),
        "expected_dwell_days": [round(float(v), 1) for v in an.expected_duration()],
        "stationary_distribution": [round(float(v), 3) for v in an.stationary()],
        "filtered_vs_viterbi_disagreement": round(dis["filtered_vs_viterbi"], 4),
        "bic_by_k": {str(kk): round(v["bic"], 1) for kk, v in scores.items()},
        "regime_at_train_end": st.regime.name,
        "regime_confidence": round(st.confidence, 3),
        "state_ann_return": [round(float(252 * m[0]), 4) for m in model.raw_means()],
        "state_ann_vol": [round(float(math.sqrt(math.exp(m[1]))), 4) for m in model.raw_means()],
    }
    log.info("HMM: K=%d labels=%s, regime at %s: %s (p=%.2f)", k, info["labels"],
             info["train_end"], st.regime.name, st.confidence)
    return info


# ===========================================================================
# PHASE 21: NEURAL SDE
# ===========================================================================

def _gbm_nll(x_train: np.ndarray, x_test: np.ndarray, dt: float) -> float:
    d = np.diff(x_train)
    mu, var = d.mean() / dt, d.var() / dt
    r = np.diff(x_test) - mu * dt
    v = var * dt
    return float(np.mean(0.5 * np.log(2 * math.pi * v) + r * r / (2 * v)))


def train_neural_sde(index_df: pd.DataFrame, years: int = 10) -> Dict:
    from neural_sde_discovery import fit_transition_nll, transition_nll

    close = index_df["Close"].iloc[-years * TRADING_DAYS:]
    x = np.log(close.to_numpy())
    dt = 1.0 / TRADING_DAYS

    # honest check first: does the network beat a constant-coefficient GBM on
    # the last 20% of the data, having been fitted on the first 80% only?
    cut = int(0.8 * x.size)
    probe = fit_transition_nll(x[:cut], dt, steps=2500, seed=0)
    xt = torch.from_numpy(x[cut:])
    with torch.no_grad():
        nll_net = float(transition_nll(probe, xt[:-1], xt[1:], dt))
    nll_gbm = _gbm_nll(x[:cut], x[cut:], dt)

    model = fit_transition_nll(x, dt, steps=2500, seed=0)
    q01, q99 = np.quantile(x, [0.01, 0.99])
    x_now = float(np.clip(x[-1], q01, q99))
    with torch.no_grad():
        mu_now = float(model.mu(torch.tensor([x_now])))
        sg_now = float(model.sigma(torch.tensor([x_now])))
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": 32, "depth": 2, "time_dependent": False,
        "x_loc": model.x_loc, "x_scale": model.x_scale, "t_scale": model.t_scale,
        "dt_years": dt, "support_q01": float(q01), "support_q99": float(q99),
        "span_years": float(x.size * dt),
    }, SDE_FILE)
    d = np.diff(x)
    info = {
        "file": os.path.basename(SDE_FILE),
        "state": "log NIFTY close, time unit = years (outputs are annualised)",
        "train_start": str(close.index[0].date()), "train_end": str(close.index[-1].date()),
        "span_years": round(float(x.size * dt), 2),
        "holdout_nll_neural": round(nll_net, 5), "holdout_nll_gbm": round(nll_gbm, 5),
        "neural_beats_gbm_out_of_sample": bool(nll_net < nll_gbm),
        "enabled": bool(nll_net < nll_gbm),
        "rule": "use the network's drift/diffusion only if it beats a constant-coefficient "
                "GBM on held-out data; otherwise use the GBM (sample) estimates",
        "drift_now_annual": round(mu_now, 4), "diffusion_now_annual": round(sg_now, 4),
        "sample_drift_annual": round(float(d.mean() / dt), 4),
        "sample_vol_annual": round(float(d.std() / math.sqrt(dt)), 4),
        "state_clamped_to_support": bool(x[-1] > q99 or x[-1] < q01),
    }
    log.info("Neural SDE: drift %.1f%%, diffusion %.1f%% at the current state "
             "(sample %.1f%% / %.1f%%); holdout NLL neural %.5f vs GBM %.5f",
             100 * mu_now, 100 * sg_now, 100 * info["sample_drift_annual"],
             100 * info["sample_vol_annual"], nll_net, nll_gbm)
    return info


# ===========================================================================
# PHASE 5: PPO POLICY
# ===========================================================================

def ppo_frame(index_df: pd.DataFrame) -> pd.DataFrame:
    """The same preprocessing as local_ai_trader.py: indicators appended, then
    dropna, which removes the first 33 rows. Only Close reaches the policy, but
    the dropna fixes where each training frame starts."""
    import pandas_ta as ta  # noqa: F401  (registers the .ta accessor)

    df = index_df.loc["2015-01-01":, ["Open", "High", "Low", "Close", "Volume"]].copy()
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.atr(append=True)
    return df.dropna()


def make_ppo_env(df: pd.DataFrame, frame: Tuple[int, int]):
    from sortino_reward_env import SortinoStocksEnv
    return SortinoStocksEnv(df=df, frame_bound=frame, **PPO_ENV_KW)


def normalise_obs(obs: np.ndarray, rms: Dict[str, np.ndarray]) -> np.ndarray:
    z = (obs - rms["mean"]) / np.sqrt(rms["var"] + float(rms["epsilon"]))
    return np.clip(z, -float(rms["clip_obs"]), float(rms["clip_obs"])).astype(np.float32)


def run_policy(model, env, rms, deterministic: bool, seed: int = 0):
    """One episode. Returns (raw observations, position after each decision)."""
    obs, _ = env.reset(seed=seed)
    raws, positions = [obs], []
    while True:
        a, _ = model.predict(normalise_obs(obs, rms)[None, :], deterministic=deterministic)
        obs, _r, _term, trunc, _info = env.step(int(np.asarray(a).ravel()[0]))
        raws.append(obs)
        positions.append(env._position.value)          # 0 = Short, 1 = Long
        if trunc:
            return np.asarray(raws), np.asarray(positions)


def rebuild_obs_rms(model, df: pd.DataFrame, passes: int = 3, episodes: int = 6) -> Dict:
    """
    VecNormalize kept a running mean/variance of every observation seen in
    training. Rebuild it by replaying the training frames: first pass with
    placeholder statistics, later passes with the statistics of the previous
    pass, sampling actions stochastically as training did.
    """
    env = make_ppo_env(df, PPO_TRAIN_FRAME)
    dim = env.observation_space.shape[0]
    rms = {"mean": np.zeros(dim), "var": np.ones(dim), "epsilon": 1e-8, "clip_obs": 10.0}
    first = run_policy(model, env, rms, deterministic=False, seed=0)[0]
    # price and diff features do not depend on the policy: exact from the first pass
    rms = {"mean": first.mean(0), "var": first.var(0) + 1e-12, "epsilon": 1e-8, "clip_obs": 10.0}
    for p in range(passes):
        obs = np.concatenate([run_policy(model, env, rms, deterministic=False, seed=100 * p + e)[0]
                              for e in range(episodes)])
        rms = {"mean": obs.mean(0), "var": obs.var(0) + 1e-12, "epsilon": 1e-8, "clip_obs": 10.0}
    rms["count"] = float(len(obs))
    return rms


def evaluate_ppo(model, df: pd.DataFrame, rms: Dict) -> Dict:
    """Out of sample: frames after the training range. The decision taken at
    the close of day t earns the index return from t to t+1."""
    start = PPO_TRAIN_FRAME[1]
    env = make_ppo_env(df, (start, len(df)))
    _, pos = run_policy(model, env, rms, deterministic=True)
    close = df["Close"].to_numpy()
    # gym-anytrading timing: the action chosen from the observation ending at df
    # row start+k takes effect at row start+k+1 (pos[k]), so it earns the return
    # from row start+k+1 to start+k+2. That is r[k+1] below. Crediting it with
    # r[k] would let each decision see one day into its own future.
    r_all = np.diff(np.log(close[start:]))          # r_all[j]: row start+j -> start+j+1
    n = min(pos.size, r_all.size - 1)
    long = (pos[:n] == 1).astype(float)
    r = r_all[1: n + 1]
    switch = np.abs(np.diff(np.concatenate([[long[0]], long])))
    strat_long_flat = long * r - switch * LONG_FLAT_SWITCH_COST
    sign = np.where(long > 0, 1.0, -1.0)
    flips = np.abs(np.diff(np.concatenate([[1.0], sign]))) / 2
    strat_long_short = sign * r - flips * 2 * LONG_FLAT_SWITCH_COST
    bh = perf(r)
    lf = perf(strat_long_flat)
    ls = perf(strat_long_short)
    decision = bool(lf.get("sortino", 0) > bh.get("sortino", 0))
    return {
        "oos_start": str(df.index[start].date()), "oos_end": str(df.index[-1].date()),
        "pct_days_long": round(float(long.mean()), 3),
        "switches": int(switch.sum()),
        "buy_and_hold": bh, "long_flat": lf, "long_short": ls,
        "beats_buy_and_hold": decision,
        "rule": "enable only if long/flat Sortino > buy-and-hold Sortino, out of sample",
    }


def train_ppo(index_df: pd.DataFrame) -> Dict:
    from stable_baselines3 import PPO

    df = ppo_frame(index_df)
    model = PPO.load(PPO_ZIP, device="cpu")
    shape = tuple(model.observation_space.shape)
    if shape != (65,) or int(model.action_space.n) != 2:
        raise RuntimeError(f"unexpected PPO spaces {shape} / {model.action_space}")
    rms = rebuild_obs_rms(model, df)
    np.savez(PPO_RMS_FILE, **{k: np.asarray(v) for k, v in rms.items()})
    ev = evaluate_ppo(model, df, rms)
    info = {
        "policy_file": os.path.basename(PPO_ZIP), "policy_sha256": sha256(PPO_ZIP),
        "obs_rms_file": os.path.basename(PPO_RMS_FILE),
        "trained_timesteps": int(getattr(model, "num_timesteps", 0)),
        "actions": {"0": "Sell (risk off)", "1": "Buy (risk on)"},
        "train_frame_dates": [str(df.index[PPO_TRAIN_FRAME[0]].date()),
                              str(df.index[PPO_TRAIN_FRAME[1] - 1].date())],
        "obs_rms_note": "rebuilt by replaying the training frames; price features exact, "
                        "reward-state features approximate",
        "evaluation": ev,
        "enabled": ev["beats_buy_and_hold"],
    }
    log.info("PPO out of sample %s..%s: long/flat Sortino %.2f vs buy-and-hold %.2f "
             "(long %.0f%% of days, %d switches) -> %s",
             ev["oos_start"], ev["oos_end"], ev["long_flat"]["sortino"],
             ev["buy_and_hold"]["sortino"], 100 * ev["pct_days_long"], ev["switches"],
             "ENABLED" if info["enabled"] else "DISABLED")
    return info


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--asof", type=str, default=None,
                    help="train on data strictly before this date (YYYY-MM-DD); default today")
    args = ap.parse_args()
    asof = date.fromisoformat(args.asof) if args.asof else date.today()

    os.makedirs(MODELS_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    daily = DailyStore().load([INDEX_SYMBOL])
    idx = daily[INDEX_SYMBOL]
    idx = idx[idx.index < pd.Timestamp(asof)]
    log.info("NIFTY daily %s .. %s (%d rows), training as of %s",
             idx.index[0].date(), idx.index[-1].date(), len(idx), asof)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "asof": str(asof),
        "data_end": str(idx.index[-1].date()),
        "components": {},
    }
    for name, fn in (("regime_hmm", train_regime), ("neural_sde", train_neural_sde),
                     ("ppo_policy", train_ppo)):
        t1 = time.time()
        try:
            manifest["components"][name] = fn(idx)
            manifest["components"][name]["status"] = "ok"
        except Exception as exc:
            log.exception("training %s failed", name)
            manifest["components"][name] = {"status": "failed", "error": repr(exc)}
        manifest["components"][name]["seconds"] = round(time.time() - t1, 1)

    unavailable = "needs order-book depth; Yahoo Finance provides bars only"
    manifest["components"]["tft_lob_forecaster"] = {"status": "unavailable", "reason": unavailable}
    manifest["components"]["spoof_autoencoder"] = {"status": "unavailable", "reason": unavailable}
    manifest["components"]["hawkes_filter"] = {
        "status": "unavailable",
        "reason": "needs trade-by-trade timestamps; Yahoo Finance provides 1-minute bars only"}

    for comp in manifest["components"].values():
        for key in ("file", "obs_rms_file"):
            if comp.get(key):
                comp[key + "_sha256"] = sha256(os.path.join(MODELS_DIR, comp[key]))
    with open(MANIFEST_FILE, "w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2)
    log.info("manifest written to %s (%.0fs total)", MANIFEST_FILE, time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

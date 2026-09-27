"""
titanq_brain.py
===============
The decision layer. It turns market data and trained models into a target
portfolio, and the target portfolio into orders.

    build_daily_context(daily, asof, equity)
        Once per session, from daily bars dated BEFORE the session (no
        look-ahead). All the slow work happens here, off the event loop.

          Phase 11  regime        causal HMM filter -> exposure multiplier
          Phase 5   PPO timing    only if it beat buy-and-hold out of sample
          Phase 21  market drift  neural SDE only if it beat a GBM out of
                                  sample, else the GBM (sample) estimates
          alpha                   12-1 momentum score -> expected returns
                                  (a documented baseline, not a claimed edge)
          covariance              Ledoit-Wolf shrinkage on one year of returns
          Phase 20  selection     best-Sharpe K of N, exact enumeration when
                                  feasible, swap annealing otherwise
          Phase 12  weights       Hierarchical Risk Parity on the selection
          Phase 4   sizing        Kelly with posterior shrinkage and costs, on
                                  the actual basket
          Phase 23  tail risk     POT/GPD expected shortfall of the basket vs
                                  the index -> trim
          Phase 24  diagnostics   transfer entropy index <-> each holding,
                                  with surrogate significance

    ToxicityMonitor
        Intraday VPIN (Phase 14, bulk volume classification on 1-minute bars),
        scored as a percentile of each stock's own recent days. Buys into toxic
        flow are held back; sells are not.

    Rebalancer
        Target weights x equity -> target shares -> the orders that close the
        gap, with a no-trade band, minimum trade size, cooldown, cash check and
        no new risk in the last minutes of the session.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from titanq_config import (CONFIG, Config, INDEX_SYMBOL, LOGS_DIR, MANIFEST_FILE, MODELS_DIR,
                           PROJECT_DIR, UNIVERSE)
from market_data import IST, Session, close_panel

log = logging.getLogger("brain")

TRADING_DAYS = 252


# ===========================================================================
# 1. DAILY CONTEXT
# ===========================================================================

@dataclass
class DailyContext:
    asof: str
    data_end: str
    built_at: str
    tradeable: bool = True
    reasons: List[str] = field(default_factory=list)

    regime: str = "UNKNOWN"
    regime_confidence: float = 0.0
    regime_novel: bool = False
    regime_multiplier: float = 0.5

    ppo_status: str = "disabled"
    ppo_action: Optional[str] = None
    ppo_multiplier: float = 1.0

    market_mu: float = 0.0
    market_sigma: float = 0.0
    market_source: str = "gbm"
    market_span_years: float = 0.0

    selection: List[str] = field(default_factory=list)
    selection_method: str = ""
    selection_sharpe: float = 0.0
    weighting: str = ""
    basket_weights: Dict[str, float] = field(default_factory=dict)
    basket_beta: float = 1.0
    basket_mu: float = 0.0
    basket_sigma: float = 0.0
    alpha_z: Dict[str, float] = field(default_factory=dict)

    kelly_f: float = 0.0
    kelly_binding: str = ""
    kelly_detail: Dict[str, float] = field(default_factory=dict)

    basket_es: Optional[float] = None
    index_es: Optional[float] = None
    evt_multiplier: float = 1.0

    exposure: float = 0.0
    target_weights: Dict[str, float] = field(default_factory=dict)
    adv_shares: Dict[str, float] = field(default_factory=dict)

    transfer_entropy: List[Dict[str, float]] = field(default_factory=list)
    components: Dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        top = sorted(self.target_weights.items(), key=lambda kv: -kv[1])[:5]
        sel = ", ".join(f"{s} {100 * w:.1f}%" for s, w in top)
        more = len(self.target_weights) - len(top)
        return (f"regime {self.regime} (x{self.regime_multiplier:.2f}), PPO x{self.ppo_multiplier:.2f}"
                f", market mu {100 * self.market_mu:.1f}% sigma {100 * self.market_sigma:.1f}% "
                f"[{self.market_source}], basket sigma {100 * self.basket_sigma:.1f}%, Kelly f "
                f"{self.kelly_f:.3f}, EVT x{self.evt_multiplier:.2f} -> exposure "
                f"{100 * self.exposure:.1f}% in {len(self.target_weights)} stocks ({self.weighting}): "
                f"{sel}{f' + {more} more' if more > 0 else ''}")


def load_manifest() -> Dict:
    if not os.path.exists(MANIFEST_FILE):
        return {}
    with open(MANIFEST_FILE, "r", encoding="ascii") as f:
        return json.load(f)


def _component(manifest: Dict, name: str) -> Dict:
    return manifest.get("components", {}).get(name, {})


# ---- Phase 11 ------------------------------------------------------------

def regime_state(index_df: pd.DataFrame, cfg: Config) -> Tuple[str, float, bool]:
    from regime_hmm_detector import MarketRegimeFilter

    with open(os.path.join(MODELS_DIR, "regime_hmm.pkl"), "rb") as f:
        obj = pickle.load(f)
    model, pipe, d = obj["model"], obj["pipeline"], int(obj["n_features"])
    vol = index_df["Volume"].to_numpy() if obj.get("use_volume") else None
    F, _rows = pipe.build(index_df["Close"].to_numpy(), vol)
    X = pipe.transform(F[:, :d])
    filt = MarketRegimeFilter(model)
    st = None
    for x in X:                       # forward filter only: causal, no smoothing
        st = filt.update(x, store=False)
    return st.regime.name, float(st.confidence), bool(st.novel)


# ---- Phase 5 -------------------------------------------------------------

def ppo_action(index_df: pd.DataFrame) -> int:
    from stable_baselines3 import PPO
    from train_models import PPO_ZIP, PPO_RMS_FILE, make_ppo_env, normalise_obs, ppo_frame

    model = PPO.load(PPO_ZIP, device="cpu")
    rms = dict(np.load(PPO_RMS_FILE))
    df = ppo_frame(index_df)
    tail = df.iloc[-1500:]                      # one training-length episode
    env = make_ppo_env(tail, (30, len(tail)))
    obs, _ = env.reset(seed=0)
    while True:
        a, _ = model.predict(normalise_obs(obs, rms)[None, :], deterministic=True)
        obs, _r, _t, trunc, _i = env.step(int(np.asarray(a).ravel()[0]))
        if trunc:
            break
    a, _ = model.predict(normalise_obs(obs, rms)[None, :], deterministic=True)
    return int(np.asarray(a).ravel()[0])


# ---- Phase 21 ------------------------------------------------------------

def market_drift_diffusion(index_close: pd.Series, manifest: Dict,
                           years: int = 10) -> Tuple[float, float, str, float]:
    """(mu, sigma) annualised, source label, span in years."""
    x = np.log(index_close.iloc[-years * TRADING_DAYS:].to_numpy())
    span = x.size / TRADING_DAYS
    comp = _component(manifest, "neural_sde")
    if comp.get("status") == "ok" and comp.get("enabled"):
        import torch
        from neural_sde_discovery import NeuralSDE

        ck = torch.load(os.path.join(MODELS_DIR, "neural_sde.pt"), weights_only=False)
        m = NeuralSDE(hidden=ck["hidden"], depth=ck["depth"], time_dependent=False,
                      x_loc=ck["x_loc"], x_scale=ck["x_scale"], t_scale=ck["t_scale"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        xs = float(np.clip(x[-1], ck["support_q01"], ck["support_q99"]))
        with torch.no_grad():
            return (float(m.mu(torch.tensor([xs]))), float(m.sigma(torch.tensor([xs]))),
                    "neural_sde", span)
    d = np.diff(x)
    return float(d.mean() * TRADING_DAYS), float(d.std(ddof=1) * math.sqrt(TRADING_DAYS)), "gbm", span


# ---- alpha and risk model ----------------------------------------------------

def momentum_alpha(closes: pd.DataFrame, vols: pd.Series, cfg: Config) -> Tuple[pd.Series, pd.Series]:
    """
    12-1 momentum, cross-sectionally z-scored and winsorised at 3, turned into an
    expected annual excess return by Grinold's rule alpha = IC x sigma x z
    (IC per month, so x sqrt(12) to annualise).
    """
    lb, skip = cfg.momentum_lookback_days, cfg.momentum_skip_days
    mom = np.log(closes.iloc[-1 - skip] / closes.iloc[-1 - lb])
    z = ((mom - mom.mean()) / mom.std(ddof=1)).clip(-3, 3)
    alpha = cfg.alpha_ic * vols * z * math.sqrt(12)
    return alpha, z


def shrunk_covariance(rets: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(rets.to_numpy()).covariance_ * TRADING_DAYS


def select_assets(mu: np.ndarray, Sigma: np.ndarray, K: int) -> Tuple[np.ndarray, float, str]:
    from quantum_portfolio_annealer import anneal_swap, brute_force

    N = mu.size
    if math.comb(N, K) <= 2_000_000:
        x, s, _n, _t = brute_force(mu, Sigma, K)
        return np.flatnonzero(x > 0.5), float(s), f"exact enumeration of C({N},{K})"
    best_x, best_s = None, -np.inf
    for seed in range(8):
        x, s = anneal_swap(mu, Sigma, K, n_steps=40_000, seed=seed)
        if s > best_s:
            best_x, best_s = x, s
    return np.flatnonzero(best_x > 0.5), float(best_s), "swap annealing, 8 restarts"


# ---- Phase 23 ----------------------------------------------------------------

def expected_shortfall(losses: np.ndarray, level: float) -> Optional[float]:
    import warnings
    from extreme_value_theory_risk import POTModel
    try:
        with warnings.catch_warnings():
            # scipy's finite-difference Hessian probes outside the GPD support
            # and warns; the fit itself is unaffected
            warnings.simplefilter("ignore", RuntimeWarning)
            es = POTModel(np.asarray(losses, dtype=float), threshold_q=0.95).es(level)
        return float(es) if np.isfinite(es) else None
    except Exception as exc:
        log.warning("EVT fit failed: %s", exc)
        return None


# ---- Phase 24 ----------------------------------------------------------------

def te_diagnostics(idx_ret: np.ndarray, rets: pd.DataFrame, symbols: Sequence[str],
                   cfg: Config, n: int = 750) -> List[Dict[str, float]]:
    from transfer_entropy_causality import te_test
    out = []
    x = idx_ret[-n:]
    for s in symbols:
        y = rets[s].to_numpy()[-n:]
        f = te_test(x, y, bins=cfg.te_bins, n_surr=cfg.te_surrogates, seed=1)
        b = te_test(y, x, bins=cfg.te_bins, n_surr=cfg.te_surrogates, seed=2)
        out.append({"symbol": s,
                    "index_to_stock_bits": round(f.effective, 5), "index_to_stock_p": round(f.p_value, 3),
                    "stock_to_index_bits": round(b.effective, 5), "stock_to_index_p": round(b.p_value, 3)})
    return out


# ---- assembly ----------------------------------------------------------------

def build_daily_context(daily: Dict[str, pd.DataFrame], asof: date, equity: float,
                        cfg: Config = CONFIG, manifest: Optional[Dict] = None,
                        with_te: bool = True, regime_fn=None,
                        holdings: Optional[Sequence[str]] = None) -> DailyContext:
    """
    Everything here uses daily bars dated strictly before `asof`.

    `holdings` are the symbols currently held; if they form a full selection
    they are kept unless the new best subset is clearly better (see
    Config.selection_switch_margin).

    `regime_fn(index_df) -> (name, confidence, novel)` replaces the saved
    regime model; the walk-forward backtest passes one backed by a model
    trained only on data before each test year.
    """
    t0 = time.time()
    manifest = load_manifest() if manifest is None else manifest
    idx = daily[INDEX_SYMBOL]
    idx = idx[idx.index < pd.Timestamp(asof)]
    ctx = DailyContext(asof=str(asof), data_end=str(idx.index[-1].date()),
                       built_at=datetime.now(IST).isoformat(timespec="seconds"))
    for name in ("tft_lob_forecaster", "spoof_autoencoder", "hawkes_filter"):
        comp = _component(manifest, name)
        ctx.components[name] = "unavailable: " + comp.get("reason", "not in manifest")

    if not manifest:
        ctx.tradeable = False
        ctx.reasons.append("models/manifest.json missing: run train_models.py")

    # Phase 11
    try:
        ctx.regime, ctx.regime_confidence, ctx.regime_novel = (
            regime_fn(idx) if regime_fn is not None else regime_state(idx, cfg))
        mult = cfg.regime_multipliers.get(ctx.regime, cfg.regime_multipliers["UNKNOWN"])
        if ctx.regime_novel:
            mult = min(mult, cfg.novel_regime_multiplier)
        ctx.regime_multiplier = mult
        ctx.components["regime_hmm"] = "ok"
    except Exception as exc:
        ctx.tradeable = False
        ctx.reasons.append(f"regime model unavailable ({exc}); refusing to trade")
        ctx.components["regime_hmm"] = f"error: {exc}"

    # Phase 5
    ppo = _component(manifest, "ppo_policy")
    use_ppo = cfg.ppo_timing == "on" or (cfg.ppo_timing == "auto" and ppo.get("enabled"))
    if use_ppo:
        try:
            a = ppo_action(idx)
            ctx.ppo_action = "BUY" if a == 1 else "SELL"
            ctx.ppo_multiplier = 1.0 if a == 1 else cfg.ppo_risk_off_multiplier
            ctx.ppo_status = "on"
        except Exception as exc:
            ctx.ppo_status = f"error: {exc}"
    else:
        ev = ppo.get("evaluation", {})
        ctx.ppo_status = ("disabled: out-of-sample long/flat Sortino %.2f vs buy-and-hold %.2f"
                          % (ev.get("long_flat", {}).get("sortino", float("nan")),
                             ev.get("buy_and_hold", {}).get("sortino", float("nan")))
                          if ev else "disabled")
    ctx.components["ppo_policy"] = ctx.ppo_status

    # Phase 21
    ctx.market_mu, ctx.market_sigma, ctx.market_source, ctx.market_span_years = \
        market_drift_diffusion(idx["Close"], manifest)
    ctx.components["neural_sde"] = ("ok (drives market drift)" if ctx.market_source == "neural_sde"
                                    else "disabled: lost to a constant-coefficient GBM out of sample; "
                                         "GBM estimates used")

    # risk model and alpha
    closes = close_panel(daily, list(UNIVERSE) + [INDEX_SYMBOL], before=asof)
    need = max(cfg.momentum_lookback_days, cfg.cov_lookback_days) + 2
    window = closes.iloc[-need:]
    valid = [s for s in UNIVERSE if s in window and window[s].notna().mean() > 0.97]
    dropped = sorted(set(UNIVERSE) - set(valid))
    if dropped:
        ctx.reasons.append(f"excluded for missing history: {', '.join(dropped)}")
    logret = np.log(closes[valid + [INDEX_SYMBOL]]).diff()
    cov_win = logret.iloc[-cfg.cov_lookback_days:].fillna(0.0)
    Sigma_all = shrunk_covariance(cov_win[valid])
    vols = pd.Series(np.sqrt(np.diag(Sigma_all)), index=valid)
    idx_r = cov_win[INDEX_SYMBOL].to_numpy()
    betas = pd.Series([np.cov(cov_win[s], idx_r)[0, 1] / idx_r.var(ddof=1) for s in valid], index=valid)
    alpha, z = momentum_alpha(window[valid].ffill(), vols, cfg)
    mu = betas * ctx.market_mu + alpha

    # Phase 20: which stocks
    from quantum_portfolio_annealer import sharpe as subset_sharpe
    if cfg.selection_mode == "optimize":
        sel_idx, ctx.selection_sharpe, ctx.selection_method = select_assets(
            mu.to_numpy(), Sigma_all, cfg.n_select)
        # incumbent hysteresis: re-optimising from scratch every day flips the
        # selection on noise; only switch when the new subset is clearly better
        inc = [s for s in (holdings or []) if s in valid]
        if len(inc) == cfg.n_select and set(inc) != {valid[i] for i in sel_idx}:
            x = np.zeros(len(valid))
            inc_idx = np.array(sorted(valid.index(s) for s in inc))
            x[inc_idx] = 1.0
            s_inc = subset_sharpe(x, mu.to_numpy(), Sigma_all)
            if ctx.selection_sharpe <= s_inc + cfg.selection_switch_margin * abs(s_inc):
                ctx.selection_method += (f"; kept holdings (best {ctx.selection_sharpe:.2f} vs held "
                                         f"{s_inc:.2f}, margin {100 * cfg.selection_switch_margin:.0f}%)")
                sel_idx, ctx.selection_sharpe = inc_idx, float(s_inc)
    else:
        sel_idx = np.arange(len(valid))
        ctx.selection_sharpe = float(subset_sharpe(np.ones(len(valid)), mu.to_numpy(), Sigma_all))
        ctx.selection_method = f"all {len(valid)} eligible stocks (no selection signal showed an edge)"
    sel = [valid[i] for i in sel_idx]
    ctx.selection = sel

    # Phase 12: how much of each
    S_sel = Sigma_all[np.ix_(sel_idx, sel_idx)]
    if cfg.weighting == "hrp":
        from portfolio_hrp_allocator import HRPAllocator
        w = HRPAllocator(method="single", bisection="dendrogram").allocate(S_sel, labels=sel).weights
    else:
        iv = 1.0 / np.sqrt(np.diag(S_sel))
        w = iv / iv.sum()
    ctx.weighting = cfg.weighting
    ctx.basket_weights = {s: float(v) for s, v in zip(sel, w)}
    ctx.alpha_z = {s: round(float(z[s]), 3) for s in sel}
    mu_sel = mu[sel].to_numpy()
    ctx.basket_mu = float(w @ mu_sel)
    ctx.basket_sigma = float(math.sqrt(w @ S_sel @ w))

    # liquidity for sizing and for VPIN bucket sizes
    for s in sel:
        v = daily[s][daily[s].index < pd.Timestamp(asof)].iloc[-20:]
        ctx.adv_shares[s] = float(v["Volume"].median())
    adv_notional = float(sum(w[i] * ctx.adv_shares[s] * closes[s].iloc[-1] for i, s in enumerate(sel)))

    # Phase 4
    from fusion_and_sizing import CostModel, KellySizer
    # The basket's return is beta x market + alpha. Its prior is centred on beta
    # times the long-run market return (the alpha part's prior is centred on 0).
    ctx.basket_beta = float(w @ betas[sel].to_numpy())
    sizer = KellySizer(kelly_fraction=cfg.kelly_fraction, prior_edge_vol=cfg.kelly_prior_edge_vol,
                       max_leverage=cfg.max_exposure, max_position_weight=cfg.max_exposure,
                       prior_mean=ctx.basket_beta * cfg.prior_market_return)
    costs = CostModel(impact_coefficient=0.75, half_spread=cfg.slippage_bps / 1e4,
                      fees_bps=1e4 * (cfg.exchange_rate + cfg.sebi_rate + cfg.stamp_rate_buy / 2),
                      stt_bps=1e4 * cfg.stt_rate)
    kr = sizer.size(mu_hat=ctx.basket_mu, sigma_annual=max(ctx.basket_sigma, 1e-4),
                    sample_years=ctx.market_span_years, equity=equity, adv_notional=adv_notional,
                    daily_vol=ctx.basket_sigma / math.sqrt(TRADING_DAYS), cost_model=costs)
    ctx.kelly_f, ctx.kelly_binding = kr.f_final, kr.binding_constraint
    ctx.kelly_detail = {"f_naive": round(kr.f_naive, 4), "f_shrunk": round(kr.f_shrunk, 4),
                        "mu_posterior": round(kr.mu_posterior, 5), "mu_net": round(kr.mu_net, 5),
                        "se_mu": round(kr.se_mu, 4)}

    # Phase 23
    hist = logret.iloc[-1500:]
    basket_loss = -(hist[sel].fillna(0.0).to_numpy() @ w)
    ctx.basket_es = expected_shortfall(basket_loss, cfg.es_level)
    ctx.index_es = expected_shortfall(-hist[INDEX_SYMBOL].fillna(0.0).to_numpy(), cfg.es_level)
    if ctx.basket_es is not None and ctx.index_es is not None \
            and ctx.basket_es > cfg.es_relative_threshold * ctx.index_es:
        ctx.evt_multiplier = cfg.es_trim
    ctx.components["evt"] = "ok" if ctx.basket_es is not None else "error: POT fit failed"

    # exposure and targets
    ctx.exposure = float(min(cfg.max_exposure, max(0.0, ctx.kelly_f * ctx.regime_multiplier
                                                   * ctx.ppo_multiplier * ctx.evt_multiplier)))
    if not ctx.tradeable:
        ctx.exposure = 0.0
    ctx.target_weights = {s: ctx.exposure * ctx.basket_weights[s] for s in sel}

    # Phase 24 (diagnostic only): the five largest positions
    if with_te:
        try:
            top = sorted(sel, key=lambda s: -ctx.basket_weights[s])[:5]
            ctx.transfer_entropy = te_diagnostics(logret[INDEX_SYMBOL].fillna(0.0).to_numpy(),
                                                  logret[top].fillna(0.0), top, cfg)
            ctx.components["transfer_entropy"] = "ok (diagnostic)"
        except Exception as exc:
            ctx.components["transfer_entropy"] = f"error: {exc}"
    ctx.components["vpin_toxicity"] = "ok (1-minute bars, bulk volume classification)"
    ctx.seconds = round(time.time() - t0, 1)
    return ctx


def save_context(ctx: DailyContext) -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"context_{ctx.asof}.json")
    with open(path, "w", encoding="ascii") as f:
        json.dump(ctx.to_dict(), f, indent=2, default=float)
    return path


# ===========================================================================
# 2. INTRADAY TOXICITY (Phase 14, VPIN half)
# ===========================================================================

class ToxicityMonitor:
    """
    VPIN on 1-minute bars: buckets of 1/`buckets_per_day` of the stock's median
    daily volume, bulk volume classification, VPIN over the last `window`
    buckets. The score is the percentile of the current VPIN against the
    stock's own VPIN history from earlier sessions, through ToxicityScore with
    the Hawkes weight set to zero (no trade-by-trade data to drive it).
    """

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self._cache: Dict[str, Tuple[pd.Timestamp, Optional[float], Optional[float]]] = {}

    def score(self, symbol: str, bars: pd.DataFrame, adv_shares: float,
              session_date: date) -> Tuple[Optional[float], Optional[float]]:
        """(score in [0, 1], current VPIN), or (None, None) if not computable."""
        from hft_hawkes_toxicity import ToxicityScore, VPINEstimator

        if bars is None or len(bars) < 50 or not adv_shares or adv_shares <= 0:
            return None, None
        last = bars.index[-1]
        hit = self._cache.get(symbol)
        if hit is not None and hit[0] == last:
            return hit[1], hit[2]
        b = bars[bars["Volume"] > 0]
        try:
            est = VPINEstimator(bucket_volume=adv_shares / self.cfg.vpin_buckets_per_day,
                                n_buckets=self.cfg.vpin_window_buckets, classifier="bvc",
                                sigma_mode="rolling")
            ts = b.index.asi8 / 1e9
            res = est.compute(ts, b["Close"].to_numpy(), b["Volume"].to_numpy())
        except Exception as exc:
            log.debug("VPIN %s not computable: %s", symbol, exc)
            self._cache[symbol] = (last, None, None)
            return None, None
        vpin = res["vpin"]
        t_end = pd.to_datetime(res["t_end"], unit="s", utc=True).tz_convert(IST)
        start = pd.Timestamp(datetime.combine(session_date, datetime.min.time()), tz=IST)
        ref = vpin[(t_end < start) & np.isfinite(vpin)]
        cur = vpin[np.isfinite(vpin)]
        if ref.size < 20 or cur.size == 0:
            self._cache[symbol] = (last, None, None)
            return None, None
        tox = ToxicityScore(combine="geometric", weight_hawkes=0.0,
                            alert_threshold=self.cfg.toxicity_threshold)
        tox.fit_reference(np.zeros(ref.size), ref)
        s = float(tox.score(np.zeros(1), cur[-1:])[0])
        self._cache[symbol] = (last, s, float(cur[-1]))
        return s, float(cur[-1])


# ===========================================================================
# 3. REBALANCER
# ===========================================================================

@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: int
    price: float
    reason: str
    reduce_only: bool


class Rebalancer:
    def __init__(self, cfg: Config = CONFIG, session: Optional[Session] = None) -> None:
        self.cfg = cfg
        self.session = session or Session()
        self.last_trade: Dict[str, datetime] = {}

    def plan(self, target_weights: Dict[str, float], positions: Dict[str, int],
             prices: Dict[str, Optional[float]], cash: float, equity: float, now: datetime,
             toxicity: Dict[str, Optional[float]]) -> Tuple[List[PlannedOrder], List[str]]:
        cfg = self.cfg
        sells: List[PlannedOrder] = []
        buys: List[PlannedOrder] = []
        notes: List[str] = []
        late = self.session.minutes_to_close(now) < cfg.no_new_risk_last_minutes

        for s in sorted(set(target_weights) | {k for k, q in positions.items() if q}):
            px = prices.get(s)
            if not px or px <= 0:
                notes.append(f"{s}: no price")
                continue
            held = int(positions.get(s, 0))
            tgt = int(math.floor(equity * target_weights.get(s, 0.0) / px))
            diff = tgt - held
            if diff == 0:
                continue
            exit_all = tgt == 0 and held > 0
            if not exit_all:
                if abs(diff) * px < cfg.min_trade_value:
                    continue
                if held > 0 and abs(diff) / max(tgt, 1) < cfg.rebalance_band:
                    continue
                last = self.last_trade.get(s)
                if last is not None and (now - last) < timedelta(minutes=cfg.rebalance_cooldown_minutes):
                    continue
            if diff < 0:
                sells.append(PlannedOrder(s, "SELL", -diff, px,
                                          "exit" if exit_all else "trim to target", True))
            else:
                if late:
                    notes.append(f"{s}: buy of {diff} held back (last {cfg.no_new_risk_last_minutes} min)")
                    continue
                tox = toxicity.get(s)
                if tox is not None and tox > cfg.toxicity_threshold:
                    notes.append(f"{s}: buy of {diff} held back (VPIN toxicity {tox:.2f})")
                    continue
                buys.append(PlannedOrder(s, "BUY", diff, px, "build to target", False))

        # cash check: sells first, and buys only what the cash (after sells) covers
        cost_buffer = 1.0 + cfg.stt_rate + cfg.stamp_rate_buy + cfg.slippage_bps / 1e4 + 0.001
        avail = cash + sum(o.qty * o.price * (1 - cfg.stt_rate) for o in sells)
        fitted: List[PlannedOrder] = []
        for o in buys:
            afford = int(avail // (o.price * cost_buffer))
            q = min(o.qty, afford)
            if q * o.price < cfg.min_trade_value:
                if q < o.qty:
                    notes.append(f"{o.symbol}: buy skipped, not enough cash")
                continue
            if q < o.qty:
                notes.append(f"{o.symbol}: buy cut from {o.qty} to {q} by cash")
            fitted.append(PlannedOrder(o.symbol, "BUY", q, o.price, o.reason, False))
            avail -= q * o.price * cost_buffer
        orders = sells + fitted
        for o in orders:
            self.last_trade[o.symbol] = now
        return orders, notes


def portfolio_equity(cash: float, positions: Dict[str, int], prices: Dict[str, Optional[float]]) -> float:
    return float(cash + sum(q * (prices.get(s) or 0.0) for s, q in positions.items()))


# ===========================================================================
# SELF-TESTS
# ===========================================================================

def _test_rebalancer_rules() -> None:
    cfg = Config()
    rb = Rebalancer(cfg)
    now = datetime(2026, 9, 25, 11, 0, tzinfo=IST)
    prices = {"A": 100.0, "B": 250.0, "C": 50.0, "D": 1000.0}
    # empty book, 2 targets -> two buys sized from equity
    orders, _ = rb.plan({"A": 0.2, "B": 0.1}, {}, prices, 500_000, 500_000, now, {})
    got = {(o.symbol, o.side, o.qty) for o in orders}
    assert got == {("A", "BUY", 1000), ("B", "BUY", 200)}, got
    # within band -> nothing; cooldown also active
    rb2 = Rebalancer(cfg)
    orders, _ = rb2.plan({"A": 0.2}, {"A": 950}, prices, 405_000, 500_000, now, {})
    assert orders == [], orders
    # dropped from target -> full exit even though small, and even if toxic
    orders, _ = rb2.plan({}, {"C": 3}, prices, 0, 150, now, {"C": 0.99})
    assert [(o.symbol, o.side, o.qty, o.reduce_only) for o in orders] == [("C", "SELL", 3, True)]
    # toxic flow blocks buys
    rb3 = Rebalancer(cfg)
    orders, notes = rb3.plan({"A": 0.2}, {}, prices, 500_000, 500_000, now, {"A": 0.95})
    assert orders == [] and "toxicity" in notes[0]
    # no new risk in the last minutes
    late = datetime(2026, 9, 25, 15, 25, tzinfo=IST)
    orders, notes = Rebalancer(cfg).plan({"A": 0.2}, {}, prices, 500_000, 500_000, late, {})
    assert orders == [] and "last" in notes[0]
    # cash caps buys, sells fund buys
    orders, notes = Rebalancer(cfg).plan({"D": 0.5}, {"A": 100}, prices, 20_000, 520_000, now, {})
    buy = [o for o in orders if o.side == "BUY"][0]
    assert orders[0].side == "SELL" and buy.qty * buy.price <= 20_000 + 100 * 100
    print("[ok] rebalancer: sizes from equity, no-trade band, full exits always allowed, toxic "
          "flow and the last minutes block buys only, cash caps buys, sells go first")


def _test_toxicity_detects_one_sided_flow() -> None:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2026-09-21", periods=5)
    frames = []
    for k, d in enumerate(days):
        idx = pd.date_range(pd.Timestamp(d).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15),
                            periods=375, freq="1min")
        step = rng.normal(0, 0.0008, idx.size)
        if k == len(days) - 1:
            step[200:] = np.abs(step[200:]) + 0.0015      # one-sided buying late on the last day
        px = 100 * np.exp(np.cumsum(step))
        vol = rng.integers(800, 1200, idx.size).astype(float)
        frames.append(pd.DataFrame({"Open": px, "High": px, "Low": px, "Close": px, "Volume": vol},
                                   index=idx))
    bars = pd.concat(frames)
    mon = ToxicityMonitor(Config())
    quiet, _ = mon.score("X", bars[bars.index < days[-1].tz_localize(IST) + pd.Timedelta(hours=12)],
                         375_000, days[-1].date())
    mon2 = ToxicityMonitor(Config())
    toxic, v = mon2.score("X", bars, 375_000, days[-1].date())
    assert toxic is not None and toxic > 0.9 and (quiet is None or quiet < toxic), (quiet, toxic)
    print(f"[ok] VPIN toxicity: one-sided flow scores {toxic:.2f} (percentile vs earlier "
          f"sessions), before it {quiet if quiet is None else round(quiet, 2)}")


def _test_context_on_real_data() -> None:
    from market_data import ALL_SYMBOLS, DailyStore

    if not os.path.exists(MANIFEST_FILE):
        print("[skip] no models/manifest.json: run train_models.py first")
        return
    daily = DailyStore().load(ALL_SYMBOLS, refresh=False)
    if INDEX_SYMBOL not in daily:
        print("[skip] no cached daily data: run market_data.py --fetch first")
        return
    asof = daily[INDEX_SYMBOL].index[-1].date()          # pretend we trade the last cached day
    ctx = build_daily_context(daily, asof, 500_000.0)
    assert ctx.data_end < ctx.asof, "context used data from the traded day"
    if CONFIG.selection_mode == "optimize":
        assert len(ctx.selection) == CONFIG.n_select
    else:
        assert len(ctx.selection) >= 40, len(ctx.selection)
    assert abs(sum(ctx.basket_weights.values()) - 1.0) < 1e-9
    assert 0.0 <= ctx.exposure <= CONFIG.max_exposure
    assert abs(sum(ctx.target_weights.values()) - ctx.exposure) < 1e-9
    print(f"[ok] daily context on real data for {ctx.asof} (data to {ctx.data_end}, {ctx.seconds}s): "
          f"{ctx.summary()}")
    for k, v in ctx.components.items():
        print(f"       {k:<20} {v}")
    if ctx.transfer_entropy:
        for t in ctx.transfer_entropy:
            print(f"       TE {t['symbol']:<14} index->stock {t['index_to_stock_bits']:+.4f} bits "
                  f"(p={t['index_to_stock_p']:.2f})  stock->index {t['stock_to_index_bits']:+.4f} "
                  f"(p={t['stock_to_index_p']:.2f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _test_rebalancer_rules()
    _test_toxicity_detects_one_sided_flow()
    _test_context_on_real_data()

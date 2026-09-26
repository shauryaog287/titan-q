"""
fusion_and_sizing.py
====================

Phase 5, Parts II and III: sentiment channel construction, IC-optimal signal
combination, and uncertainty-adjusted Kelly sizing.

Modules
-------
SentimentAggregator : syndication dedup, time decay, Kish effective sample
                      size, confidence shrinkage.
SignalCombiner      : Grinold-Kahn optimal weights with IC shrinkage, and the
                      go/no-go test for whether the second signal earns any
                      weight at all.
EdgeHead            : calibrated forward-excess-return estimator. This, NOT the
                      PPO value function, is what feeds Kelly.
GarchVolForecaster  : GARCH(1,1) conditional variance with horizon aggregation.
KellySizer          : continuous-time Kelly with parameter-uncertainty
                      shrinkage, higher-moment correction, fractional scaling,
                      and cost-recursive net edge.

Dependencies: numpy, pandas, scipy, scikit-learn (isotonic), arch (optional).
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

_EPS = 1e-12

__all__ = [
    "Article",
    "SentimentAggregator",
    "SentimentSnapshot",
    "SignalCombiner",
    "CombinationResult",
    "EdgeHead",
    "GarchVolForecaster",
    "KellySizer",
    "KellyResult",
    "CostModel",
]


# ===========================================================================
# PART II-A: SENTIMENT CHANNEL
# ===========================================================================

@dataclass(frozen=True)
class Article:
    """One scored news item, point-in-time."""

    article_id: str
    body: str
    published_ts: pd.Timestamp
    available_ts: pd.Timestamp      # when OUR pipeline could first have used it
    finbert_score: float            # [-1, +1], calibrated
    source: str
    entity_salience: float = 1.0    # [0, 1] from NER + position weighting

    def __post_init__(self) -> None:
        if self.available_ts < self.published_ts:
            raise ValueError(
                f"{self.article_id}: available_ts precedes published_ts. This is "
                "the signature of a revised archive and invalidates the sample."
            )


@dataclass(frozen=True)
class SentimentSnapshot:
    """What the policy observation and the F3 gate consume."""

    score: float          # shrunk, decayed, [-1, +1]
    raw_mean: float       # before confidence shrinkage
    confidence: float     # [0, 1]
    present: int          # 0/1 mask -- NOT redundant with score == 0.0
    k_eff: float          # Kish effective sample size
    bars_since_news: float
    n_raw_articles: int
    n_clusters: int

    def as_features(self, max_bars: float = 500.0) -> np.ndarray:
        """Observation channel. Order is fixed and must match training."""
        return np.array(
            [
                self.score,
                float(self.present),
                math.log1p(min(self.bars_since_news, max_bars)) / math.log1p(max_bars),
                self.confidence,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def empty(bars_since: float = 1e9) -> "SentimentSnapshot":
        return SentimentSnapshot(0.0, 0.0, 0.0, 0, 0.0, bars_since, 0, 0)


def _simhash(text: str, bits: int = 64) -> int:
    """64-bit SimHash over token shingles. Near-duplicate wire copy detection."""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0
    shingles = [" ".join(tokens[i : i + 3]) for i in range(max(len(tokens) - 2, 1))]
    v = np.zeros(bits, dtype=np.int64)
    for sh in shingles:
        h = int(hashlib.blake2b(sh.encode(), digest_size=8).hexdigest(), 16)
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if v[b] > 0:
            out |= 1 << b
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class SentimentAggregator:
    """
    Articles -> one snapshot per symbol per bar.

    The four transformations, in order: de-duplicate syndicated copies, weight
    by source quality and entity salience, decay by age, and shrink the result
    toward neutral by effective sample size.

    Parameters
    ----------
    half_life_bars : float
        News decay half-life. FIT THIS -- see `fit_half_life`. Macro items
        decay in hours, single-name earnings coverage over one to three days,
        and they should not share a kernel.
    prior_strength : float
        k0 in the shrinkage c = k_eff / (k_eff + k0). Larger means more
        evidence required before the channel speaks. Fit on held-out data.
    simhash_threshold : int
        Hamming distance below which two articles are the same story. 3 is
        tight (near-identical wire copy); 12 starts merging distinct stories.
    source_quality : dict
        Per-source weight. Unknown sources take `default_quality`.
    """

    def __init__(
        self,
        half_life_bars: float = 8.0,
        prior_strength: float = 3.0,
        simhash_threshold: int = 3,
        source_quality: Optional[Dict[str, float]] = None,
        default_quality: float = 0.5,
        lookback_bars: float = 40.0,
    ) -> None:
        self.half_life_bars = float(half_life_bars)
        self.prior_strength = float(prior_strength)
        self.simhash_threshold = int(simhash_threshold)
        self.source_quality = source_quality or {}
        self.default_quality = float(default_quality)
        self.lookback_bars = float(lookback_bars)

    # -- dedup ------------------------------------------------------------ #

    def deduplicate(self, articles: Sequence[Article]) -> List[List[Article]]:
        """
        Greedy SimHash clustering. Returns clusters, earliest article first
        within each -- the earliest is the one whose timestamp matters, since
        that is when the information became available.
        """
        clusters: List[List[Article]] = []
        hashes: List[int] = []
        for art in sorted(articles, key=lambda a: a.available_ts):
            h = _simhash(art.body)
            placed = False
            for i, ch in enumerate(hashes):
                if _hamming(h, ch) <= self.simhash_threshold:
                    clusters[i].append(art)
                    placed = True
                    break
            if not placed:
                clusters.append([art])
                hashes.append(h)
        return clusters

    # -- aggregation ------------------------------------------------------ #

    def aggregate(
        self,
        articles: Sequence[Article],
        now: pd.Timestamp,
        bar_seconds: float,
    ) -> SentimentSnapshot:
        """
        Point-in-time aggregation. Only articles with available_ts <= now are
        admitted; publication time is irrelevant to what we could have known.
        """
        usable = [a for a in articles if a.available_ts <= now]
        if not usable:
            return SentimentSnapshot.empty()

        clusters = self.deduplicate(usable)

        weights: List[float] = []
        scores: List[float] = []
        ages: List[float] = []

        for cluster in clusters:
            lead = cluster[0]  # earliest available copy
            age_bars = (now - lead.available_ts).total_seconds() / bar_seconds
            if age_bars > self.lookback_bars:
                continue

            q = self.source_quality.get(lead.source, self.default_quality)
            q *= float(np.clip(lead.entity_salience, 0.0, 1.0))
            kappa = 0.5 ** (age_bars / self.half_life_bars)

            # Representative score: median across the cluster resists a single
            # mis-scored copy without letting cluster size inflate influence.
            s = float(np.median([a.finbert_score for a in cluster]))

            weights.append(q * kappa)
            scores.append(s)
            ages.append(age_bars)

        if not weights:
            return SentimentSnapshot.empty()

        w = np.asarray(weights, dtype=float)
        s = np.asarray(scores, dtype=float)
        wsum = w.sum()
        if wsum < _EPS:
            return SentimentSnapshot.empty()

        raw_mean = float(np.dot(w, s) / wsum)

        # Kish effective sample size: penalises concentration, so one dominant
        # source cannot masquerade as broad coverage.
        k_eff = float(wsum**2 / max(np.dot(w, w), _EPS))
        confidence = k_eff / (k_eff + self.prior_strength)

        return SentimentSnapshot(
            score=float(np.clip(raw_mean * confidence, -1.0, 1.0)),
            raw_mean=raw_mean,
            confidence=float(confidence),
            present=1,
            k_eff=k_eff,
            bars_since_news=float(min(ages)),
            n_raw_articles=len(usable),
            n_clusters=len(clusters),
        )

    # -- calibration of the decay kernel ---------------------------------- #

    @staticmethod
    def fit_half_life(
        sentiment_by_age: pd.DataFrame,
        forward_returns: pd.Series,
        grid: Sequence[float] = tuple(np.linspace(0.5, 48.0, 40)),
    ) -> Tuple[float, pd.Series]:
        """
        Choose H by maximising |IC| of the decayed signal against forward
        returns. `sentiment_by_age` has columns 'score' and 'age_bars'.

        Returns (best_half_life, ic_by_half_life). A flat curve means the decay
        kernel is not the binding constraint and the channel is weak for other
        reasons.
        """
        ics = {}
        for H in grid:
            kappa = 0.5 ** (sentiment_by_age["age_bars"].values / H)
            decayed = sentiment_by_age["score"].values * kappa
            ok = np.isfinite(decayed) & np.isfinite(forward_returns.values)
            if ok.sum() < 30:
                continue
            ics[float(H)] = abs(
                float(np.corrcoef(decayed[ok], forward_returns.values[ok])[0, 1])
            )
        if not ics:
            return float(grid[0]), pd.Series(dtype=float)
        ser = pd.Series(ics).sort_index()
        return float(ser.idxmax()), ser


# ===========================================================================
# PART II-B: OPTIMAL COMBINATION
# ===========================================================================

@dataclass(frozen=True)
class CombinationResult:
    ic_technical_raw: float
    ic_sentiment_raw: float
    ic_technical_shrunk: float
    ic_sentiment_shrunk: float
    signal_correlation: float
    se_ic: float
    n_eff: float

    w_technical: float
    w_sentiment: float
    ic_combined: float
    ic_technical_alone: float

    @property
    def marginal_gain(self) -> float:
        """Fractional IC improvement from adding the second signal."""
        base = abs(self.ic_technical_alone)
        return (abs(self.ic_combined) - base) / base if base > _EPS else 0.0

    @property
    def sentiment_earns_weight(self) -> bool:
        """IC2 > rho * IC1 -- the go/no-go condition."""
        return self.ic_sentiment_shrunk > self.signal_correlation * self.ic_technical_shrunk

    @property
    def t_stat_sentiment(self) -> float:
        return self.ic_sentiment_raw / self.se_ic if self.se_ic > _EPS else 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                "Signal combination",
                f"  IC technical  : {self.ic_technical_raw:+.4f} raw -> "
                f"{self.ic_technical_shrunk:+.4f} shrunk",
                f"  IC sentiment  : {self.ic_sentiment_raw:+.4f} raw -> "
                f"{self.ic_sentiment_shrunk:+.4f} shrunk  (t={self.t_stat_sentiment:+.2f})",
                f"  signal corr   : {self.signal_correlation:+.4f}",
                f"  SE(IC)        : {self.se_ic:.4f}   n_eff={self.n_eff:.0f}",
                "",
                f"  w_technical   : {self.w_technical:.4f}",
                f"  w_sentiment   : {self.w_sentiment:.4f}",
                f"  IC combined   : {self.ic_combined:+.4f} "
                f"(vs {self.ic_technical_alone:+.4f} alone, "
                f"{100*self.marginal_gain:+.1f}%)",
                "",
                f"  sentiment earns weight: {self.sentiment_earns_weight}",
            ]
        )


class SignalCombiner:
    """
    Grinold-Kahn optimal signal weighting: w* proportional to Sigma^-1 * IC.

    The output that matters is not the weight but `sentiment_earns_weight`.
    A signal with genuine standalone predictive power earns zero weight when
    it is sufficiently correlated with a signal you already have.
    """

    def __init__(
        self,
        cross_sectional_corr: float = 0.3,
        max_sentiment_weight: float = 0.4,
        allow_negative_weight: bool = False,
    ) -> None:
        self.cross_sectional_corr = float(cross_sectional_corr)
        self.max_sentiment_weight = float(max_sentiment_weight)
        self.allow_negative_weight = bool(allow_negative_weight)

    # -- effective sample size -------------------------------------------- #

    def effective_n(self, n_periods: int, n_names: int) -> float:
        """
        Panel effective sample size. Cross-sectional correlation means n names
        do not give n independent observations:

            n_eff_per_period = n / (1 + (n-1) * rho_c)

        For Nifty 50 at rho_c = 0.3 this is ~3.2, not 50.
        """
        rho = self.cross_sectional_corr
        per_period = n_names / (1.0 + (n_names - 1) * rho) if n_names > 1 else 1.0
        return float(n_periods * per_period)

    @staticmethod
    def shrink_ic(ic_hat: float, se: float) -> float:
        """IC_shrunk = IC_hat * t^2/(t^2+1). Retains 76% at t=1.8, 39% at t=0.8."""
        if se <= _EPS:
            return ic_hat
        t = ic_hat / se
        return float(ic_hat * (t * t) / (t * t + 1.0))

    # -- core -------------------------------------------------------------- #

    def combine(
        self,
        z_technical: np.ndarray,
        z_sentiment: np.ndarray,
        forward_returns: np.ndarray,
        n_periods: int,
        n_names: int,
    ) -> CombinationResult:
        z1 = np.asarray(z_technical, dtype=float)
        z2 = np.asarray(z_sentiment, dtype=float)
        r = np.asarray(forward_returns, dtype=float)

        ok = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(r)
        if ok.sum() < 50:
            raise ValueError(f"insufficient joint observations: {ok.sum()}")
        z1, z2, r = z1[ok], z2[ok], r[ok]

        ic1 = float(np.corrcoef(z1, r)[0, 1])
        ic2 = float(np.corrcoef(z2, r)[0, 1])
        rho = float(np.corrcoef(z1, z2)[0, 1])

        n_eff = self.effective_n(n_periods, n_names)
        se = 1.0 / math.sqrt(max(n_eff, 1.0))

        ic1_s = self.shrink_ic(ic1, se)
        ic2_s = self.shrink_ic(ic2, se)

        denom = 1.0 - rho * rho
        if abs(denom) < 1e-6:
            # Signals are collinear: the second carries no independent content.
            warnings.warn("signals near-collinear; sentiment weight forced to zero")
            w1_raw, w2_raw = 1.0, 0.0
        else:
            w1_raw = (ic1_s - rho * ic2_s) / denom
            w2_raw = (ic2_s - rho * ic1_s) / denom

        if not self.allow_negative_weight:
            w2_raw = max(w2_raw, 0.0)

        total = abs(w1_raw) + abs(w2_raw)
        if total < _EPS:
            w1, w2 = 1.0, 0.0
        else:
            w1, w2 = w1_raw / total, w2_raw / total

        # cap sentiment regardless of what the estimate says
        if abs(w2) > self.max_sentiment_weight:
            w2 = math.copysign(self.max_sentiment_weight, w2)
            w1 = math.copysign(1.0 - abs(w2), w1)

        ic_c_sq = (ic1_s**2 - 2 * rho * ic1_s * ic2_s + ic2_s**2) / max(denom, 1e-6)
        ic_c = math.copysign(math.sqrt(max(ic_c_sq, 0.0)), ic1_s)

        return CombinationResult(
            ic_technical_raw=ic1,
            ic_sentiment_raw=ic2,
            ic_technical_shrunk=ic1_s,
            ic_sentiment_shrunk=ic2_s,
            signal_correlation=rho,
            se_ic=se,
            n_eff=n_eff,
            w_technical=w1,
            w_sentiment=w2,
            ic_combined=ic_c,
            ic_technical_alone=ic1_s,
        )


# ===========================================================================
# PART III-A: CALIBRATED EDGE
# ===========================================================================

class EdgeHead:
    """
    Forward excess return estimator with isotonic calibration.

    This exists because the PPO value function cannot serve this role. Under a
    differential Sortino reward, V(s) is a discounted sum of dimensionless
    ratio sensitivities: no return units, no calibration to forward P&L, and
    not even monotone in expected return (D_t rises when downside variance
    falls, independently of the mean). Kelly's numerator needs mu in return
    units, estimated separately from sigma.

    The calibration curve is also the diagnostic. If it is flat, the head has
    no edge and the Kelly fraction should be zero regardless of what the
    policy signals.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = int(n_bins)
        self._iso = None
        self._fitted = False
        self.reliability_: Optional[pd.DataFrame] = None

    def fit_calibration(
        self, predicted_edge: np.ndarray, realised_excess: np.ndarray
    ) -> "EdgeHead":
        """Fit on a HELD-OUT fold. Fitting in-sample defeats the purpose."""
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(predicted_edge, dtype=float)
        y = np.asarray(realised_excess, dtype=float)
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        if p.size < 100:
            raise ValueError("need >=100 held-out observations to calibrate")

        self._iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        self._fitted = True

        qs = np.quantile(p, np.linspace(0, 1, self.n_bins + 1))
        qs[-1] += 1e-9
        idx = np.clip(np.digitize(p, qs[1:-1]), 0, self.n_bins - 1)
        rows = []
        for b in range(self.n_bins):
            m = idx == b
            if m.sum() == 0:
                continue
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_predicted": float(p[m].mean()),
                    "mean_realised": float(y[m].mean()),
                    "se_realised": float(y[m].std(ddof=1) / math.sqrt(m.sum())),
                }
            )
        self.reliability_ = pd.DataFrame(rows)
        return self

    def is_flat(self, alpha: float = 0.05) -> bool:
        """
        Test the reliability curve for slope. A flat curve means the Kelly
        numerator is noise; size fixed-fraction until this is fixed.
        """
        if self.reliability_ is None or len(self.reliability_) < 3:
            return True
        x = self.reliability_["mean_predicted"].values
        y = self.reliability_["mean_realised"].values
        slope, _, _, pval, _ = stats.linregress(x, y)
        return bool(pval > alpha or slope <= 0)

    def calibrated_edge(self, raw_edge: float) -> float:
        if not self._fitted:
            raise RuntimeError("calibration not fitted; refusing to size a position")
        return float(self._iso.predict([raw_edge])[0])


# ===========================================================================
# PART III-B: VOLATILITY
# ===========================================================================

class GarchVolForecaster:
    """
    GARCH(1,1) conditional variance with horizon aggregation.

    Trailing realised volatility lags; it over-sizes immediately after a
    volatility spike because the spike has not yet worked through the window.
    GARCH responds through the alpha term on the current shock.
    """

    def __init__(self, omega: float, alpha: float, beta: float) -> None:
        if alpha + beta >= 1.0:
            raise ValueError("alpha + beta >= 1: non-stationary, no long-run variance")
        self.omega, self.alpha, self.beta = float(omega), float(alpha), float(beta)

    @property
    def unconditional_var(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)

    @classmethod
    def fit(cls, returns: np.ndarray) -> "GarchVolForecaster":
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError("pip install arch") from exc
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)] * 100.0  # arch prefers percent scale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant").fit(disp="off")
        p = res.params
        return cls(
            omega=float(p["omega"]) / 1e4,
            alpha=float(p["alpha[1]"]),
            beta=float(p["beta[1]"]),
        )

    def forecast_path(self, current_var: float, horizon: int) -> np.ndarray:
        """sigma^2_{t+h|t} = var_bar + (alpha+beta)^h (sigma^2_t - var_bar)."""
        vbar = self.unconditional_var
        persist = self.alpha + self.beta
        h = np.arange(1, horizon + 1)
        return vbar + (persist**h) * (current_var - vbar)

    def cumulative_var(self, current_var: float, horizon: int) -> float:
        """Variance of the h-period cumulative return."""
        return float(self.forecast_path(current_var, horizon).sum())


# ===========================================================================
# PART III-C: KELLY
# ===========================================================================

@dataclass(frozen=True)
class CostModel:
    """
    Square-root impact plus linear frictions.

        dP/P ~ Y * sigma_daily * sqrt(Q / V_daily)

    Y is dimensionless, typically 0.5-1.0, and MUST be fitted on your own
    fills. The default is a placeholder that will be wrong for your venue.
    """

    impact_coefficient: float = 0.75
    half_spread: float = 0.0005
    fees_bps: float = 3.0
    stt_bps: float = 10.0

    def round_trip_cost(
        self, notional: float, adv_notional: float, daily_vol: float
    ) -> float:
        if adv_notional <= 0 or notional <= 0:
            return self.half_spread * 2
        participation = notional / adv_notional
        impact = self.impact_coefficient * daily_vol * math.sqrt(participation)
        linear = self.half_spread + (self.fees_bps + self.stt_bps) / 1e4
        return float(2.0 * (impact + linear))


@dataclass(frozen=True)
class KellyResult:
    f_naive: float
    f_shrunk: float
    f_final: float
    mu_raw: float
    mu_posterior: float
    mu_net: float
    sigma: float
    se_mu: float
    lam: float
    growth_fraction: float
    drawdown_prob_50pct: float
    binding_constraint: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"  mu raw / posterior / net : {self.mu_raw:+.4f} / "
                f"{self.mu_posterior:+.4f} / {self.mu_net:+.4f}",
                f"  sigma / SE(mu)           : {self.sigma:.4f} / {self.se_mu:.4f}",
                f"  f naive / shrunk / final : {self.f_naive:.4f} / "
                f"{self.f_shrunk:.4f} / {self.f_final:.4f}",
                f"  lambda                   : {self.lam:.2f} "
                f"({100*self.growth_fraction:.1f}% of max growth)",
                f"  P(50% drawdown ever)     : {100*self.drawdown_prob_50pct:.2f}%",
                f"  binding constraint       : {self.binding_constraint}",
            ]
        )


class KellySizer:
    """
    Continuous-time Kelly with the corrections that make it usable.

    Base result, from maximising g(f) = f*mu - f^2*sigma^2/2:

        f* = (mu - r) / sigma^2

    Corrections applied here, in order:

    1. Posterior shrinkage of mu. SE(mu_hat) = sigma / sqrt(T) where T is the
       CALENDAR SPAN in years -- independent of sampling frequency (Merton).
       At sigma=25% and T=10, SE = 7.9% p.a. With a prior mu ~ N(0, tau^2),
       mu_post = mu_hat * tau^2/(tau^2 + SE^2). At tau=3% the multiplier is
       0.126: an 87% haircut, and it is what the posterior says, not caution.

    2. Predictive variance. Under parameter uncertainty the denominator is
       sigma^2 + SE(mu)^2, not sigma^2.

    3. Cost recursion. Q depends on f, and f depends on mu net of the cost of
       trading Q. The sqrt in the impact law makes the map a contraction;
       two or three fixed-point passes converge.

    4. Higher moments. m1 - f*m2 + f^2*m3 - f^3*m4 = 0 from the fourth-order
       expansion of E[log(1+fR)]. Negative skew and excess kurtosis both push
       f down.

    5. Fractional scaling. g(lam*f*)/g(f*) = 2*lam - lam^2. Half Kelly buys
       75% of the growth for 50% of the volatility, and the drawdown law
       P(DD >= 1-a) = a^(2/lam - 1) is what should actually set lambda.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        prior_edge_vol: float = 0.03,
        risk_free: float = 0.0,
        max_leverage: float = 1.0,
        max_position_weight: float = 0.10,
        use_higher_moments: bool = True,
    ) -> None:
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1]")
        self.lam = float(kelly_fraction)
        self.tau = float(prior_edge_vol)
        self.rf = float(risk_free)
        self.max_leverage = float(max_leverage)
        self.max_position_weight = float(max_position_weight)
        self.use_higher_moments = bool(use_higher_moments)

    # -- component pieces -------------------------------------------------- #

    @staticmethod
    def se_mu(sigma_annual: float, years: float) -> float:
        """Independent of sampling frequency. This is the whole problem."""
        return float(sigma_annual / math.sqrt(max(years, _EPS)))

    def posterior_mu(self, mu_hat: float, se: float) -> float:
        t2 = self.tau**2
        return float(mu_hat * t2 / (t2 + se**2))

    @staticmethod
    def growth_fraction(lam: float) -> float:
        """g(lam f*)/g(f*) = 2*lam - lam^2."""
        return float(2.0 * lam - lam**2)

    @staticmethod
    def drawdown_probability(lam: float, alpha: float = 0.5) -> float:
        """P(wealth ever falls to fraction alpha of peak) = alpha^(2/lam - 1)."""
        if lam <= 0:
            return 0.0
        exponent = 2.0 / lam - 1.0
        return float(min(alpha**exponent, 1.0))

    @staticmethod
    def higher_moment_f(moments: Tuple[float, float, float, float]) -> float:
        """
        Smallest positive root of m1 - f*m2 + f^2*m3 - f^3*m4 = 0.

        Falls back to m1/m2 (the Gaussian answer) if no positive root exists.
        """
        m1, m2, m3, m4 = moments
        coeffs = [-m4, m3, -m2, m1]  # descending powers of f
        try:
            roots = np.roots(coeffs)
        except Exception:
            return m1 / m2 if m2 > _EPS else 0.0
        real = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
        if not real:
            return m1 / m2 if m2 > _EPS else 0.0
        return min(real)

    # -- main -------------------------------------------------------------- #

    def size(
        self,
        mu_hat: float,
        sigma_annual: float,
        sample_years: float,
        equity: float,
        adv_notional: float,
        daily_vol: float,
        cost_model: CostModel,
        moments: Optional[Tuple[float, float, float, float]] = None,
        max_iter: int = 6,
    ) -> KellyResult:
        """
        mu_hat and sigma_annual must be on the same (annualised) basis.
        Returns the fraction of equity to allocate, after every correction.
        """
        if sigma_annual <= _EPS:
            raise ValueError("sigma must be positive")

        se = self.se_mu(sigma_annual, sample_years)
        mu_post = self.posterior_mu(mu_hat, se)

        # predictive variance
        var_pred = sigma_annual**2 + se**2

        f_naive = (mu_hat - self.rf) / sigma_annual**2
        f_shrunk = (mu_post - self.rf) / var_pred

        # cost recursion: Q depends on f, f depends on mu net of cost(Q)
        f = f_shrunk
        mu_net = mu_post
        for _ in range(max_iter):
            notional = abs(f) * equity
            cost = cost_model.round_trip_cost(notional, adv_notional, daily_vol)
            mu_net = mu_post - cost
            if mu_net <= 0:
                f = 0.0
                break
            f_new = (mu_net - self.rf) / var_pred
            if abs(f_new - f) < 1e-6:
                f = f_new
                break
            f = f_new

        # higher-moment correction, applied as a ratio so it composes with
        # the shrinkage above rather than replacing it
        if self.use_higher_moments and moments is not None and f != 0.0:
            m1, m2, _, _ = moments
            f_gauss = m1 / m2 if m2 > _EPS else 0.0
            f_hm = self.higher_moment_f(moments)
            if f_gauss > _EPS:
                f *= float(np.clip(f_hm / f_gauss, 0.0, 1.0))

        f *= self.lam

        # constraints, tracking which one actually binds
        binding = "none"
        if abs(f) > self.max_position_weight:
            f = math.copysign(self.max_position_weight, f)
            binding = "position_weight"
        if abs(f) > self.max_leverage:
            f = math.copysign(self.max_leverage, f)
            binding = "leverage"
        if mu_net <= 0:
            f, binding = 0.0, "negative_net_edge"

        return KellyResult(
            f_naive=float(f_naive),
            f_shrunk=float(f_shrunk),
            f_final=float(f),
            mu_raw=float(mu_hat),
            mu_posterior=float(mu_post),
            mu_net=float(mu_net),
            sigma=float(sigma_annual),
            se_mu=float(se),
            lam=self.lam,
            growth_fraction=self.growth_fraction(self.lam),
            drawdown_prob_50pct=self.drawdown_probability(self.lam),
            binding_constraint=binding,
        )


# ===========================================================================
# DEMONSTRATIONS
# ===========================================================================

def demo_orthogonality_dominates() -> None:
    """
    The two rows from the whitepaper. What you buy with a sentiment channel is
    orthogonality, not sentiment.
    """
    print("IC1 = 0.050 in both cases\n")
    print(f"{'IC2':>7} {'rho':>6} {'w_sent':>8} {'IC_comb':>9} {'gain':>8}")
    for ic2, rho in [(0.020, 0.30), (0.040, 0.10)]:
        ic1 = 0.050
        d = 1 - rho**2
        w1, w2 = (ic1 - rho * ic2) / d, (ic2 - rho * ic1) / d
        tot = w1 + w2
        icc = math.sqrt((ic1**2 - 2 * rho * ic1 * ic2 + ic2**2) / d)
        print(
            f"{ic2:>7.3f} {rho:>6.2f} {w2/tot:>7.1%} {icc:>9.4f} "
            f"{(icc-ic1)/ic1:>7.1%}"
        )
    print("\nFirst row is the usual outcome for news sentiment bolted onto a")
    print("technical model: real but weak, correlated with price action, worth")
    print("essentially nothing.")


def demo_estimation_error() -> None:
    """Why full Kelly on a 10-year sample is not a conservative-vs-aggressive call."""
    sigma, years, tau = 0.25, 10.0, 0.03
    se = sigma / math.sqrt(years)
    print(f"sigma = {sigma:.0%} p.a., sample = {years:.0f} years")
    print(f"  SE(mu_hat) = sigma/sqrt(T) = {se:.1%} p.a.")
    print(f"  years needed for SE = 2%:  {sigma**2/0.02**2:.0f}")
    print()
    for mu_hat in (0.04, 0.08, 0.15):
        mu_post = mu_hat * tau**2 / (tau**2 + se**2)
        f_naive = mu_hat / sigma**2
        f_adj = mu_post / (sigma**2 + se**2)
        print(
            f"  mu_hat={mu_hat:.0%}  95% CI=[{mu_hat-2*se:+.1%}, {mu_hat+2*se:+.1%}]  "
            f"mu_post={mu_post:.2%}  f: {f_naive:.2f}x -> {f_adj:.2f}x"
        )


def demo_fractional_kelly() -> None:
    print(f"{'lambda':>8} {'growth':>9} {'vol':>7} {'P(50% DD)':>11}")
    for lam in (1.0, 0.75, 0.5, 0.25):
        print(
            f"{lam:>8.2f} {KellySizer.growth_fraction(lam):>8.1%} "
            f"{lam:>6.0%} {KellySizer.drawdown_probability(lam):>10.2%}"
        )
    print("\nGrowth is quadratic and flat at the optimum; volatility is linear.")
    print("That asymmetry is the entire argument for fractional Kelly.")


def demo_full_sizing() -> None:
    sizer = KellySizer(kelly_fraction=0.25, prior_edge_vol=0.03, max_position_weight=0.10)
    costs = CostModel(impact_coefficient=0.75, half_spread=0.0004)
    res = sizer.size(
        mu_hat=0.09,
        sigma_annual=0.26,
        sample_years=10.0,
        equity=10_000_000.0,
        adv_notional=850_000_000.0,
        daily_vol=0.26 / math.sqrt(252),
        cost_model=costs,
        moments=(0.0004, 0.00027, -1.2e-6, 2.4e-7),
    )
    print(res.summary())


if __name__ == "__main__":
    demo_orthogonality_dominates()
    print("\n" + "=" * 72 + "\n")
    demo_estimation_error()
    print("\n" + "=" * 72 + "\n")
    demo_fractional_kelly()
    print("\n" + "=" * 72 + "\n")
    demo_full_sizing()

"""
statarb_cointegration.py
========================

Phase 6 statistical arbitrage: cointegration discovery, spread construction,
z-score signal generation.

Modules
-------
1. CointegrationTester : Engle-Granger (both orderings), ADF on residuals with
                         correct MacKinnon EG critical values, Johansen trace /
                         max-eigenvalue, OU half-life, Hurst, Lo-MacKinlay
                         variance ratio.
2. SpreadModel         : static OLS, Total Least Squares, rolling OLS, and a
                         Kalman filter for time-varying hedge ratios.
3. ZScoreEngine        : causal rolling z-score with lagged moments, plus the
                         Kalman-native standardised innovation.
4. SignalEngine        : entry/exit/stop state machine with hysteresis,
                         half-life-scaled time stops, and a live cointegration
                         break monitor.
5. PairScreener        : universe screening with Benjamini-Hochberg FDR control
                         and formation/trading window separation.

Dependencies: numpy, pandas, scipy, statsmodels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_EPS = 1e-12

__all__ = [
    "CointegrationTester",
    "CointegrationResult",
    "SpreadModel",
    "HedgeRatioMethod",
    "ZScoreEngine",
    "SignalEngine",
    "SignalConfig",
    "PairScreener",
    "PositionState",
]


# ===========================================================================
# 1. COINTEGRATION TESTING
# ===========================================================================

@dataclass(frozen=True)
class EngleGrangerDirection:
    """Result of one ordering of the Engle-Granger regression."""

    dependent: str
    independent: str
    alpha: float
    beta: float
    beta_stderr: float
    r_squared: float

    # Correctly sized test (MacKinnon critical values for EG residuals)
    eg_stat: float
    eg_pvalue: float
    eg_crit_1pct: float
    eg_crit_5pct: float
    eg_crit_10pct: float

    # Naive ADF on the same residuals -- reported ONLY to expose the size
    # distortion. Do not use this p-value for an inference decision.
    naive_adf_stat: float
    naive_adf_pvalue: float
    naive_adf_crit_5pct: float

    adf_used_lags: int
    n_obs: int

    @property
    def size_distortion(self) -> float:
        """Difference between the naive and correct 5% critical values."""
        return self.naive_adf_crit_5pct - self.eg_crit_5pct

    @property
    def naive_would_falsely_reject(self) -> bool:
        """True if naive ADF rejects at 5% where the correct EG test does not."""
        return (self.naive_adf_stat < self.naive_adf_crit_5pct) and (
            self.eg_stat >= self.eg_crit_5pct
        )


@dataclass(frozen=True)
class CointegrationResult:
    """Aggregated evidence for a candidate pair."""

    symbol_y: str
    symbol_x: str
    n_obs: int

    forward: EngleGrangerDirection
    reverse: EngleGrangerDirection

    johansen_trace_stat: Optional[float] = None
    johansen_trace_crit_5pct: Optional[float] = None
    johansen_maxeig_stat: Optional[float] = None
    johansen_maxeig_crit_5pct: Optional[float] = None
    johansen_beta: Optional[float] = None

    half_life: Optional[float] = None
    half_life_pvalue: Optional[float] = None
    hurst: Optional[float] = None
    variance_ratio: Optional[float] = None
    variance_ratio_pvalue: Optional[float] = None

    spread_kpss_pvalue: Optional[float] = None
    correlation: Optional[float] = None

    # -- decision surface ------------------------------------------------- #

    @property
    def conservative(self) -> EngleGrangerDirection:
        """
        The less favourable of the two orderings.

        Engle-Granger is not symmetric: regressing y on x and x on y yields
        different test statistics, and on finite samples one direction can
        reject while the other does not. Taking the weaker of the two is the
        honest reading; taking the stronger is selection on noise.
        """
        return self.forward if self.forward.eg_pvalue >= self.reverse.eg_pvalue else self.reverse

    @property
    def orderings_disagree_at_5pct(self) -> bool:
        return (self.forward.eg_pvalue < 0.05) != (self.reverse.eg_pvalue < 0.05)

    def passes(
        self,
        alpha: float = 0.05,
        require_johansen: bool = True,
        max_half_life: Optional[float] = None,
        min_half_life: float = 1.0,
        max_hurst: float = 0.5,
    ) -> bool:
        """All-of conjunction. Each additional filter costs power; that is
        intentional, since the base rate of genuine cointegration in an
        arbitrary equity universe is very low."""
        if self.conservative.eg_pvalue >= alpha:
            return False
        if require_johansen:
            if self.johansen_trace_stat is None or self.johansen_trace_crit_5pct is None:
                return False
            if self.johansen_trace_stat <= self.johansen_trace_crit_5pct:
                return False
        if self.half_life is None or not np.isfinite(self.half_life):
            return False
        if self.half_life < min_half_life:
            return False
        if max_half_life is not None and self.half_life > max_half_life:
            return False
        if self.hurst is not None and self.hurst > max_hurst:
            return False
        return True

    def summary(self) -> str:
        c = self.conservative
        lines = [
            f"Pair: {self.symbol_y} / {self.symbol_x}   (n={self.n_obs})",
            f"  corr(log returns)        : {self.correlation:+.4f}"
            if self.correlation is not None
            else "  corr                     : n/a",
            "",
            "  Engle-Granger, both orderings",
            f"    {self.forward.dependent}~{self.forward.independent}: "
            f"beta={self.forward.beta:+.5f} (se {self.forward.beta_stderr:.5f})  "
            f"tau={self.forward.eg_stat:+.4f}  p={self.forward.eg_pvalue:.4f}",
            f"    {self.reverse.dependent}~{self.reverse.independent}: "
            f"beta={self.reverse.beta:+.5f} (se {self.reverse.beta_stderr:.5f})  "
            f"tau={self.reverse.eg_stat:+.4f}  p={self.reverse.eg_pvalue:.4f}",
            f"    orderings disagree @5%  : {self.orderings_disagree_at_5pct}",
            "",
            "  Critical value check (the trap)",
            f"    EG 5% crit              : {c.eg_crit_5pct:+.4f}",
            f"    naive ADF 5% crit       : {c.naive_adf_crit_5pct:+.4f}",
            f"    distortion              : {c.size_distortion:+.4f}",
            f"    naive test would falsely reject: {c.naive_would_falsely_reject}",
            "",
            "  Johansen",
            f"    trace  stat / 5% crit   : "
            f"{self.johansen_trace_stat:+.4f} / {self.johansen_trace_crit_5pct:+.4f}"
            if self.johansen_trace_stat is not None
            else "    trace                   : n/a",
            f"    maxeig stat / 5% crit   : "
            f"{self.johansen_maxeig_stat:+.4f} / {self.johansen_maxeig_crit_5pct:+.4f}"
            if self.johansen_maxeig_stat is not None
            else "    maxeig                  : n/a",
            f"    beta (normalised)       : {self.johansen_beta:+.5f}"
            if self.johansen_beta is not None
            else "",
            "",
            "  Mean-reversion characterisation",
            f"    OU half-life (bars)     : {self.half_life:.2f}"
            if self.half_life is not None and np.isfinite(self.half_life)
            else "    OU half-life            : undefined (b >= 0)",
            f"    half-life coef p-value  : {self.half_life_pvalue:.4g}"
            if self.half_life_pvalue is not None
            else "",
            f"    Hurst exponent          : {self.hurst:.4f}"
            if self.hurst is not None
            else "",
            f"    variance ratio / p      : {self.variance_ratio:.4f} / "
            f"{self.variance_ratio_pvalue:.4g}"
            if self.variance_ratio is not None
            else "",
            f"    spread KPSS p-value     : {self.spread_kpss_pvalue:.4g}"
            if self.spread_kpss_pvalue is not None
            else "",
        ]
        return "\n".join(l for l in lines if l is not None)


class CointegrationTester:
    """
    Engle-Granger two-step plus corroborating diagnostics.

    Parameters
    ----------
    use_log_prices : bool
        Test on log prices. For assets whose ratio is the economically
        meaningful quantity this is the right space; a cointegrating
        relationship in levels is not scale-invariant and will fail after a
        split or a large price move.
    adf_regression : {"n", "c", "ct"}
        Deterministic terms in the residual ADF regression. "n" is correct for
        OLS residuals, which are mean-zero by construction; "c" wastes a degree
        of freedom and lowers power.
    autolag : {"AIC", "BIC", "t-stat", None}
        Lag order selection for the ADF augmentation.
    maxlag : int, optional
        Defaults to Schwert's rule, floor(12 * (T/100)^(1/4)).
    """

    def __init__(
        self,
        use_log_prices: bool = True,
        adf_regression: Literal["n", "c", "ct"] = "n",
        autolag: Optional[str] = "AIC",
        maxlag: Optional[int] = None,
        johansen_det_order: int = 0,
        johansen_k_ar_diff: int = 1,
    ) -> None:
        self.use_log_prices = use_log_prices
        self.adf_regression = adf_regression
        self.autolag = autolag
        self.maxlag = maxlag
        self.johansen_det_order = johansen_det_order
        self.johansen_k_ar_diff = johansen_k_ar_diff

    # -- helpers ---------------------------------------------------------- #

    @staticmethod
    def _schwert_maxlag(n: int) -> int:
        return int(np.floor(12.0 * (n / 100.0) ** 0.25))

    def _prepare(self, s: pd.Series) -> pd.Series:
        s = pd.Series(s).astype(float).dropna()
        if self.use_log_prices:
            if (s <= 0).any():
                raise ValueError("non-positive values present; cannot take logs")
            s = np.log(s)
        return s

    # -- Engle-Granger ---------------------------------------------------- #

    def _engle_granger_one_way(
        self, y: pd.Series, x: pd.Series, name_y: str, name_x: str
    ) -> EngleGrangerDirection:
        X = sm.add_constant(x.values)
        ols = sm.OLS(y.values, X).fit()
        alpha, beta = float(ols.params[0]), float(ols.params[1])
        beta_se = float(ols.bse[1])
        resid = pd.Series(ols.resid, index=y.index)

        maxlag = self.maxlag or self._schwert_maxlag(len(resid))

        # Correctly sized test. statsmodels' `coint` applies MacKinnon's
        # response surface for the Engle-Granger null, accounting for the
        # estimated cointegrating vector.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eg_stat, eg_p, eg_crit = coint(
                y.values,
                x.values,
                trend="c",
                method="aeg",
                maxlag=maxlag,
                autolag=self.autolag,
            )

        # Naive ADF on the residual series, using standard Dickey-Fuller
        # critical values. Reported for contrast only.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_out = adfuller(
                resid.values,
                maxlag=maxlag,
                regression=self.adf_regression,
                autolag=self.autolag,
            )
        naive_stat, naive_p, used_lags, nobs, naive_crit = adf_out[:5]

        return EngleGrangerDirection(
            dependent=name_y,
            independent=name_x,
            alpha=alpha,
            beta=beta,
            beta_stderr=beta_se,
            r_squared=float(ols.rsquared),
            eg_stat=float(eg_stat),
            eg_pvalue=float(eg_p),
            eg_crit_1pct=float(eg_crit[0]),
            eg_crit_5pct=float(eg_crit[1]),
            eg_crit_10pct=float(eg_crit[2]),
            naive_adf_stat=float(naive_stat),
            naive_adf_pvalue=float(naive_p),
            naive_adf_crit_5pct=float(naive_crit["5%"]),
            adf_used_lags=int(used_lags),
            n_obs=int(nobs),
        )

    # -- Johansen --------------------------------------------------------- #

    def _johansen(self, y: pd.Series, x: pd.Series) -> Dict[str, Optional[float]]:
        """
        Johansen trace and maximum-eigenvalue tests.

        Advantages over Engle-Granger: invariant to the ordering of the
        series, estimates the cointegrating vector by maximum likelihood
        rather than OLS (removing the normalisation bias), and extends to
        k > 2 assets with an explicit rank hypothesis.
        """
        data = np.column_stack([y.values, x.values])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = coint_johansen(
                    data, det_order=self.johansen_det_order, k_ar_diff=self.johansen_k_ar_diff
                )
        except Exception:
            return dict.fromkeys(
                ["trace_stat", "trace_crit", "maxeig_stat", "maxeig_crit", "beta"], None
            )

        # r = 0 row; column index 1 is the 95% critical value
        evec = res.evec[:, 0]
        beta = float(-evec[1] / evec[0]) if abs(evec[0]) > _EPS else np.nan
        return {
            "trace_stat": float(res.lr1[0]),
            "trace_crit": float(res.cvt[0, 1]),
            "maxeig_stat": float(res.lr2[0]),
            "maxeig_crit": float(res.cvm[0, 1]),
            "beta": beta,
        }

    # -- mean reversion diagnostics --------------------------------------- #

    @staticmethod
    def half_life(spread: Sequence[float]) -> Tuple[float, float, float]:
        """
        Ornstein-Uhlenbeck half-life via the discrete AR(1) proxy.

            delta s_t = a + b * s_{t-1} + u_t
            H = -ln(2) / b        (defined only for b < 0)

        Returns (half_life, b, p_value_of_b). A half-life below ~1 bar is a
        microstructure artefact, not tradable mean reversion; above a few
        hundred bars the position is a buy-and-hold in disguise with far more
        leverage than the label suggests.
        """
        s = np.asarray(spread, dtype=float)
        s = s[np.isfinite(s)]
        if s.size < 20:
            return np.nan, np.nan, np.nan

        lag = s[:-1]
        delta = np.diff(s)
        X = sm.add_constant(lag)
        fit = sm.OLS(delta, X).fit()
        b = float(fit.params[1])
        p = float(fit.pvalues[1])
        if b >= -_EPS:
            return np.inf, b, p
        return float(-np.log(2.0) / b), b, p

    @staticmethod
    def hurst(series: Sequence[float], max_lag: int = 100) -> float:
        """
        Hurst exponent from the scaling of the variance of lagged differences.

            Var[s_{t+k} - s_t] ~ k^{2H}

        H < 0.5 mean reverting, H = 0.5 random walk, H > 0.5 trending.
        Sensitive to `max_lag` and to sample length; treat it as corroborating
        evidence, never as a standalone gate.
        """
        s = np.asarray(series, dtype=float)
        s = s[np.isfinite(s)]
        n = s.size
        if n < 100:
            return np.nan
        max_lag = int(min(max_lag, n // 2))
        lags = np.arange(2, max_lag)
        tau = []
        for k in lags:
            d = s[k:] - s[:-k]
            tau.append(np.sqrt(np.std(d)) if d.size else np.nan)
        tau = np.asarray(tau)
        ok = np.isfinite(tau) & (tau > 0)
        if ok.sum() < 10:
            return np.nan
        slope = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
        return float(slope * 2.0)

    @staticmethod
    def variance_ratio(series: Sequence[float], q: int = 5) -> Tuple[float, float]:
        """
        Lo-MacKinlay variance ratio with the heteroskedasticity-robust
        statistic. VR < 1 indicates mean reversion at horizon q.
        """
        s = np.asarray(series, dtype=float)
        s = s[np.isfinite(s)]
        n = s.size - 1
        if n < 5 * q:
            return np.nan, np.nan

        r = np.diff(s)
        mu = r.mean()
        var1 = np.sum((r - mu) ** 2) / (n - 1)

        m = q * (n - q + 1) * (1.0 - q / n)
        rq = s[q:] - s[:-q]
        varq = np.sum((rq - q * mu) ** 2) / m
        if var1 < _EPS:
            return np.nan, np.nan
        vr = varq / var1

        # heteroskedasticity-consistent variance of VR (Lo-MacKinlay z2)
        theta = 0.0
        e = (r - mu) ** 2
        denom = np.sum(e) ** 2
        for j in range(1, q):
            num = np.sum(e[j:] * e[:-j])
            delta = n * num / denom if denom > _EPS else 0.0
            theta += ((2.0 * (q - j) / q) ** 2) * delta
        if theta <= _EPS:
            return float(vr), np.nan
        z = (vr - 1.0) / np.sqrt(theta)
        return float(vr), float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    # -- public entry point ----------------------------------------------- #

    def test(
        self,
        series_y: pd.Series,
        series_x: pd.Series,
        name_y: str = "Y",
        name_x: str = "X",
        run_johansen: bool = True,
    ) -> CointegrationResult:
        y = self._prepare(series_y)
        x = self._prepare(series_x)
        idx = y.index.intersection(x.index)
        y, x = y.loc[idx], x.loc[idx]
        if len(idx) < 60:
            raise ValueError(f"insufficient overlapping observations: {len(idx)}")

        fwd = self._engle_granger_one_way(y, x, name_y, name_x)
        rev = self._engle_granger_one_way(x, y, name_x, name_y)

        joh = self._johansen(y, x) if run_johansen else dict.fromkeys(
            ["trace_stat", "trace_crit", "maxeig_stat", "maxeig_crit", "beta"], None
        )

        spread = (y - fwd.alpha - fwd.beta * x).values
        hl, _, hl_p = self.half_life(spread)
        h = self.hurst(spread)
        vr, vr_p = self.variance_ratio(spread)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kpss_p = float(kpss(spread, regression="c", nlags="auto")[1])
            except Exception:
                kpss_p = np.nan

        corr = float(np.corrcoef(np.diff(y.values), np.diff(x.values))[0, 1])

        return CointegrationResult(
            symbol_y=name_y,
            symbol_x=name_x,
            n_obs=len(idx),
            forward=fwd,
            reverse=rev,
            johansen_trace_stat=joh["trace_stat"],
            johansen_trace_crit_5pct=joh["trace_crit"],
            johansen_maxeig_stat=joh["maxeig_stat"],
            johansen_maxeig_crit_5pct=joh["maxeig_crit"],
            johansen_beta=joh["beta"],
            half_life=hl,
            half_life_pvalue=hl_p,
            hurst=h,
            variance_ratio=vr,
            variance_ratio_pvalue=vr_p,
            spread_kpss_pvalue=kpss_p,
            correlation=corr,
        )


# ===========================================================================
# 2. SPREAD CONSTRUCTION
# ===========================================================================

class HedgeRatioMethod(IntEnum):
    STATIC_OLS = 0
    TOTAL_LEAST_SQUARES = 1
    ROLLING_OLS = 2
    KALMAN = 3


@dataclass
class KalmanState:
    """Filter output for a time-varying hedge ratio."""

    alpha: pd.Series
    beta: pd.Series
    innovation: pd.Series          # e_t = y_t - F_t theta_{t|t-1}
    innovation_std: pd.Series      # sqrt(F_t P_{t|t-1} F_t' + R)
    z_score: pd.Series             # e_t / innovation_std  -- causal by design


class SpreadModel:
    """
    Constructs the tradable spread s_t = y_t - beta_t * x_t (- alpha_t).

    Method selection matters more than threshold tuning:

    STATIC_OLS
        Single beta over the whole sample. Uses future data at every point
        before the sample end, so it is invalid inside a backtest unless
        estimated strictly on a prior formation window.

    TOTAL_LEAST_SQUARES
        Orthogonal regression. OLS assumes x is measured without error; both
        legs are noisy prices, so OLS beta is attenuated toward zero by
        roughly var(x_true)/(var(x_true)+var(noise)). TLS removes that bias
        and is the correct estimator when the two series have comparable
        noise.

    ROLLING_OLS
        Trailing-window beta. Causal, but the window length trades adaptation
        against estimator variance, and beta jumps when an outlier enters or
        leaves the window.

    KALMAN
        Beta as a random walk. Smooth, causal, no window parameter, and the
        standardised innovation is itself a z-score requiring no separate
        lookback. `delta` controls responsiveness: 1e-5 is near-static, 1e-3
        tracks aggressively and will chase noise.
    """

    def __init__(self, use_log_prices: bool = True) -> None:
        self.use_log_prices = use_log_prices

    def _prep(self, s: pd.Series) -> pd.Series:
        s = pd.Series(s).astype(float)
        return np.log(s) if self.use_log_prices else s

    # -- estimators ------------------------------------------------------- #

    @staticmethod
    def static_ols(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        return float(fit.params[0]), float(fit.params[1])

    @staticmethod
    def total_least_squares(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
        """Orthogonal regression via the smallest principal component."""
        my, mx = y.mean(), x.mean()
        M = np.column_stack([x - mx, y - my])
        _, _, vt = np.linalg.svd(M, full_matrices=False)
        vx, vy = vt[-1]
        if abs(vy) < _EPS:
            raise np.linalg.LinAlgError("degenerate TLS solution")
        beta = float(-vx / vy)
        return float(my - beta * mx), beta

    def rolling_ols(
        self, y: pd.Series, x: pd.Series, window: int
    ) -> Tuple[pd.Series, pd.Series]:
        X = sm.add_constant(x.values)
        res = RollingOLS(y.values, X, window=window, min_nobs=window).fit()
        alpha = pd.Series(res.params[:, 0], index=y.index)
        beta = pd.Series(res.params[:, 1], index=y.index)
        return alpha, beta

    def kalman(
        self,
        y: pd.Series,
        x: pd.Series,
        delta: float = 1e-5,
        obs_var: Optional[float] = None,
        burn_in: int = 30,
    ) -> KalmanState:
        """
        Two-state Kalman filter, theta_t = [alpha_t, beta_t]'.

            theta_t = theta_{t-1} + w_t,   w ~ N(0, Q),  Q = delta/(1-delta) I
            y_t     = F_t theta_t + v_t,   v ~ N(0, R),  F_t = [1, x_t]
        """
        yv = y.values.astype(float)
        xv = x.values.astype(float)
        n = yv.size

        if obs_var is None:
            seed = min(max(burn_in, 30), n)
            a0, b0 = self.static_ols(yv[:seed], xv[:seed])
            obs_var = float(np.var(yv[:seed] - a0 - b0 * xv[:seed]))
            obs_var = max(obs_var, 1e-10)
        else:
            a0, b0 = 0.0, 1.0

        Q = (delta / (1.0 - delta)) * np.eye(2)
        R = float(obs_var)

        theta = np.array([a0, b0], dtype=float)
        P = np.eye(2) * 1.0

        alphas = np.full(n, np.nan)
        betas = np.full(n, np.nan)
        innov = np.full(n, np.nan)
        innov_sd = np.full(n, np.nan)

        for t in range(n):
            F = np.array([1.0, xv[t]])

            # predict
            P_pred = P + Q
            y_pred = float(F @ theta)

            e = yv[t] - y_pred
            S = float(F @ P_pred @ F.T + R)
            S = max(S, 1e-14)

            # update
            K = (P_pred @ F.T) / S
            theta = theta + K * e
            P = P_pred - np.outer(K, F @ P_pred)

            alphas[t] = theta[0]
            betas[t] = theta[1]
            innov[t] = e
            innov_sd[t] = np.sqrt(S)

        # burn-in is not yet converged; mask rather than silently trade on it
        alphas[:burn_in] = np.nan
        betas[:burn_in] = np.nan
        innov[:burn_in] = np.nan
        innov_sd[:burn_in] = np.nan

        z = pd.Series(innov / np.where(innov_sd > 0, innov_sd, np.nan), index=y.index)
        return KalmanState(
            alpha=pd.Series(alphas, index=y.index),
            beta=pd.Series(betas, index=y.index),
            innovation=pd.Series(innov, index=y.index),
            innovation_std=pd.Series(innov_sd, index=y.index),
            z_score=z,
        )

    # -- orchestration ---------------------------------------------------- #

    def build(
        self,
        series_y: pd.Series,
        series_x: pd.Series,
        method: HedgeRatioMethod = HedgeRatioMethod.KALMAN,
        window: int = 60,
        delta: float = 1e-5,
        formation_slice: Optional[slice] = None,
        include_intercept: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a frame with columns: y, x, alpha, beta, spread [, z_kalman].

        `formation_slice` restricts static estimators to an in-sample block so
        the resulting spread remains usable out of sample without look-ahead.
        """
        y = self._prep(series_y)
        x = self._prep(series_x)
        idx = y.index.intersection(x.index)
        y, x = y.loc[idx], x.loc[idx]

        out = pd.DataFrame({"y": y, "x": x})

        if method in (HedgeRatioMethod.STATIC_OLS, HedgeRatioMethod.TOTAL_LEAST_SQUARES):
            sl = formation_slice or slice(None)
            yf, xf = y.values[sl], x.values[sl]
            if method is HedgeRatioMethod.STATIC_OLS:
                a, b = self.static_ols(yf, xf)
            else:
                a, b = self.total_least_squares(yf, xf)
            out["alpha"] = a
            out["beta"] = b

        elif method is HedgeRatioMethod.ROLLING_OLS:
            a, b = self.rolling_ols(y, x, window)
            out["alpha"], out["beta"] = a, b

        elif method is HedgeRatioMethod.KALMAN:
            ks = self.kalman(y, x, delta=delta)
            out["alpha"], out["beta"] = ks.alpha, ks.beta
            out["z_kalman"] = ks.z_score
            out["innovation"] = ks.innovation

        else:
            raise ValueError(f"unknown method {method}")

        intercept = out["alpha"] if include_intercept else 0.0
        out["spread"] = out["y"] - intercept - out["beta"] * out["x"]
        return out


# ===========================================================================
# 3. Z-SCORE
# ===========================================================================

class ZScoreEngine:
    """
    Causal rolling standardisation of the spread.

    Two properties the naive `(s - s.rolling(w).mean()) / s.rolling(w).std()`
    formulation gets wrong:

    1. Self-inclusion. Including s_t in its own moments shrinks |z_t| by
       sqrt((w-1)/w) and, more importantly, lets the current observation move
       the threshold it is being compared against. Moments are lagged by one
       bar here.

    2. Window choice. A window materially shorter than the OU half-life
       standardises away the signal: the spread is measured against a mean
       that has already followed it. `window_from_half_life` derives the
       lookback from the estimated dynamics instead.
    """

    def __init__(
        self,
        window: int = 60,
        min_periods: Optional[int] = None,
        lag_moments: bool = True,
        vol_floor_quantile: float = 0.05,
        clip: Optional[float] = 10.0,
    ) -> None:
        self.window = int(window)
        self.min_periods = int(min_periods or window)
        self.lag_moments = lag_moments
        self.vol_floor_quantile = vol_floor_quantile
        self.clip = clip

    @staticmethod
    def window_from_half_life(half_life: float, multiplier: float = 3.0,
                              lo: int = 10, hi: int = 500) -> int:
        """
        Lookback as a multiple of the half-life. The spread decays to ~12.5%
        of a shock over three half-lives, so a 3x window covers roughly one
        full excursion-and-reversion cycle without averaging over several.
        """
        if not np.isfinite(half_life) or half_life <= 0:
            return lo
        return int(np.clip(round(multiplier * half_life), lo, hi))

    def compute(self, spread: pd.Series) -> pd.DataFrame:
        s = pd.Series(spread).astype(float)

        base = s.shift(1) if self.lag_moments else s
        mu = base.rolling(self.window, min_periods=self.min_periods).mean()
        sd = base.rolling(self.window, min_periods=self.min_periods).std(ddof=1)

        # Guard against a near-degenerate denominator: a quiet window produces
        # an sd near zero and a z-score that detonates on the next tick.
        finite = sd[np.isfinite(sd) & (sd > 0)]
        floor = float(finite.quantile(self.vol_floor_quantile)) if len(finite) else _EPS
        floor = max(floor, _EPS)
        sd_adj = sd.clip(lower=floor)

        z = (s - mu) / sd_adj
        if self.clip is not None:
            z = z.clip(-self.clip, self.clip)

        return pd.DataFrame(
            {"spread": s, "mean": mu, "std": sd_adj, "z": z,
             "vol_floor_binding": sd < floor}
        )


# ===========================================================================
# 4. SIGNAL STATE MACHINE
# ===========================================================================

class PositionState(IntEnum):
    FLAT = 0
    LONG_SPREAD = 1    # spread cheap: long y, short beta * x
    SHORT_SPREAD = -1  # spread rich: short y, long beta * x


@dataclass
class SignalConfig:
    """
    Threshold geometry.

    entry_z / exit_z form a hysteresis band. With exit_z = 0.0 the position
    is held to the full reversion but round-trips the entire band; a non-zero
    exit_z books the middle of the move and cuts time-in-market, which matters
    once financing and borrow are charged against holding period.

    stop_z is the divergence cut. Without it the strategy has unbounded loss:
    a cointegration break looks exactly like a large z-score, and adding to a
    stretched spread is the failure mode that ends pairs books.

    max_holding_half_lives converts a time stop into units of the pair's own
    dynamics. If the spread has not reverted in ~3 half-lives, the estimated
    half-life no longer describes the process.
    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_half_lives: float = 3.0
    half_life: float = 20.0
    min_bars_between_trades: int = 1
    recheck_cointegration: bool = True
    break_pvalue: float = 0.10


@dataclass
class TradeRecord:
    entry_index: int
    exit_index: Optional[int]
    direction: PositionState
    entry_z: float
    exit_z: Optional[float]
    exit_reason: Optional[str]
    beta_at_entry: float
    bars_held: Optional[int]


class SignalEngine:
    """
    Converts a z-score path into a position path with explicit exit reasons.

    Vectorised threshold comparison cannot express hysteresis, time stops, or
    path-dependent exits, so the loop is deliberate. Each bar's decision uses
    only information available at that bar.
    """

    def __init__(self, config: SignalConfig) -> None:
        if not (0.0 <= config.exit_z < config.entry_z < config.stop_z):
            raise ValueError("require 0 <= exit_z < entry_z < stop_z")
        self.cfg = config

    def _max_bars(self) -> float:
        hl = self.cfg.half_life
        if not np.isfinite(hl) or hl <= 0:
            return np.inf
        return self.cfg.max_holding_half_lives * hl

    def run(
        self,
        z: pd.Series,
        beta: Optional[pd.Series] = None,
        cointegration_pvalue: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, List[TradeRecord]]:
        zv = z.values.astype(float)
        n = zv.size
        bv = beta.values.astype(float) if beta is not None else np.ones(n)
        pv = (
            cointegration_pvalue.values.astype(float)
            if cointegration_pvalue is not None
            else np.zeros(n)
        )

        cfg = self.cfg
        max_bars = self._max_bars()

        position = np.zeros(n, dtype=np.int8)
        reason = np.empty(n, dtype=object)
        trades: List[TradeRecord] = []

        state = PositionState.FLAT
        entry_i: Optional[int] = None
        entry_z: float = np.nan
        entry_beta: float = np.nan
        last_exit_i = -10**9

        for t in range(n):
            zt = zv[t]
            broken = cfg.recheck_cointegration and np.isfinite(pv[t]) and pv[t] > cfg.break_pvalue

            if not np.isfinite(zt):
                position[t] = int(state)
                continue

            if state is PositionState.FLAT:
                cooled = (t - last_exit_i) >= cfg.min_bars_between_trades
                if cooled and not broken:
                    if zt <= -cfg.entry_z and zt > -cfg.stop_z:
                        state, entry_i, entry_z, entry_beta = (
                            PositionState.LONG_SPREAD, t, zt, bv[t],
                        )
                        reason[t] = "entry_long_spread"
                    elif zt >= cfg.entry_z and zt < cfg.stop_z:
                        state, entry_i, entry_z, entry_beta = (
                            PositionState.SHORT_SPREAD, t, zt, bv[t],
                        )
                        reason[t] = "entry_short_spread"
                    # |z| >= stop_z while flat: do not enter. A spread already
                    # past the stop is more likely broken than stretched.

            else:
                held = t - entry_i
                exit_reason: Optional[str] = None

                if broken:
                    exit_reason = "cointegration_break"
                elif held >= max_bars:
                    exit_reason = "time_stop"
                elif state is PositionState.LONG_SPREAD:
                    if zt <= -cfg.stop_z:
                        exit_reason = "stop_loss"
                    elif zt >= -cfg.exit_z:
                        exit_reason = "target"
                elif state is PositionState.SHORT_SPREAD:
                    if zt >= cfg.stop_z:
                        exit_reason = "stop_loss"
                    elif zt <= cfg.exit_z:
                        exit_reason = "target"

                if exit_reason is not None:
                    trades.append(
                        TradeRecord(
                            entry_index=entry_i,
                            exit_index=t,
                            direction=state,
                            entry_z=entry_z,
                            exit_z=zt,
                            exit_reason=exit_reason,
                            beta_at_entry=entry_beta,
                            bars_held=held,
                        )
                    )
                    reason[t] = exit_reason
                    state, entry_i, entry_z, entry_beta = PositionState.FLAT, None, np.nan, np.nan
                    last_exit_i = t

            position[t] = int(state)

        if state is not PositionState.FLAT and entry_i is not None:
            trades.append(
                TradeRecord(entry_i, None, state, entry_z, None, "open_at_end",
                            entry_beta, None)
            )

        frame = pd.DataFrame(
            {
                "z": z.values,
                "position": position,
                "beta": bv,
                "event": reason,
                # Position is applied to the NEXT bar's return. Multiplying a
                # position formed at t by the return realised at t books a
                # profit from information not yet available.
                "position_lagged": pd.Series(position, index=z.index).shift(1).fillna(0),
            },
            index=z.index,
        )
        return frame, trades

    @staticmethod
    def trade_statistics(trades: List[TradeRecord]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        rows = [
            {
                "entry": t.entry_index,
                "exit": t.exit_index,
                "direction": t.direction.name,
                "entry_z": t.entry_z,
                "exit_z": t.exit_z,
                "bars_held": t.bars_held,
                "reason": t.exit_reason,
            }
            for t in trades
        ]
        return pd.DataFrame(rows)


# ===========================================================================
# 5. UNIVERSE SCREENING
# ===========================================================================

class PairScreener:
    """
    Screens a universe with explicit multiple-testing control.

    On a universe of N assets there are N(N-1)/2 candidate pairs. At N = 100
    that is 4,950 tests; at a nominal 5% level roughly 250 independent pairs
    reject under a true null of no cointegration anywhere. Unadjusted
    screening therefore returns a list that is predominantly noise, and the
    subsequent backtest of that list is a backtest of the selection procedure.

    Benjamini-Hochberg controls the expected proportion of false discoveries
    among the pairs returned. Bonferroni controls the probability of any false
    discovery and is far more conservative.
    """

    def __init__(
        self,
        tester: Optional[CointegrationTester] = None,
        min_correlation: float = 0.7,
        fdr_alpha: float = 0.05,
        correction: Literal["bh", "bonferroni", "none"] = "bh",
    ) -> None:
        self.tester = tester or CointegrationTester()
        self.min_correlation = min_correlation
        self.fdr_alpha = fdr_alpha
        self.correction = correction

    @staticmethod
    def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
        """Returns (reject_mask, adaptive_threshold)."""
        p = np.asarray(pvalues, dtype=float)
        m = p.size
        order = np.argsort(p)
        ranked = p[order]
        thresholds = alpha * np.arange(1, m + 1) / m
        passing = np.where(ranked <= thresholds)[0]
        if passing.size == 0:
            return np.zeros(m, dtype=bool), 0.0
        k = passing.max()
        cutoff = ranked[k]
        return p <= cutoff, float(cutoff)

    def screen(
        self,
        prices: pd.DataFrame,
        formation_end: Optional[int] = None,
        run_johansen: bool = True,
        max_half_life: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        `formation_end` splits the sample. Cointegration is tested only on
        [0:formation_end]; the remainder is reserved for evaluation. Testing
        and trading on the same window is the single largest source of
        overstated stat-arb performance.
        """
        df = prices.dropna(axis=1, how="all")
        form = df.iloc[:formation_end] if formation_end else df
        cols = list(form.columns)

        logret = np.log(form).diff().dropna()
        corr = logret.corr()

        records: List[dict] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                c = float(corr.loc[a, b])
                if abs(c) < self.min_correlation:
                    continue
                try:
                    res = self.tester.test(
                        form[a], form[b], name_y=a, name_x=b, run_johansen=run_johansen
                    )
                except Exception as exc:  # degenerate series, gaps, etc.
                    warnings.warn(f"{a}/{b} skipped: {exc}")
                    continue

                cons = res.conservative
                records.append(
                    {
                        "y": a,
                        "x": b,
                        "correlation": c,
                        "eg_pvalue": cons.eg_pvalue,
                        "eg_stat": cons.eg_stat,
                        "beta": res.forward.beta,
                        "orderings_disagree": res.orderings_disagree_at_5pct,
                        "johansen_trace": res.johansen_trace_stat,
                        "johansen_trace_crit": res.johansen_trace_crit_5pct,
                        "half_life": res.half_life,
                        "hurst": res.hurst,
                        "variance_ratio": res.variance_ratio,
                        "naive_false_reject": cons.naive_would_falsely_reject,
                    }
                )

        if not records:
            return pd.DataFrame()

        out = pd.DataFrame(records)
        m = len(out)
        out["n_tests"] = m

        if self.correction == "bh":
            reject, cutoff = self.benjamini_hochberg(out["eg_pvalue"].values, self.fdr_alpha)
            out["significant"] = reject
            out["adjusted_threshold"] = cutoff
        elif self.correction == "bonferroni":
            cutoff = self.fdr_alpha / m
            out["significant"] = out["eg_pvalue"] <= cutoff
            out["adjusted_threshold"] = cutoff
        else:
            out["significant"] = out["eg_pvalue"] <= self.fdr_alpha
            out["adjusted_threshold"] = self.fdr_alpha

        if max_half_life is not None:
            out["significant"] &= out["half_life"].between(1.0, max_half_life)

        return out.sort_values("eg_pvalue").reset_index(drop=True)


# ===========================================================================
# DEMONSTRATIONS
# ===========================================================================

def _simulate_cointegrated_pair(
    n: int = 1500, beta: float = 1.4, half_life: float = 18.0, seed: int = 7
) -> Tuple[pd.Series, pd.Series]:
    """x is a random walk; the spread is a discretised OU process."""
    rng = np.random.default_rng(seed)
    theta = np.log(2.0) / half_life
    x = np.cumsum(rng.normal(0.0004, 0.012, n)) + np.log(100.0)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - theta * s[t - 1] + rng.normal(0.0, 0.010)
    y = 0.35 + beta * x + s
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(np.exp(y), idx, name="Y"), pd.Series(np.exp(x), idx, name="X")


def demo_critical_value_trap(n_trials: int = 300, n_obs: int = 500, seed: int = 0) -> None:
    """
    Two INDEPENDENT random walks -- no cointegration exists by construction.

    Empirical rejection rate at a nominal 5% level, naive ADF on the OLS
    residual versus the correctly sized Engle-Granger test. The naive
    procedure should reject far more often than 5%.
    """
    rng = np.random.default_rng(seed)
    naive_rej = eg_rej = 0
    for _ in range(n_trials):
        a = np.cumsum(rng.normal(0, 1, n_obs))
        b = np.cumsum(rng.normal(0, 1, n_obs))
        resid = sm.OLS(a, sm.add_constant(b)).fit().resid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            naive_p = adfuller(resid, regression="n", autolag="AIC")[1]
            eg_p = coint(a, b, trend="c", autolag="AIC")[1]
        naive_rej += naive_p < 0.05
        eg_rej += eg_p < 0.05

    print("Independent random walks, nominal size 5%")
    print(f"  naive ADF on residuals rejects : {100*naive_rej/n_trials:5.1f}% of trials")
    print(f"  Engle-Granger (MacKinnon)      : {100*eg_rej/n_trials:5.1f}% of trials")
    print("  -> the first number is the false-positive rate of the common implementation")


def demo_pipeline() -> None:
    y, x = _simulate_cointegrated_pair()

    tester = CointegrationTester(use_log_prices=True, adf_regression="n")
    res = tester.test(y, x, "Y", "X")
    print(res.summary())
    print()
    print(f"  passes gate: {res.passes(max_half_life=120)}")
    print()

    model = SpreadModel(use_log_prices=True)
    frame = model.build(y, x, method=HedgeRatioMethod.KALMAN, delta=1e-5)

    w = ZScoreEngine.window_from_half_life(res.half_life, multiplier=3.0)
    print(f"  half-life {res.half_life:.1f} bars -> z-score window {w}")

    zf = ZScoreEngine(window=w, lag_moments=True).compute(frame["spread"])

    cfg = SignalConfig(
        entry_z=2.0, exit_z=0.5, stop_z=3.5,
        half_life=res.half_life, max_holding_half_lives=3.0,
    )
    signals, trades = SignalEngine(cfg).run(zf["z"], beta=frame["beta"])

    stats_df = SignalEngine.trade_statistics(trades)
    print(f"  trades: {len(trades)}")
    if not stats_df.empty:
        print(f"  median bars held: {stats_df['bars_held'].median():.0f} "
              f"(half-life {res.half_life:.1f})")
        print("  exit reason mix:")
        for k, v in stats_df["reason"].value_counts().items():
            print(f"    {k:<22}: {v}")
        print()
        print("  Rolling window z vs Kalman innovation z, correlation: "
              f"{zf['z'].corr(frame['z_kalman']):.4f}")


if __name__ == "__main__":
    demo_critical_value_trap()
    print("\n" + "=" * 72 + "\n")
    demo_pipeline()
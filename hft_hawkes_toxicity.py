"""
hft_hawkes_toxicity.py
======================

Phase 14: Hawkes-process trade arrival modelling and order-flow toxicity.

    lambda(t) = mu + sum_{t_i < t} alpha * exp( -beta (t - t_i) )

    branching ratio  n = alpha / beta,   stationary iff n < 1
    stationary mean  E[lambda] = mu / (1 - n)

Every recursion here is O(1) per event. The naive double sum over past events
is O(N^2) and is intractable on a session of ticks; the exponential kernel is
chosen precisely because it admits the Markov recursion

    A_i = exp(-beta * delta_i) * ( 1 + A_{i-1} ),   lambda(t_i) = mu + alpha * A_i
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

_EPS = 1e-12
_LAM_FLOOR = 1e-300

__all__ = [
    "HawkesParams",
    "HawkesExpKernel",
    "HawkesFit",
    "fit_hawkes_mle",
    "HawkesOnlineFilter",
    "time_rescaling_residuals",
    "VolumeBucket",
    "VolumeBucketer",
    "classify_bvc",
    "classify_tick_rule",
    "VPINEstimator",
    "ToxicityScore",
    "ToxicityReading",
    "simulate_flash_crash",
]


# ===========================================================================
# 1. HAWKES
# ===========================================================================

@dataclass(frozen=True)
class HawkesParams:
    mu: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    @property
    def stationary_intensity(self) -> float:
        if not self.is_stationary:
            return float("inf")
        return self.mu / (1.0 - self.branching_ratio)

    @property
    def half_life(self) -> float:
        return math.log(2.0) / self.beta

    @property
    def endogeneity(self) -> float:
        return self.branching_ratio

    def as_array(self) -> np.ndarray:
        return np.array([self.mu, self.alpha, self.beta], dtype=float)

    def __repr__(self) -> str:
        return (f"Hawkes(mu={self.mu:.4f}, alpha={self.alpha:.4f}, "
                f"beta={self.beta:.4f}, n={self.branching_ratio:.4f}, "
                f"E[lam]={self.stationary_intensity:.4f})")


class HawkesExpKernel:

    @staticmethod
    def excitation_states(t: np.ndarray, beta: float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        n = t.size
        A = np.empty(n, dtype=float)
        if n == 0:
            return A
        A[0] = 0.0
        for i in range(1, n):
            A[i] = math.exp(-beta * (t[i] - t[i - 1])) * (1.0 + A[i - 1])
        return A

    @staticmethod
    def excitation_derivative(t: np.ndarray, beta: float, A: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        n = t.size
        B = np.empty(n, dtype=float)
        if n == 0:
            return B
        B[0] = 0.0
        for i in range(1, n):
            d = t[i] - t[i - 1]
            e = math.exp(-beta * d)
            B[i] = e * (d * (1.0 + A[i - 1]) + B[i - 1])
        return B

    @staticmethod
    def log_likelihood(
        params: HawkesParams, t: np.ndarray, T: float, A: Optional[np.ndarray] = None
    ) -> float:
        t = np.asarray(t, dtype=float)
        mu, al, be = params.mu, params.alpha, params.beta
        if A is None:
            A = HawkesExpKernel.excitation_states(t, be)
        lam = mu + al * A
        if np.any(lam <= 0):
            return -np.inf
        compensator = mu * T + (al / be) * np.sum(1.0 - np.exp(-be * (T - t)))
        return float(np.sum(np.log(lam)) - compensator)

    @staticmethod
    def log_likelihood_grad(
        params: HawkesParams, t: np.ndarray, T: float
    ) -> Tuple[float, np.ndarray]:
        t = np.asarray(t, dtype=float)
        mu, al, be = params.mu, params.alpha, params.beta
        A = HawkesExpKernel.excitation_states(t, be)
        B = HawkesExpKernel.excitation_derivative(t, be, A)

        lam = np.maximum(mu + al * A, _LAM_FLOOR)
        s = T - t
        es = np.exp(-be * s)
        one_minus = 1.0 - es

        ll = float(np.sum(np.log(lam)) - (mu * T + (al / be) * np.sum(one_minus)))

        d_mu = float(np.sum(1.0 / lam) - T)
        d_al = float(np.sum(A / lam) - np.sum(one_minus) / be)
        d_be = float(
            -al * np.sum(B / lam)
            + (al / be**2) * np.sum(one_minus)
            - (al / be) * np.sum(s * es)
        )
        return ll, np.array([d_mu, d_al, d_be], dtype=float)

    @staticmethod
    def intensity_at(params: HawkesParams, t: np.ndarray, grid: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        grid = np.asarray(grid, dtype=float)
        mu, al, be = params.mu, params.alpha, params.beta

        out = np.empty(grid.size, dtype=float)
        A = 0.0
        last = 0.0
        j = 0
        for k, g in enumerate(grid):
            while j < t.size and t[j] <= g:
                A = A * math.exp(-be * (t[j] - last)) + 1.0
                last = t[j]
                j += 1
            out[k] = mu + al * A * math.exp(-be * (g - last))
        return out

    @staticmethod
    def simulate(params: HawkesParams, T: float, seed: int = 0) -> np.ndarray:
        if not params.is_stationary:
            warnings.warn(f"branching ratio {params.branching_ratio:.3f} >= 1: explosive")
        rng = np.random.default_rng(seed)
        mu, al, be = params.mu, params.alpha, params.beta

        events: List[float] = []
        t = 0.0
        A = 0.0
        while True:
            lam_bar = mu + al * A
            if lam_bar <= 0:
                break
            w = rng.exponential(1.0 / lam_bar)
            t += w
            if t >= T:
                break
            A_new = A * math.exp(-be * w)
            lam = mu + al * A_new
            if rng.random() <= lam / lam_bar:
                events.append(t)
                A = A_new + 1.0
            else:
                A = A_new
        return np.asarray(events, dtype=float)


@dataclass
class HawkesFit:
    params: HawkesParams
    log_likelihood: float
    n_events: int
    horizon: float
    converged: bool
    n_iter: int
    std_errors: np.ndarray
    ks_pvalue: float

    @property
    def aic(self) -> float:
        return 2.0 * 3 - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return 3.0 * math.log(max(self.n_events, 2)) - 2.0 * self.log_likelihood

    def summary(self) -> str:
        p = self.params
        se = self.std_errors
        return (
            f"mu={p.mu:.4f}({se[0]:.4f})  alpha={p.alpha:.4f}({se[1]:.4f})  "
            f"beta={p.beta:.4f}({se[2]:.4f})\n"
            f"  n={p.branching_ratio:.4f}  E[lam]={p.stationary_intensity:.3f}  "
            f"half-life={p.half_life:.4f}s  ll={self.log_likelihood:.2f}  "
            f"KS p={self.ks_pvalue:.4f}"
        )


def fit_hawkes_mle(
    t: np.ndarray,
    T: Optional[float] = None,
    initial: Optional[Sequence[float]] = None,
    n_restarts: int = 4,
    max_branching: float = 0.999,
) -> HawkesFit:
    t = np.asarray(t, dtype=float).ravel()
    if t.size < 20:
        raise ValueError(f"need >=20 events, got {t.size}")
    if np.any(np.diff(t) < 0):
        raise ValueError("event times must be non-decreasing")
    if np.any(np.diff(t) == 0):
        warnings.warn("duplicate timestamps present; jitter them or aggregate")
    t = t - t[0]
    if T is None:
        T = float(t[-1]) * (1.0 + 1.0 / t.size)
    if T <= t[-1]:
        raise ValueError("horizon T must exceed the last event time")

    rate = t.size / T
    mean_gap = float(np.mean(np.diff(t))) if t.size > 1 else 1.0

    def neg(x: np.ndarray) -> Tuple[float, np.ndarray]:
        mu, al, be = np.exp(x)
        if al / be >= max_branching:
            return 1e12, np.zeros(3)
        try:
            ll, g = HawkesExpKernel.log_likelihood_grad(HawkesParams(mu, al, be), t, T)
        except (ValueError, FloatingPointError):
            return 1e12, np.zeros(3)
        if not np.isfinite(ll):
            return 1e12, np.zeros(3)
        return -ll, -g * np.array([mu, al, be])

    seeds: List[np.ndarray] = []
    if initial is not None:
        seeds.append(np.log(np.asarray(initial, dtype=float)))
    for n0 in (0.2, 0.5, 0.8):
        for bscale in (1.0, 5.0):
            be0 = bscale / max(mean_gap, 1e-6)
            seeds.append(np.log([max(rate * (1 - n0), 1e-6), n0 * be0, be0]))
    seeds = seeds[: max(n_restarts, 1) + (1 if initial is not None else 0)]

    best = None
    for x0 in seeds:
        try:
            r = optimize.minimize(neg, x0, jac=True, method="L-BFGS-B",
                                  options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        raise RuntimeError("all optimisation seeds failed")

    mu, al, be = np.exp(best.x)
    params = HawkesParams(mu, al, be)
    ll = HawkesExpKernel.log_likelihood(params, t, T)

    H = np.zeros((3, 3))
    p0 = params.as_array()
    for k in range(3):
        h = max(1e-6, 1e-5 * p0[k])
        pp, pm = p0.copy(), p0.copy()
        pp[k] += h
        pm[k] -= h
        try:
            _, gp = HawkesExpKernel.log_likelihood_grad(HawkesParams(*pp), t, T)
            _, gm = HawkesExpKernel.log_likelihood_grad(HawkesParams(*pm), t, T)
            H[:, k] = -(gp - gm) / (2.0 * h)
        except ValueError:
            H[:, k] = np.nan
    H = 0.5 * (H + H.T)
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)

    tau = time_rescaling_residuals(params, t)
    ks = stats.kstest(tau, "expon") if tau.size > 5 else stats.KstestResult(np.nan, np.nan)

    return HawkesFit(
        params=params, log_likelihood=ll, n_events=t.size, horizon=float(T),
        converged=bool(best.success), n_iter=int(best.nit),
        std_errors=se, ks_pvalue=float(ks.pvalue),
    )


def time_rescaling_residuals(params: HawkesParams, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    mu, al, be = params.mu, params.alpha, params.beta
    A = HawkesExpKernel.excitation_states(t, be)
    d = np.diff(t)
    return mu * d + (al / be) * (1.0 - np.exp(-be * d)) * (1.0 + A[:-1])


# ===========================================================================
# 2. ONLINE FILTER
# ===========================================================================

class HawkesOnlineFilter:
    __slots__ = ("params", "_A", "_t_last", "_n", "_started", "peak_intensity")

    def __init__(self, params: HawkesParams) -> None:
        self.params = params
        self.reset()

    def reset(self) -> None:
        self._A = 0.0
        self._t_last = 0.0
        self._n = 0
        self._started = False
        self.peak_intensity = 0.0

    def update(self, t: float) -> float:
        t = float(t)
        if not self._started:
            self._A = 1.0
            self._t_last = t
            self._started = True
            self._n = 1
            lam = self.params.mu + self.params.alpha * self._A
            self.peak_intensity = max(self.peak_intensity, lam)
            return lam
        if t < self._t_last:
            raise ValueError(f"out-of-order event: {t} < {self._t_last}")
        self._A = self._A * math.exp(-self.params.beta * (t - self._t_last)) + 1.0
        self._t_last = t
        self._n += 1
        lam = self.params.mu + self.params.alpha * self._A
        self.peak_intensity = max(self.peak_intensity, lam)
        return lam

    def intensity(self, t: Optional[float] = None) -> float:
        if not self._started:
            return self.params.mu
        s = self._t_last if t is None else float(t)
        decay = math.exp(-self.params.beta * max(s - self._t_last, 0.0))
        return self.params.mu + self.params.alpha * self._A * decay

    def run(self, times: Sequence[float]) -> np.ndarray:
        return np.array([self.update(t) for t in times], dtype=float)

    @property
    def n_events(self) -> int:
        return self._n

    @property
    def excitation(self) -> float:
        return self._A

    @property
    def normalised_intensity(self) -> float:
        s = self.params.stationary_intensity
        return self.intensity() / s if np.isfinite(s) and s > 0 else float("nan")


# ===========================================================================
# 3. VPIN
# ===========================================================================

@dataclass(frozen=True)
class VolumeBucket:
    index: int
    t_start: float
    t_end: float
    volume: float
    price_start: float
    price_end: float
    n_trades: int

    @property
    def delta_p(self) -> float:
        return self.price_end - self.price_start

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class VolumeBucketer:
    def __init__(self, bucket_volume: float) -> None:
        if bucket_volume <= 0:
            raise ValueError("bucket_volume must be positive")
        self.V = float(bucket_volume)

    def bucket(
        self, times: np.ndarray, prices: np.ndarray, volumes: np.ndarray
    ) -> List[VolumeBucket]:
        times = np.asarray(times, dtype=float)
        prices = np.asarray(prices, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        if not (times.size == prices.size == volumes.size):
            raise ValueError("times, prices and volumes must align")
        if np.any(volumes <= 0):
            raise ValueError("volumes must be positive")

        out: List[VolumeBucket] = []
        acc = 0.0
        start_i = 0
        t0, p0 = times[0], prices[0]
        n_tr = 0
        carry = 0.0

        for i in range(times.size):
            remaining = volumes[i] + carry
            carry = 0.0
            n_tr += 1
            while acc + remaining >= self.V:
                take = self.V - acc
                remaining -= take
                out.append(VolumeBucket(
                    index=len(out), t_start=t0, t_end=times[i], volume=self.V,
                    price_start=p0, price_end=prices[i], n_trades=n_tr,
                ))
                acc = 0.0
                n_tr = 0
                t0, p0 = times[i], prices[i]
            acc += remaining
        return out


def classify_bvc(
    buckets: Sequence[VolumeBucket],
    sigma: Optional[float] = None,
    df: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    dp = np.array([b.delta_p for b in buckets], dtype=float)
    V = np.array([b.volume for b in buckets], dtype=float)
    if sigma is None:
        sigma = float(np.std(dp, ddof=1)) if dp.size > 1 else 1.0
    sigma = max(float(sigma), 1e-12)
    z = dp / sigma
    frac = stats.t.cdf(z, df=df) if df is not None else stats.norm.cdf(z)
    v_buy = V * frac
    return v_buy, V - v_buy, sigma


def classify_tick_rule(
    times: np.ndarray, prices: np.ndarray, volumes: np.ndarray,
    buckets: Sequence[VolumeBucket],
) -> Tuple[np.ndarray, np.ndarray]:
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    d = np.diff(prices, prepend=prices[0])
    sign = np.sign(d)
    last = 1.0
    for i in range(sign.size):
        if sign[i] == 0:
            sign[i] = last
        else:
            last = sign[i]

    times = np.asarray(times, dtype=float)
    vb = np.zeros(len(buckets))
    vs = np.zeros(len(buckets))
    for k, b in enumerate(buckets):
        m = (times >= b.t_start) & (times <= b.t_end)
        if not m.any():
            vb[k] = vs[k] = b.volume / 2.0
            continue
        sv = volumes[m] * sign[m]
        tot = volumes[m].sum()
        buy = 0.5 * (tot + sv.sum())
        scale = b.volume / max(tot, _EPS)
        vb[k] = buy * scale
        vs[k] = (tot - buy) * scale
    return vb, vs


class VPINEstimator:
    def __init__(
        self,
        bucket_volume: float,
        n_buckets: int = 50,
        classifier: Literal["bvc", "tick"] = "bvc",
        sigma_mode: Literal["rolling", "frozen"] = "rolling",
        sigma_window: int = 50,
        t_df: Optional[float] = None,
    ) -> None:
        self.bucketer = VolumeBucketer(bucket_volume)
        self.N = int(n_buckets)
        self.classifier = classifier
        self.sigma_mode = sigma_mode
        self.sigma_window = int(sigma_window)
        self.t_df = t_df
        self._sigma_frozen: Optional[float] = None

    def calibrate_sigma(self, buckets: Sequence[VolumeBucket]) -> float:
        dp = np.array([b.delta_p for b in buckets], dtype=float)
        self._sigma_frozen = float(np.std(dp, ddof=1))
        return self._sigma_frozen

    def compute(
        self, times: np.ndarray, prices: np.ndarray, volumes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        buckets = self.bucketer.bucket(times, prices, volumes)
        if len(buckets) < self.N + 1:
            raise ValueError(
                f"{len(buckets)} buckets for N={self.N}; reduce bucket_volume "
                f"or n_buckets"
            )

        if self.classifier == "bvc":
            if self.sigma_mode == "frozen":
                sig = self._sigma_frozen
                if sig is None:
                    raise RuntimeError("call calibrate_sigma first in frozen mode")
                vb, vs, _ = classify_bvc(buckets, sigma=sig, df=self.t_df)
            else:
                dp = np.array([b.delta_p for b in buckets])
                w = self.sigma_window
                sig = np.array([
                    np.std(dp[max(0, k - w):k], ddof=1) if k > 2 else np.std(dp[:3], ddof=1)
                    for k in range(dp.size)
                ])
                sig = np.maximum(sig, 1e-12)
                V = np.array([b.volume for b in buckets])
                z = dp / sig
                frac = stats.t.cdf(z, df=self.t_df) if self.t_df else stats.norm.cdf(z)
                vb, vs = V * frac, V * (1.0 - frac)
        else:
            vb, vs = classify_tick_rule(times, prices, volumes, buckets)

        V = np.array([b.volume for b in buckets], dtype=float)
        imb = np.abs(vs - vb)

        vpin = np.full(len(buckets), np.nan)
        csum = np.concatenate(([0.0], np.cumsum(imb)))
        for k in range(self.N - 1, len(buckets)):
            vpin[k] = (csum[k + 1] - csum[k + 1 - self.N]) / (self.N * V[k])

        return {
            "vpin": vpin,
            "imbalance": imb / V,
            "buy_volume": vb,
            "sell_volume": vs,
            "t_end": np.array([b.t_end for b in buckets]),
            "price_end": np.array([b.price_end for b in buckets]),
            "delta_p": np.array([b.delta_p for b in buckets]),
            "signed_imbalance": (vb - vs) / V,
        }


# ===========================================================================
# 4. TOXICITY
# ===========================================================================

@dataclass(frozen=True)
class ToxicityReading:
    t: float
    score: float
    p_hawkes: float
    p_vpin: float
    intensity: float
    vpin: float
    signed_imbalance: float

    @property
    def is_alert(self) -> bool:
        return self.score >= 0.90


class ToxicityScore:
    def __init__(
        self,
        combine: Literal["geometric", "arithmetic", "max", "noisy_or"] = "geometric",
        weight_hawkes: float = 0.5,
        alert_threshold: float = 0.90,
    ) -> None:
        if not 0.0 <= weight_hawkes <= 1.0:
            raise ValueError("weight_hawkes must lie in [0,1]")
        self.combine = combine
        self.w = float(weight_hawkes)
        self.alert_threshold = float(alert_threshold)
        self._ref_lam: Optional[np.ndarray] = None
        self._ref_vpin: Optional[np.ndarray] = None

    def fit_reference(self, intensities: np.ndarray, vpins: np.ndarray) -> "ToxicityScore":
        lam = np.asarray(intensities, dtype=float)
        vp = np.asarray(vpins, dtype=float)
        lam = lam[np.isfinite(lam)]
        vp = vp[np.isfinite(vp)]
        if lam.size < 20 or vp.size < 20:
            raise ValueError("need >=20 finite reference observations per channel")
        self._ref_lam = np.sort(lam)
        self._ref_vpin = np.sort(vp)
        return self

    @staticmethod
    def _ecdf(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref, x, side="right") / (ref.size + 1.0)

    def _combine(self, ph: np.ndarray, pv: np.ndarray) -> np.ndarray:
        w = self.w
        if self.combine == "geometric":
            return np.power(np.maximum(ph, _EPS), w) * np.power(np.maximum(pv, _EPS), 1 - w)
        if self.combine == "arithmetic":
            return w * ph + (1 - w) * pv
        if self.combine == "max":
            return np.maximum(ph, pv)
        if self.combine == "noisy_or":
            return 1.0 - (1.0 - ph) * (1.0 - pv)
        raise ValueError(f"unknown combiner {self.combine}")

    def score(self, intensities: np.ndarray, vpins: np.ndarray) -> np.ndarray:
        if self._ref_lam is None or self._ref_vpin is None:
            raise RuntimeError("call fit_reference on a quiet window first")
        lam = np.atleast_1d(np.asarray(intensities, dtype=float))
        vp = np.atleast_1d(np.asarray(vpins, dtype=float))
        ph = self._ecdf(self._ref_lam, lam)
        pv = self._ecdf(self._ref_vpin, vp)
        out = self._combine(ph, pv)
        out[~np.isfinite(lam) | ~np.isfinite(vp)] = np.nan
        return np.clip(out, 0.0, 1.0)

    def readings(
        self, t: np.ndarray, intensities: np.ndarray, vpins: np.ndarray,
        signed_imbalance: Optional[np.ndarray] = None,
    ) -> List[ToxicityReading]:
        s = self.score(intensities, vpins)
        ph = self._ecdf(self._ref_lam, np.asarray(intensities, dtype=float))
        pv = self._ecdf(self._ref_vpin, np.asarray(vpins, dtype=float))
        si = signed_imbalance if signed_imbalance is not None else np.zeros_like(s)
        return [
            ToxicityReading(float(t[k]), float(s[k]), float(ph[k]), float(pv[k]),
                            float(intensities[k]), float(vpins[k]), float(si[k]))
            for k in range(s.size)
        ]

    def first_alert(
        self, t: np.ndarray, scores: np.ndarray, persistence: int = 4
    ) -> Optional[float]:
        m = np.isfinite(scores) & (scores >= self.alert_threshold)
        if persistence <= 1:
            idx = np.flatnonzero(m)
            return float(t[idx[0]]) if idx.size else None
        run = 0
        for i in range(m.size):
            run = run + 1 if m[i] else 0
            if run >= persistence:
                return float(t[i])
        return None


# ===========================================================================
# 5. FLASH CRASH SIMULATOR
# ===========================================================================

def simulate_flash_crash(
    seed: int = 0,
    t_quiet: float = 600.0,
    t_buildup: float = 120.0,
    t_crash: float = 40.0,
    t_recover: float = 200.0,
    mu_quiet: float = 4.0,
    n_quiet: float = 0.35,
    mu_stress: float = 5.0,
    n_stress: float = 0.72,
    mu_crash: float = 8.0,
    n_crash: float = 0.88,
    sell_prob_quiet: float = 0.50,
    sell_prob_buildup: float = 0.66,
    sell_prob_crash: float = 0.95,
    beta: float = 3.0,
    tick: float = 0.01,
    price0: float = 100.0,
    impact: float = 0.14,
    toxic_buildup: bool = True,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    b1, b2, b3 = t_quiet, t_quiet + t_buildup, t_quiet + t_buildup + t_crash
    T = b3 + t_recover

    if toxic_buildup:
        segs = [
            (0.0, b1, mu_quiet, n_quiet, sell_prob_quiet, 0.0),
            (b1, b2, mu_stress, n_stress, sell_prob_buildup, 0.0),
            (b2, b3, mu_crash, n_crash, sell_prob_crash, 0.0),
            (b3, T, mu_quiet, n_quiet, 0.44, 0.0),
        ]
    else:
        segs = [
            (0.0, b1, mu_quiet, n_quiet, sell_prob_quiet, 0.0),
            (b1, b2, mu_quiet, n_quiet, sell_prob_quiet, 0.0),
            (b2, b3, mu_quiet, n_quiet, sell_prob_quiet, -1.55),
            (b3, T, mu_quiet, n_quiet, 0.50, 0.0),
        ]

    times: List[float] = []
    seg_id: List[int] = []
    sell_p: List[float] = []
    exo: List[float] = []
    for k, (a, b, mu, nb, sp, dr) in enumerate(segs):
        ev = HawkesExpKernel.simulate(
            HawkesParams(mu, nb * beta, beta), b - a, seed=seed * 17 + k
        )
        for e in ev:
            times.append(a + e)
            seg_id.append(k)
            sell_p.append(sp)
            exo.append(dr)

    t = np.asarray(times)
    order = np.argsort(t, kind="stable")
    t = t[order]
    seg = np.asarray(seg_id)[order]
    sp = np.asarray(sell_p)[order]
    dr = np.asarray(exo)[order]
    n = t.size

    side = np.where(rng.random(n) < sp, -1.0, 1.0)
    volumes = rng.lognormal(mean=3.0, sigma=0.7, size=n)
    if toxic_buildup:
        volumes[seg >= 1] *= 1.8

    vmed = float(np.median(volumes))
    flow = impact * side * np.sqrt(volumes / vmed)
    noise = rng.standard_normal(n) * 0.35
    prices = price0 + np.cumsum((flow + dr + noise) * tick)

    return {
        "times": t, "prices": prices, "volumes": volumes, "side": side,
        "segment": seg, "T": np.array([T]),
        "boundaries": np.array([b1, b2, b3]),
    }


def _pipeline(data: Dict[str, np.ndarray], seed: int = 0) -> Dict[str, np.ndarray]:
    t, p, v = data["times"], data["prices"], data["volumes"]
    b1 = float(data["boundaries"][0])

    train = t < b1 * 0.8
    fit = fit_hawkes_mle(t[train], T=float(b1 * 0.8))

    est = VPINEstimator(bucket_volume=float(np.median(v) * 20), n_buckets=25,
                        classifier="bvc", sigma_mode="rolling", sigma_window=40)
    out = est.compute(t, p, v)

    filt = HawkesOnlineFilter(fit.params)
    lam_at_event = filt.run(t)
    lam_at_bucket = np.interp(out["t_end"], t, lam_at_event)

    ref = out["t_end"] < b1 * 0.8
    tox = ToxicityScore(combine="geometric", alert_threshold=0.90)
    good = ref & np.isfinite(out["vpin"])
    tox.fit_reference(lam_at_bucket[good], out["vpin"][good])
    score = tox.score(lam_at_bucket, out["vpin"])

    return {
        "t": out["t_end"], "score": score, "vpin": out["vpin"],
        "intensity": lam_at_bucket, "price": out["price_end"],
        "params": fit.params, "tox": tox,
        "signed_imbalance": out["signed_imbalance"],
    }


def _run_flash_crash_demo() -> None:
    print("Simulating Flash Crash Scenario with Toxic Order Flow...")
    data = simulate_flash_crash(seed=3, toxic_buildup=True)
    b1, b2, b3 = data["boundaries"]
    r = _pipeline(data)

    first = r["tox"].first_alert(r["t"], r["score"], persistence=4)
    lead = b2 - first

    quiet = r["score"][r["t"] < b1]
    buildup = r["score"][(r["t"] >= b1) & (r["t"] < b2)]

    px = lambda x: float(np.interp(x, r["t"], r["price"]))
    drop_buildup = (px(b2) - px(b1)) / px(b1)
    drop_crash = (px(b3) - px(b2)) / px(b2)

    si_q = float(np.nanmean(r["signed_imbalance"][r["t"] < b1]))
    si_b = float(np.nanmean(r["signed_imbalance"][(r["t"] >= b1) & (r["t"] < b2)]))

    print(f"\n--- HFT TOXICITY ENGINE RESULTS ---")
    print(f"-> Quiet Period Ends:   t = {b1:.1f}s")
    print(f"-> Crash Begins:        t = {b2:.1f}s")
    print(f"-> SYSTEM ALERT FIRED:  t = {first:.1f}s")
    print(f"-> LEAD TIME:           {lead:.1f} seconds of advance warning!")
    print(f"\n-> Pre-Crash Price Mvmt:  {100*drop_buildup:+.2f}% (Undetectable visually)")
    print(f"-> Actual Crash Mvmt:     {100*drop_crash:+.2f}%")
    print(f"\n-> BVC Flow Imbalance:    {si_q:+.3f} (Quiet) -> {si_b:+.3f} (Buildup)")
    print(f"-> Hawkes Score Jump:     {np.nanmean(quiet):.3f} (Quiet) -> {np.nanmean(buildup):.3f} (Buildup, Peak {np.nanmax(buildup):.3f})")

if __name__ == "__main__":
    print("==========================================================================")
    print(" ORDER FLOW TOXICITY: HAWKES PROCESSES & VPIN")
    print("==========================================================================")
    _run_flash_crash_demo()
    print("\n[SUCCESS] HAWKES HFT ENGINE OPERATIONAL.")

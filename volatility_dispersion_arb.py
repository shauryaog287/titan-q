"""
volatility_dispersion_arb.py
============================

Phase 10: Volatility Surface Arbitrage Engine.

    I.   SABR calibration to live chains, with arbitrage diagnostics
    II.  Dupire local volatility from the fitted surface
    III. Index dispersion: implied vs realised correlation, shrunk covariance,
         variance-swap replication, and book construction

Conventions
-----------
F   forward, K strike, T tenor in years, k = log(K/F) log-moneyness.
w   total implied variance, w(k) = sigma_imp(k)^2 * T.
All volatilities are annualised lognormal (Black) vols unless suffixed _n.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy import linalg, optimize, stats

_EPS = 1e-12

__all__ = [
    "SABRParams",
    "SABRCalibrationResult",
    "hagan_lognormal_vol",
    "sabr_atm_vol",
    "solve_alpha_from_atm",
    "SABRCalibrator",
    "SmileArbitrageCheck",
    "SABRSurface",
    "DupireLocalVol",
    "black_price",
    "black_vega",
    "VarianceSwapReplicator",
    "LedoitWolfShrinkage",
    "nearest_psd",
    "ImpliedCorrelation",
    "RealisedCorrelation",
    "DispersionSignal",
    "DispersionBook",
    "DispersionEngine",
]


# ===========================================================================
# BLACK-76
# ===========================================================================

def black_price(
    F: np.ndarray, K: np.ndarray, T: float, sigma: np.ndarray, df: float = 1.0,
    right: Literal["C", "P"] = "C",
) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    s = np.asarray(sigma, dtype=float)
    vt = np.maximum(s * math.sqrt(max(T, _EPS)), _EPS)
    d1 = (np.log(np.maximum(F, _EPS) / np.maximum(K, _EPS)) + 0.5 * vt**2) / vt
    d2 = d1 - vt
    if right == "C":
        return df * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    return df * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))


def black_vega(
    F: np.ndarray, K: np.ndarray, T: float, sigma: np.ndarray, df: float = 1.0
) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    s = np.asarray(sigma, dtype=float)
    rt = math.sqrt(max(T, _EPS))
    vt = np.maximum(s * rt, _EPS)
    d1 = (np.log(np.maximum(F, _EPS) / np.maximum(K, _EPS)) + 0.5 * vt**2) / vt
    return df * F * stats.norm.pdf(d1) * rt


def black_implied_vol(
    price: float, F: float, K: float, T: float, df: float = 1.0,
    right: Literal["C", "P"] = "C", lo: float = 1e-6, hi: float = 5.0,
) -> float:
    intrinsic = df * max(F - K, 0.0) if right == "C" else df * max(K - F, 0.0)
    if price <= intrinsic + 1e-14:
        return float("nan")

    def f(s: float) -> float:
        return float(black_price(F, K, T, s, df, right)) - price

    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(optimize.brentq(f, lo, hi, xtol=1e-12, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


# ===========================================================================
# I. SABR
# ===========================================================================

@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must lie in [0,1], got {self.beta}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie in (-1,1), got {self.rho}")
        if self.nu < 0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.rho, self.nu], dtype=float)


def _z_over_x(z: np.ndarray, rho: float, series_cutoff: float = 1e-4) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < series_cutoff

    zs = z[small]
    out[small] = 1.0 - rho * zs / 2.0 + (2.0 - 3.0 * rho**2) * zs**2 / 12.0

    zl = z[~small]
    if zl.size:
        root = np.sqrt(1.0 - 2.0 * rho * zl + zl**2)
        x = np.log(np.maximum((root + zl - rho) / (1.0 - rho), _EPS))
        out[~small] = np.where(np.abs(x) < _EPS, 1.0, zl / x)
    return out


def hagan_lognormal_vol(
    F: float, K: np.ndarray, T: float, p: SABRParams
) -> np.ndarray:
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = float(F)
    if F <= 0:
        raise ValueError("forward must be positive")

    a, b, rho, nu = p.alpha, p.beta, p.rho, p.nu
    one_mb = 1.0 - b

    FK = F * K
    log_fk = np.log(F / np.maximum(K, _EPS))
    fk_half = FK ** (one_mb / 2.0)

    denom = fk_half * (
        1.0
        + (one_mb**2 / 24.0) * log_fk**2
        + (one_mb**4 / 1920.0) * log_fk**4
    )

    if nu < _EPS:
        zx = np.ones_like(K)
    else:
        z = (nu / a) * fk_half * log_fk
        zx = _z_over_x(z, rho)

    correction = 1.0 + (
        (one_mb**2 / 24.0) * a**2 / FK**one_mb
        + 0.25 * rho * b * nu * a / fk_half
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    ) * T

    return (a / denom) * zx * correction


def sabr_atm_vol(F: float, T: float, p: SABRParams) -> float:
    a, b, rho, nu = p.alpha, p.beta, p.rho, p.nu
    pw = F ** (1.0 - b)
    return float(
        (a / pw)
        * (
            1.0
            + (
                (1.0 - b) ** 2 * a**2 / (24.0 * pw**2)
                + 0.25 * rho * b * nu * a / pw
                + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
            )
            * T
        )
    )


def solve_alpha_from_atm(
    sigma_atm: float, F: float, T: float, beta: float, rho: float, nu: float
) -> float:
    pw = F ** (1.0 - beta)
    c3 = (1.0 - beta) ** 2 * T / (24.0 * pw**2)
    c2 = rho * beta * nu * T / (4.0 * pw)
    c1 = 1.0 + (2.0 - 3.0 * rho**2) * nu**2 * T / 24.0
    c0 = -sigma_atm * pw

    coeffs = np.array([c3, c2, c1, c0], dtype=float)
    nz = np.flatnonzero(np.abs(coeffs) > 1e-18)
    if nz.size == 0:
        raise ValueError("degenerate ATM cubic")
    coeffs = coeffs[nz[0]:]

    if coeffs.size == 1:
        raise ValueError("degenerate ATM cubic")

    roots = np.roots(coeffs)
    real = np.array(
        [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 1e-12], dtype=float
    )
    if real.size == 0:
        return float(sigma_atm * pw)
    return float(real.min())


# ---------------------------------------------------------------------------

@dataclass
class SABRCalibrationResult:
    params: SABRParams
    F: float
    T: float
    strikes: np.ndarray
    market_vols: np.ndarray
    model_vols: np.ndarray
    weights: np.ndarray
    rmse_vol: float
    max_abs_error: float
    vega_weighted_rmse: float
    n_iterations: int
    converged: bool
    method: str
    residuals_bps: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def atm_error(self) -> float:
        i = int(np.argmin(np.abs(self.strikes - self.F)))
        return float(self.model_vols[i] - self.market_vols[i])

    def mispricing(self, vegas: Optional[np.ndarray] = None) -> np.ndarray:
        d = self.market_vols - self.model_vols
        return d if vegas is None else d * vegas

    def summary(self) -> str:
        p = self.params
        return (
            f"SABR({self.method})  F={self.F:.2f} T={self.T:.4f}\n"
            f"  alpha={p.alpha:.6f} beta={p.beta:.2f} "
            f"rho={p.rho:+.4f} nu={p.nu:.4f}\n"
            f"  rmse={1e4*self.rmse_vol:.2f}bps  max={1e4*self.max_abs_error:.2f}bps  "
            f"vega-wtd={1e4*self.vega_weighted_rmse:.2f}bps  "
            f"atm_err={1e4*self.atm_error:+.2f}bps"
        )


class SABRCalibrator:
    def __init__(
        self,
        beta: float = 1.0,
        mode: Literal["atm_constrained", "free"] = "atm_constrained",
        weighting: Literal["vega", "inverse_spread", "uniform"] = "vega",
        max_nfev: int = 2000,
        rho_bound: float = 0.9999,
        nu_max: float = 5.0,
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0,1]")
        self.beta = float(beta)
        self.mode = mode
        self.weighting = weighting
        self.max_nfev = int(max_nfev)
        self.rho_bound = float(rho_bound)
        self.nu_max = float(nu_max)

    def _weights(
        self,
        F: float,
        K: np.ndarray,
        T: float,
        vols: np.ndarray,
        spreads: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.weighting == "vega":
            w = black_vega(F, K, T, vols)
        elif self.weighting == "inverse_spread":
            if spreads is None:
                raise ValueError("inverse_spread weighting needs `spreads`")
            w = 1.0 / np.maximum(np.asarray(spreads, dtype=float), _EPS)
        else:
            w = np.ones_like(K, dtype=float)
        s = w.sum()
        return w / s * w.size if s > _EPS else np.ones_like(w)

    def initial_guess(
        self, F: float, K: np.ndarray, T: float, vols: np.ndarray
    ) -> Tuple[float, float, float]:
        i_atm = int(np.argmin(np.abs(K - F)))
        atm = float(vols[i_atm])

        k = np.log(np.maximum(K, _EPS) / F)
        if k.size >= 3 and np.ptp(k) > _EPS:
            quad = np.polyfit(k, vols, 2)
            curvature, slope = float(quad[0]), float(quad[1])
        else:
            curvature, slope = 0.0, 0.0

        nu0 = float(np.clip(math.sqrt(max(2.0 * abs(curvature), 1e-4)), 0.05, 2.0))
        rho0 = float(np.clip(2.0 * slope / max(nu0, 1e-3), -0.9, 0.9))
        alpha0 = solve_alpha_from_atm(atm, F, T, self.beta, rho0, nu0)
        return alpha0, rho0, nu0

    def _residuals_atm(
        self, u: np.ndarray, F: float, K: np.ndarray, T: float,
        vols: np.ndarray, w: np.ndarray, atm: float,
    ) -> np.ndarray:
        rho = self.rho_bound * math.tanh(u[0])
        nu = min(math.exp(u[1]), self.nu_max)
        try:
            alpha = solve_alpha_from_atm(atm, F, T, self.beta, rho, nu)
            model = hagan_lognormal_vol(F, K, T, SABRParams(alpha, self.beta, rho, nu))
        except (ValueError, FloatingPointError):
            return np.full_like(vols, 1e3)
        return w * (model - vols)

    def _residuals_free(
        self, u: np.ndarray, F: float, K: np.ndarray, T: float,
        vols: np.ndarray, w: np.ndarray,
    ) -> np.ndarray:
        alpha = math.exp(np.clip(u[0], -20.0, 5.0))
        rho = self.rho_bound * math.tanh(u[1])
        nu = min(math.exp(np.clip(u[2], -20.0, 5.0)), self.nu_max)
        try:
            model = hagan_lognormal_vol(F, K, T, SABRParams(alpha, self.beta, rho, nu))
        except (ValueError, FloatingPointError):
            return np.full_like(vols, 1e3)
        return w * (model - vols)

    def calibrate(
        self,
        F: float,
        strikes: np.ndarray,
        T: float,
        market_vols: np.ndarray,
        spreads: Optional[np.ndarray] = None,
        atm_vol: Optional[float] = None,
        initial: Optional[Tuple[float, float, float]] = None,
        n_restarts: int = 3,
    ) -> SABRCalibrationResult:
        K = np.atleast_1d(np.asarray(strikes, dtype=float))
        v = np.atleast_1d(np.asarray(market_vols, dtype=float))
        if K.shape != v.shape:
            raise ValueError("strikes and market_vols must align")

        ok = np.isfinite(K) & np.isfinite(v) & (K > 0) & (v > 0)
        K, v = K[ok], v[ok]
        sp = None if spreads is None else np.asarray(spreads, dtype=float)[ok]
        if K.size < 3:
            raise ValueError(f"need >=3 usable quotes, got {K.size}")

        w = self._weights(F, K, T, v, sp)
        a0, r0, n0 = initial or self.initial_guess(F, K, T, v)

        if atm_vol is None:
            atm_vol = float(np.interp(F, K, v))

        best: Optional[optimize.OptimizeResult] = None
        best_u: Optional[np.ndarray] = None

        if self.mode == "atm_constrained":
            seeds = [np.array([math.atanh(np.clip(r0, -0.95, 0.95) / self.rho_bound),
                               math.log(max(n0, 1e-3))])]
            for s in range(max(n_restarts - 1, 0)):
                rng = np.random.default_rng(s)
                seeds.append(np.array([
                    math.atanh(rng.uniform(-0.85, 0.85)),
                    math.log(rng.uniform(0.05, 1.5)),
                ]))
            fun = lambda u: self._residuals_atm(u, F, K, T, v, w, atm_vol)
        else:
            seeds = [np.array([math.log(max(a0, 1e-6)),
                               math.atanh(np.clip(r0, -0.95, 0.95) / self.rho_bound),
                               math.log(max(n0, 1e-3))])]
            for s in range(max(n_restarts - 1, 0)):
                rng = np.random.default_rng(100 + s)
                seeds.append(np.array([
                    math.log(max(a0, 1e-6) * rng.uniform(0.5, 2.0)),
                    math.atanh(rng.uniform(-0.85, 0.85)),
                    math.log(rng.uniform(0.05, 1.5)),
                ]))
            fun = lambda u: self._residuals_free(u, F, K, T, v, w)

        for u0 in seeds:
            try:
                res = optimize.least_squares(
                    fun, u0, method="lm", max_nfev=self.max_nfev,
                    xtol=1e-14, ftol=1e-14, gtol=1e-14,
                )
            except Exception as exc:
                warnings.warn(f"LM failed from seed {u0}: {exc}")
                continue
            if best is None or res.cost < best.cost:
                best, best_u = res, res.x

        if best is None or best_u is None:
            raise RuntimeError("all calibration seeds failed")

        if self.mode == "atm_constrained":
            rho = self.rho_bound * math.tanh(best_u[0])
            nu = min(math.exp(best_u[1]), self.nu_max)
            alpha = solve_alpha_from_atm(atm_vol, F, T, self.beta, rho, nu)
        else:
            alpha = math.exp(np.clip(best_u[0], -20.0, 5.0))
            rho = self.rho_bound * math.tanh(best_u[1])
            nu = min(math.exp(np.clip(best_u[2], -20.0, 5.0)), self.nu_max)

        params = SABRParams(alpha, self.beta, rho, nu)
        model = hagan_lognormal_vol(F, K, T, params)
        err = model - v
        vegas = black_vega(F, K, T, v)

        return SABRCalibrationResult(
            params=params, F=float(F), T=float(T), strikes=K, market_vols=v,
            model_vols=model, weights=w,
            rmse_vol=float(np.sqrt(np.mean(err**2))),
            max_abs_error=float(np.max(np.abs(err))),
            vega_weighted_rmse=float(
                np.sqrt(np.sum(vegas * err**2) / max(vegas.sum(), _EPS))
            ),
            n_iterations=int(best.nfev),
            converged=bool(best.success),
            method=f"lm/{self.mode}",
            residuals_bps=1e4 * (v - model),
        )


# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmileArbitrageCheck:
    k_grid: np.ndarray
    g: np.ndarray                 
    density: np.ndarray
    min_g: float
    butterfly_ok: bool
    first_violation_k: Optional[float]
    calendar_ok: Optional[bool] = None

    @property
    def violating_moneyness(self) -> np.ndarray:
        return self.k_grid[self.g < 0.0]


class SmileArbitrageDetector:
    def __init__(self, k_min: float = -1.5, k_max: float = 1.5, n_points: int = 401) -> None:
        self.k_grid = np.linspace(k_min, k_max, n_points)

    def check(self, F: float, T: float, p: SABRParams) -> SmileArbitrageCheck:
        k = self.k_grid
        K = F * np.exp(k)
        sig = hagan_lognormal_vol(F, K, T, p)
        w = np.maximum(sig**2 * T, _EPS)

        dk = k[1] - k[0]
        w1 = np.gradient(w, dk, edge_order=2)
        w2 = np.gradient(w1, dk, edge_order=2)

        g = (
            (1.0 - k * w1 / (2.0 * w)) ** 2
            - (w1**2 / 4.0) * (1.0 / w + 0.25)
            + w2 / 2.0
        )

        d2 = -k / np.sqrt(w) - np.sqrt(w) / 2.0
        dens = g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d2**2)

        bad = np.flatnonzero(g < 0.0)
        return SmileArbitrageCheck(
            k_grid=k, g=g, density=dens, min_g=float(g.min()),
            butterfly_ok=bool(bad.size == 0),
            first_violation_k=float(k[bad[0]]) if bad.size else None,
        )

    @staticmethod
    def calendar_ok(
        F: float, slices: Sequence[Tuple[float, SABRParams]],
        k_grid: Optional[np.ndarray] = None,
    ) -> bool:
        k = k_grid if k_grid is not None else np.linspace(-1.0, 1.0, 101)
        K = F * np.exp(k)
        ordered = sorted(slices, key=lambda s: s[0])
        prev = None
        for T, p in ordered:
            w = hagan_lognormal_vol(F, K, T, p) ** 2 * T
            if prev is not None and np.any(w < prev - 1e-10):
                return False
            prev = w
        return True


# ===========================================================================
# II. SURFACE AND LOCAL VOLATILITY
# ===========================================================================

class SABRSurface:
    def __init__(self, F_by_expiry: Mapping[float, float]) -> None:
        self.forwards: Dict[float, float] = dict(F_by_expiry)
        self.slices: Dict[float, SABRParams] = {}

    def add(self, T: float, params: SABRParams) -> None:
        self.slices[float(T)] = params

    @property
    def expiries(self) -> List[float]:
        return sorted(self.slices)

    def forward(self, T: float) -> float:
        Ts = sorted(self.forwards)
        return float(np.interp(T, Ts, [self.forwards[t] for t in Ts]))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        Ts = self.expiries
        if not Ts:
            raise RuntimeError("no slices fitted")
        k = np.atleast_1d(np.asarray(k, dtype=float))

        if T <= Ts[0]:
            F = self.forward(Ts[0])
            return hagan_lognormal_vol(F, F * np.exp(k), Ts[0], self.slices[Ts[0]])
        if T >= Ts[-1]:
            F = self.forward(Ts[-1])
            return hagan_lognormal_vol(F, F * np.exp(k), Ts[-1], self.slices[Ts[-1]])

        j = int(np.searchsorted(Ts, T))
        T0, T1 = Ts[j - 1], Ts[j]
        F0, F1 = self.forward(T0), self.forward(T1)
        w0 = hagan_lognormal_vol(F0, F0 * np.exp(k), T0, self.slices[T0]) ** 2 * T0
        w1 = hagan_lognormal_vol(F1, F1 * np.exp(k), T1, self.slices[T1]) ** 2 * T1
        lam = (T - T0) / (T1 - T0)
        return np.sqrt(np.maximum((1.0 - lam) * w0 + lam * w1, _EPS) / T)

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        return self.implied_vol(k, T) ** 2 * T


class DupireLocalVol:
    def __init__(self, surface: SABRSurface, dT: float = 1e-4, dk: float = 1e-3) -> None:
        self.surface = surface
        self.dT = float(dT)
        self.dk = float(dk)

    def local_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        k = np.atleast_1d(np.asarray(k, dtype=float))
        h, hk = self.dT, self.dk

        w = self.surface.total_variance(k, T)
        wT = (self.surface.total_variance(k, T + h)
              - self.surface.total_variance(k, max(T - h, h))) / (2.0 * h)
        wk = (self.surface.total_variance(k + hk, T)
              - self.surface.total_variance(k - hk, T)) / (2.0 * hk)
        wkk = (
            self.surface.total_variance(k + hk, T)
            - 2.0 * w
            + self.surface.total_variance(k - hk, T)
        ) / hk**2

        w = np.maximum(w, _EPS)
        denom = (
            1.0
            - (k / w) * wk
            + 0.25 * (-0.25 - 1.0 / w + k**2 / w**2) * wk**2
            + 0.5 * wkk
        )
        return np.where(denom > 1e-8, wT / denom, np.nan)

    def local_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        lv = self.local_variance(k, T)
        return np.sqrt(np.maximum(lv, 0.0))

    def grid(self, k_grid: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
        return np.vstack([self.local_vol(k_grid, float(T)) for T in T_grid])


# ===========================================================================
# III. VARIANCE REPLICATION
# ===========================================================================

@dataclass(frozen=True)
class ReplicationResult:
    fair_variance: float
    fair_vol: float
    truncation_error_var: float
    k_lower: float
    k_upper: float
    n_strikes: int

    @property
    def truncation_bps_of_vol(self) -> float:
        if self.fair_variance <= 0:
            return float("nan")
        v = math.sqrt(max(self.fair_variance - self.truncation_error_var, 0.0))
        return 1e4 * (self.fair_vol - v)


class VarianceSwapReplicator:
    def __init__(self, n_strikes: int = 200, k_range: float = 4.0) -> None:
        self.n_strikes = int(n_strikes)
        self.k_range = float(k_range)

    def fair_variance(
        self,
        F: float,
        T: float,
        vol_fn: Callable[[np.ndarray], np.ndarray],
        k_lower: Optional[float] = None,
        k_upper: Optional[float] = None,
        df: float = 1.0,
    ) -> ReplicationResult:
        atm = float(vol_fn(np.array([0.0]))[0])
        span = self.k_range * atm * math.sqrt(max(T, _EPS))
        kl_full, ku_full = -span, span
        kl = kl_full if k_lower is None else max(k_lower, kl_full)
        ku = ku_full if k_upper is None else min(k_upper, ku_full)

        full = self._integrate(F, T, vol_fn, kl_full, ku_full, df)
        trunc = self._integrate(F, T, vol_fn, kl, ku, df)

        return ReplicationResult(
            fair_variance=full,
            fair_vol=math.sqrt(max(full, 0.0)),
            truncation_error_var=full - trunc,
            k_lower=kl, k_upper=ku, n_strikes=self.n_strikes,
        )

    def _integrate(
        self, F: float, T: float, vol_fn: Callable[[np.ndarray], np.ndarray],
        kl: float, ku: float, df: float,
    ) -> float:
        if ku <= kl:
            return 0.0
        k = np.linspace(kl, ku, self.n_strikes)
        K = F * np.exp(k)
        sig = np.asarray(vol_fn(k), dtype=float)

        put = K <= F
        prices = np.where(
            put,
            black_price(F, K, T, sig, df, "P"),
            black_price(F, K, T, sig, df, "C"),
        )
        integrand = prices / K**2
        return float(2.0 / T * np.trapezoid(integrand, K))


# ===========================================================================
# IV. COVARIANCE
# ===========================================================================

def nearest_psd(a: np.ndarray, eig_floor: float = 0.0) -> np.ndarray:
    s = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(s)
    np.clip(w, eig_floor, None, out=w)
    return (v * w) @ v.T


@dataclass(frozen=True)
class ShrinkageResult:
    covariance: np.ndarray
    sample: np.ndarray
    target: np.ndarray
    intensity: float
    mean_correlation: float
    condition_number: float
    sample_condition_number: float


class LedoitWolfShrinkage:
    def __init__(self, target: Literal["constant_correlation"] = "constant_correlation") -> None:
        self.target = target

    def fit(self, returns: np.ndarray) -> ShrinkageResult:
        X = np.asarray(returns, dtype=float)
        if X.ndim != 2:
            raise ValueError("returns must be (T, N)")
        T, N = X.shape
        if T < 2 or N < 2:
            raise ValueError(f"need T>=2 and N>=2, got ({T},{N})")

        X = X - X.mean(axis=0, keepdims=True)
        S = X.T @ X / T
        sd = np.sqrt(np.maximum(np.diag(S), _EPS))
        R = S / np.outer(sd, sd)

        off = ~np.eye(N, dtype=bool)
        rbar = float(R[off].mean())

        F = rbar * np.outer(sd, sd)
        np.fill_diagonal(F, np.diag(S))

        X2 = X**2
        pi_mat = (X2.T @ X2) / T - S**2
        pi = float(pi_mat.sum())

        M3 = (X**3).T @ X / T
        s_diag = np.diag(S)
        theta_ii = M3 - np.outer(s_diag, np.ones(N)) * S
        theta_jj = M3.T - np.outer(np.ones(N), s_diag) * S

        ratio = np.sqrt(np.outer(1.0 / np.maximum(s_diag, _EPS), s_diag))
        cross = (rbar / 2.0) * (ratio * theta_ii + (1.0 / ratio) * theta_jj)
        rho = float(np.trace(pi_mat) + cross[off].sum())

        gamma = float(np.sum((F - S) ** 2))
        delta = 0.0 if gamma < _EPS else float(np.clip((pi - rho) / gamma / T, 0.0, 1.0))

        sigma = delta * F + (1.0 - delta) * S
        sigma = nearest_psd(sigma, eig_floor=1e-14)

        return ShrinkageResult(
            covariance=sigma, sample=S, target=F, intensity=delta,
            mean_correlation=rbar,
            condition_number=float(np.linalg.cond(sigma)),
            sample_condition_number=float(np.linalg.cond(S)),
        )


# ===========================================================================
# V. DISPERSION
# ===========================================================================

class CorrelationMeasure(Enum):
    CLEAN = auto()
    DIRTY = auto()


class ImpliedCorrelation:
    def __init__(self, measure: CorrelationMeasure = CorrelationMeasure.CLEAN) -> None:
        self.measure = measure

    @staticmethod
    def _align(
        vols: Mapping[str, float], weights: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
        names = tuple(sorted(set(vols) & set(weights)))
        if len(names) < 2:
            raise ValueError("need >=2 names present in both vols and weights")
        s = np.array([vols[n] for n in names], dtype=float)
        w = np.array([weights[n] for n in names], dtype=float)
        tot = w.sum()
        if tot <= _EPS:
            raise ValueError("weights sum to zero")
        return s, w / tot, names

    @staticmethod
    def budget(sigma: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
        ws = float(np.dot(w, sigma))
        idio = float(np.dot(w**2, sigma**2))
        return ws**2 - idio, idio

    def implied(
        self, index_vol: float, vols: Mapping[str, float], weights: Mapping[str, float]
    ) -> float:
        s, w, _ = self._align(vols, weights)
        b_corr, b_idio = self.budget(s, w)
        if self.measure is CorrelationMeasure.DIRTY:
            ws = float(np.dot(w, s))
            return float(index_vol**2 / max(ws**2, _EPS))
        return float((index_vol**2 - b_idio) / max(b_corr, _EPS))

    def both(
        self, index_vol: float, vols: Mapping[str, float], weights: Mapping[str, float]
    ) -> Tuple[float, float]:
        s, w, _ = self._align(vols, weights)
        b_corr, b_idio = self.budget(s, w)
        ws = float(np.dot(w, s))
        clean = (index_vol**2 - b_idio) / max(b_corr, _EPS)
        dirty = index_vol**2 / max(ws**2, _EPS)
        return float(clean), float(dirty)

    @staticmethod
    def herfindahl(weights: Mapping[str, float]) -> float:
        w = np.array(list(weights.values()), dtype=float)
        w = w / max(w.sum(), _EPS)
        return float(np.dot(w, w))

    def index_vol_from_correlation(
        self, rho: float, vols: Mapping[str, float], weights: Mapping[str, float]
    ) -> float:
        s, w, _ = self._align(vols, weights)
        b_corr, b_idio = self.budget(s, w)
        return float(math.sqrt(max(b_idio + rho * b_corr, 0.0)))

    def decompose_change(
        self,
        index_vol_0: float, vols_0: Mapping[str, float], weights_0: Mapping[str, float],
        index_vol_1: float, vols_1: Mapping[str, float], weights_1: Mapping[str, float],
    ) -> Dict[str, float]:
        r0 = self.implied(index_vol_0, vols_0, weights_0)
        r1 = self.implied(index_vol_1, vols_1, weights_1)
        r_w = self.implied(index_vol_0, vols_0, weights_1)
        return {
            "total_change": r1 - r0,
            "weight_effect": r_w - r0,
            "repricing_effect": r1 - r_w,
            "herfindahl_0": self.herfindahl(weights_0),
            "herfindahl_1": self.herfindahl(weights_1),
        }


class RealisedCorrelation:
    def __init__(self, annualisation: int = 252, use_shrinkage: bool = True) -> None:
        self.annualisation = int(annualisation)
        self.shrinker = LedoitWolfShrinkage() if use_shrinkage else None

    def from_returns(
        self, returns: np.ndarray, weights: np.ndarray
    ) -> Dict[str, float]:
        X = np.asarray(returns, dtype=float)
        w = np.asarray(weights, dtype=float)
        w = w / max(w.sum(), _EPS)

        if self.shrinker is not None:
            res = self.shrinker.fit(X)
            cov = res.covariance
            intensity = res.intensity
        else:
            Xd = X - X.mean(axis=0, keepdims=True)
            cov = Xd.T @ Xd / Xd.shape[0]
            intensity = 0.0

        cov_ann = cov * self.annualisation
        sig = np.sqrt(np.maximum(np.diag(cov_ann), _EPS))

        port_var = float(w @ cov_ann @ w)
        ws = float(np.dot(w, sig))
        idio = float(np.dot(w**2, sig**2))
        b_corr = ws**2 - idio

        return {
            "portfolio_vol": math.sqrt(max(port_var, 0.0)),
            "rho_clean": (port_var - idio) / max(b_corr, _EPS),
            "rho_dirty": port_var / max(ws**2, _EPS),
            "correlation_budget": b_corr,
            "idio_budget": idio,
            "shrinkage_intensity": intensity,
            "mean_constituent_vol": float(np.dot(w, sig)),
        }


@dataclass(frozen=True)
class DispersionSignal:
    expiry: float
    rho_implied_clean: float
    rho_implied_dirty: float
    rho_realised: float
    rho_forecast: float
    correlation_budget: float
    index_notional: float
    truncation_cost_var: float
    herfindahl: float

    @property
    def spread(self) -> float:
        return self.rho_implied_clean - self.rho_forecast

    @property
    def clean_dirty_gap(self) -> float:
        return self.rho_implied_dirty - self.rho_implied_clean

    @property
    def gross_pnl_per_corr_point(self) -> float:
        return self.index_notional * self.correlation_budget

    @property
    def expected_pnl(self) -> float:
        return self.gross_pnl_per_corr_point * self.spread

    @property
    def net_expected_pnl(self) -> float:
        return self.expected_pnl - self.index_notional * self.truncation_cost_var

    @property
    def direction(self) -> Literal["short_correlation", "long_correlation", "flat"]:
        if self.net_expected_pnl > 0 and self.spread > 0:
            return "short_correlation"
        if self.net_expected_pnl > 0 and self.spread < 0:
            return "long_correlation"
        return "flat"


@dataclass
class DispersionBook:
    index_symbol: str
    expiry: float
    names: Tuple[str, ...]
    index_weights: Dict[str, float]
    variance_notionals: Dict[str, float]
    index_notional: float
    constituent_vols: Dict[str, float]
    index_vol: float
    full_index_weights: Optional[Dict[str, float]] = None

    @property
    def index_coverage(self) -> float:
        if self.full_index_weights is None:
            return 1.0
        tot = sum(self.full_index_weights.values())
        return sum(self.full_index_weights.get(n, 0.0) for n in self.names) / max(tot, _EPS)

    @property
    def uncovered_idio(self) -> float:
        if self.full_index_weights is None:
            return 0.0
        tot = sum(self.full_index_weights.values())
        out = 0.0
        for n, w in self.full_index_weights.items():
            if n in self.names:
                continue
            s = self.constituent_vols.get(n)
            if s is not None:
                out += (w / tot) ** 2 * s**2
        return out

    def net_vega(self) -> float:
        cv = sum(
            2.0 * self.variance_notionals[n] * self.constituent_vols[n]
            for n in self.names
        )
        return cv - 2.0 * self.index_notional * self.index_vol

    def pnl(
        self, realised_index_var: float, realised_constituent_vars: Mapping[str, float],
        strike_index_var: float, strike_constituent_vars: Mapping[str, float],
    ) -> Dict[str, float]:
        legs = sum(
            self.variance_notionals[n]
            * (realised_constituent_vars[n] - strike_constituent_vars[n])
            for n in self.names
        )
        idx = self.index_notional * (realised_index_var - strike_index_var)
        return {"constituent_pnl": legs, "index_pnl": -idx, "total": legs - idx}


class DispersionEngine:
    def __init__(
        self,
        implied: Optional[ImpliedCorrelation] = None,
        realised: Optional[RealisedCorrelation] = None,
        replicator: Optional[VarianceSwapReplicator] = None,
        min_names: int = 8,
        max_names: int = 12,
        min_index_coverage: float = 0.45,
        entry_spread: float = 0.05,
    ) -> None:
        self.implied = implied or ImpliedCorrelation(CorrelationMeasure.CLEAN)
        self.realised = realised or RealisedCorrelation()
        self.replicator = replicator or VarianceSwapReplicator()
        self.min_names = int(min_names)
        self.max_names = int(max_names)
        self.min_index_coverage = float(min_index_coverage)
        self.entry_spread = float(entry_spread)

    def select_basket(
        self,
        index_weights: Mapping[str, float],
        liquidity_rank: Mapping[str, float],
        eligible: Optional[Iterable[str]] = None,
        sector_map: Optional[Mapping[str, str]] = None,
        max_sector_weight: float = 0.45,
    ) -> Tuple[str, ...]:
        pool = set(index_weights) & set(liquidity_rank)
        if eligible is not None:
            pool &= set(eligible)
        if len(pool) < self.min_names:
            raise ValueError(f"only {len(pool)} eligible names, need {self.min_names}")

        scored = sorted(
            pool, key=lambda n: index_weights[n] * liquidity_rank[n], reverse=True
        )

        chosen: List[str] = []
        sector_w: Dict[str, float] = {}
        total_w = sum(index_weights[n] for n in pool)

        for n in scored:
            if len(chosen) >= self.max_names:
                break
            sec = sector_map.get(n, n) if sector_map else n
            w = index_weights[n] / max(total_w, _EPS)
            if sector_map and sector_w.get(sec, 0.0) + w > max_sector_weight:
                continue
            chosen.append(n)
            sector_w[sec] = sector_w.get(sec, 0.0) + w

        if len(chosen) < self.min_names:
            for n in scored:
                if n not in chosen:
                    chosen.append(n)
                if len(chosen) >= self.min_names:
                    break
        return tuple(chosen)

    def signal(
        self,
        index_vol: float,
        constituent_vols: Mapping[str, float],
        weights: Mapping[str, float],
        historical_returns: np.ndarray,
        expiry: float,
        index_notional: float,
        rho_forecast: Optional[float] = None,
        truncation_cost_var: float = 0.0,
    ) -> DispersionSignal:
        clean, dirty = self.implied.both(index_vol, constituent_vols, weights)

        names = tuple(sorted(set(constituent_vols) & set(weights)))
        w = np.array([weights[n] for n in names], dtype=float)
        rz = self.realised.from_returns(historical_returns, w)

        return DispersionSignal(
            expiry=float(expiry),
            rho_implied_clean=clean,
            rho_implied_dirty=dirty,
            rho_realised=rz["rho_clean"],
            rho_forecast=float(rho_forecast if rho_forecast is not None else rz["rho_clean"]),
            correlation_budget=self.implied.budget(
                np.array([constituent_vols[n] for n in names]), w / w.sum()
            )[0],
            index_notional=float(index_notional),
            truncation_cost_var=float(truncation_cost_var),
            herfindahl=self.implied.herfindahl({n: weights[n] for n in names}),
        )

    def build_book(
        self,
        index_symbol: str,
        signal: DispersionSignal,
        names: Sequence[str],
        index_weights: Mapping[str, float],
        constituent_vols: Mapping[str, float],
        index_vol: float,
        full_index_weights: Optional[Mapping[str, float]] = None,
    ) -> DispersionBook:
        sub = {n: index_weights[n] for n in names}
        tot = sum(sub.values())
        wn = {n: v / max(tot, _EPS) for n, v in sub.items()}
        notionals = {n: signal.index_notional * wn[n] ** 2 for n in names}

        return DispersionBook(
            index_symbol=index_symbol,
            expiry=signal.expiry,
            names=tuple(names),
            index_weights=wn,
            variance_notionals=notionals,
            index_notional=signal.index_notional,
            constituent_vols={n: constituent_vols[n] for n in constituent_vols},
            index_vol=index_vol,
            full_index_weights=dict(full_index_weights) if full_index_weights else None,
        )


if __name__ == "__main__":
    print("============================================================================")
    print(" DISPERSION ARBITRAGE : SABR CALIBRATION & VOLATILITY SURFACE ANALYSIS")
    print("============================================================================")
    
    print("\n[BOOT] Initializing Options Chain Calibrator...")
    F, T = 20000.0, 0.25
    K = F * np.exp(np.linspace(-0.25, 0.25, 21))
    truth = SABRParams(alpha=0.30, beta=1.0, rho=-0.45, nu=0.75)
    v = hagan_lognormal_vol(F, K, T, truth)
    
    cal = SABRCalibrator(beta=1.0, mode="atm_constrained")
    res = cal.calibrate(F, K, T, v, atm_vol=sabr_atm_vol(F, T, truth))
    
    print(f"[TEST 1] SABR Model Volatility Calibration (ATM Constrained)")
    print(f"-> Fitted Alpha: {res.params.alpha:.4f} (Target: {truth.alpha:.4f})")
    print(f"-> Fitted Rho  : {res.params.rho:.4f} (Target: {truth.rho:.4f})")
    print(f"-> Fitted Nu   : {res.params.nu:.4f} (Target: {truth.nu:.4f})")
    print(f"-> Calibration RMSE: {res.rmse_vol * 1e4:.2f} bps")

    print("\n[BOOT] Detecting Smile Arbitrage & Dupire Local Volatility...")
    det = SmileArbitrageDetector(k_min=-1.0, k_max=1.0, n_points=301)
    benign = det.check(F, T, res.params)
    print(f"[TEST 2] Butterfly Arbitrage Check")
    print(f"-> Arbitrage Free (Density > 0): {benign.butterfly_ok}")
    
    print("\n[BOOT] Constructing Dispersion Matrix...")
    names = [f"Stock_{i}" for i in range(10)]
    rng = np.random.default_rng(5)
    vols = {n: float(v) for n, v in zip(names, rng.uniform(0.18, 0.45, len(names)))}
    raw = rng.uniform(0.02, 0.20, len(names))
    weights = {n: float(w / raw.sum()) for n, w in zip(names, raw)}

    ic = ImpliedCorrelation(CorrelationMeasure.CLEAN)
    rho_target = 0.35
    iv = ic.index_vol_from_correlation(rho_target, vols, weights)
    rho_recovered = ic.implied(iv, vols, weights)
    
    print(f"[TEST 3] Implied Correlation Extraction")
    print(f"-> Target Correlation: {rho_target:.4f}")
    print(f"-> Implied Index Volatility: {iv:.4f}")
    print(f"-> Recovered Clean Correlation: {rho_recovered:.4f}")
    
    print("\n[SUCCESS] VOLATILITY ARBITRAGE ENGINE OPERATIONAL.")

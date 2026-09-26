"""
optimal_execution_sor.py
========================

Phase 9: Institutional Smart Order Router.

Two solvers and the routing layer that sits on top of them.

    I.  Almgren-Chriss optimal trajectory  (parent-order scheduling)
    II. Avellaneda-Stoikov / Gueant HJB    (child-order passive quoting)

Conventions
-----------
Time in years unless suffixed `_s`. Prices in currency. `side = +1` for a BUY
(acquire), `-1` for a SELL (liquidate). Inventory `q` is signed, in lots.
All rates (sigma, mu, A) are annualised to match.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
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
    "ImpactModel",
    "LinearImpact",
    "PowerLawImpact",
    "ACParameters",
    "ExecutionSchedule",
    "AlmgrenChrissSolver",
    "NonlinearImpactSolver",
    "EfficientFrontier",
    "ASParameters",
    "QuotePair",
    "AvellanedaStoikovQuoter",
    "GueantHJBSolver",
    "IntensityCalibrator",
    "MicrostructureState",
    "SORConfig",
    "SmartOrderRouter",
]


# ===========================================================================
# STABLE PRIMITIVES
# ===========================================================================

def _arccosh1p(x: np.ndarray | float) -> np.ndarray | float:
    """
    arccosh(1 + x), numerically stable for x -> 0.

        cosh(2u) = 1 + 2 sinh^2(u)
        => arccosh(1 + x) = 2 asinh( sqrt(x/2) )

    Direct np.arccosh(1 + x) loses ~half the mantissa for x < 1e-8, which is
    precisely the regime of the Almgren-Chriss kappa solve at fine discretisation.
    """
    return 2.0 * np.arcsinh(np.sqrt(np.maximum(np.asarray(x, dtype=float), 0.0) / 2.0))


def _sinh_ratio(a: np.ndarray | float, b: float) -> np.ndarray | float:
    """
    sinh(a) / sinh(b) for 0 <= a <= b, without overflowing for large b.

        sinh(a)/sinh(b) = exp(a-b) * (1 - exp(-2a)) / (1 - exp(-2b))
    """
    a = np.asarray(a, dtype=float)
    if abs(b) < 1e-9:
        return a / b if abs(b) > _EPS else np.zeros_like(a)
    return np.exp(a - b) * (1.0 - np.exp(-2.0 * a)) / (1.0 - np.exp(-2.0 * b))


def _cosh_ratio(a: np.ndarray | float, b: float) -> np.ndarray | float:
    """cosh(a) / sinh(b), same treatment."""
    a = np.asarray(a, dtype=float)
    if abs(b) < 1e-9:
        return np.full_like(a, np.inf)
    return np.exp(a - b) * (1.0 + np.exp(-2.0 * a)) / (1.0 - np.exp(-2.0 * b))


# ===========================================================================
# I. MARKET IMPACT
# ===========================================================================

class ImpactModel:
    """
    Almgren-Chriss impact decomposition.

    PERMANENT impact g(v) shifts the efficient price and is not recovered:

        S_k = S_{k-1} + sigma sqrt(tau) xi_k - tau * g(n_k / tau)

    TEMPORARY impact h(v) is the concession paid on this trade only; it decays
    before the next interval and does not enter the price path:

        S~_k = S_{k-1} - h(n_k / tau)

    Implementation shortfall for a liquidation of X over N intervals:

        E[C] = sum_k tau * x_k * g(n_k/tau)  +  sum_k n_k * h(n_k/tau)
        V[C] = sigma^2 * tau * sum_k x_k^2

    where x_k is holdings AFTER the k-th trade, n_k = x_{k-1} - x_k.
    """

    def permanent(self, rate: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def temporary(self, rate: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def permanent_cost(self, x: np.ndarray, n: np.ndarray, tau: float) -> float:
        return float(np.sum(tau * x * self.permanent(n / tau)))

    def temporary_cost(self, n: np.ndarray, tau: float) -> float:
        return float(np.sum(n * self.temporary(n / tau)))


@dataclass(frozen=True)
class LinearImpact(ImpactModel):
    """
        g(v) = gamma * v
        h(v) = epsilon * sgn(v) + eta * v

    The only specification admitting the closed-form sinh trajectory. With it,

        E[C] = (1/2) gamma X^2 + epsilon sum|n_k| + (eta_tilde / tau) sum n_k^2
        eta_tilde = eta - gamma * tau / 2

    eta_tilde < 0 means the discretisation is coarse enough that permanent
    impact dominates temporary within a bucket; the problem becomes unbounded
    below and the solver must refuse.
    """

    gamma: float          # permanent, currency per (share / year)
    eta: float            # temporary, currency per (share / year)
    epsilon: float = 0.0  # fixed half-spread + fee per share

    def permanent(self, rate: np.ndarray) -> np.ndarray:
        return self.gamma * np.asarray(rate, dtype=float)

    def temporary(self, rate: np.ndarray) -> np.ndarray:
        r = np.asarray(rate, dtype=float)
        return self.epsilon * np.sign(r) + self.eta * r

    def eta_tilde(self, tau: float) -> float:
        return self.eta - 0.5 * self.gamma * tau

    @classmethod
    def from_square_root_law(
        cls,
        daily_volatility: float,
        adv_shares: float,
        y_coefficient: float = 0.75,
        price: float = 1.0,
        reference_participation: float = 0.10,
        permanent_fraction: float = 0.33,
        epsilon: float = 0.0,
        trading_days: int = 252,
    ) -> "LinearImpact":
        """
        Linearise the empirical square-root law

            dP/P = Y * sigma_daily * sqrt(Q / V_daily)

        about a reference participation rate, so the closed-form solver can be
        used with parameters calibrated the way the desk actually measures
        impact. The linearisation is only valid near `reference_participation`;
        outside roughly [0.5x, 2x] of it, use NonlinearImpactSolver instead.

        `permanent_fraction` splits total measured impact into the permanent
        component; the conventional estimate is between 1/3 and 1/2.
        """
        if adv_shares <= 0 or reference_participation <= 0:
            raise ValueError("adv_shares and reference_participation must be positive")

        q_ref = reference_participation * adv_shares
        impact_ref = y_coefficient * daily_volatility * math.sqrt(q_ref / adv_shares)
        cost_per_share = impact_ref * price

        # Rate at the reference point, in shares per YEAR.
        rate_ref = q_ref * trading_days
        slope = cost_per_share / rate_ref

        return cls(
            gamma=permanent_fraction * slope,
            eta=(1.0 - permanent_fraction) * slope,
            epsilon=epsilon,
        )


@dataclass(frozen=True)
class PowerLawImpact(ImpactModel):
    """
        g(v) = gamma * v
        h(v) = epsilon * sgn(v) + eta * |v|^alpha * sgn(v)

    alpha = 0.5 recovers the square-root law. No closed form exists; solved by
    NonlinearImpactSolver.
    """

    gamma: float
    eta: float
    alpha: float = 0.5
    epsilon: float = 0.0

    def permanent(self, rate: np.ndarray) -> np.ndarray:
        return self.gamma * np.asarray(rate, dtype=float)

    def temporary(self, rate: np.ndarray) -> np.ndarray:
        r = np.asarray(rate, dtype=float)
        return self.epsilon * np.sign(r) + self.eta * np.abs(r) ** self.alpha * np.sign(r)


# ===========================================================================
# II. ALMGREN-CHRISS
# ===========================================================================

@dataclass(frozen=True)
class ACParameters:
    total_shares: float
    horizon: float                 # years
    n_intervals: int
    volatility: float              # annualised, absolute currency units
    risk_aversion: float           # lambda, 1/currency
    side: int = -1                 # -1 sell/liquidate, +1 buy/acquire
    drift: float = 0.0             # mu, expected price change per year
    adv_shares: Optional[float] = None
    max_participation: Optional[float] = None
    trading_days: int = 252

    @property
    def tau(self) -> float:
        return self.horizon / self.n_intervals

    def __post_init__(self) -> None:
        if self.total_shares <= 0:
            raise ValueError("total_shares must be positive; use `side` for direction")
        if self.horizon <= 0 or self.n_intervals < 1:
            raise ValueError("horizon and n_intervals must be positive")
        if self.side not in (-1, 1):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")


@dataclass
class ExecutionSchedule:
    times: np.ndarray              # (N+1,) interval boundaries, years
    holdings: np.ndarray           # (N+1,) x_0 = X ... x_N = 0
    trades: np.ndarray             # (N,)   n_k = x_{k-1} - x_k
    side: int
    kappa: float
    expected_cost: float
    cost_variance: float
    risk_aversion: float
    params: ACParameters
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def utility(self) -> float:
        return self.expected_cost + self.risk_aversion * self.cost_variance

    @property
    def cost_stdev(self) -> float:
        return math.sqrt(max(self.cost_variance, 0.0))

    @property
    def half_life(self) -> float:
        """1/kappa: the e-folding time of the remaining position."""
        return float("inf") if self.kappa < _EPS else 1.0 / self.kappa

    @property
    def participation(self) -> Optional[np.ndarray]:
        p = self.params
        if p.adv_shares is None:
            return None
        interval_volume = p.adv_shares * p.tau * p.trading_days
        return self.trades / max(interval_volume, _EPS)

    def signed_trades(self) -> np.ndarray:
        """Trade list in signed share terms: negative = sell."""
        return self.side * self.trades

    def cost_bps(self, reference_price: float) -> float:
        notional = self.params.total_shares * reference_price
        return 1e4 * self.expected_cost / max(notional, _EPS)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "t": self.times[:-1],
                "t_end": self.times[1:],
                "holdings_start": self.holdings[:-1],
                "holdings_end": self.holdings[1:],
                "trade": self.trades,
                "signed_trade": self.signed_trades(),
                "participation": (
                    self.participation
                    if self.participation is not None
                    else np.full(self.trades.shape, np.nan)
                ),
            }
        )


class AlmgrenChrissSolver:
    """
    Closed-form optimal trajectory under linear impact.

    Minimise  U = E[C] + lambda * V[C]  subject to x_0 = X, x_N = 0.

    Discrete Euler-Lagrange condition:

        x_{k-1} - 2 x_k + x_{k+1} = kappa_tilde^2 tau^2 x_k,
        kappa_tilde^2 = lambda sigma^2 / eta_tilde,
        eta_tilde     = eta - gamma tau / 2

    whose solution is

        x_j = X * sinh( kappa (T - t_j) ) / sinh( kappa T )

        kappa = (1/tau) arccosh( tau^2 kappa_tilde^2 / 2 + 1 )   -> kappa_tilde as tau -> 0

    Trade list, by the sinh difference identity:

        n_j = X * 2 sinh(kappa tau / 2) cosh( kappa (T - t_{j-1/2}) ) / sinh(kappa T)

    Closed-form moments (Almgren & Chriss 2000):

        E[C] = (1/2) gamma X^2 + epsilon sum|n_k|
               + eta_tilde X^2 * tanh(kappa tau/2)
                 * ( tau sinh(2 kappa T) + 2T sinh(kappa tau) )
                 / ( 2 tau^2 sinh^2(kappa T) )

        V[C] = (1/2) sigma^2 X^2
               * ( tau sinh(kappa T) cosh(kappa (T - tau)) - T sinh(kappa tau) )
               / ( sinh^2(kappa T) sinh(kappa tau) )

    WITH DRIFT. Adding an expected price change mu makes the continuous
    objective  int [ eta_tilde x'^2 + lambda sigma^2 x^2 - mu x ] dt,  whose
    Euler-Lagrange equation is  x'' = kappa^2 x - mu / (2 eta_tilde)  with
    particular solution x_p = mu / (2 lambda sigma^2). Imposing the boundary
    conditions:

        x(t) = x_p [ 1 - cosh(kappa (T-t)) ]
             + [ X - x_p + x_p cosh(kappa T) ] sinh(kappa (T-t)) / sinh(kappa T)

    For a SELL, mu > 0 (price expected to rise) slows liquidation; for a BUY
    the drift is negated on entry. This is the natural coupling point for the
    Phase 8 GNN residual forecast.
    """

    def __init__(self, allow_non_monotone: bool = False) -> None:
        self.allow_non_monotone = bool(allow_non_monotone)

    # -- kappa ------------------------------------------------------------ #

    @staticmethod
    def kappa(params: ACParameters, impact: LinearImpact) -> Tuple[float, float]:
        """Returns (kappa_discrete, kappa_tilde_continuous)."""
        tau = params.tau
        eta_t = impact.eta_tilde(tau)
        if eta_t <= 0:
            raise ValueError(
                f"eta_tilde = {eta_t:.6g} <= 0: permanent impact dominates temporary "
                f"within a bucket of {tau:.6g}y. Increase n_intervals or re-calibrate."
            )
        if params.risk_aversion <= 0:
            return 0.0, 0.0

        k_tilde_sq = params.risk_aversion * params.volatility**2 / eta_t
        k_tilde = math.sqrt(k_tilde_sq)
        k_disc = float(_arccosh1p(0.5 * tau**2 * k_tilde_sq)) / tau
        return k_disc, k_tilde

    # -- trajectory -------------------------------------------------------- #

    def solve(self, params: ACParameters, impact: LinearImpact) -> ExecutionSchedule:
        tau, T, X, N = params.tau, params.horizon, params.total_shares, params.n_intervals
        k, k_tilde = self.kappa(params, impact)
        t = np.linspace(0.0, T, N + 1)

        mu_eff = params.drift * (-params.side)  # sell: +mu slows; buy: +mu hurries

        if k < 1e-10:
            # Risk-neutral limit: linear (TWAP) plus a drift tilt that is only
            # bounded when lambda > 0; at lambda == 0 exactly, it is TWAP.
            x = X * (1.0 - t / T)
        elif abs(mu_eff) < _EPS:
            x = X * np.asarray(_sinh_ratio(k * (T - t), k * T))
        else:
            x_p = mu_eff / (2.0 * params.risk_aversion * params.volatility**2)
            cosh_kT = math.cosh(min(k * T, 700.0))
            amp = X - x_p + x_p * cosh_kT
            x = (
                x_p * (1.0 - np.cosh(np.clip(k * (T - t), -700.0, 700.0)))
                + amp * np.asarray(_sinh_ratio(k * (T - t), k * T))
            )

        x[0], x[-1] = X, 0.0
        n = x[:-1] - x[1:]

        if not self.allow_non_monotone and np.any(n < -1e-9 * X):
            warnings.warn(
                "trajectory requires trading against the parent side "
                f"(min slice {n.min():.4g}); clipping. Drift may be too large "
                "relative to risk aversion."
            )
            x = np.minimum.accumulate(x)
            x[-1] = 0.0
            n = x[:-1] - x[1:]

        e_cost, e_var = self.cost_moments(x, n, params, impact)
        e_closed, v_closed = self._closed_form_moments(params, impact, k)

        return ExecutionSchedule(
            times=t,
            holdings=x,
            trades=n,
            side=params.side,
            kappa=k,
            expected_cost=e_cost,
            cost_variance=e_var,
            risk_aversion=params.risk_aversion,
            params=params,
            diagnostics={
                "kappa_discrete": k,
                "kappa_continuous": k_tilde,
                "eta_tilde": impact.eta_tilde(tau),
                "half_life_years": (1.0 / k) if k > _EPS else float("inf"),
                "closed_form_cost": e_closed,
                "closed_form_variance": v_closed,
                "cost_relative_error": abs(e_cost - e_closed) / max(abs(e_closed), _EPS),
                "max_participation": (
                    float(np.max(n / max(params.adv_shares * tau * params.trading_days, _EPS)))
                    if params.adv_shares
                    else float("nan")
                ),
            },
        )

    # -- moments ----------------------------------------------------------- #

    @staticmethod
    def cost_moments(
        x: np.ndarray, n: np.ndarray, params: ACParameters, impact: ImpactModel
    ) -> Tuple[float, float]:
        """
        Exact moments for the DISCRETE problem, evaluated directly from the
        trajectory. Valid for any impact specification; used to cross-check the
        closed form and as the objective for the nonlinear solver.
        """
        tau = params.tau
        perm = impact.permanent_cost(x[1:], n, tau)
        temp = impact.temporary_cost(n, tau)
        drift_term = -params.drift * (-params.side) * tau * float(np.sum(x[1:]))
        var = params.volatility**2 * tau * float(np.sum(x[1:] ** 2))
        return perm + temp + drift_term, var

    @staticmethod
    def _closed_form_moments(
        params: ACParameters, impact: LinearImpact, k: float
    ) -> Tuple[float, float]:
        tau, T, X = params.tau, params.horizon, params.total_shares
        eta_t = impact.eta_tilde(tau)

        if k < 1e-10:
            e = 0.5 * impact.gamma * X**2 + impact.epsilon * X + eta_t * X**2 / T
            v = params.volatility**2 * X**2 * T / 3.0
            return e, v

        kT, ktau = k * T, k * tau
        sinh_kT = math.sinh(min(kT, 700.0))
        sinh_ktau = math.sinh(ktau)

        e_impact = (
            eta_t
            * X**2
            * math.tanh(0.5 * ktau)
            * (tau * math.sinh(min(2.0 * kT, 700.0)) + 2.0 * T * sinh_ktau)
            / (2.0 * tau**2 * sinh_kT**2)
        )
        e = 0.5 * impact.gamma * X**2 + impact.epsilon * X + e_impact

        v = (
            0.5
            * params.volatility**2
            * X**2
            * (
                tau * sinh_kT * math.cosh(min(k * (T - tau), 700.0))
                - T * sinh_ktau
            )
            / (sinh_kT**2 * sinh_ktau)
        )
        return e, v


# ===========================================================================
# III. NONLINEAR IMPACT
# ===========================================================================

class NonlinearImpactSolver:
    """
    Trajectory optimisation for power-law temporary impact and/or hard
    participation constraints, where no closed form exists.

    Decision variable: interior holdings x_1 ... x_{N-1}, with x_0 = X fixed
    and x_N = 0 fixed.

        min  gamma * sum_k x_k n_k
             + epsilon * sum_k |n_k|
             + eta tau^{-alpha} * sum_k |n_k|^{1+alpha}
             - mu tau * sum_k x_k
             + lambda sigma^2 tau * sum_k x_k^2

    Analytic gradient (i = 1 .. N-1):

        d/dx_i = gamma (x_{i-1} - 2 x_i + x_{i+1})
               + epsilon ( sgn(n_{i+1}) - sgn(n_i) )
               + eta tau^{-alpha} (1+alpha) ( |n_{i+1}|^alpha sgn(n_{i+1})
                                            - |n_i|^alpha sgn(n_i) )
               - mu tau
               + 2 lambda sigma^2 tau x_i

    The |n|^{1+alpha} term is convex for alpha >= 0, and the quadratic risk
    term is convex, so with monotonicity imposed the programme is convex and
    SLSQP converges to the global optimum. Without monotonicity the epsilon
    term makes it non-smooth at n_k = 0; a multi-start is then advisable.
    """

    def __init__(
        self,
        enforce_monotone: bool = True,
        max_iter: int = 500,
        tol: float = 1e-10,
        n_restarts: int = 1,
    ) -> None:
        self.enforce_monotone = bool(enforce_monotone)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_restarts = int(n_restarts)

    def solve(
        self,
        params: ACParameters,
        impact: PowerLawImpact,
        warm_start: Optional[np.ndarray] = None,
    ) -> ExecutionSchedule:
        X, N, tau, T = params.total_shares, params.n_intervals, params.tau, params.horizon
        lam_sig2_tau = params.risk_aversion * params.volatility**2 * tau
        c = impact.eta * tau ** (-impact.alpha)
        a = impact.alpha
        mu_eff = params.drift * (-params.side)

        if N < 2:
            x = np.array([X, 0.0])
            n = np.array([X])
            e, v = AlmgrenChrissSolver.cost_moments(x, n, params, impact)
            return self._package(x, n, e, v, params, {"n_intervals": 1})

        def unpack(u: np.ndarray) -> np.ndarray:
            return np.concatenate(([X], u, [0.0]))

        def objective(u: np.ndarray) -> float:
            x = unpack(u)
            n = x[:-1] - x[1:]
            return (
                impact.gamma * float(np.dot(x[1:], n))
                + impact.epsilon * float(np.sum(np.abs(n)))
                + c * float(np.sum(np.abs(n) ** (1.0 + a)))
                - mu_eff * tau * float(np.sum(x[1:]))
                + lam_sig2_tau * float(np.dot(x[1:], x[1:]))
            )

        def gradient(u: np.ndarray) -> np.ndarray:
            x = unpack(u)
            n = x[:-1] - x[1:]
            sgn = np.sign(n)
            pw = np.abs(n) ** a * sgn

            g = np.empty(N - 1, dtype=float)
            for i in range(1, N):
                g[i - 1] = (
                    impact.gamma * (x[i - 1] - 2.0 * x[i] + x[i + 1])
                    + impact.epsilon * (sgn[i] - sgn[i - 1])
                    + c * (1.0 + a) * (pw[i] - pw[i - 1])
                    - mu_eff * tau
                    + 2.0 * lam_sig2_tau * x[i]
                )
            return g

        constraints: List[dict] = []
        if self.enforce_monotone:
            # n_k >= 0 for every k: no trading against the parent side.
            def mono(u: np.ndarray) -> np.ndarray:
                x = unpack(u)
                return x[:-1] - x[1:]

            def mono_jac(u: np.ndarray) -> np.ndarray:
                j = np.zeros((N, N - 1))
                for k in range(N):
                    if k - 1 >= 0 and k - 1 < N - 1:
                        j[k, k - 1] = 1.0
                    if k < N - 1:
                        j[k, k] = -1.0
                return j

            constraints.append({"type": "ineq", "fun": mono, "jac": mono_jac})

        if params.max_participation is not None and params.adv_shares:
            cap = params.max_participation * params.adv_shares * tau * params.trading_days

            def part(u: np.ndarray, cap: float = cap) -> np.ndarray:
                x = unpack(u)
                return cap - (x[:-1] - x[1:])

            constraints.append({"type": "ineq", "fun": part})

            if cap * N < X * (1.0 - 1e-9):
                raise ValueError(
                    f"participation cap infeasible: max {cap * N:.0f} shares over "
                    f"{N} intervals against a parent of {X:.0f}. Extend the horizon."
                )

        bounds = [(0.0, X)] * (N - 1)

        starts: List[np.ndarray] = []
        if warm_start is not None:
            starts.append(np.asarray(warm_start, dtype=float)[1:-1])
        starts.append(X * (1.0 - np.arange(1, N) / N))  # TWAP
        for s in range(max(self.n_restarts - len(starts), 0)):
            rng = np.random.default_rng(s)
            pert = np.sort(rng.uniform(0.0, X, N - 1))[::-1]
            starts.append(pert)

        best: Optional[optimize.OptimizeResult] = None
        for u0 in starts:
            res = optimize.minimize(
                objective,
                np.clip(u0, 0.0, X),
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
            if best is None or (res.success and res.fun < best.fun):
                best = res

        assert best is not None
        if not best.success:
            warnings.warn(f"SLSQP did not converge: {best.message}")

        x = unpack(best.x)
        x[0], x[-1] = X, 0.0
        n = x[:-1] - x[1:]
        e, v = AlmgrenChrissSolver.cost_moments(x, n, params, impact)

        return self._package(
            x,
            n,
            e,
            v,
            params,
            {
                "converged": float(best.success),
                "objective": float(best.fun),
                "n_iterations": float(best.nit),
                "alpha": impact.alpha,
                "max_participation": (
                    float(np.max(n / max(params.adv_shares * tau * params.trading_days, _EPS)))
                    if params.adv_shares
                    else float("nan")
                ),
            },
        )

    @staticmethod
    def _package(
        x: np.ndarray,
        n: np.ndarray,
        e: float,
        v: float,
        params: ACParameters,
        diag: Dict[str, float],
    ) -> ExecutionSchedule:
        return ExecutionSchedule(
            times=np.linspace(0.0, params.horizon, params.n_intervals + 1),
            holdings=x,
            trades=n,
            side=params.side,
            kappa=float("nan"),
            expected_cost=e,
            cost_variance=v,
            risk_aversion=params.risk_aversion,
            params=params,
            diagnostics=diag,
        )


class EfficientFrontier:
    """
    Sweep lambda to trace the (E[C], V[C]) frontier. Every point is optimal
    for some risk aversion; the desk's choice of lambda is the only free
    parameter and should be pinned to a stated cost-variance tradeoff rather
    than tuned to a backtest.
    """

    def __init__(self, solver: Optional[AlmgrenChrissSolver] = None) -> None:
        self.solver = solver or AlmgrenChrissSolver()

    def trace(
        self,
        params: ACParameters,
        impact: LinearImpact,
        lambdas: Sequence[float],
    ) -> List[ExecutionSchedule]:
        out: List[ExecutionSchedule] = []
        for lam in lambdas:
            p = ACParameters(**{**params.__dict__, "risk_aversion": float(lam)})
            out.append(self.solver.solve(p, impact))
        return out

    def lambda_for_horizon(
        self, params: ACParameters, impact: LinearImpact, target_half_life: float
    ) -> float:
        """
        Invert kappa -> lambda for a desired position half-life, which is a far
        more interpretable dial than lambda itself.

            kappa_tilde = 1 / t_half  =>  lambda = eta_tilde / (sigma^2 t_half^2)
        """
        eta_t = impact.eta_tilde(params.tau)
        return eta_t / (params.volatility**2 * max(target_half_life, _EPS) ** 2)


# ===========================================================================
# IV. AVELLANEDA-STOIKOV
# ===========================================================================

@dataclass(frozen=True)
class ASParameters:
    """
    sigma      : annualised absolute price volatility
    gamma      : CARA risk aversion (inventory penalty)
    k          : order-arrival decay, lambda(delta) = A exp(-k delta)
    A          : arrival intensity at the touch, per year
    horizon    : T, years to the terminal liquidation point (session close)
    q_max      : hard inventory bound in lots
    rebate     : maker rebate per share, subtracted from the quoted distance
    drift      : mu, alpha forecast per year -- skews the reservation price
    """

    sigma: float
    gamma: float
    k: float
    A: float
    horizon: float
    q_max: int = 50
    rebate: float = 0.0
    drift: float = 0.0
    tick_size: float = 0.01
    lot_size: int = 1

    def __post_init__(self) -> None:
        for name in ("sigma", "gamma", "k", "A"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.q_max < 1:
            raise ValueError("q_max must be at least 1")


@dataclass(frozen=True)
class QuotePair:
    bid: float
    ask: float
    reservation_price: float
    delta_bid: float
    delta_ask: float
    inventory: int
    time_remaining: float
    quote_bid: bool = True
    quote_ask: bool = True

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def skew(self) -> float:
        """Reservation price minus mid of the quotes; zero at flat inventory."""
        return self.reservation_price - 0.5 * (self.bid + self.ask)


class AvellanedaStoikovQuoter:
    """
    Closed-form Avellaneda-Stoikov (2008).

    Mid price:   dS = mu dt + sigma dW
    Fill flow:   lambda(delta) = A exp(-k delta), Poisson, independent per side
    Objective:   max E[ -exp( -gamma (X_T + q_T S_T) ) ]

    HJB for u(t, x, s, q):

        u_t + (1/2) sigma^2 u_ss
            + max_{delta^b} lambda(delta^b) [ u(t, x - s + delta^b, s, q+1) - u ]
            + max_{delta^a} lambda(delta^a) [ u(t, x + s + delta^a, s, q-1) - u ] = 0

        u(T, x, s, q) = -exp( -gamma (x + q s) )

    With the ansatz u = -exp(-gamma x) exp(-gamma q s) exp(-gamma theta(t,q)),
    the system linearises and yields

        RESERVATION PRICE
            r(t, q) = s - q gamma sigma^2 (T - t)          [+ mu (T - t)]

        TOTAL OPTIMAL SPREAD
            delta^a + delta^b = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)

        INDIVIDUAL DISTANCES
            delta^a(t,q) = (1/gamma) ln(1 + gamma/k) - (2q - 1) gamma sigma^2 (T-t) / 2
            delta^b(t,q) = (1/gamma) ln(1 + gamma/k) + (2q + 1) gamma sigma^2 (T-t) / 2

    Sign check: q > 0 (long) shrinks delta^a and widens delta^b, pulling both
    quotes down so the book works the inventory off. The quote midpoint is
    exactly r, and at q = 0 the quotes are symmetric.

    MAKER REBATE. A fill at distance delta earns delta + m, while the arrival
    intensity still depends on the posted delta. For the one-sided problem
    max_delta (delta + m) A exp(-k delta), the first-order condition gives
    delta* = 1/k - m: the rebate shifts the quote inwards by exactly m. The
    same shift applies here, so the posted distance is the utility-optimal
    distance less the rebate, floored at one tick.

    DRIFT. mu enters the reservation price linearly, which is exact to first
    order in mu and is the coupling point for the Phase 8 residual forecast.
    """

    def __init__(self, params: ASParameters) -> None:
        self.p = params
        self._base_half_spread = math.log(1.0 + params.gamma / params.k) / params.gamma

    # -- components -------------------------------------------------------- #

    def reservation_price(self, mid: float, q: int, time_remaining: float) -> float:
        p = self.p
        return (
            mid
            + p.drift * time_remaining
            - q * p.gamma * p.sigma**2 * time_remaining
        )

    def optimal_spread(self, time_remaining: float) -> float:
        p = self.p
        return p.gamma * p.sigma**2 * time_remaining + 2.0 * self._base_half_spread

    def inventory_risk_scale(self, time_remaining: float) -> float:
        """gamma sigma^2 (T - t): the per-lot skew, in currency."""
        return self.p.gamma * self.p.sigma**2 * time_remaining

    # -- quoting ----------------------------------------------------------- #

    def distances(self, q: int, time_remaining: float) -> Tuple[float, float]:
        """
        Raw optimal (delta_bid, delta_ask), net of rebate but WITHOUT the tick
        floor. Interface parity with GueantHJBSolver.distances.

        These may be negative at large |q|: the optimal action at deep
        inventory is to quote through the mid. Callers that floor them at a
        positive tick destroy exactly that signal, so the floor belongs in the
        placement policy, not here.
        """
        p = self.p
        scale = self.inventory_risk_scale(max(time_remaining, 0.0))
        d_ask = self._base_half_spread - (2 * q - 1) * scale / 2.0 - p.rebate
        d_bid = self._base_half_spread + (2 * q + 1) * scale / 2.0 - p.rebate
        return d_bid, d_ask

    def quote(
        self,
        mid: float,
        q: int,
        time_remaining: float,
        min_half_spread: Optional[float] = None,
    ) -> QuotePair:
        p = self.p
        tr = max(time_remaining, 0.0)
        scale = self.inventory_risk_scale(tr)

        d_ask = self._base_half_spread - (2 * q - 1) * scale / 2.0
        d_bid = self._base_half_spread + (2 * q + 1) * scale / 2.0

        d_ask -= p.rebate
        d_bid -= p.rebate

        floor = min_half_spread if min_half_spread is not None else p.tick_size
        d_ask = max(d_ask, floor)
        d_bid = max(d_bid, floor)

        quote_bid = q < p.q_max
        quote_ask = q > -p.q_max

        ask = self._round_to_tick(mid + d_ask, "up")
        bid = self._round_to_tick(mid - d_bid, "down")
        if ask <= bid:
            ask = bid + p.tick_size

        return QuotePair(
            bid=bid,
            ask=ask,
            reservation_price=self.reservation_price(mid, q, tr),
            delta_bid=mid - bid,
            delta_ask=ask - mid,
            inventory=q,
            time_remaining=tr,
            quote_bid=quote_bid,
            quote_ask=quote_ask,
        )

    def _round_to_tick(self, price: float, direction: Literal["up", "down"]) -> float:
        t = self.p.tick_size
        if t <= 0:
            return price
        return math.ceil(price / t) * t if direction == "up" else math.floor(price / t) * t

    # -- diagnostics -------------------------------------------------------- #

    def expected_fill_rate(self, delta: float) -> float:
        return self.p.A * math.exp(-self.p.k * max(delta, 0.0))

    def quote_grid(self, mid: float, time_remaining: float) -> Dict[int, QuotePair]:
        return {
            q: self.quote(mid, q, time_remaining)
            for q in range(-self.p.q_max, self.p.q_max + 1)
        }


class GueantHJBSolver:
    """
    Exact numerical solution of the AS problem on a bounded inventory grid,
    via the Gueant-Lehalle-Fernandez-Tapia linearisation.

    Under exponential intensity and CARA utility the HJB collapses to a LINEAR
    ODE system in omega_q(t), q = -Q .. Q:

        d/dt omega_q(t) = alpha q^2 omega_q(t) - eta ( omega_{q-1}(t) + omega_{q+1}(t) )

        alpha = k gamma sigma^2 / 2
        eta   = A ( 1 + gamma/k )^{-(1 + k/gamma)}

    with terminal condition omega_q(T) = exp( -k * L(q) ) for a terminal
    liquidation penalty L(q) IN CURRENCY (L == 0 gives omega_q(T) = 1). This
    follows from the CARA ansatz u = -exp(-gamma (x + q s + theta_q)) with
    terminal wealth x + q s - L(q), so theta_q(T) = -L(q) and
    omega_q = exp(k theta_q).

    FIXED IN PHASE 22. This previously read exp(-k * gamma * L(q)), silently
    applying gamma * L -- one tenth of the intended penalty at gamma = 0.1. The
    error is invisible at the start of the session, where the terminal
    condition has washed out, and grows toward the close: a maker long 5 lots
    bid 1.18 below mid instead of 2.90, and kept buying into the close. No
    test exercised the penalty path. Caught by the independent nonlinear-HJB
    solve in hjb_market_maker.py, which now guards it. Integrating
    backwards is a single matrix exponential of a tridiagonal generator:

        omega(t) = exp( -M (T - t) ) omega(T),
        M = diag(alpha q^2) - eta (shift_up + shift_down)

    Optimal distances follow directly:

        delta^b*(t,q) = (1/k) ln( omega_q / omega_{q+1} ) + (1/gamma) ln(1 + gamma/k)
        delta^a*(t,q) = (1/k) ln( omega_q / omega_{q-1} ) + (1/gamma) ln(1 + gamma/k)

    CONSISTENCY. Expanding omega_q(t) ~ 1 - (T-t)(alpha q^2 - 2 eta) for small
    (T-t) gives ln(omega_q/omega_{q+1}) ~ (T-t) alpha (2q+1), so

        delta^b -> (1/gamma) ln(1 + gamma/k) + (2q+1) gamma sigma^2 (T-t) / 2

    which is exactly the closed-form AS bid distance. The solvers agree in the
    limit and diverge where AS's unbounded-inventory approximation breaks down
    -- near the inventory limits and at long horizons, which is where a
    production desk actually operates.

    This class is the production quoter; AvellanedaStoikovQuoter is the fast
    path and the analytic reference.
    """

    def __init__(
        self,
        params: ASParameters,
        n_time_steps: int = 200,
        terminal_penalty: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.p = params
        self.Q = int(params.q_max)
        self.n_steps = int(n_time_steps)
        self.terminal_penalty = terminal_penalty

        self.alpha = params.k * params.gamma * params.sigma**2 / 2.0
        self.eta = params.A * (1.0 + params.gamma / params.k) ** (
            -(1.0 + params.k / params.gamma)
        )
        self._base_half_spread = math.log(1.0 + params.gamma / params.k) / params.gamma

        self.q_grid = np.arange(-self.Q, self.Q + 1, dtype=int)
        self.time_grid: Optional[np.ndarray] = None
        self.log_omega: Optional[np.ndarray] = None  # (n_steps+1, 2Q+1), row t
        self.omega: Optional[np.ndarray] = None      # exp view; may underflow
        self._solved = False

    # -- generator --------------------------------------------------------- #

    def _generator(self) -> np.ndarray:
        m = self.q_grid.size
        M = np.zeros((m, m), dtype=float)
        np.fill_diagonal(M, self.alpha * self.q_grid.astype(float) ** 2)
        idx = np.arange(m - 1)
        M[idx, idx + 1] = -self.eta
        M[idx + 1, idx] = -self.eta
        return M

    def solve(self) -> "GueantHJBSolver":
        """
        Backward integration in LOG SPACE.

        omega decays like exp(-alpha q^2 (T-t)) across the inventory grid. At
        alpha = k gamma sigma^2 / 2 with realistic parameters, omega spans tens
        of orders of magnitude between q = 0 and q = +/- Q over a single
        session. The quotes depend on RATIOS of adjacent omega:

            delta^b = (1/k) ln( omega_q / omega_{q+1} ) + (1/gamma) ln(1+gamma/k)

        so a linear-space recursion that floors omega at machine epsilon
        silently collapses those ratios to ln(1) = 0 at the grid edges and
        returns garbage distances -- non-monotone in both q and (T-t), which
        is how the failure presents rather than as an overflow.

        The generator -M has non-negative off-diagonals, so exp(-M dt) is a
        strictly positive matrix and the matrix-vector product is a valid
        log-sum-exp:

            log w'_i = logsumexp_j ( log step_ij + log w_j )

        exact to full double precision regardless of dynamic range.
        """
        M = self._generator()
        T = self.p.horizon
        dt = T / self.n_steps
        step = linalg.expm(-M * dt)

        # expm of a tridiagonal generator with non-negative off-diagonals is
        # strictly positive; guard against tiny negatives from round-off.
        np.clip(step, 1e-300, None, out=step)
        log_step = np.log(step)

        m = self.q_grid.size
        log_omega = np.empty((self.n_steps + 1, m), dtype=float)

        if self.terminal_penalty is not None:
            lw = -self.p.k * self.terminal_penalty(self.q_grid.astype(float))
        else:
            lw = np.zeros(m, dtype=float)
        log_omega[self.n_steps] = lw - lw.max()

        for i in range(self.n_steps - 1, -1, -1):
            # (m, m) + (m,) broadcast, logsumexp over the source axis
            z = log_step + log_omega[i + 1][None, :]
            mx = z.max(axis=1, keepdims=True)
            lw = (mx[:, 0] + np.log(np.exp(z - mx).sum(axis=1)))
            log_omega[i] = lw - lw.max()   # anchor; ratios are unaffected

        self.time_grid = np.linspace(0.0, T, self.n_steps + 1)
        self.log_omega = log_omega
        self.omega = np.exp(log_omega)     # convenience view; may underflow
        self._solved = True
        return self

    # -- quoting ----------------------------------------------------------- #

    def _time_index(self, time_remaining: float) -> int:
        assert self.time_grid is not None
        t = self.p.horizon - np.clip(time_remaining, 0.0, self.p.horizon)
        return int(np.clip(np.searchsorted(self.time_grid, t), 0, self.n_steps))

    def distances(self, q: int, time_remaining: float) -> Tuple[float, float]:
        """Returns (delta_bid, delta_ask) before rebate, tick and floor."""
        if not self._solved:
            self.solve()
        assert self.log_omega is not None

        i = self._time_index(time_remaining)
        lw = self.log_omega[i]
        j = int(np.clip(q + self.Q, 0, self.q_grid.size - 1))

        # Differences of logs, never a ratio of underflowed linear values.
        if j + 1 < self.q_grid.size:
            d_bid = (lw[j] - lw[j + 1]) / self.p.k + self._base_half_spread
        else:
            d_bid = float("inf")  # at max long inventory: do not bid

        if j - 1 >= 0:
            d_ask = (lw[j] - lw[j - 1]) / self.p.k + self._base_half_spread
        else:
            d_ask = float("inf")  # at max short inventory: do not offer

        return d_bid, d_ask

    def quote(
        self,
        mid: float,
        q: int,
        time_remaining: float,
        min_half_spread: Optional[float] = None,
    ) -> QuotePair:
        p = self.p
        d_bid, d_ask = self.distances(q, time_remaining)

        quote_bid = math.isfinite(d_bid)
        quote_ask = math.isfinite(d_ask)

        floor = min_half_spread if min_half_spread is not None else p.tick_size
        d_bid = max(d_bid - p.rebate, floor) if quote_bid else float("inf")
        d_ask = max(d_ask - p.rebate, floor) if quote_ask else float("inf")

        drift_shift = p.drift * max(time_remaining, 0.0)
        bid = (mid + drift_shift - d_bid) if quote_bid else float("-inf")
        ask = (mid + drift_shift + d_ask) if quote_ask else float("inf")

        if quote_bid:
            bid = math.floor(bid / p.tick_size) * p.tick_size
        if quote_ask:
            ask = math.ceil(ask / p.tick_size) * p.tick_size
        if quote_bid and quote_ask and ask <= bid:
            ask = bid + p.tick_size

        r = mid + drift_shift - q * p.gamma * p.sigma**2 * max(time_remaining, 0.0)

        return QuotePair(
            bid=bid,
            ask=ask,
            reservation_price=r,
            delta_bid=d_bid,
            delta_ask=d_ask,
            inventory=q,
            time_remaining=max(time_remaining, 0.0),
            quote_bid=quote_bid,
            quote_ask=quote_ask,
        )

    def quote_surface(self, time_remaining: float) -> Dict[int, Tuple[float, float]]:
        return {int(q): self.distances(int(q), time_remaining) for q in self.q_grid}

    def stationary_distances(self, q: int) -> Tuple[float, float]:
        """
        GLFT asymptotic (T - t -> infinity), useful as a sanity bound and for
        an always-on book with no natural terminal time:

            delta^b(q) ~ (1/gamma) ln(1 + gamma/k)
                       + (2q+1)/2 * sqrt( sigma^2 gamma / (2 k A)
                                          * (1 + gamma/k)^{1 + k/gamma} )

        Treat as an approximation; validate against `distances` at long
        time_remaining before relying on it.
        """
        p = self.p
        scale = math.sqrt(
            p.sigma**2 * p.gamma / (2.0 * p.k * p.A)
            * (1.0 + p.gamma / p.k) ** (1.0 + p.k / p.gamma)
        )
        d_bid = self._base_half_spread + (2 * q + 1) * scale / 2.0
        d_ask = self._base_half_spread - (2 * q - 1) * scale / 2.0
        return d_bid, d_ask


class IntensityCalibrator:
    """
    Fit lambda(delta) = A exp(-k delta) from observed passive fills.

    Procedure: bucket resting orders by distance from the contemporaneous mid,
    accumulate exposure time and fill counts per bucket, and fit the Poisson
    rate by weighted log-linear regression:

        ln( fills_b / exposure_b ) = ln A - k delta_b

    Weighting by fill count is essential -- unweighted OLS lets a far-touch
    bucket with two fills over a week dominate the slope, and k is the single
    parameter the whole quoting stack is most sensitive to.
    """

    def __init__(self, n_buckets: int = 12, min_fills_per_bucket: int = 5) -> None:
        self.n_buckets = int(n_buckets)
        self.min_fills = int(min_fills_per_bucket)
        self.last_fit_: Dict[str, float] = {}

    def fit(
        self,
        quote_distances: np.ndarray,
        exposure_times: np.ndarray,
        fill_counts: np.ndarray,
    ) -> Tuple[float, float]:
        d = np.asarray(quote_distances, dtype=float)
        e = np.asarray(exposure_times, dtype=float)
        f = np.asarray(fill_counts, dtype=float)

        if not (d.shape == e.shape == f.shape):
            raise ValueError("inputs must share a shape")

        edges = np.quantile(d, np.linspace(0.0, 1.0, self.n_buckets + 1))
        edges[-1] += 1e-12
        idx = np.clip(np.digitize(d, edges[1:-1]), 0, self.n_buckets - 1)

        xs: List[float] = []
        ys: List[float] = []
        ws: List[float] = []
        for b in range(self.n_buckets):
            m = idx == b
            if not m.any():
                continue
            tot_f, tot_e = f[m].sum(), e[m].sum()
            if tot_f < self.min_fills or tot_e <= 0:
                continue
            xs.append(float(np.average(d[m], weights=np.maximum(e[m], _EPS))))
            ys.append(math.log(tot_f / tot_e))
            ws.append(float(tot_f))

        if len(xs) < 3:
            raise ValueError(
                f"only {len(xs)} usable buckets; need >=3. Widen buckets or "
                "lower min_fills_per_bucket."
            )

        x = np.asarray(xs)
        y = np.asarray(ys)
        w = np.asarray(ws)

        W = w / w.sum()
        xm = float(np.dot(W, x))
        ym = float(np.dot(W, y))
        cov = float(np.dot(W, (x - xm) * (y - ym)))
        var = float(np.dot(W, (x - xm) ** 2))
        slope = cov / max(var, _EPS)
        intercept = ym - slope * xm

        k = -slope
        A = math.exp(intercept)

        resid = y - (intercept + slope * x)
        ss_res = float(np.dot(W, resid**2))
        ss_tot = float(np.dot(W, (y - ym) ** 2))

        self.last_fit_ = {
            "A": A,
            "k": k,
            "r_squared": 1.0 - ss_res / max(ss_tot, _EPS),
            "n_buckets_used": float(len(xs)),
            "total_fills": float(w.sum()),
        }

        if k <= 0:
            warnings.warn(
                f"fitted k = {k:.4g} <= 0: fill intensity rises with distance. "
                "Symptom of adverse selection or a stale mid reference."
            )
        return A, k

    def half_spread_floor(self, adverse_selection_bps: float, price: float) -> float:
        """
        AS assumes fill arrivals are independent of the subsequent price move.
        They are not: passive fills are adversely selected. The correction that
        survives contact with production is a floor on the quoted half-spread
        equal to the measured post-fill markout, rather than an adjustment to
        gamma -- inflating gamma distorts the inventory skew as a side effect.
        """
        return adverse_selection_bps * 1e-4 * price


# ===========================================================================
# V. SMART ORDER ROUTER
# ===========================================================================

class ChildOrderAction(Enum):
    QUOTE_PASSIVE = auto()
    PEG_AGGRESSIVE = auto()
    CROSS_SPREAD = auto()
    PAUSE = auto()


@dataclass
class MicrostructureState:
    mid: float
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    interval_volume: float
    realised_vol: float
    time_in_interval: float
    ofi: float = 0.0
    alpha_forecast: float = 0.0     # Phase 8 residual return, per year

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def queue_imbalance(self) -> float:
        tot = self.bid_size + self.ask_size
        return (self.bid_size - self.ask_size) / tot if tot > 0 else 0.0


@dataclass
class SORConfig:
    max_participation: float = 0.10
    urgency_floor: float = 0.0
    urgency_ceiling: float = 3.0
    cross_threshold: float = 0.85      # fraction of slice elapsed before crossing
    behind_tolerance: float = 0.15     # fraction of slice target
    min_child_lots: int = 1
    lot_size: int = 1
    tick_size: float = 0.05
    reprice_ticks: int = 1
    pause_on_spread_multiple: float = 3.0


@dataclass
class ChildOrder:
    action: ChildOrderAction
    side: int
    lots: int
    limit_price: Optional[float]
    urgency: float
    reason: str


class SmartOrderRouter:
    """
    Two-level controller.

    OUTER (minutes to hours): Almgren-Chriss fixes the schedule x_k, balancing
    impact against exposure. Recomputed on a cadence, and on any material
    change to volatility or realised impact.

    INNER (seconds): within interval k the router must acquire n_k. It quotes
    passively at Gueant/AS levels, treating the residual slice quantity as the
    inventory target, and escalates as the interval elapses. The escalation is
    not a heuristic overlay -- it is the same tradeoff AC solves, applied to a
    one-interval horizon where the terminal constraint is hard.

    URGENCY. Define

        u = ( remaining_in_slice / slice_target )
          / max( fraction_of_slice_remaining, eps )

    u = 1 exactly on schedule; u > 1 behind. Urgency compresses the quoted
    half-spread toward the touch by dividing the optimal distance by u, which
    raises the fill intensity as A exp(-k delta / u). At u >= urgency_ceiling
    the router crosses.
    """

    def __init__(
        self,
        ac_solver: AlmgrenChrissSolver,
        quoter: GueantHJBSolver | AvellanedaStoikovQuoter,
        config: SORConfig,
        calibrator: Optional[IntensityCalibrator] = None,
    ) -> None:
        self.ac = ac_solver
        self.quoter = quoter
        self.cfg = config
        self.calibrator = calibrator

        self.schedule: Optional[ExecutionSchedule] = None
        self.filled: float = 0.0
        self.current_interval: int = 0
        self.slice_filled: float = 0.0
        self._fill_log: List[Tuple[float, float, float]] = []

    # -- planning ---------------------------------------------------------- #

    def plan(
        self, params: ACParameters, impact: LinearImpact | PowerLawImpact
    ) -> ExecutionSchedule:
        if isinstance(impact, PowerLawImpact) or params.max_participation is not None:
            solver = NonlinearImpactSolver()
            if isinstance(impact, LinearImpact):
                impact = PowerLawImpact(
                    gamma=impact.gamma, eta=impact.eta, alpha=1.0, epsilon=impact.epsilon
                )
            self.schedule = solver.solve(params, impact)
        else:
            self.schedule = self.ac.solve(params, impact)

        self.filled = 0.0
        self.current_interval = 0
        self.slice_filled = 0.0
        return self.schedule

    def replan(
        self,
        realised_volatility: float,
        realised_impact: Optional[LinearImpact] = None,
        elapsed: float = 0.0,
    ) -> ExecutionSchedule:
        """
        Re-solve over the remaining horizon with the unexecuted quantity. The
        AC solution is time-consistent, so re-solving on the residual problem
        reproduces the tail of the original schedule when inputs are unchanged
        -- any divergence is attributable to the updated parameters, which is
        what makes this diagnosable.
        """
        if self.schedule is None:
            raise RuntimeError("no schedule to replan")

        p = self.schedule.params
        remaining_shares = max(p.total_shares - self.filled, 0.0)
        remaining_h = max(p.horizon - elapsed, _EPS)
        if remaining_shares <= 0:
            return self.schedule

        n_left = max(int(round(p.n_intervals * remaining_h / p.horizon)), 1)
        new_params = ACParameters(
            **{
                **p.__dict__,
                "total_shares": remaining_shares,
                "horizon": remaining_h,
                "n_intervals": n_left,
                "volatility": realised_volatility,
            }
        )
        impact = realised_impact or LinearImpact(gamma=0.0, eta=1.0)
        return self.plan(new_params, impact)

    # -- slice targets ------------------------------------------------------ #

    def slice_target(self, interval: int) -> float:
        if self.schedule is None:
            raise RuntimeError("no schedule")
        return float(self.schedule.trades[min(interval, len(self.schedule.trades) - 1)])

    def urgency(self, interval: int, fraction_elapsed: float) -> float:
        target = self.slice_target(interval)
        if target <= 0:
            return 0.0
        remaining = max(target - self.slice_filled, 0.0) / target
        time_left = max(1.0 - np.clip(fraction_elapsed, 0.0, 1.0), 1e-3)
        return float(np.clip(remaining / time_left, self.cfg.urgency_floor, self.cfg.urgency_ceiling))

    # -- routing ------------------------------------------------------------ #

    def route(
        self, state: MicrostructureState, interval: int, fraction_elapsed: float
    ) -> ChildOrder:
        if self.schedule is None:
            raise RuntimeError("no schedule")

        side = self.schedule.side
        target = self.slice_target(interval)
        remaining = max(target - self.slice_filled, 0.0)

        if remaining < self.cfg.min_child_lots * self.cfg.lot_size:
            return ChildOrder(ChildOrderAction.PAUSE, side, 0, None, 0.0, "slice_complete")

        u = self.urgency(interval, fraction_elapsed)

        # Liquidity gate: a spread blown out well beyond its own norm is not a
        # trading opportunity, and crossing it converts a schedule deviation
        # into a realised cost.
        if self.calibrator is not None and state.spread > 0:
            normal = self.cfg.tick_size * self.cfg.pause_on_spread_multiple
            if state.spread > normal and u < self.cfg.urgency_ceiling:
                return ChildOrder(
                    ChildOrderAction.PAUSE, side, 0, None, u, "spread_dislocated"
                )

        cap = self.cfg.max_participation * max(state.interval_volume, 0.0)
        lots = int(min(remaining, cap if cap > 0 else remaining) // self.cfg.lot_size)
        lots = max(lots, self.cfg.min_child_lots)

        if u >= self.cfg.urgency_ceiling or fraction_elapsed >= self.cfg.cross_threshold:
            px = state.best_ask if side > 0 else state.best_bid
            return ChildOrder(
                ChildOrderAction.CROSS_SPREAD, side, lots, px, u, "behind_schedule"
            )

        # Passive quoting. Two things must be mapped onto the quoter, and
        # getting either wrong produces a router that looks plausible and
        # behaves badly.
        #
        # (1) SCALE. q_max is a market-making inventory bound of order tens of
        #     lots; a slice residual is tens of thousands of shares. Passing
        #     shares directly pins every call to the grid edge and the quoter
        #     returns one saturated distance regardless of state -- silently,
        #     since the clip raises nothing.
        #
        # (2) MEANING. The quoter's inventory must be DEVIATION FROM THE
        #     INTENDED INTRA-SLICE PATH, not the raw residual. Feeding it the
        #     raw residual asserts "you are at your risk limit" at the start of
        #     every slice, and the optimal response to that is to cross
        #     immediately -- so the router would take liquidity on every slice
        #     open. On-schedule must map to q ~ 0 (symmetric quotes at the base
        #     half-spread); only genuine lateness should skew them.
        #
        # Inventory sets the SKEW; urgency scales the DISTANCE. Separating them
        # this way is what stops the two controls from double-counting.
        q_max = int(getattr(self.quoter, "Q", 0) or self.quoter.p.q_max)
        expected_remaining = target * max(1.0 - fraction_elapsed, 0.0)
        deviation = (remaining - expected_remaining) / max(target, _EPS)
        q_signed = int(np.clip(round(-side * q_max * deviation), -q_max, q_max))
        tr_years = max(1.0 - fraction_elapsed, 0.0) * self._interval_length()

        # Read the RAW optimal distance. `quote()` applies a positive tick
        # floor for display, which would convert a through-mid distance into
        # +1 tick and silently discard the aggression the solver asked for.
        d_bid_raw, d_ask_raw = self.quoter.distances(q_signed, tr_years)
        d_raw = d_ask_raw if side < 0 else d_bid_raw
        if not math.isfinite(d_raw):
            return ChildOrder(
                ChildOrderAction.PAUSE, side, 0, None, u, "quoter_side_disabled"
            )

        # Urgency must reduce the quote distance MONOTONICALLY over the whole
        # real line. The Gueant solver returns negative distances at deep
        # inventory near the close -- correctly, since the optimal action there
        # is to quote through the mid -- and d/u would make a negative distance
        # LESS negative, reversing the control exactly when it matters most.
        u_eff = max(u, 1e-3)
        d = d_raw / u_eff if d_raw > 0.0 else d_raw * u_eff

        tick = self.cfg.tick_size
        if side < 0:
            px = math.ceil((state.mid + d) / tick) * tick
            px = min(px, state.best_ask)
            marketable = px <= state.best_bid
            if marketable:
                px = state.best_bid
        else:
            px = math.floor((state.mid - d) / tick) * tick
            px = max(px, state.best_bid)
            marketable = px >= state.best_ask
            if marketable:
                px = state.best_ask

        if marketable:
            action = ChildOrderAction.CROSS_SPREAD
            reason = "quoter_marketable"
        elif (side < 0 and px <= state.best_ask) or (side > 0 and px >= state.best_bid):
            action = ChildOrderAction.PEG_AGGRESSIVE
            reason = "passive_quote"
        else:
            action = ChildOrderAction.QUOTE_PASSIVE
            reason = "passive_quote"

        return ChildOrder(action, side, lots, px, u, reason)

    def _interval_length(self) -> float:
        assert self.schedule is not None
        return self.schedule.params.tau

    # -- fills and attribution ---------------------------------------------- #

    def on_fill(self, qty: float, price: float, decision_price: float) -> None:
        self.filled += qty
        self.slice_filled += qty
        self._fill_log.append((qty, price, decision_price))

    def advance_interval(self) -> None:
        self.current_interval += 1
        self.slice_filled = 0.0

    def implementation_shortfall(self, arrival_price: float) -> Dict[str, float]:
        """
        Decompose realised cost against the arrival benchmark:

            IS = (P_fill - P_decision) * side   =   DELAY + IMPACT

        Delay is latency in our own pipeline; impact is what the market charged.
        They have different fixes -- delay is an engineering problem, impact is
        a scheduling one -- so a single blended slippage number is not
        actionable.
        """
        if not self._fill_log:
            return {}
        side = self.schedule.side if self.schedule else -1
        qty = np.array([f[0] for f in self._fill_log])
        px = np.array([f[1] for f in self._fill_log])
        dp = np.array([f[2] for f in self._fill_log])
        w = qty / max(qty.sum(), _EPS)

        vwap = float(np.dot(w, px))
        decision_vwap = float(np.dot(w, dp))
        sign = -side  # cost is positive when we pay up

        return {
            "filled_qty": float(qty.sum()),
            "vwap": vwap,
            "total_is_bps": 1e4 * sign * (vwap - arrival_price) / max(arrival_price, _EPS),
            "delay_bps": 1e4 * sign * (decision_vwap - arrival_price) / max(arrival_price, _EPS),
            "impact_bps": 1e4 * sign * (vwap - decision_vwap) / max(arrival_price, _EPS),
            "completion": float(qty.sum() / max(self.schedule.params.total_shares, _EPS))
            if self.schedule
            else float("nan"),
        }


# ===========================================================================
# SELF-TESTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Reference calibration used throughout the tests.
#
# UNITS. `volatility` and ASParameters.sigma are ABSOLUTE currency volatility
# per sqrt(year), not fractional. A 1500-rupee name at 30% annual vol has
# sigma = 450. Passing 0.30 here makes kappa*T ~ 0 and every schedule silently
# collapses to TWAP, which is the single easiest way to mis-wire this module.
# ---------------------------------------------------------------------------

PRICE = 1500.0
ANNUAL_VOL_FRAC = 0.30
SIGMA_ABS = ANNUAL_VOL_FRAC * PRICE          # 450.0 currency / sqrt(year)
ADV = 10_000_000.0                            # shares/day
TRADING_DAYS = 252
DAY = 1.0 / TRADING_DAYS
HOURS_PER_DAY = 6.25                          # NSE continuous session

# Square-root law anchored at 10% participation:
#   dP/P = 0.75 * sigma_daily * sqrt(0.10) = 44.8 bps  ->  6.72 currency/share
# Rate at that participation = 0.10 * ADV * 252 = 2.52e8 shares/year.
_REF_COST_PER_SHARE = 0.75 * (ANNUAL_VOL_FRAC / math.sqrt(TRADING_DAYS)) * math.sqrt(0.10) * PRICE
_REF_RATE = 0.10 * ADV * TRADING_DAYS
_SLOPE = _REF_COST_PER_SHARE / _REF_RATE      # ~2.67e-8
GAMMA_PERM = _SLOPE / 3.0
ETA_TEMP = 2.0 * _SLOPE / 3.0
ETA_SQRT = _REF_COST_PER_SHARE / math.sqrt(_REF_RATE)   # h(v) = eta |v|^0.5

# kappa^2 = lambda sigma^2 / eta_tilde  ->  lambda = kappa^2 eta / sigma^2.
# kappa = 252 means a one-day half-life; kappa = 1008 means a quarter-day.
def _lambda_for_kappa(kappa: float) -> float:
    return kappa**2 * ETA_TEMP / SIGMA_ABS**2


def _hours(years: float) -> float:
    return years * TRADING_DAYS * HOURS_PER_DAY


def _test_ac_twap_limit() -> None:
    p = ACParameters(
        total_shares=1_000_000, horizon=DAY, n_intervals=20,
        volatility=SIGMA_ABS, risk_aversion=0.0,
    )
    imp = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP, epsilon=0.01)
    s = AlmgrenChrissSolver().solve(p, imp)
    assert np.allclose(s.trades, p.total_shares / p.n_intervals, rtol=1e-10)
    assert s.kappa == 0.0
    print(f"[ok] lambda=0 -> TWAP exactly, slice={s.trades[0]:,.0f}")


def _test_ac_closed_form_agreement() -> None:
    print(f"    {'kappa':>8} {'half-life':>11} {'E[C] bps':>10} {'sd[C] bps':>10} {'rel err':>10}")
    prev_first_slice = None
    for kappa in (126.0, 252.0, 756.0, 2016.0):
        lam = _lambda_for_kappa(kappa)
        p = ACParameters(
            total_shares=1_000_000, horizon=DAY, n_intervals=50,
            volatility=SIGMA_ABS, risk_aversion=lam,
        )
        imp = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP, epsilon=0.0)
        s = AlmgrenChrissSolver().solve(p, imp)

        err = s.diagnostics["cost_relative_error"]
        assert err < 1e-8, f"closed form vs trajectory disagree: {err:.3e}"

        # arccosh(1+x) = sqrt(2x)(1 - x/12 + ...) with x = tau^2 kappa~^2 / 2,
        # so the discrete root sits below the continuous one by exactly
        # kappa~ tau^2 kappa~^2 / 24 to leading order. Assert the observed gap
        # matches that, rather than asserting it is small -- the gap IS the
        # discretisation and should be verified, not tolerated.
        k_c = s.diagnostics["kappa_continuous"]
        predicted = k_c * (1.0 - (p.tau * k_c) ** 2 / 24.0)
        assert abs(s.kappa - predicted) / k_c < 1e-4, (
            f"discrete kappa {s.kappa:.3f} != predicted {predicted:.3f}"
        )

        notional = p.total_shares * PRICE
        print(
            f"    {s.kappa:8.1f} {_hours(s.half_life):9.2f}h "
            f"{1e4*s.expected_cost/notional:10.2f} "
            f"{1e4*s.cost_stdev/notional:10.2f} {err:10.2e}"
        )
        if prev_first_slice is not None:
            assert s.trades[0] > prev_first_slice, "higher lambda must front-load more"
        prev_first_slice = s.trades[0]
    print("[ok] closed-form moments match trajectory; schedules front-load with lambda")


def _test_ac_monotone_in_lambda() -> None:
    p0 = ACParameters(
        total_shares=500_000, horizon=DAY, n_intervals=40,
        volatility=SIGMA_ABS, risk_aversion=0.0,
    )
    imp = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP)
    lams = [_lambda_for_kappa(k) for k in (100.0, 250.0, 500.0, 1000.0, 2000.0)]
    scheds = EfficientFrontier().trace(p0, imp, lams)
    costs = np.array([s.expected_cost for s in scheds])
    vars_ = np.array([s.cost_variance for s in scheds])
    assert np.all(np.diff(costs) > 0), "cost must rise with risk aversion"
    assert np.all(np.diff(vars_) < 0), "variance must fall with risk aversion"
    print(
        f"[ok] frontier monotone over kappa 100->2000: "
        f"cost +{100*(costs[-1]/costs[0]-1):.1f}%, sd -{100*(1-np.sqrt(vars_[-1]/vars_[0])):.1f}%"
    )


def _test_lambda_for_horizon_inverts() -> None:
    p = ACParameters(
        total_shares=1_000_000, horizon=DAY, n_intervals=60,
        volatility=SIGMA_ABS, risk_aversion=0.0,
    )
    imp = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP)
    fr = EfficientFrontier()
    target_hl = DAY / 3.0
    lam = fr.lambda_for_horizon(p, imp, target_hl)
    s = fr.trace(p, imp, [lam])[0]
    rel = abs(s.half_life - target_hl) / target_hl
    assert rel < 5e-3, f"half-life inversion off by {rel:.3e}"
    print(f"[ok] lambda_for_horizon inverts: target {_hours(target_hl):.2f}h -> "
          f"realised {_hours(s.half_life):.2f}h")


def _test_nonlinear_matches_linear() -> None:
    lam = _lambda_for_kappa(500.0)
    p = ACParameters(
        total_shares=400_000, horizon=DAY, n_intervals=25,
        volatility=SIGMA_ABS, risk_aversion=lam,
    )
    lin = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP, epsilon=0.0)
    s_lin = AlmgrenChrissSolver().solve(p, lin)
    pw = PowerLawImpact(gamma=GAMMA_PERM, eta=ETA_TEMP, alpha=1.0, epsilon=0.0)
    s_nl = NonlinearImpactSolver().solve(p, pw, warm_start=s_lin.holdings)

    assert s_lin.kappa * p.horizon > 0.5, "test must exercise real curvature"
    rel_cost = abs(s_nl.expected_cost - s_lin.expected_cost) / abs(s_lin.expected_cost)
    max_dev = float(np.max(np.abs(s_nl.holdings - s_lin.holdings))) / p.total_shares
    assert rel_cost < 1e-3, f"alpha=1 must reproduce closed form: {rel_cost:.3e}"
    assert max_dev < 5e-3, f"trajectories diverge: {max_dev:.3e}"
    print(
        f"[ok] SLSQP(alpha=1) reproduces sinh solution: "
        f"cost rel_err={rel_cost:.2e}, max holdings dev={max_dev:.2e} of parent"
    )


def _test_sqrt_impact_and_participation_cap() -> None:
    lam = _lambda_for_kappa(400.0)
    p = ACParameters(
        total_shares=1_000_000, horizon=DAY, n_intervals=30,
        volatility=SIGMA_ABS, risk_aversion=lam,
        adv_shares=ADV, max_participation=0.12,
    )
    sqrt_imp = PowerLawImpact(gamma=GAMMA_PERM, eta=ETA_SQRT, alpha=0.5)
    s = NonlinearImpactSolver().solve(p, sqrt_imp)

    part = s.participation
    assert part is not None
    assert part.max() <= 0.12 + 1e-6, f"POV cap violated: {part.max():.4f}"
    assert abs(s.trades.sum() - p.total_shares) / p.total_shares < 1e-6
    assert np.all(s.trades >= -1e-6), "monotonicity violated"
    print(
        f"[ok] sqrt-law under 12% POV: max part={part.max():.4f}, "
        f"front/back slice={s.trades[0]/s.trades[-1]:.2f}, "
        f"cost={1e4*s.expected_cost/(p.total_shares*PRICE):.1f}bps"
    )


def _test_participation_infeasibility_detected() -> None:
    p = ACParameters(
        total_shares=2_000_000, horizon=DAY, n_intervals=30,
        volatility=SIGMA_ABS, risk_aversion=_lambda_for_kappa(400.0),
        adv_shares=ADV, max_participation=0.15,
    )
    try:
        NonlinearImpactSolver().solve(
            p, PowerLawImpact(gamma=GAMMA_PERM, eta=ETA_SQRT, alpha=0.5)
        )
    except ValueError as exc:
        assert "infeasible" in str(exc)
        print(f"[ok] infeasible POV cap refused up front ({str(exc).split(':')[1].strip()[:44]}...)")
        return
    raise AssertionError("must refuse a parent larger than the cap permits")


def _test_drift_tilts_schedule() -> None:
    lam = _lambda_for_kappa(500.0)
    base = dict(
        total_shares=1_000_000, horizon=DAY, n_intervals=40,
        volatility=SIGMA_ABS, risk_aversion=lam, side=-1,
    )
    imp = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP)
    solver = AlmgrenChrissSolver()

    flat = solver.solve(ACParameters(**base), imp)
    # Selling into an expected rally: hold longer.
    up = solver.solve(ACParameters(**{**base, "drift": 0.20 * PRICE}), imp)
    down = solver.solve(ACParameters(**{**base, "drift": -0.20 * PRICE}), imp)

    mid = len(flat.holdings) // 2
    assert up.holdings[mid] > flat.holdings[mid] > down.holdings[mid], (
        "positive drift must slow a sell, negative must hurry it"
    )
    print(
        f"[ok] drift tilts schedule: holdings at midpoint "
        f"down={down.holdings[mid]/1e6:.3f}M < flat={flat.holdings[mid]/1e6:.3f}M "
        f"< up={up.holdings[mid]/1e6:.3f}M"
    )


def _test_eta_tilde_guard() -> None:
    p = ACParameters(
        total_shares=100_000, horizon=DAY, n_intervals=2,
        volatility=SIGMA_ABS, risk_aversion=_lambda_for_kappa(250.0),
    )
    try:
        AlmgrenChrissSolver().solve(p, LinearImpact(gamma=1e-3, eta=1e-9))
    except ValueError as exc:
        assert "eta_tilde" in str(exc)
        print("[ok] eta_tilde <= 0 refused (permanent impact dominates within a bucket)")
        return
    raise AssertionError("must refuse eta_tilde <= 0")


def _test_kappa_numerical_stability() -> None:
    """
    arccosh(1+x) for x -> 0 is where a naive implementation loses precision.

    The discrete root satisfies kappa_d = kappa~ (1 - (tau kappa~)^2/24 + ...),
    so the relative gap must fall as 1/N^2 and must MATCH that prediction at
    every N. A naive np.arccosh(1.0 + x) agrees at coarse N but saturates
    around sqrt(eps) once x drops below ~1e-16, which is the regime reached at
    N = 50,000 -- so this test discriminates the stable implementation from
    the naive one rather than merely bounding the error.
    """
    imp = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP)
    rows = []
    for n in (50, 500, 5_000, 50_000):
        p = ACParameters(
            total_shares=1e6, horizon=DAY, n_intervals=n,
            volatility=SIGMA_ABS, risk_aversion=_lambda_for_kappa(504.0),
        )
        k_d, k_c = AlmgrenChrissSolver.kappa(p, imp)
        observed = (k_c - k_d) / k_c
        predicted = (p.tau * k_c) ** 2 / 24.0
        assert abs(observed - predicted) / predicted < 0.02, (
            f"N={n}: gap {observed:.3e} != predicted {predicted:.3e}"
        )
        rows.append((n, observed))

    for (n0, e0), (n1, e1) in zip(rows, rows[1:]):
        ratio = e0 / e1
        assert 90.0 < ratio < 110.0, f"gap fell {ratio:.1f}x for 10x N; expected ~100x"
    print(
        "[ok] kappa gap tracks (tau kappa)^2/24 and falls 100x per 10x N: "
        + ", ".join(f"N={n}:{e:.1e}" for n, e in rows)
    )


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov / Gueant
# ---------------------------------------------------------------------------

AS_GAMMA = 1e-4      # CARA, 1/currency
AS_K = 20.0          # intensity decay, 1/currency -> base half-spread ~ 1/k = 0.05
AS_A = 50.0 * TRADING_DAYS   # ~50 touch fills per session


def _as_params(**kw) -> ASParameters:
    base = dict(
        sigma=SIGMA_ABS, gamma=AS_GAMMA, k=AS_K, A=AS_A,
        horizon=DAY, q_max=30, tick_size=0.05,
    )
    base.update(kw)
    return ASParameters(**base)


def _test_as_sign_convention() -> None:
    p = _as_params()
    q = AvellanedaStoikovQuoter(p)
    flat = q.quote(PRICE, 0, p.horizon)
    long_ = q.quote(PRICE, 10, p.horizon)
    short = q.quote(PRICE, -10, p.horizon)

    assert abs(flat.delta_bid - flat.delta_ask) < 1e-9, "flat must be symmetric"
    assert long_.reservation_price < flat.reservation_price < short.reservation_price
    assert long_.ask <= flat.ask and long_.bid <= flat.bid, "long skews down"
    assert short.bid >= flat.bid and short.ask >= flat.ask, "short skews up"

    total = q.optimal_spread(p.horizon)
    assert abs((flat.delta_bid + flat.delta_ask) - total) < 2 * p.tick_size
    print(
        f"[ok] AS: r(+10)={long_.reservation_price:.3f} < r(0)={flat.reservation_price:.3f} "
        f"< r(-10)={short.reservation_price:.3f}; spread={total:.4f}"
    )


def _test_as_spread_identity() -> None:
    p = _as_params()
    q = AvellanedaStoikovQuoter(p)
    for tr_frac in (0.05, 0.5, 1.0):
        tr = p.horizon * tr_frac
        scale = q.inventory_risk_scale(tr)
        for inv in (-8, 0, 8):
            d_a = q._base_half_spread - (2 * inv - 1) * scale / 2.0
            d_b = q._base_half_spread + (2 * inv + 1) * scale / 2.0
            assert abs((d_a + d_b) - q.optimal_spread(tr)) < 1e-12
            assert abs((d_a - d_b) / 2.0 + inv * scale) < 1e-12
    print("[ok] delta_a + delta_b = gamma sigma^2 (T-t) + (2/gamma)ln(1+gamma/k) exactly")


def _test_gueant_matches_as_skew() -> None:
    """
    The AS closed form is the small-(T-t) expansion of the Gueant system.
    Compare the SKEW, which is the inventory-dependent part and therefore the
    discriminating quantity -- the base half-spread agrees by construction.
    """
    p = _as_params(q_max=30)
    g = GueantHJBSolver(p, n_time_steps=800).solve()
    a = AvellanedaStoikovQuoter(p)

    tr = p.horizon * 0.01
    worst = 0.0
    for inv in (-6, -3, -1, 0, 1, 3, 6):
        gb, ga = g.distances(inv, tr)
        skew_g = (gb - ga) / 2.0
        skew_a = inv * a.inventory_risk_scale(tr)
        denom = max(abs(skew_a), 1e-9)
        if inv != 0:
            worst = max(worst, abs(skew_g - skew_a) / denom)
        else:
            assert abs(skew_g) < 1e-9, "zero inventory must give zero skew"
    assert worst < 0.05, f"Gueant/AS skew disagree near terminal: {worst:.3e}"
    print(f"[ok] Gueant HJB reproduces AS skew near T, max rel diff={worst:.2e}")


def _test_gueant_base_half_spread() -> None:
    p = _as_params()
    g = GueantHJBSolver(p, n_time_steps=400).solve()
    tr = p.horizon * 0.005
    gb, ga = g.distances(0, tr)
    analytic = math.log(1.0 + p.gamma / p.k) / p.gamma
    rel = abs(0.5 * (gb + ga) - analytic) / analytic
    assert rel < 0.02, f"half-spread off: {0.5*(gb+ga):.5f} vs {analytic:.5f}"
    print(f"[ok] Gueant half-spread at q=0 -> (1/gamma)ln(1+gamma/k)={analytic:.5f} "
          f"(rel {rel:.2e})")


def _test_gueant_inventory_bounds_and_monotone() -> None:
    p = _as_params(q_max=10)
    g = GueantHJBSolver(p, n_time_steps=300).solve()
    at_max = g.quote(PRICE, 10, p.horizon * 0.5)
    at_min = g.quote(PRICE, -10, p.horizon * 0.5)
    assert not at_max.quote_bid and at_max.quote_ask, "max long must stop bidding"
    assert at_min.quote_bid and not at_min.quote_ask, "max short must stop offering"

    prev = None
    for inv in range(-9, 10):
        db, _ = g.distances(inv, p.horizon * 0.5)
        if prev is not None:
            assert db >= prev - 1e-9, f"bid distance non-monotone at q={inv}"
        prev = db
    print("[ok] Gueant: inventory bounds kill the offending side; skew monotone in q")


def _test_gueant_widens_with_horizon() -> None:
    p = _as_params()
    g = GueantHJBSolver(p, n_time_steps=600).solve()
    skews = []
    for frac in (0.05, 0.25, 0.60, 1.00):
        db, da = g.distances(5, p.horizon * frac)
        skews.append((db - da) / 2.0)
    assert all(np.diff(skews) > 0), f"skew must grow with time to close: {skews}"
    print(f"[ok] inventory skew grows with horizon: {[f'{s:.4f}' for s in skews]}")


def _test_rebate_shifts_quotes_in() -> None:
    rebate = 0.02
    q0 = AvellanedaStoikovQuoter(_as_params(tick_size=1e-6))
    q1 = AvellanedaStoikovQuoter(_as_params(rebate=rebate, tick_size=1e-6))
    a = q0.quote(PRICE, 0, DAY)
    b = q1.quote(PRICE, 0, DAY)
    assert abs((a.delta_bid - b.delta_bid) - rebate) < 1e-9
    assert abs((a.delta_ask - b.delta_ask) - rebate) < 1e-9
    print(f"[ok] maker rebate {rebate} shifts each quote in by exactly {rebate}")


def _test_intensity_calibration_roundtrip() -> None:
    rng = np.random.default_rng(11)
    A_true, k_true = 180.0 * TRADING_DAYS, 20.0
    d = rng.uniform(0.0, 0.25, 60_000)
    e = rng.uniform(0.5, 2.0, 60_000) / (TRADING_DAYS * HOURS_PER_DAY * 3600)
    f = rng.poisson(A_true * np.exp(-k_true * d) * e)

    cal = IntensityCalibrator(n_buckets=14, min_fills_per_bucket=3)
    A_hat, k_hat = cal.fit(d, e, f)
    assert abs(k_hat - k_true) / k_true < 0.10, f"k off: {k_hat:.2f} vs {k_true}"
    assert abs(A_hat - A_true) / A_true < 0.35, f"A off: {A_hat:.0f} vs {A_true}"
    print(
        f"[ok] intensity roundtrip: k={k_hat:6.2f} (true {k_true}), "
        f"A={A_hat:9.0f} (true {A_true:.0f}), R2={cal.last_fit_['r_squared']:.4f}"
    )


def _test_calibration_flags_adverse_selection() -> None:
    """Fill intensity rising with distance is the signature of a stale mid."""
    d = np.repeat(np.linspace(0.01, 0.2, 8), 400)
    e = np.full_like(d, 1.0 / (TRADING_DAYS * HOURS_PER_DAY * 3600))
    f = np.repeat(np.linspace(1.0, 9.0, 8), 400)
    cal = IntensityCalibrator(n_buckets=8, min_fills_per_bucket=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, k_hat = cal.fit(d, e, f)
        assert any("adverse selection" in str(x.message) for x in w)
    print(f"[ok] negative k ({k_hat:.2f}) raises the adverse-selection warning")


# ---------------------------------------------------------------------------
# Smart Order Router
# ---------------------------------------------------------------------------

def _make_sor() -> Tuple[SmartOrderRouter, ACParameters]:
    p = ACParameters(
        total_shares=300_000, horizon=DAY, n_intervals=12,
        volatility=SIGMA_ABS, risk_aversion=_lambda_for_kappa(500.0),
        adv_shares=ADV,
    )
    imp = LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP, epsilon=0.02)
    asp = _as_params(horizon=p.tau, q_max=60)
    sor = SmartOrderRouter(
        AlmgrenChrissSolver(),
        GueantHJBSolver(asp, n_time_steps=120).solve(),
        SORConfig(tick_size=0.05, lot_size=1, max_participation=0.15),
    )
    sor.plan(p, imp)
    return sor, p


def _book(spread_ticks: int = 1) -> MicrostructureState:
    half = 0.05 * spread_ticks / 2.0
    return MicrostructureState(
        mid=PRICE, best_bid=PRICE - half, best_ask=PRICE + half,
        bid_size=500, ask_size=500, interval_volume=ADV / 12,
        realised_vol=SIGMA_ABS, time_in_interval=0.0,
    )


def _test_sor_escalation() -> None:
    sor, _ = _make_sor()
    st = _book()
    early = sor.route(st, 0, 0.10)
    mid = sor.route(st, 0, 0.60)
    late = sor.route(st, 0, 0.95)

    assert early.urgency < mid.urgency < late.urgency
    assert late.action is ChildOrderAction.CROSS_SPREAD
    assert early.action in (ChildOrderAction.QUOTE_PASSIVE, ChildOrderAction.PEG_AGGRESSIVE)
    print(
        f"[ok] SOR escalates: 10%->{early.action.name}(u={early.urgency:.2f}) "
        f"60%->{mid.action.name}(u={mid.urgency:.2f}) "
        f"95%->{late.action.name}(u={late.urgency:.2f})"
    )


def _test_sor_urgency_tightens_quote() -> None:
    """
    Needs a book wide enough for the quote to move. At a 1-tick spread the
    placement clips to the touch at every urgency level, so a narrow book
    makes this assertion vacuous rather than passing on merit.
    """
    sor, _ = _make_sor()
    st = _book(spread_ticks=40)          # 1.00 wide, room to work inside
    target = sor.slice_target(0)

    # Hold the residual fixed at a mid-grid inventory so the quoter is not
    # pinned at its bound, and sweep only elapsed time. Then urgency is the
    # single varying input and the effect is attributable.
    quotes = []
    for frac in (0.05, 0.25, 0.45, 0.65):
        sor.slice_filled = 0.80 * target
        o = sor.route(st, 0, frac)
        assert o.limit_price is not None
        quotes.append((o.urgency, o.limit_price, o.action))

    urg = [q[0] for q in quotes]
    px = [q[1] for q in quotes]
    assert all(np.diff(urg) > 0), f"urgency must rise as the slice elapses: {urg}"
    # Selling: tightening means a strictly lower ask.
    assert all(np.diff(px) < 1e-9), f"quote must tighten monotonically: {px}"
    assert min(px) >= st.best_bid, "passive quote must not sit through the bid"
    print(
        "[ok] urgency tightens the ask monotonically: "
        + " -> ".join(f"{p:.2f}(u={u:.2f})" for u, p, _ in quotes)
        + f"  [touch {st.best_bid:.2f}/{st.best_ask:.2f}]"
    )


def _test_negative_distance_crosses() -> None:
    """
    Deep inventory near the close drives the Gueant distance negative -- the
    optimal action is to quote through the mid. The router must translate that
    into a cross, and urgency must make it MORE aggressive, not less.
    """
    sor, _ = _make_sor()
    st = _book(spread_ticks=1)           # 1499.975 / 1500.025
    frac = 0.50                          # below cross_threshold (0.85), and
    sor.slice_filled = 0.0               # urgency 2.0 < ceiling 3.0, so the
                                         # cross must come from the quoter
    q_max = sor.quoter.Q
    dev = 1.0 - (1.0 - frac)
    q = int(round(q_max * dev))
    tr = (1.0 - frac) * sor._interval_length()
    _, d_ask = sor.quoter.distances(q, tr)
    assert d_ask < 0.0, f"expected a through-mid distance at q={q}, got {d_ask:+.5f}"

    o = sor.route(st, 0, frac)
    assert o.urgency < sor.cfg.urgency_ceiling, "urgency must not be the trigger"
    assert o.action is ChildOrderAction.CROSS_SPREAD
    assert o.reason == "quoter_marketable", f"crossed for the wrong reason: {o.reason}"
    print(
        f"[ok] behind at q={q} gives d_ask={d_ask:+.5f} (through mid) -> "
        f"{o.action.name} via {o.reason} at {o.limit_price:.3f} (u={o.urgency:.2f})"
    )


def _test_on_schedule_is_not_skewed() -> None:
    """On-schedule must map to q ~ 0 and quote symmetrically, not cross."""
    sor, _ = _make_sor()
    st = _book(spread_ticks=20)
    target = sor.slice_target(0)
    for frac in (0.1, 0.3, 0.5, 0.7):
        sor.slice_filled = target * frac          # exactly on the linear path
        o = sor.route(st, 0, frac)
        assert o.action is not ChildOrderAction.CROSS_SPREAD, (
            f"on-schedule at t={frac} must not take liquidity"
        )
        assert abs(o.urgency - 1.0) < 1e-9, f"on-schedule urgency must be 1: {o.urgency}"
    print("[ok] on-schedule quotes passively at u=1.00 throughout the slice")


def _test_sor_respects_slice_completion() -> None:
    sor, _ = _make_sor()
    target = sor.slice_target(0)
    sor.slice_filled = target
    order = sor.route(_book(), 0, 0.30)
    assert order.action is ChildOrderAction.PAUSE and order.lots == 0
    print(f"[ok] completed slice pauses ({target:,.0f} shares filled)")


def _test_sor_participation_cap() -> None:
    sor, p = _make_sor()
    st = _book()
    st.interval_volume = 10_000
    order = sor.route(st, 0, 0.20)
    cap = sor.cfg.max_participation * st.interval_volume
    assert order.lots <= max(cap, sor.cfg.min_child_lots) + 1
    print(f"[ok] child size capped at {sor.cfg.max_participation:.0%} of "
          f"{st.interval_volume:,.0f} -> {order.lots:,} lots")


def _test_sor_shortfall_decomposition() -> None:
    sor, p = _make_sor()
    arrival = PRICE
    for i in range(5):
        sor.on_fill(10_000, PRICE + 0.10 + 0.02 * i, PRICE + 0.04 * i)
    d = sor.implementation_shortfall(arrival)
    assert abs(d["total_is_bps"] - (d["delay_bps"] + d["impact_bps"])) < 1e-6
    print(
        f"[ok] IS decomposes exactly: total={d['total_is_bps']:.3f}bps = "
        f"delay={d['delay_bps']:.3f} + impact={d['impact_bps']:.3f}"
    )


def _test_sor_replan_time_consistency() -> None:
    """
    AC is time-consistent: re-solving on the residual problem with unchanged
    inputs must reproduce the tail of the original schedule. Divergence is
    then attributable to the updated parameters, which is what makes a
    replan diagnosable rather than mysterious.
    """
    sor, p = _make_sor()
    original = sor.schedule
    assert original is not None
    k_orig = original.kappa

    executed = original.trades[:4].sum()
    sor.filled = executed
    elapsed = 4 * p.tau
    new = sor.replan(
        realised_volatility=SIGMA_ABS,
        realised_impact=LinearImpact(gamma=GAMMA_PERM, eta=ETA_TEMP, epsilon=0.02),
        elapsed=elapsed,
    )
    assert abs(new.kappa - k_orig) / k_orig < 1e-6, "kappa must be unchanged"
    tail = original.holdings[4:] / original.holdings[4]
    fresh = new.holdings / new.holdings[0]
    assert np.max(np.abs(tail - fresh)) < 1e-6, "replan must reproduce the tail"
    print(f"[ok] replan is time-consistent (kappa {k_orig:.1f} preserved, "
          f"max tail dev {np.max(np.abs(tail - fresh)):.2e})")


def _run(title: str, tests) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    for t in tests:
        t()
    print()


if __name__ == "__main__":
    print(
        f"calibration: price={PRICE:.0f}  sigma_abs={SIGMA_ABS:.1f}/sqrt(yr)  "
        f"ADV={ADV:,.0f}  gamma={GAMMA_PERM:.3e}  eta={ETA_TEMP:.3e}\n"
    )
    _run("ALMGREN-CHRISS", [
        _test_ac_twap_limit,
        _test_ac_closed_form_agreement,
        _test_ac_monotone_in_lambda,
        _test_lambda_for_horizon_inverts,
        _test_nonlinear_matches_linear,
        _test_sqrt_impact_and_participation_cap,
        _test_participation_infeasibility_detected,
        _test_drift_tilts_schedule,
        _test_eta_tilde_guard,
        _test_kappa_numerical_stability,
    ])
    _run("AVELLANEDA-STOIKOV / GUEANT HJB", [
        _test_as_sign_convention,
        _test_as_spread_identity,
        _test_gueant_matches_as_skew,
        _test_gueant_base_half_spread,
        _test_gueant_inventory_bounds_and_monotone,
        _test_gueant_widens_with_horizon,
        _test_rebate_shifts_quotes_in,
        _test_intensity_calibration_roundtrip,
        _test_calibration_flags_adverse_selection,
    ])
    _run("SMART ORDER ROUTER", [
        _test_sor_escalation,
        _test_sor_urgency_tightens_quote,
        _test_negative_distance_crosses,
        _test_on_schedule_is_not_skewed,
        _test_sor_respects_slice_completion,
        _test_sor_participation_cap,
        _test_sor_shortfall_decomposition,
        _test_sor_replan_time_consistency,
    ])
    print("all checks passed")

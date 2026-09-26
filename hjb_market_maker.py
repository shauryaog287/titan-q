"""
hjb_market_maker.py
===================

Phase 22: Market making by stochastic optimal control -- the Avellaneda-Stoikov
problem solved through its Hamilton-Jacobi-Bellman equation.

MODEL (Avellaneda & Stoikov 2008)
    mid        dS_t = sigma dW_t
    quotes     bid S - delta^b, ask S + delta^a
    fills      Poisson, intensity lambda(delta) = A exp(-k delta) on each side
    inventory  q in [-Q, Q]; no bid at +Q, no offer at -Q
    objective  max E[ -exp( -gamma ( X_T + q_T S_T - L(q_T) ) ) ]

L(q) is a terminal liquidation penalty in currency; L = 0 is the original paper.

THE HJB, AND HOW IT COLLAPSES
-----------------------------
For u(t, x, s, q):

    0 = u_t + (1/2) sigma^2 u_ss
        + sup_{db} A e^{-k db} [ u(t, x - s + db, s, q + 1) - u ]
        + sup_{da} A e^{-k da} [ u(t, x + s + da, s, q - 1) - u ]

CARA utility makes wealth and price separable. The ansatz
u = -exp( -gamma ( x + q s + theta_q(t) ) ) reduces the four-variable PDE EXACTLY
to a system of ODEs in theta_q(t), one per inventory level:

    theta_q' = (1/2) sigma^2 gamma q^2 - H( theta_q - theta_{q+1} ) - H( theta_q - theta_{q-1} )

    H(p) = sup_delta (A/gamma) e^{-k delta} ( 1 - e^{-gamma (delta - p)} )
         = (eta / k) e^{-k p},        eta = A ( 1 + gamma/k )^{-(1 + k/gamma)}

attained at delta* = p + (1/gamma) ln(1 + gamma/k). Hence the optimal quotes

    delta^b = theta_q - theta_{q+1} + (1/gamma) ln(1 + gamma/k)
    delta^a = theta_q - theta_{q-1} + (1/gamma) ln(1 + gamma/k)

This module solves that NONLINEAR system directly, with an implicit stiff
integrator, and checks that the result satisfies the original four-variable
HJB. Separately, the substitution w_q = exp(k theta_q) makes it LINEAR,
w' = M w with M tridiagonal (Gueant, Lehalle & Fernandez-Tapia 2013); that is
the path Phase 9's GueantHJBSolver takes. Two independent routes to the same
answer, and they are required to agree.

WHAT THIS PHASE FOUND IN PHASE 9
--------------------------------
The two routes agreed to 2e-14 without a terminal penalty and DISAGREED with
one. Phase 9 set w_q(T) = exp(-k gamma L(q)); the ansatz gives theta_q(T) =
-L(q), so w_q(T) = exp(-k L(q)). Phase 9 was applying gamma * L -- a tenth of
the intended penalty at gamma = 0.1. Near the close, a maker long 5 lots bid
1.18 below mid instead of 2.90 and kept buying into the liquidation it was
meant to avoid. At the start of the session the two agree, because the terminal
condition has washed out, so the bug could only be seen near the close. No
Phase 9 test exercised the penalty. It is fixed there and guarded here.

THE CLOSED FORM IS A SHORT-HORIZON APPROXIMATION
------------------------------------------------
Avellaneda-Stoikov's reservation price and spread,

    r = s - q gamma sigma^2 (T - t),    spread = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)

are the first-order expansion of the exact quotes in (T - t). Both grow
LINEARLY and without bound in the horizon. The exact quotes do not: as
T - t -> infinity they converge to a stationary limit set by the ground state
of M. Measured below.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp

__all__ = [
    "MMParams", "hamiltonian", "optimal_delta",
    "solve_theta", "HJBQuoter", "ASQuoter", "SymmetricQuoter",
    "stationary_quotes", "glft_asymptotic", "simulate",
]


# ===========================================================================
# 1. PARAMETERS
# ===========================================================================

@dataclass(frozen=True)
class MMParams:
    """
    Defaults are the Avellaneda-Stoikov (2008) experiment: s0 = 100, T = 1,
    sigma = 2, gamma = 0.1, k = 1.5, A = 140, dt = 0.005.
    """

    sigma: float = 2.0
    gamma: float = 0.1
    k: float = 1.5
    A: float = 140.0
    T: float = 1.0
    Q: int = 10
    penalty: float = 0.0          # b in L(q) = b q^2, currency per lot^2

    @property
    def alpha(self) -> float:
        return self.k * self.gamma * self.sigma**2 / 2.0

    @property
    def eta(self) -> float:
        return self.A * (1.0 + self.gamma / self.k) ** (-(1.0 + self.k / self.gamma))

    @property
    def base(self) -> float:
        """(1/gamma) ln(1 + gamma/k): the half-spread at zero inventory risk."""
        return math.log(1.0 + self.gamma / self.k) / self.gamma

    @property
    def qs(self) -> np.ndarray:
        return np.arange(-self.Q, self.Q + 1, dtype=float)


# ===========================================================================
# 2. THE HAMILTONIAN
# ===========================================================================

def hamiltonian(p: float, P: MMParams) -> float:
    """H(p) = (eta/k) e^{-k p}: the supremum of one side's jump term."""
    return P.eta / P.k * math.exp(-P.k * p)


def optimal_delta(p: float, P: MMParams) -> float:
    """
    delta* = p + (1/gamma) ln(1 + gamma/k), from the first-order condition of

        (A/gamma) e^{-k delta} ( 1 - e^{-gamma (delta - p)} ).

    p is the change in theta from the trade: the value lost or gained by moving
    inventory one lot. The constant is the half-spread a maker charges with no
    inventory at stake -- the monopoly markup against exponential demand.
    """
    return p + P.base


# ===========================================================================
# 3. SOLVING THE HJB
# ===========================================================================

def _rhs(tau: float, th: np.ndarray, P: MMParams) -> np.ndarray:
    """
    d theta / d tau, tau = T - t. Integrating forward in tau is integrating
    backward in calendar time from the terminal condition.
    """
    q = P.qs
    bid = np.zeros_like(th)
    ask = np.zeros_like(th)
    bid[:-1] = np.exp(-P.k * (th[:-1] - th[1:]))      # no bid at q = +Q
    ask[1:] = np.exp(-P.k * (th[1:] - th[:-1]))       # no offer at q = -Q
    return -0.5 * P.sigma**2 * P.gamma * q**2 + (P.eta / P.k) * (bid + ask)


def _jac(tau: float, th: np.ndarray, P: MMParams) -> np.ndarray:
    n = th.size
    J = np.zeros((n, n))
    e_b = np.exp(-P.k * (th[:-1] - th[1:]))
    e_a = np.exp(-P.k * (th[1:] - th[:-1]))
    i = np.arange(n - 1)
    J[i, i] -= P.eta * e_b
    J[i, i + 1] += P.eta * e_b
    J[i + 1, i + 1] -= P.eta * e_a
    J[i + 1, i] += P.eta * e_a
    return J


def solve_theta(P: MMParams, taus: Sequence[float]) -> np.ndarray:
    """
    Solve the NONLINEAR reduced HJB for theta_q at the requested times-to-go,
    from theta_q(T) = -L(q) = -b q^2.

    An IMPLICIT stiff integrator (Radau IIA, order 5, analytic Jacobian) is
    required, not a convenience. With a terminal penalty the theta differences
    at the inventory edges reach ~10, the exponentials push the Jacobian to
    ~1e8, and explicit RK4 at any reasonable step overflows to NaN -- measured
    while building this. Phase 9's linearised log-space route has no such
    difficulty, which is a genuine advantage of the linearisation.

    Returns theta of shape (len(taus), 2Q + 1).
    """
    taus = np.asarray(taus, dtype=float)
    th0 = -P.penalty * P.qs**2
    if taus.max() <= 0:
        return np.tile(th0, (taus.size, 1))
    sol = solve_ivp(lambda t, y: _rhs(t, y, P), (0.0, float(taus.max())), th0,
                    method="Radau", jac=lambda t, y: _jac(t, y, P),
                    t_eval=np.sort(np.unique(taus)), rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    lookup = {float(t): sol.y[:, i] for i, t in enumerate(sol.t)}
    return np.stack([lookup[float(t)] if t > 0 else th0 for t in taus])


def _distances_from_theta(th: np.ndarray, P: MMParams) -> Tuple[np.ndarray, np.ndarray]:
    """(delta^b, delta^a) for every q; +inf where the side is not quoted."""
    n = th.size
    db = np.full(n, np.inf)
    da = np.full(n, np.inf)
    db[:-1] = th[:-1] - th[1:] + P.base
    da[1:] = th[1:] - th[:-1] + P.base
    return db, da


def stationary_quotes(P: MMParams) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    The T - t -> infinity limit, exactly.

    Linearised, w(tau) = exp(-M tau) w(0). As tau grows every mode but the one
    with the smallest eigenvalue of M decays relative to it, so w aligns with
    that GROUND STATE v0 -- positive by Perron-Frobenius, since -M has
    non-negative off-diagonals -- and the quotes converge to

        delta^b_inf(q) = (1/k) ln( v0_q / v0_{q+1} ) + (1/gamma) ln(1 + gamma/k)

    independent of the terminal condition. M = diag(alpha q^2) - eta (shifts)
    is a discrete Schrodinger operator: a harmonic potential alpha q^2 with
    hopping eta. Returns (delta^b, delta^a, lambda_0).
    """
    q = P.qs
    n = q.size
    M = np.diag(P.alpha * q**2)
    i = np.arange(n - 1)
    M[i, i + 1] = -P.eta
    M[i + 1, i] = -P.eta
    evals, evecs = linalg.eigh(M)
    v0 = np.abs(evecs[:, 0])
    lv = np.log(v0)
    db = np.full(n, np.inf)
    da = np.full(n, np.inf)
    db[:-1] = (lv[:-1] - lv[1:]) / P.k + P.base
    da[1:] = (lv[1:] - lv[:-1]) / P.k + P.base
    return db, da, float(evals[0])


def glft_asymptotic(q: np.ndarray, P: MMParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gueant-Lehalle-Fernandez-Tapia closed-form asymptotic quotes:

        delta^b ~ base + (2q + 1)/2 * omega,   delta^a ~ base - (2q - 1)/2 * omega
        omega = sqrt( gamma sigma^2 / (2 k A) * (1 + gamma/k)^{1 + k/gamma} )

    Derived here rather than quoted. For smooth v, (S+ + S-) v ~ 2v + v'', so
    M v ~ -eta v'' + ( alpha q^2 - 2 eta ) v: a continuous harmonic oscillator
    with Gaussian ground state v ~ exp( -(1/2) sqrt(alpha/eta) q^2 ). Then
    ln(v_q / v_{q+1}) = (2q + 1) sqrt(alpha/eta) / 2, and (1/k) sqrt(alpha/eta)
    is exactly omega. So the closed form is the CONTINUUM approximation of the
    discrete ground state, accurate when the Gaussian spans many lots
    (alpha/eta small) and not otherwise.
    """
    omega = (1.0 / P.k) * math.sqrt(P.alpha / P.eta)
    q = np.asarray(q, dtype=float)
    return P.base + (2 * q + 1) / 2 * omega, P.base - (2 * q - 1) / 2 * omega


# ===========================================================================
# 4. QUOTERS
# ===========================================================================

class HJBQuoter:
    """
    Exact optimal quotes from the solved HJB, tabulated on a time grid.

    `distances(q, tau)` returns (delta^b, delta^a); `quote(s, q, tau)` returns
    the reservation price r = s + (delta^a - delta^b)/2 and the spread
    delta^a + delta^b. At the inventory bounds one side is +inf: the maker
    stops quoting it.
    """

    def __init__(self, P: MMParams, n_grid: int = 400) -> None:
        self.P = P
        self.taus = np.linspace(0.0, P.T, n_grid + 1)
        theta = solve_theta(P, self.taus)
        self.db = np.empty_like(theta)
        self.da = np.empty_like(theta)
        for i, th in enumerate(theta):
            self.db[i], self.da[i] = _distances_from_theta(th, P)

    def _i(self, tau: float) -> int:
        return int(np.clip(np.searchsorted(self.taus, tau), 0, self.taus.size - 1))

    def distances(self, q: int, tau: float) -> Tuple[float, float]:
        j = int(np.clip(q + self.P.Q, 0, 2 * self.P.Q))
        i = self._i(tau)
        return float(self.db[i, j]), float(self.da[i, j])

    def table(self, tau_idx: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        j = np.clip(q + self.P.Q, 0, 2 * self.P.Q).astype(int)
        return self.db[tau_idx, j], self.da[tau_idx, j]

    def quote(self, s: float, q: int, tau: float) -> Tuple[float, float]:
        db, da = self.distances(q, tau)
        return s + 0.5 * (da - db), da + db


class ASQuoter:
    """Avellaneda-Stoikov closed form: the first-order expansion in T - t."""

    def __init__(self, P: MMParams) -> None:
        self.P = P

    def distances(self, q, tau):
        P = self.P
        c = P.gamma * P.sigma**2 * tau / 2.0
        return P.base + (2 * q + 1) * c, P.base - (2 * q - 1) * c

    def quote(self, s: float, q: int, tau: float) -> Tuple[float, float]:
        P = self.P
        return (s - q * P.gamma * P.sigma**2 * tau,
                P.gamma * P.sigma**2 * tau + 2.0 * P.base)


class SymmetricQuoter:
    """
    The benchmark from Avellaneda-Stoikov's own experiment: the SAME average
    spread as the inventory strategy, centred on the mid, ignoring inventory.
    """

    def __init__(self, P: MMParams) -> None:
        self.P = P

    def distances(self, q, tau):
        P = self.P
        h = P.base + P.gamma * P.sigma**2 * tau / 2.0
        return h + 0.0 * np.asarray(q, float), h + 0.0 * np.asarray(q, float)


# ===========================================================================
# 5. SIMULATION
# ===========================================================================

@dataclass
class SimResult:
    pnl: np.ndarray            # (n_paths,) terminal mark-to-market
    q_final: np.ndarray
    q_path: np.ndarray         # (n_steps + 1, n_paths)
    s_path: np.ndarray
    r_path: np.ndarray         # reservation price
    spread_path: np.ndarray
    fills: np.ndarray          # total fills per path


def simulate(quoter, P: MMParams, n_paths: int = 1000, dt: float = 0.005,
             s0: float = 100.0, seed: int = 0, bounded: Optional[bool] = None,
             liquidate: bool = False) -> SimResult:
    """
    Avellaneda-Stoikov's own Monte Carlo, vectorised across paths: on each step
    the mid moves by +/- sigma sqrt(dt), and each side fills with probability
    A e^{-k delta} dt.

    Terminal PnL is marked at the final mid, X_T + q_T S_T; with `liquidate`
    the penalty L(q_T) = b q_T^2 is charged as well. The inventory bound is
    enforced for the HJB quoter, which never quotes past it; the closed-form
    and symmetric quoters are unbounded, as in the original paper.
    """
    rng = np.random.default_rng(seed)
    n = int(round(P.T / dt))
    bounded = isinstance(quoter, HJBQuoter) if bounded is None else bounded
    s = np.full(n_paths, s0)
    x = np.zeros(n_paths)
    q = np.zeros(n_paths, dtype=int)
    fills = np.zeros(n_paths, dtype=int)
    q_path = np.empty((n + 1, n_paths), dtype=int)
    s_path = np.empty((n + 1, n_paths))
    r_path = np.empty((n + 1, n_paths))
    sp_path = np.empty((n + 1, n_paths))
    q_path[0], s_path[0] = q, s

    for step in range(n):
        tau = P.T - step * dt
        if isinstance(quoter, HJBQuoter):
            db, da = quoter.table(np.full(n_paths, quoter._i(tau)), q)
        else:
            db, da = quoter.distances(q, tau)
            db, da = np.asarray(db, float), np.asarray(da, float)
        if bounded:
            db = np.where(q >= P.Q, np.inf, db)
            da = np.where(q <= -P.Q, np.inf, da)
        fin_b, fin_a = np.isfinite(db), np.isfinite(da)
        r_path[step] = np.where(fin_b & fin_a, s + 0.5 * (da - db), np.nan)
        sp_path[step] = np.where(fin_b & fin_a, da + db, np.nan)

        pb = np.where(fin_b, np.minimum(P.A * np.exp(-P.k * np.where(fin_b, db, 0)) * dt, 1.0), 0.0)
        pa = np.where(fin_a, np.minimum(P.A * np.exp(-P.k * np.where(fin_a, da, 0)) * dt, 1.0), 0.0)
        hit_b = rng.random(n_paths) < pb
        hit_a = rng.random(n_paths) < pa
        x -= np.where(hit_b, s - db, 0.0)
        x += np.where(hit_a, s + da, 0.0)
        q += hit_b.astype(int) - hit_a.astype(int)
        fills += hit_b.astype(int) + hit_a.astype(int)
        s = s + P.sigma * math.sqrt(dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)
        q_path[step + 1], s_path[step + 1] = q, s

    r_path[n], sp_path[n] = np.nan, np.nan
    pnl = x + q * s
    if liquidate:
        pnl = pnl - P.penalty * q.astype(float) ** 2
    return SimResult(pnl, q.astype(float), q_path, s_path, r_path, sp_path, fills)


# ===========================================================================
# SELF-TESTS
# ===========================================================================

P0 = MMParams()


def _test_hamiltonian_maximiser() -> None:
    """The closed-form sup and argmax against a brute-force grid search."""
    grid = np.linspace(-5.0, 10.0, 600_001)
    worst_d, worst_h = 0.0, 0.0
    for p in (-2.0, -0.5, 0.0, 0.3, 1.7, 4.0):
        f = (P0.A / P0.gamma) * np.exp(-P0.k * grid) * (1 - np.exp(-P0.gamma * (grid - p)))
        j = int(np.argmax(f))
        worst_d = max(worst_d, abs(grid[j] - optimal_delta(p, P0)))
        worst_h = max(worst_h, abs(f[j] - hamiltonian(p, P0)) / hamiltonian(p, P0))
    assert worst_d < 1e-4 and worst_h < 1e-9, (worst_d, worst_h)
    print(f"[ok] Hamiltonian: closed-form argmax delta* = p + (1/gamma)ln(1+gamma/k) and "
          f"sup (eta/k)e^(-kp) match a 600k-point grid search (argmax {worst_d:.1e}, "
          f"value {worst_h:.1e} relative)")


def _test_solution_satisfies_the_full_hjb() -> None:
    """
    The decisive check that the dimension reduction is right. Build
    u = -exp(-gamma (x + q s + theta_q(t))) from the solved theta and evaluate
    the ORIGINAL four-variable HJB at random (t, x, s, q): time and price
    derivatives by central differences, each supremum by brute-force search over
    delta. The residual must vanish to finite-difference precision.
    """
    P = replace(P0, penalty=0.05)
    rng = np.random.default_rng(0)
    h = 1e-4
    grid = np.linspace(-3.0, 8.0, 110_001)
    worst = 0.0
    for _ in range(12):
        tau = float(rng.uniform(0.1, 0.9))
        q = int(rng.integers(-P.Q + 1, P.Q))
        x, s = float(rng.uniform(-50, 50)), float(rng.uniform(95, 105))
        th = solve_theta(P, [tau - h, tau, tau + h])
        j = q + P.Q

        def u(xx, ss, qq, row=1):
            return -math.exp(-P.gamma * (xx + qq * ss + th[row, qq + P.Q]))

        u0 = u(x, s, q)
        u_t = -(u(x, s, q, 2) - u(x, s, q, 0)) / (2 * h)       # d/dt = -d/dtau
        hs = 1e-3
        u_ss = (u(x, s + hs, q) - 2 * u0 + u(x, s - hs, q)) / hs**2
        # jump targets: buying one lot at s - d, selling one at s + d
        dg = grid[::100]
        ub = np.array([u(x - s + d, s, q + 1) for d in dg])
        ua = np.array([u(x + s + d, s, q - 1) for d in dg])
        sup_b = float(np.max(P.A * np.exp(-P.k * dg) * (ub - u0)))
        sup_a = float(np.max(P.A * np.exp(-P.k * dg) * (ua - u0)))
        R = u_t + 0.5 * P.sigma**2 * u_ss + sup_b + sup_a
        worst = max(worst, abs(R) / abs(u0))
    assert worst < 5e-4, worst
    print(f"[ok] solved theta satisfies the ORIGINAL four-variable HJB at 12 random "
          f"(t, x, s, q), with terminal penalty: max |residual|/|u| = {worst:.1e} "
          f"(finite differences + brute-force sup)")


def _test_linearisation_agrees_with_nonlinear() -> None:
    """
    w = exp(k theta) must turn the nonlinear system into w' = M w. Solve the
    linear system by matrix exponential and compare with the nonlinear solve.
    """
    P = replace(P0, penalty=0.05)
    q = P.qs
    n = q.size
    M = np.diag(P.alpha * q**2)
    i = np.arange(n - 1)
    M[i, i + 1] = -P.eta
    M[i + 1, i] = -P.eta
    w0 = np.exp(-P.k * P.penalty * q**2)
    worst = 0.0
    for tau in (0.01, 0.2, 0.7):
        lw = np.log(linalg.expm(-M * tau) @ w0) / P.k
        th = solve_theta(P, [tau])[0]
        worst = max(worst, float(np.max(np.abs(np.diff(lw) - np.diff(th)))))
    assert worst < 1e-8, worst
    print(f"[ok] linearisation w = exp(k theta), w' = M w, agrees with the nonlinear "
          f"solve on every quote-determining difference (max {worst:.1e})")


def _test_cross_phase_against_phase9() -> None:
    """
    Regression guard for the Phase 9 fix. Phase 9's GueantHJBSolver takes the
    linearised log-space route; this module solves the nonlinear system. They
    must agree -- including near the close with a terminal penalty, which is
    where they used to differ by up to 1.7 in the bid distance.
    """
    try:
        from optimal_execution_sor import ASParameters, GueantHJBSolver
    except ImportError:
        print("[skip] optimal_execution_sor.py not importable -- cross-phase check skipped")
        return
    rows = []
    for b in (0.0, 0.5):
        P = replace(P0, penalty=b)
        g = GueantHJBSolver(ASParameters(sigma=P.sigma, gamma=P.gamma, k=P.k, A=P.A,
                                         horizon=P.T, q_max=P.Q), n_time_steps=400,
                            terminal_penalty=(lambda qq, b=b: b * qq**2) if b else None).solve()
        taus = [0.0025, 0.01, 0.05, 1.0]
        th = solve_theta(P, taus)
        worst = 0.0
        for k_, tau in enumerate(taus):
            db, da = _distances_from_theta(th[k_], P)
            for qq in range(-P.Q + 1, P.Q):
                p9b, p9a = g.distances(qq, tau)
                worst = max(worst, abs(p9b - db[qq + P.Q]), abs(p9a - da[qq + P.Q]))
        rows.append((b, worst))
        assert worst < 1e-6, f"penalty {b}: Phase 9 disagrees by {worst:.3e}"
    print("[ok] Phase 9 GueantHJBSolver agrees with the independent nonlinear solve "
          "at every q and at times-to-go down to 0.0025:")
    for b, w in rows:
        print(f"       terminal penalty b = {b}: max quote difference {w:.1e}")


def _test_short_horizon_limit_is_avellaneda_stoikov() -> None:
    """
    Exact quotes -> the AS closed form as T - t -> 0, and the error is THIRD
    order, not second.

    AS is the first-order expansion of theta in tau. The second-order term is
    theta_q''(0) = eta * (discrete Laplacian of -(1/2) sigma^2 gamma q^2)
    = -eta sigma^2 gamma, which is the SAME for every q. Quotes depend only on
    differences theta_q - theta_{q+1}, so it cancels exactly and the first
    surviving error is O(tau^3).
    """
    q = np.arange(-4, 5)
    rows = []
    for tau in (0.2, 0.05, 0.0125):
        db, _ = _distances_from_theta(solve_theta(P0, [tau])[0], P0)
        ab, _ = ASQuoter(P0).distances(q, tau)
        rows.append((tau, float(np.max(np.abs(db[q + P0.Q] - ab)))))
    orders = [math.log(rows[i][1] / rows[i + 1][1]) / math.log(4.0) for i in range(2)]
    # theta''(0) exactly, no differencing: d/dtau f(theta) = J(theta) f(theta)
    th0 = np.zeros(P0.qs.size)
    second = _jac(0.0, th0, P0) @ _rhs(0.0, th0, P0)
    interior = second[2:-2]          # away from the bounds, where a neighbour is missing
    assert all(o > 2.5 for o in orders), orders
    assert float(np.ptp(interior)) < 1e-9 * float(np.abs(interior).max()), "tau^2 term not uniform in q"
    assert abs(interior.mean() + P0.eta * P0.sigma**2 * P0.gamma) < 1e-9
    print("[ok] exact quotes -> AS closed form as T-t -> 0: "
          + ", ".join(f"T-t={t}: {e:.1e}" for t, e in rows)
          + f"; local order {orders[0]:.2f}, {orders[1]:.2f} -- THIRD order")
    print(f"     the tau^2 correction theta''(0) = -eta sigma^2 gamma = "
          f"{-P0.eta*P0.sigma**2*P0.gamma:.4f} is uniform in q (evaluated exactly as "
          f"J f: {interior.mean():.4f}, spread {float(np.ptp(interior)):.1e}) and cancels "
          f"from every quote")


def _test_long_horizon_closed_form_diverges() -> None:
    """
    AS's spread and skew grow linearly in T - t. The exact quotes converge to
    the stationary ground-state limit.
    """
    db_inf, da_inf, _ = stationary_quotes(P0)
    rows = []
    for tau in (1.0, 5.0, 20.0):
        db, da = _distances_from_theta(solve_theta(P0, [tau])[0], P0)
        j = P0.Q
        ex_spread = db[j] + da[j]
        ex_skew = 0.5 * (da[j + 1] - db[j + 1])
        r_as, sp_as = ASQuoter(P0).quote(0.0, 1, tau)
        rows.append((tau, ex_spread, sp_as, ex_skew, r_as))
    inf_spread = db_inf[P0.Q] + da_inf[P0.Q]
    assert abs(rows[-1][1] - inf_spread) < 1e-6
    assert rows[-1][2] > 5 * rows[-1][1], "AS spread should blow up relative to exact"
    q = P0.qs
    M = np.diag(P0.alpha * q**2)
    i = np.arange(q.size - 1)
    M[i, i + 1] = -P0.eta
    M[i + 1, i] = -P0.eta
    ev = linalg.eigvalsh(M)
    t_relax = 1.0 / (ev[1] - ev[0])
    skew_inf = -0.5 * (da_inf[P0.Q + 1] - db_inf[P0.Q + 1])
    t_valid = skew_inf / (P0.gamma * P0.sigma**2)
    print(f"[ok] spread at q = 0 and reservation shift at q = +1 versus horizon T - t:")
    print(f"       {'T-t':>6} {'exact spread':>13} {'AS spread':>10} {'exact r-s':>10} {'AS r-s':>9}")
    for tau, es, as_, ek, ar in rows:
        print(f"       {tau:>6g} {es:>13.4f} {as_:>10.4f} {ek:>10.4f} {ar:>9.4f}")
    print(f"       {'inf':>6} {inf_spread:>13.4f} {'unbounded':>10} "
          f"{0.5*(da_inf[P0.Q+1]-db_inf[P0.Q+1]):>10.4f} {'unbounded':>9}")
    print(f"     the exact quotes relax to stationarity on the spectral-gap timescale")
    print(f"     1/(lambda_1 - lambda_0) = {t_relax:.3f}; AS's skew gamma sigma^2 (T-t) matches")
    print(f"     the stationary skew only for T-t < {t_valid:.3f}. With T = 1 -- Avellaneda and")
    print(f"     Stoikov's own experiment -- the closed form over-skews for the first "
          f"{100*(1-t_valid):.0f}% of")
    print(f"     the session, by {abs(rows[0][4]/rows[0][3]):.1f}x at the open")
    assert abs(t_valid - t_relax) < 0.5 * t_relax, "validity window should track the relaxation time"


def _test_closed_form_asymptotic_is_a_continuum_limit() -> None:
    """
    The GLFT asymptotic formula is the continuum (harmonic-oscillator)
    approximation of the discrete ground state. Accurate when alpha/eta is small
    -- the ground state spans many lots -- and not otherwise.
    """
    q = np.arange(-3, 4)
    rows = []
    for g in (0.01, 0.1, 1.0):
        P = replace(P0, gamma=g, Q=25)
        db, _, _ = stationary_quotes(P)
        cb, _ = glft_asymptotic(q, P)
        skew_exact = db[q + P.Q] - P.base
        skew_cf = cb - P.base
        err = float(np.max(np.abs(skew_exact - skew_cf)) / np.max(np.abs(skew_exact)))
        rows.append((g, P.alpha / P.eta, err))
    assert rows[0][2] < rows[-1][2], rows
    print("[ok] GLFT closed-form asymptotic vs exact ground state (inventory-driven "
          "part of the bid, |q| <= 3):")
    for g, ratio, e in rows:
        print(f"       gamma={g:<5} alpha/eta={ratio:.2e}  relative error {100*e:.2f}%")


def _test_penalty_flattens_inventory_into_the_close() -> None:
    """
    Without a terminal penalty the exact solution stops caring about inventory
    as T - t -> 0 -- the terminal condition values it at the mid -- and the maker
    carries it into the close. The penalty is what makes it flatten.
    """
    none = simulate(HJBQuoter(replace(P0, penalty=0.0)), replace(P0, penalty=0.0),
                    n_paths=2000, seed=3)
    pen = simulate(HJBQuoter(replace(P0, penalty=0.5)), replace(P0, penalty=0.5),
                   n_paths=2000, seed=3)
    q_none = float(np.mean(np.abs(none.q_final)))
    q_pen = float(np.mean(np.abs(pen.q_final)))
    assert q_pen < 0.7 * q_none, (q_none, q_pen)
    print(f"[ok] mean |inventory| at the close over 2,000 sessions: {q_none:.2f} lots "
          f"with no terminal penalty, {q_pen:.2f} with b = 0.5 -- the penalty is what "
          f"returns inventory to zero at the end")


def _cara_ce(pnl: np.ndarray, gamma: float) -> float:
    """Certainty equivalent -(1/gamma) ln E[exp(-gamma PnL)], via log-sum-exp."""
    z = -gamma * pnl
    m = float(z.max())
    return -(m + math.log(float(np.mean(np.exp(z - m))))) / gamma


def _test_avellaneda_stoikov_experiment() -> None:
    """
    Avellaneda-Stoikov's own comparison, re-run, and scored on the objective
    the controls were derived for: CARA expected utility, reported as a
    certainty equivalent. The exact HJB policy is the optimal control, so it
    must score highest on it -- an end-to-end check of the whole solve that no
    mean/std table provides.
    """
    rows = []
    for name, qt in (("exact HJB (Q=10)", HJBQuoter(P0)),
                     ("AS closed form", ASQuoter(P0)),
                     ("symmetric", SymmetricQuoter(P0))):
        r = simulate(qt, P0, n_paths=4000, seed=11)
        rows.append((name, float(r.pnl.mean()), float(r.pnl.std()),
                     float(r.q_final.std()), _cara_ce(r.pnl, P0.gamma), float(r.fills.mean())))
    hjb, as_, sym = rows
    assert as_[2] < 0.6 * sym[2] and hjb[2] < 0.6 * sym[2], "inventory control should cut dispersion"
    assert hjb[4] > as_[4] and hjb[4] > sym[4], "the optimal control must win on its own objective"
    print("[ok] Avellaneda-Stoikov experiment (T=1, sigma=2, gamma=0.1, k=1.5, A=140, 4,000 paths):")
    print(f"       {'strategy':<18}{'mean PnL':>10}{'std PnL':>9}{'std q_T':>9}"
          f"{'CARA CE':>9}{'fills':>7}")
    for name, m, sd, sq, ce, f in rows:
        print(f"       {name:<18}{m:>10.2f}{sd:>9.2f}{sq:>9.2f}{ce:>9.2f}{f:>7.0f}")
    print(f"     the exact HJB keeps the symmetric strategy's mean at "
          f"{100*hjb[2]/sym[2]:.0f}% of its dispersion and has the highest certainty")
    print(f"     equivalent; the closed form over-skews early and gives up "
          f"{hjb[1]-as_[1]:.2f} of mean PnL to do it")


def _run(title: str, tests) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    for f in tests:
        f()
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def _main(path: str = "D:/Quant_Project/hjb_market_maker.png") -> None:
    print("=" * 78)
    print("MAIN: 1,000 ORDER ARRIVALS AGAINST THE EXACT HJB QUOTER")
    print("=" * 78)
    P = replace(P0, T=10.0, penalty=0.5)
    quoter = HJBQuoter(P, n_grid=2000)

    print(f"  sigma={P.sigma}, gamma={P.gamma}, k={P.k}, A={P.A}, inventory bound "
          f"Q={P.Q}, terminal penalty b={P.penalty}, session T={P.T}")
    print(f"\n  optimal quotes mid-session (T-t = {P.T/2:g}) as inventory piles up:")
    print(f"  {'q':>4} {'r - s':>9} {'spread':>8} {'bid dist':>9} {'ask dist':>9}")
    for q in (-10, -8, -5, -2, 0, 2, 5, 8, 10):
        db, da = quoter.distances(q, P.T / 2)
        r = 0.5 * (da - db) if np.isfinite(db) and np.isfinite(da) else float("nan")
        sp = db + da if np.isfinite(db) and np.isfinite(da) else float("inf")
        fb = f"{db:9.4f}" if np.isfinite(db) else f"{'no bid':>9}"
        fa = f"{da:9.4f}" if np.isfinite(da) else f"{'no offer':>9}"
        print(f"  {q:>4} {r:>9.4f} {sp:>8.4f} {fb} {fa}")

    res = simulate(quoter, P, n_paths=1, dt=0.005, seed=2024)
    n_fill = int(res.fills[0])
    q = res.q_path[:, 0]
    t = np.arange(q.size) * 0.005

    lag = 50
    ac = float(np.corrcoef(q[:-lag], q[lag:])[0, 1])
    print(f"\n  one session: {n_fill:,} order arrivals filled over T = {P.T}")
    print(f"  inventory: mean {q.mean():+.2f}, std {q.std():.2f}, max |q| {int(np.abs(q).max())}, "
          f"final {int(q[-1])}")
    print(f"  a random walk of the same {n_fill:,} fills would reach |q| ~ "
          f"{np.sqrt(n_fill):.0f}; this path stays within +/-{int(np.abs(q).max())}")
    print(f"  inventory autocorrelation at {lag*0.005:g} time units: {ac:.2f} -- "
          f"mean-reverting, not a random walk")

    print(f"\n  quote adjustment along the path (sampled):")
    print(f"  {'t':>6} {'q':>4} {'r - s':>9} {'spread':>8}")
    for i in np.linspace(0, q.size - 2, 10).astype(int):
        rs = res.r_path[i, 0] - res.s_path[i, 0]
        sp = res.spread_path[i, 0]
        print(f"  {t[i]:>6.2f} {int(q[i]):>4} "
              f"{(f'{rs:+9.4f}' if np.isfinite(rs) else f'{chr(8212):>9}')} "
              f"{(f'{sp:8.4f}' if np.isfinite(sp) else f'{chr(8212):>8}')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
        s_ = res.s_path[:, 0]
        r_ = res.r_path[:, 0]
        sp_ = res.spread_path[:, 0]
        ax[0].plot(t, s_, color="0.4", lw=0.8, label="mid s")
        ax[0].plot(t, r_, color="C3", lw=0.8, label="reservation price r")
        ax[0].fill_between(t, r_ - sp_ / 2, r_ + sp_ / 2, color="C3", alpha=0.12,
                           label="quoted bid-ask")
        ax[0].set_ylabel("price")
        ax[0].legend(fontsize=8, loc="best")
        ax[0].set_title(f"Exact HJB market maker: {n_fill:,} fills, inventory bound "
                        f"+/-{P.Q}, terminal penalty b={P.penalty}")
        ax[1].plot(t, q, color="C0", lw=0.9)
        ax[1].axhline(0, color="0.5", lw=0.6)
        ax[1].set_ylabel("inventory q")
        ax[2].plot(t, r_ - s_, color="C3", lw=0.8)
        ax[2].axhline(0, color="0.5", lw=0.6)
        ax[2].set_ylabel("r - s")
        ax[2].set_xlabel("t")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"\n  figure written to {path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n  figure skipped: {exc}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    _run("THE HJB", [
        _test_hamiltonian_maximiser,
        _test_solution_satisfies_the_full_hjb,
        _test_linearisation_agrees_with_nonlinear,
        _test_cross_phase_against_phase9,
    ])
    _run("CLOSED FORMS AND THEIR LIMITS", [
        _test_short_horizon_limit_is_avellaneda_stoikov,
        _test_long_horizon_closed_form_diverges,
        _test_closed_form_asymptotic_is_a_continuum_limit,
    ])
    _run("SIMULATION", [
        _test_penalty_flattens_inventory_into_the_close,
        _test_avellaneda_stoikov_experiment,
    ])
    _main()
    print(f"\ncompleted in {time.time()-t0:.0f}s")

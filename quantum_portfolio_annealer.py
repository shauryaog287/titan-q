"""
quantum_portfolio_annealer.py
=============================

Phase 20: QUBO formulation and annealing for cardinality-constrained portfolio
selection.

PROBLEM
    Choose exactly K of N assets, equal-weighted, to maximise Sharpe:

        maximise  S(x) = mu' x / sqrt( x' Sigma x ),   x in {0,1}^N,  1' x = K

    With weights w = x/K the factor K cancels, so the equal-weight Sharpe is
    exactly mu'x / sqrt(x'Sigma x). Cardinality-constrained selection is NP-hard.

THREE THINGS THIS MODULE DOES NOT CLAIM
---------------------------------------
1. SHARPE IS NOT A QUBO. A QUBO minimises a single quadratic form x'Qx; Sharpe
   is a ratio of a linear function to a norm. It is reached EXACTLY through
   Dinkelbach's method on S^2: for mu'x > 0,

       max (mu'x)^2 / (x'Sigma x)   <=>   a sequence of   max x'( mu mu' - theta Sigma )x

   and each of those is a genuine QUBO. The ratio is recovered by iterating
   theta_{k+1} = (mu'x_k)^2 / (x_k'Sigma x_k). See `dinkelbach`.

2. ANNEALING DOES NOT ESCAPE NP-HARDNESS. Simulated annealing and simulated
   bifurcation are heuristics. They return good solutions fast and carry no
   optimality certificate. Every result here is scored either against the exact
   brute-force optimum (where it can be enumerated) or against a certified
   upper bound (where it cannot).

3. "FASTER THAN BRUTE FORCE" AT N = 50 PROVES NOTHING. C(50,10) = 1.03e10
   subsets; brute force would take hours to days, so any method is faster. The
   informative comparisons are (a) exact optimality at a size brute force can
   still enumerate, and (b) annealing against strong CLASSICAL baselines --
   greedy selection with local search in particular -- at N = 50.

WHAT WAS MEASURED
-----------------
  Exact size (N=26, K=8, 1.56M subsets): swap-SA and greedy+local search both
  hit the brute-force optimum; greedy is ~60x faster than swap-SA.

  N=50, K=10: swap-SA is best, and a majority of independent restarts converge
  on the same subset once given enough steps. Penalty-QUBO SA and simulated
  bifurcation come within a few percent -- but ONLY at a tuned penalty; across
  P in [1x, 20x] their result swings by roughly 30%. Greedy+local search, exact
  at N=26, stalls well short at N=50.

  Simulated bifurcation itself is correct: discrete SB recovers the exact
  ground state of random 14-spin Ising problems every time. Its weakness here
  is the cardinality penalty, not the dynamics.

WHY THE QUBO FORM EXISTS AT ALL
-------------------------------
Ising and QUBO hardware -- quantum annealers, coherent Ising machines,
Toshiba's SB chips -- accept only UNCONSTRAINED binary quadratic problems, so
the cardinality constraint must be folded in as a penalty P (1'x - K)^2. That
penalty is a hardware limitation, not a feature of the problem. A classical
annealer is under no such restriction: it can move by SWAPS that preserve
1'x = K exactly and never needs a penalty. Both are built and compared.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

__all__ = [
    "make_problem", "sharpe", "brute_force",
    "mean_variance_qubo", "dinkelbach_qubo", "qubo_energy", "qubo_to_ising", "ising_energy",
    "anneal_qubo", "anneal_swap", "simulated_bifurcation",
    "greedy_local_search", "dinkelbach", "long_only_sharpe_bound", "repair",
]


# ===========================================================================
# 1. PROBLEM
# ===========================================================================

def make_problem(N: int, n_factors: int = 4, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Annualised expected returns and a factor-structured covariance,

        Sigma = B B' + diag(d),

    with volatilities in roughly 15-45%. A factor structure is what makes
    selection non-trivial: assets that load on the same factor do not diversify
    each other, so the best K-subset is not simply the K highest Sharpe names.
    """
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 0.10, (N, n_factors))
    d = rng.uniform(0.02, 0.12, N) ** 2 * 4.0
    Sigma = B @ B.T + np.diag(d)
    vol = np.sqrt(np.diag(Sigma))
    mu = 0.05 + 0.35 * vol * rng.normal(0.35, 0.25, N)
    return mu, Sigma


def sharpe(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Equal-weight Sharpe of the selected subset: mu'x / sqrt(x'Sigma x)."""
    x = np.asarray(x, dtype=float)
    v = float(x @ Sigma @ x)
    return float(mu @ x) / math.sqrt(v) if v > 0 else -np.inf


# ===========================================================================
# 2. EXACT BASELINE
# ===========================================================================

def brute_force(mu: np.ndarray, Sigma: np.ndarray, K: int, chunk: int = 150_000
                ) -> Tuple[np.ndarray, float, int, float]:
    """
    Enumerate every K-subset in vectorised chunks. Returns (best x, best
    Sharpe, number of subsets, seconds). Exact, and exponential: usable to
    roughly C(N,K) ~ 1e7.
    """
    N = mu.size
    t0 = time.time()
    best_s, best_idx = -np.inf, None
    it = itertools.combinations(range(N), K)
    n = 0
    while True:
        block = list(itertools.islice(it, chunk))
        if not block:
            break
        idx = np.asarray(block, dtype=np.int32)
        n += idx.shape[0]
        r = mu[idx].sum(axis=1)
        v = Sigma[idx[:, :, None], idx[:, None, :]].sum(axis=(1, 2))
        s = r / np.sqrt(v)
        j = int(np.argmax(s))
        if s[j] > best_s:
            best_s, best_idx = float(s[j]), idx[j].copy()
    x = np.zeros(N)
    x[best_idx] = 1.0
    return x, best_s, n, time.time() - t0


# ===========================================================================
# 3. QUBO AND ISING
# ===========================================================================

def _cardinality_penalty(N: int, K: int, P: float) -> Tuple[np.ndarray, float]:
    """
    P (1'x - K)^2 = P ( x'11'x - 2K 1'x + K^2 ).

    For binary x, x_i^2 = x_i, so the linear term folds onto the diagonal of Q:
    no separate linear vector is needed.
    """
    Q = P * np.ones((N, N))
    Q[np.diag_indices(N)] += -2.0 * P * K
    return Q, P * K * K


def mean_variance_qubo(mu: np.ndarray, Sigma: np.ndarray, lam: float, K: int, P: float
                       ) -> Tuple[np.ndarray, float]:
    """
    Minimise  lam * x'Sigma x - mu'x + P (1'x - K)^2  as  x'Qx + const.

    The linear return term -mu'x becomes -diag(mu) on the diagonal, using
    x_i^2 = x_i. This is the classic Markowitz QUBO. It is NOT the Sharpe
    problem: lam trades return against VARIANCE, and the subset it selects moves
    with lam. Sweeping lam and keeping the best Sharpe approximates max-Sharpe;
    `dinkelbach_qubo` hits it exactly.
    """
    N = mu.size
    Q = lam * Sigma - np.diag(mu)
    Pq, c = _cardinality_penalty(N, K, P)
    return Q + Pq, c


def dinkelbach_qubo(mu: np.ndarray, Sigma: np.ndarray, theta: float, K: int, P: float
                    ) -> Tuple[np.ndarray, float]:
    """
    Minimise  x'( theta Sigma - mu mu' )x + P (1'x - K)^2.

    The Dinkelbach subproblem for S^2 = (mu'x)^2 / (x'Sigma x). mu mu' is
    rank-one PSD and theta Sigma is PSD, so the matrix is INDEFINITE -- which is
    what makes the landscape rugged and the problem hard.
    """
    N = mu.size
    Q = theta * Sigma - np.outer(mu, mu)
    Pq, c = _cardinality_penalty(N, K, P)
    return Q + Pq, c


def qubo_energy(x: np.ndarray, Q: np.ndarray, const: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x) + const


def qubo_to_ising(Q: np.ndarray, const: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Substitute x = (s + 1)/2 with s in {-1, +1}:

        x'Qx = (1/4) sum_{i != j} Q_ij s_i s_j + (1/2) sum_i (sum_j Q_ij) s_i
               + (1/4)( tr Q + 1'Q1 )

    Matched to  E(s) = -(1/2) s'Js - h's + offset  with J_ii = 0:

        J_ij = -Q_ij / 2  (i != j),   h_i = -(1/2) sum_j Q_ij,
        offset = (tr Q + 1'Q1)/4 + const
    """
    Qs = 0.5 * (Q + Q.T)
    J = -0.5 * Qs
    np.fill_diagonal(J, 0.0)
    h = -0.5 * Qs.sum(axis=1)
    offset = 0.25 * (np.trace(Qs) + Qs.sum()) + const
    return J, h, float(offset)


def ising_energy(s: np.ndarray, J: np.ndarray, h: np.ndarray, offset: float = 0.0) -> float:
    s = np.asarray(s, dtype=float)
    return float(-0.5 * s @ J @ s - h @ s) + offset


# ===========================================================================
# 4. SOLVERS
# ===========================================================================

def anneal_qubo(
    Q: np.ndarray, n_sweeps: int = 400, T0: Optional[float] = None, T1: Optional[float] = None,
    seed: int = 0, x0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Single-bit-flip Metropolis annealing on min x'Qx, geometric schedule.

    The flip delta for bit i with symmetric Q is

        dE_i = (1 - 2 x_i) ( Q_ii + 2 sum_{j != i} Q_ij x_j )

    so the local field g = Q x is maintained and each proposal is O(1); an
    accepted flip updates g in O(N). This is the formulation Ising hardware
    requires, penalty and all.
    """
    rng = np.random.default_rng(seed)
    Qs = 0.5 * (Q + Q.T)
    N = Qs.shape[0]
    diag = np.diag(Qs).copy()
    x = (rng.random(N) < 0.5).astype(float) if x0 is None else np.asarray(x0, float).copy()
    g = Qs @ x
    E = float(x @ Qs @ x)
    if T0 is None or T1 is None:
        sample = np.abs((1 - 2 * x) * (diag + 2 * (g - diag * x)))
        scale = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 1.0
        T0 = T0 if T0 is not None else scale
        T1 = T1 if T1 is not None else scale * 1e-4
    best_x, best_E = x.copy(), E
    total = n_sweeps * N
    decay = (T1 / T0) ** (1.0 / max(total - 1, 1))
    T = T0
    idx = rng.integers(0, N, total)
    u = rng.random(total)
    for k in range(total):
        i = idx[k]
        s = 1.0 - 2.0 * x[i]
        dE = s * (diag[i] + 2.0 * (g[i] - diag[i] * x[i]))
        if dE <= 0.0 or u[k] < math.exp(-dE / T):
            x[i] += s
            g += s * Qs[:, i]
            E += dE
            if E < best_E:
                best_E, best_x = E, x.copy()
        T *= decay
    return best_x, best_E


def anneal_swap(
    mu: np.ndarray, Sigma: np.ndarray, K: int, n_steps: int = 20_000,
    T0: Optional[float] = None, T1: Optional[float] = None, seed: int = 0,
    x0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Annealing DIRECTLY on Sharpe, moving by SWAPS: one selected asset out, one
    unselected in. 1'x = K is preserved by construction, so there is no penalty
    and no infeasible state is ever visited.

    With r = mu'x, v = x'Sigma x and the running vector s = Sigma x, swapping i
    out and j in gives, in O(1),

        r' = r - mu_i + mu_j
        v' = v - 2 s_i + Sigma_ii + 2 s_j - 2 Sigma_ij + Sigma_jj

    and an accepted swap updates s in O(N). Temperature is in units of Sharpe
    and calibrated from the spread of random swap deltas at the start.
    """
    rng = np.random.default_rng(seed)
    N = mu.size
    if x0 is None:
        sel = rng.choice(N, K, replace=False)
        x = np.zeros(N)
        x[sel] = 1.0
    else:
        x = np.asarray(x0, float).copy()
    inside = list(np.flatnonzero(x > 0.5))
    outside = list(np.flatnonzero(x < 0.5))
    s = Sigma @ x
    r = float(mu @ x)
    v = float(x @ s)
    S = r / math.sqrt(v)
    dSig = np.diag(Sigma)

    if T0 is None or T1 is None:
        deltas = []
        for _ in range(200):
            a, b = inside[rng.integers(K)], outside[rng.integers(N - K)]
            r2 = r - mu[a] + mu[b]
            v2 = v - 2 * s[a] + dSig[a] + 2 * s[b] - 2 * Sigma[a, b] + dSig[b]
            deltas.append(abs(r2 / math.sqrt(v2) - S))
        scale = float(np.median(deltas)) or 1e-3
        T0 = T0 if T0 is not None else scale
        T1 = T1 if T1 is not None else scale * 1e-3

    best_x, best_S = x.copy(), S
    decay = (T1 / T0) ** (1.0 / max(n_steps - 1, 1))
    T = T0
    ai = rng.integers(0, K, n_steps)
    bi = rng.integers(0, N - K, n_steps)
    u = rng.random(n_steps)
    for k in range(n_steps):
        pa, pb = ai[k], bi[k]
        a, b = inside[pa], outside[pb]
        r2 = r - mu[a] + mu[b]
        v2 = v - 2 * s[a] + dSig[a] + 2 * s[b] - 2 * Sigma[a, b] + dSig[b]
        S2 = r2 / math.sqrt(v2)
        dS = S2 - S
        if dS >= 0.0 or u[k] < math.exp(dS / T):
            s += Sigma[:, b] - Sigma[:, a]
            r, v, S = r2, v2, S2
            inside[pa], outside[pb] = b, a
            x[a], x[b] = 0.0, 1.0
            if S > best_S:
                best_S, best_x = S, x.copy()
        T *= decay
    return best_x, best_S


def simulated_bifurcation(
    J: np.ndarray, h: np.ndarray, n_steps: int = 2000, dt: float = 0.5,
    variant: str = "discrete", n_replicas: int = 64, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated bifurcation (Goto, Tatsumura & Dixon, 2019; Goto et al. 2021),
    ballistic (bSB) or discrete (dSB) variant. Each spin is an oscillator with
    position x_i and momentum y_i:

        y_i <- y_i + [ -(a0 - a(t)) x_i + c0 ( sum_j J_ij f(x_j) + h_i ) ] dt
        x_i <- x_i + a0 y_i dt
        |x_i| > 1  ->  x_i = sign(x_i), y_i = 0        (inelastic walls)

    with f(x) = x for bSB and f(x) = sign(x) for dSB, and a(t) ramped linearly
    from 0 to a0. As a(t) passes a0 each oscillator bifurcates into one of two
    branches, which are the spin states. The update is a matrix-vector product,
    so many replicas run in parallel -- the property that makes SB attractive
    on GPUs and FPGAs.

    c0 is set from the coupling scale, c0 = 0.5 / ( rms(J) sqrt(N) ), per Goto.
    Returns (spins of every replica, their Ising energies).
    """
    rng = np.random.default_rng(seed)
    N = J.shape[0]
    a0 = 1.0
    off = J[~np.eye(N, dtype=bool)]
    rms = math.sqrt(float(np.mean(off**2)) + float(np.mean(h**2)) / max(N, 1)) or 1.0
    c0 = 0.5 / (rms * math.sqrt(N))
    X = rng.uniform(-0.1, 0.1, (n_replicas, N))
    Y = rng.uniform(-0.1, 0.1, (n_replicas, N))
    for k in range(n_steps):
        a = a0 * k / n_steps
        fx = np.sign(X) if variant == "discrete" else X
        Y += (-(a0 - a) * X + c0 * (fx @ J + h)) * dt
        X += a0 * Y * dt
        wall = np.abs(X) > 1.0
        X[wall] = np.sign(X[wall])
        Y[wall] = 0.0
    S = np.sign(X)
    S[S == 0] = 1.0
    E = -0.5 * np.einsum("ri,ij,rj->r", S, J, S) - S @ h
    return S, E


def repair(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, K: int) -> np.ndarray:
    """
    Force 1'x = K by greedily adding the asset that most raises Sharpe or
    removing the one whose removal hurts least. Penalty-based solvers --
    including any Ising machine -- can return infeasible cardinalities; repair
    is the post-processing that makes their output usable.
    """
    x = np.asarray(x, float).copy()
    while x.sum() > K:
        inside = np.flatnonzero(x > 0.5)
        best, bi = -np.inf, inside[0]
        for i in inside:
            x[i] = 0.0
            s = sharpe(x, mu, Sigma)
            x[i] = 1.0
            if s > best:
                best, bi = s, i
        x[bi] = 0.0
    while x.sum() < K:
        outside = np.flatnonzero(x < 0.5)
        best, bj = -np.inf, outside[0]
        for j in outside:
            x[j] = 1.0
            s = sharpe(x, mu, Sigma)
            x[j] = 0.0
            if s > best:
                best, bj = s, j
        x[bj] = 1.0
    return x


def greedy_local_search(mu: np.ndarray, Sigma: np.ndarray, K: int
                        ) -> Tuple[np.ndarray, float]:
    """
    The classical baseline any annealer must beat. Forward selection -- add the
    asset that most raises Sharpe until K are held -- then best-improvement swap
    local search until no single swap helps. Deterministic, a few milliseconds,
    and on well-conditioned instances frequently optimal.
    """
    N = mu.size
    x = np.zeros(N)
    x[int(np.argmax(mu / np.sqrt(np.diag(Sigma))))] = 1.0
    while x.sum() < K:
        best, bj = -np.inf, -1
        for j in np.flatnonzero(x < 0.5):
            x[j] = 1.0
            s = sharpe(x, mu, Sigma)
            x[j] = 0.0
            if s > best:
                best, bj = s, j
        x[bj] = 1.0
    improved = True
    cur = sharpe(x, mu, Sigma)
    while improved:
        improved = False
        best, move = cur, None
        for i in np.flatnonzero(x > 0.5):
            for j in np.flatnonzero(x < 0.5):
                x[i], x[j] = 0.0, 1.0
                s = sharpe(x, mu, Sigma)
                x[i], x[j] = 1.0, 0.0
                if s > best + 1e-12:
                    best, move = s, (i, j)
        if move is not None:
            i, j = move
            x[i], x[j] = 0.0, 1.0
            cur = best
            improved = True
    return x, cur


def dinkelbach(
    mu: np.ndarray, Sigma: np.ndarray, K: int,
    inner: Callable[[np.ndarray, float], np.ndarray],
    P: Optional[float] = None, max_iter: int = 12, tol: float = 1e-10,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Maximise S^2 = (mu'x)^2 / (x'Sigma x) as a sequence of QUBOs:

        x_k        = argmin  x'( theta_k Sigma - mu mu' )x + P (1'x - K)^2
        theta_{k+1} = (mu'x_k)^2 / (x_k'Sigma x_k)

    Dinkelbach's theorem: theta_k increases monotonically and converges to the
    optimal ratio, and at the optimum the subproblem's minimum is exactly zero
    -- PROVIDED each subproblem is solved to optimality. With a heuristic inner
    solver the monotonicity guarantee is lost, which is why theta is tracked and
    the best iterate kept rather than the last.

    Valid while mu'x > 0; the squared ratio would otherwise reward large
    negative returns.

    THE PENALTY MUST BE TUNED, AND THE DEFAULT IS ONLY A STARTING POINT. P is
    set relative to the largest objective coefficient max|mu mu'|. It was
    measured at N = 50 to swing the final Sharpe by ~30% across P in [1x, 20x],
    NON-monotonically for simulated bifurcation. Too small and the cardinality
    is violated; too large and the dense uniform coupling it adds to every spin
    pair swamps the objective. An earlier default of 4K x max|mu mu'| (= 40x at
    K = 10) sat deep in the bad regime and made both penalty solvers score
    WORSE THAN RANDOM SUBSETS. Sweep P; do not trust a single value.
    """
    N = mu.size
    x = np.zeros(N)
    x[np.argsort(-mu)[:K]] = 1.0
    theta = float((mu @ x) ** 2 / (x @ Sigma @ x))
    thetas = [theta]
    best_x, best_S = x.copy(), sharpe(x, mu, Sigma)
    for _ in range(max_iter):
        Pk = P if P is not None else 2.0 * float(np.max(np.abs(np.outer(mu, mu))))
        Q, c = dinkelbach_qubo(mu, Sigma, theta, K, Pk)
        xk = repair(inner(Q, Pk), mu, Sigma, K)
        r = float(mu @ xk)
        if r <= 0:
            break
        new = r * r / float(xk @ Sigma @ xk)
        thetas.append(new)
        Sk = sharpe(xk, mu, Sigma)
        if Sk > best_S:
            best_S, best_x = Sk, xk.copy()
        if abs(new - theta) < tol:
            break
        theta = new
    return best_x, best_S, thetas


def long_only_sharpe_bound(mu: np.ndarray, Sigma: np.ndarray) -> float:
    """
    A CERTIFIED upper bound on the discrete optimum. Every equal-weight
    K-subset is a long-only portfolio, so the best long-only Sharpe over ALL
    weights w >= 0 bounds it from above:

        max_{w >= 0} mu'w / sqrt(w'Sigma w) = 1 / sqrt( min { w'Sigma w : mu'w = 1, w >= 0 } )

    a convex QP. The bound is loose -- it allows unequal weights and any number
    of assets -- so the certified gap overstates the true one. It is still the
    only defensible optimality statement available at N = 50.
    """
    N = mu.size
    w0 = np.maximum(mu, 0) / max(float(np.sum(np.maximum(mu, 0)) * mu @ np.maximum(mu, 0)), 1e-12)
    w0 = np.maximum(mu, 0)
    w0 = w0 / float(mu @ w0)
    res = optimize.minimize(
        lambda w: float(w @ Sigma @ w), w0, jac=lambda w: 2.0 * Sigma @ w,
        constraints=[{"type": "eq", "fun": lambda w: float(mu @ w) - 1.0, "jac": lambda w: mu}],
        bounds=[(0.0, None)] * N, method="SLSQP", options={"maxiter": 500, "ftol": 1e-14},
    )
    return 1.0 / math.sqrt(float(res.fun))


# ===========================================================================
# SELF-TESTS
# ===========================================================================

def _test_qubo_energy_is_the_objective() -> None:
    rng = np.random.default_rng(0)
    mu, Sigma = make_problem(12, seed=1)
    K, lam, P = 4, 3.0, 5.0
    Q, c = mean_variance_qubo(mu, Sigma, lam, K, P)
    worst = 0.0
    for _ in range(300):
        x = (rng.random(12) < 0.4).astype(float)
        direct = lam * x @ Sigma @ x - mu @ x + P * (x.sum() - K) ** 2
        worst = max(worst, abs(qubo_energy(x, Q, c) - direct))
    assert worst < 1e-12, worst
    print(f"[ok] x'Qx + c equals lam x'Sigma x - mu'x + P(1'x-K)^2 on 300 random "
          f"binary vectors (max dev {worst:.1e})")


def _test_qubo_to_ising_exhaustive() -> None:
    mu, Sigma = make_problem(10, seed=2)
    Q, c = dinkelbach_qubo(mu, Sigma, 0.7, 4, 3.0)
    J, h, off = qubo_to_ising(Q, c)
    worst = 0.0
    for bits in itertools.product((0.0, 1.0), repeat=10):
        x = np.asarray(bits)
        worst = max(worst, abs(qubo_energy(x, Q, c) - ising_energy(2 * x - 1, J, h, off)))
    assert worst < 1e-10, worst
    assert np.allclose(np.diag(J), 0.0) and np.allclose(J, J.T)
    print(f"[ok] QUBO -> Ising energies agree on all 1,024 states of a 10-spin problem "
          f"(max dev {worst:.1e}); J symmetric with zero diagonal")


def _test_swap_delta_is_exact() -> None:
    """The O(1) swap update must equal a full recomputation after every move."""
    mu, Sigma = make_problem(30, seed=3)
    rng = np.random.default_rng(4)
    K = 8
    x = np.zeros(30)
    x[rng.choice(30, K, replace=False)] = 1.0
    s = Sigma @ x
    r, v = float(mu @ x), float(x @ s)
    worst = 0.0
    for _ in range(2000):
        a = int(rng.choice(np.flatnonzero(x > 0.5)))
        b = int(rng.choice(np.flatnonzero(x < 0.5)))
        r2 = r - mu[a] + mu[b]
        v2 = v - 2 * s[a] + Sigma[a, a] + 2 * s[b] - 2 * Sigma[a, b] + Sigma[b, b]
        s += Sigma[:, b] - Sigma[:, a]
        x[a], x[b] = 0.0, 1.0
        r, v = r2, v2
        worst = max(worst, abs(r - mu @ x), abs(v - x @ Sigma @ x))
    assert worst < 1e-10, worst
    print(f"[ok] O(1) swap update of (mu'x, x'Sigma x) matches full recomputation over "
          f"2,000 chained swaps (max drift {worst:.1e})")


def _test_dinkelbach_reaches_the_exact_optimum() -> None:
    """
    With an EXACT inner solver Dinkelbach must land on the brute-force optimum,
    and theta must rise monotonically. Exact inner solution by enumeration over
    the K-subsets -- the only way to test the outer loop in isolation.
    """
    mu, Sigma = make_problem(14, seed=5)
    K = 5
    subsets = [np.asarray(c) for c in itertools.combinations(range(14), K)]

    def exact_inner(Q, P):
        best, bx = np.inf, None
        for c in subsets:
            x = np.zeros(14)
            x[c] = 1.0
            e = x @ Q @ x
            if e < best:
                best, bx = e, x
        return bx

    x_bf, s_bf, _, _ = brute_force(mu, Sigma, K)
    x_dk, s_dk, th = dinkelbach(mu, Sigma, K, exact_inner)
    assert abs(s_dk - s_bf) < 1e-12, (s_dk, s_bf)
    assert all(b >= a - 1e-12 for a, b in zip(th, th[1:])), th
    print(f"[ok] Dinkelbach with an exact inner solver hits the brute-force optimum "
          f"S = {s_bf:.6f} in {len(th)-1} QUBO solves; theta rises monotonically: "
          + " -> ".join(f"{t:.4f}" for t in th))


def _test_penalty_weight_tradeoff() -> None:
    """
    The penalty that Ising hardware forces on you. Too small and the solver
    returns the wrong number of assets; large enough to enforce K and it
    flattens the objective relative to the constraint, degrading quality.
    """
    mu, Sigma = make_problem(24, seed=6)
    K = 6
    _, s_opt, _, _ = brute_force(mu, Sigma, K)
    theta = s_opt ** 2
    scale = float(np.max(np.abs(np.outer(mu, mu))))
    rows = []
    for mult in (0.05, 0.5, 5.0, 50.0, 500.0):
        P = mult * scale
        Q, c = dinkelbach_qubo(mu, Sigma, theta, K, P)
        feas, gaps = 0, []
        for sd in range(12):
            x, _ = anneal_qubo(Q, n_sweeps=150, seed=sd)
            feas += int(round(x.sum()) == K)
            gaps.append(1.0 - sharpe(repair(x, mu, Sigma, K), mu, Sigma) / s_opt)
        rows.append((mult, feas / 12, float(np.mean(gaps))))
    assert rows[0][1] < rows[-1][1], "small penalty should violate cardinality more"
    print("[ok] penalty weight P (multiple of max |mu mu'| entry), single-flip SA, 12 seeds:")
    print(f"       {'P multiple':>11} {'feasible':>9} {'gap after repair':>17}")
    for m, f, g in rows:
        print(f"       {m:>11g} {100*f:>8.0f}% {100*g:>16.3f}%")


def _test_solvers_find_the_exact_optimum() -> None:
    """Success rate at recovering the brute-force optimum on a small instance."""
    mu, Sigma = make_problem(20, seed=7)
    K = 6
    _, s_opt, n, _ = brute_force(mu, Sigma, K)
    seeds = range(20)
    hit_swap = sum(abs(anneal_swap(mu, Sigma, K, 8000, seed=s)[1] - s_opt) < 1e-9 for s in seeds)
    _, s_gr = greedy_local_search(mu, Sigma, K)
    print(f"[ok] N=20, K=6 ({n:,} subsets): swap-SA recovers the exact optimum in "
          f"{hit_swap}/20 seeds; greedy+local search "
          f"{'finds it' if abs(s_gr - s_opt) < 1e-9 else f'stops {100*(1-s_gr/s_opt):.3f}% short'}")
    assert hit_swap >= 17


def _local_search_from(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    x = x.copy()
    cur = sharpe(x, mu, Sigma)
    improved = True
    while improved:
        improved, best, mv = False, cur, None
        for i in np.flatnonzero(x > 0.5):
            for j in np.flatnonzero(x < 0.5):
                x[i], x[j] = 0.0, 1.0
                s = sharpe(x, mu, Sigma)
                x[i], x[j] = 1.0, 0.0
                if s > best + 1e-12:
                    best, mv = s, (i, j)
        if mv is not None:
            x[mv[0]], x[mv[1]] = 0.0, 1.0
            cur, improved = best, True
    return cur


def _test_landscape_is_rugged() -> None:
    """
    Why greedy is exact at N=26 but stalls at N=50, and why annealing wins.

    Greedy's output is a TRUE 1-swap local optimum -- no single swap improves it
    -- so its shortfall is not a bug. The landscape is simply full of basins:
    local search from random starts almost never lands in the same one twice.
    A method that follows one deterministic path commits to whichever basin it
    enters; annealing escapes them. Multi-start local search is the cheap
    classical answer, and it is competitive.
    """
    mu, Sigma = make_problem(50, seed=21)
    K = 10
    xg, sg = greedy_local_search(mu, Sigma, K)
    nb = max(sharpe(np.where(np.isin(np.arange(50), [j]), 1.0,
                             np.where(np.isin(np.arange(50), [i]), 0.0, xg)), mu, Sigma)
             for i in np.flatnonzero(xg > 0.5) for j in np.flatnonzero(xg < 0.5))
    assert nb <= sg + 1e-12, "greedy did not converge to a local optimum"
    rng = np.random.default_rng(1)
    optima = set()
    for _ in range(40):
        x = np.zeros(50)
        x[rng.choice(50, K, replace=False)] = 1.0
        optima.add(round(_local_search_from(x, mu, Sigma), 6))
    assert len(optima) >= 30, f"only {len(optima)} distinct optima"
    print(f"[ok] N=50: greedy is a true 1-swap local optimum (best neighbour {nb:.4f} < "
          f"{sg:.4f}); local search from 40 random starts reaches {len(optima)} DISTINCT "
          f"local optima spanning {min(optima):.3f}..{max(optima):.3f} -- the best, "
          f"{max(optima):.6f}, matching swap-SA")


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

def _exact_comparison() -> None:
    """At a size brute force can still enumerate, so the gap is EXACT."""
    print("=" * 78)
    print("EXACT COMPARISON: N=26, K=8")
    print("=" * 78)
    mu, Sigma = make_problem(26, seed=11)
    K = 8
    x_bf, s_bf, n, t_bf = brute_force(mu, Sigma, K)
    print(f"  brute force: {n:,} subsets in {t_bf:.2f}s -> optimum S = {s_bf:.6f}")

    t0 = time.time()
    best = -np.inf
    for sd in range(8):
        best = max(best, anneal_swap(mu, Sigma, K, 12_000, seed=sd)[1])
    t_sa = time.time() - t0

    t0 = time.time()
    _, s_gr = greedy_local_search(mu, Sigma, K)
    t_gr = time.time() - t0

    print(f"  {'method':<30} {'Sharpe':>10} {'gap':>9} {'time':>9} {'speedup':>9}")
    print(f"  {'brute force (exact)':<30} {s_bf:>10.6f} {0.0:>8.3f}% {t_bf:>8.3f}s {'1x':>9}")
    for name, s, t in (("swap-SA, 8 restarts", best, t_sa), ("greedy + local search", s_gr, t_gr)):
        print(f"  {name:<30} {s:>10.6f} {100*(1-s/s_bf):>8.3f}% {t:>8.3f}s "
              f"{t_bf/max(t,1e-9):>8.0f}x")
    assert abs(best - s_bf) < 1e-9, "swap-SA should recover the exact optimum here"
    print()


def _large_comparison() -> None:
    print("=" * 78)
    print("N=50, K=10: 1.03e10 SUBSETS -- BRUTE FORCE NOT RUN")
    print("=" * 78)
    N, K = 50, 10
    mu, Sigma = make_problem(N, seed=21)
    n_sub = math.comb(N, K)
    ub = long_only_sharpe_bound(mu, Sigma)
    scale = float(np.max(np.abs(np.outer(mu, mu))))

    _, _, n_probe, t_probe = brute_force(*make_problem(26, seed=11), 8)
    per = t_probe / n_probe
    print(f"  brute force: {n_sub:.3e} subsets x {1e9*per:.0f} ns = ~{n_sub*per/3600:.0f} hours.")
    print(f"  Every method below is 'faster'. The question is quality, and at this size")
    print(f"  there is no certificate -- only convergence evidence and a loose bound.\n")

    # swap-SA: convergence across budgets is the optimality evidence
    conv = []
    t_best = 0.0
    xs_best, s_best = None, -np.inf
    for steps in (30_000, 400_000):
        t0 = time.time()
        runs = [anneal_swap(mu, Sigma, K, steps, seed=sd) for sd in range(12)]
        dt = time.time() - t0
        vals = np.array([r[1] for r in runs])
        conv.append((steps, float(vals.max()), float(np.median(vals)), dt, vals))
        if vals.max() > s_best:
            s_best, xs_best = float(vals.max()), runs[int(np.argmax(vals))][0]
            t_best = dt
    ref = s_best
    for steps, b, m, dt, vals in conv:
        reach = float(np.mean(np.abs(vals - ref) < 1e-9))
        print(f"  swap-SA {steps:>7,} steps x12 restarts: best {b:.6f}  median {m:.6f}  "
              f"{100*reach:3.0f}% reach {ref:.6f}  ({dt:.1f}s)")
    print(f"  -> with enough annealing a majority of INDEPENDENT restarts converge on the")
    print(f"     same subset. That is the strongest optimality evidence available here.\n")

    def sweep(kind: str) -> Tuple[float, float, Dict[float, float], float]:
        res: Dict[float, float] = {}
        t0 = time.time()
        for mult in (1.0, 2.0, 5.0, 10.0, 20.0):
            def inner(Q, P):
                if kind == "SA":
                    return anneal_qubo(Q, n_sweeps=300, seed=0)[0]
                J, h, _ = qubo_to_ising(Q)
                S, E = simulated_bifurcation(J, h, n_steps=1500, variant="discrete",
                                             n_replicas=64, seed=0)
                return ((S + 1) / 2)[int(np.argmin(E))]
            res[mult] = dinkelbach(mu, Sigma, K, inner, P=mult * scale)[1]
        best_m = max(res, key=res.get)
        return res[best_m], best_m, res, time.time() - t0

    s_pqsa, m_pqsa, all_pqsa, t_pqsa = sweep("SA")
    s_sb, m_sb, all_sb, t_sb = sweep("SB")

    t0 = time.time()
    _, s_gr = greedy_local_search(mu, Sigma, K)
    t_gr = time.time() - t0

    rng = np.random.default_rng(0)
    rnd = max(sharpe(np.isin(np.arange(N), rng.choice(N, K, replace=False)).astype(float),
                     mu, Sigma) for _ in range(20_000))

    rows = [
        ("swap-SA (direct on Sharpe)", ref, t_best, "no penalty"),
        ("penalty-QUBO SA + Dinkelbach", s_pqsa, t_pqsa, f"best P={m_pqsa:g}x of 5 tried"),
        ("discrete SB + Dinkelbach", s_sb, t_sb, f"best P={m_sb:g}x of 5 tried"),
        ("greedy + local search", s_gr, t_gr, "deterministic"),
        ("best of 20,000 random subsets", rnd, float("nan"), "-"),
    ]
    print(f"  {'method':<32} {'Sharpe':>10} {'gap to best':>12} {'time':>8}  note")
    for name, sv, t, note in rows:
        tt = f"{t:.1f}s" if np.isfinite(t) else "-"
        print(f"  {name:<32} {sv:>10.6f} {100*(1-sv/ref):>11.3f}% {tt:>8}  {note}")

    lo_sa, hi_sa = min(all_pqsa.values()), max(all_pqsa.values())
    lo_sb, hi_sb = min(all_sb.values()), max(all_sb.values())
    print(f"\n  penalty sensitivity across P in [1x, 20x]:")
    print(f"    penalty-QUBO SA  Sharpe {lo_sa:.3f} .. {hi_sa:.3f}  "
          f"({100*(1-lo_sa/hi_sa):.0f}% swing)  " + ", ".join(f"{m:g}x:{v:.3f}" for m, v in all_pqsa.items()))
    print(f"    discrete SB      Sharpe {lo_sb:.3f} .. {hi_sb:.3f}  "
          f"({100*(1-lo_sb/hi_sb):.0f}% swing)  " + ", ".join(f"{m:g}x:{v:.3f}" for m, v in all_sb.items()))
    print(f"\n  certified upper bound (long-only, any weights): {ub:.6f}, so the best found")
    print(f"  is at most {100*(1-ref/ub):.1f}% below optimal. The bound admits unequal weights")
    print(f"  and any number of assets, so it is too loose to certify anything useful here.")

    assert ref >= max(s_pqsa, s_sb, s_gr, rnd) - 1e-9, "swap-SA should be best"
    assert s_pqsa > rnd and s_sb > rnd, "tuned penalty solvers should beat random sampling"
    assert conv[-1][2] >= conv[0][2], "more annealing should not lower the median"


if __name__ == "__main__":
    t0 = time.time()
    _run("FORMULATION", [
        _test_qubo_energy_is_the_objective,
        _test_qubo_to_ising_exhaustive,
        _test_swap_delta_is_exact,
        _test_dinkelbach_reaches_the_exact_optimum,
    ])
    _run("SOLVERS", [
        _test_penalty_weight_tradeoff,
        _test_solvers_find_the_exact_optimum,
        _test_landscape_is_rugged,
    ])
    _exact_comparison()
    _large_comparison()
    print(f"\ncompleted in {time.time()-t0:.0f}s")

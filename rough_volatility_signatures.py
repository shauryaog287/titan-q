"""
rough_volatility_signatures.py
==============================

Phase 17: Rough paths and signature methods for LOB feature extraction.

For a path X : [0,T] -> R^d, the SIGNATURE is the collection of iterated
integrals

    S(X)^{i_1 ... i_k} = int_{0 < t_1 < ... < t_k < T} dX^{i_1}_{t_1} ... dX^{i_k}_{t_k}

indexed by words (i_1, ..., i_k) over the alphabet {1..d}. Level k has d^k
terms; the truncated signature to depth N has 1 + d + ... + d^N.

For a piecewise-linear path the signature is EXACT and needs no quadrature.
A single segment with increment D has signature equal to the tensor
exponential

    exp(D) = ( 1, D, D (x) D / 2!, D (x) D (x) D / 3!, ... )

and CHEN'S IDENTITY makes concatenation a tensor product,

    S( X * Y ) = S(X) (x) S(Y)

so the signature of a sampled path is the ordered product of the exponentials
of its increments. Everything here is built on those two facts, and both are
verified to machine precision in the self-tests.

THE ONE FACT THAT MATTERS FOR ORDER FLOW
---------------------------------------
Level 2 splits exactly into a symmetric and an antisymmetric part:

    S^{ij} + S^{ji} = D^i D^j                   (shuffle identity)
    A^{ij} = ( S^{ij} - S^{ji} ) / 2            (Levy area)

The symmetric part is a function of the TOTAL increments only; it cannot tell
two paths with the same endpoints apart. The Levy area is the signed area
between the path and its chord, and it depends on the ORDER in which the
coordinates moved. "Volume rose, then price fell" and "price fell, then volume
rose" have identical increments, identical level 1, identical symmetric level
2 -- and Levy areas that differ by exactly D^V * D^P.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "signature",
    "tensor_mul",
    "tensor_log",
    "word_index",
    "sig_term",
    "flatten",
    "levy_area",
    "lead_lag",
    "time_augment",
    "add_basepoint",
    "realised_covariation_from_leadlag",
    "fgn_davies_harte",
    "hurst_from_leadlag",
    "simulate_gbm",
    "simulate_order_task",
]


# ===========================================================================
# 1. TRUNCATED TENSOR ALGEBRA
# ===========================================================================

Sig = List[np.ndarray]   # levels 0..N, level k shaped (B, d^k)


def _outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched outer product (B, p) x (B, q) -> (B, p*q), row-major word order."""
    B = a.shape[0]
    return (a[:, :, None] * b[:, None, :]).reshape(B, a.shape[1] * b.shape[1])


def tensor_mul(a: Sig, b: Sig, depth: int) -> Sig:
    """
    Product in the truncated tensor algebra:

        ( a (x) b )_k = sum_{j=0}^{k} a_j (x) b_{k-j}

    Chen's identity says this is concatenation of paths.
    """
    B = a[0].shape[0]
    out: Sig = []
    for k in range(depth + 1):
        acc = None
        for j in range(k + 1):
            t = _outer(a[j], b[k - j])
            acc = t if acc is None else acc + t
        out.append(acc)
    return out


def signature(paths: np.ndarray, depth: int) -> Sig:
    """
    Exact truncated signature of piecewise-linear paths, batched.

    paths  (n, d) or (B, n, d)
    returns list of levels 0..depth, level k shaped (B, d^k)

    Implementation: left-to-right Chen product of segment exponentials,

        S <- S (x) exp(D_s),   ( S (x) exp(D) )_k = sum_j S_j (x) D^{(x)(k-j)} / (k-j)!

    vectorised across the batch so the only Python loop is over segments.
    Cost per segment is O(depth * d^depth) per path.

    No approximation is made anywhere: this is the signature of the
    piecewise-linear interpolant, to machine precision. Whether that
    interpolant is the right object for tick data is a modelling question
    addressed by the transforms below, not a numerical one.
    """
    X = np.asarray(paths, dtype=float)
    if X.ndim == 2:
        X = X[None]
    if X.ndim != 3:
        raise ValueError("paths must be (n, d) or (B, n, d)")
    B, n, d = X.shape
    if n < 2:
        raise ValueError("need at least 2 points")
    if depth < 1:
        raise ValueError("depth must be >= 1")

    sig: Sig = [np.ones((B, 1))] + [np.zeros((B, d**k)) for k in range(1, depth + 1)]
    inc = np.diff(X, axis=1)
    for s in range(n - 1):
        D = inc[:, s, :]
        P = [np.ones((B, 1)), D]
        for m in range(2, depth + 1):
            P.append(_outer(P[-1], D) / m)
        new: Sig = [sig[0]]
        for k in range(1, depth + 1):
            acc = sig[k] + P[k]                 # j = k and j = 0 terms
            for j in range(1, k):
                acc = acc + _outer(sig[j], P[k - j])
            new.append(acc)
        sig = new
    return sig


def tensor_log(sig: Sig, depth: int) -> Sig:
    """
    Log-signature via the truncated series

        log( 1 + x ) = sum_{m=1}^{N} (-1)^{m+1} x^{(x)m} / m

    The signature is highly redundant -- the shuffle relations mean products of
    signature terms are themselves linear in higher signature terms. The
    log-signature removes that redundancy and lives in the free Lie algebra:
    at level 2 it is exactly the Levy area and nothing else.
    """
    B = sig[0].shape[0]
    d = sig[1].shape[1]
    x: Sig = [np.zeros((B, 1))] + [s.copy() for s in sig[1:]]
    out: Sig = [np.zeros((B, 1))] + [np.zeros((B, d**k)) for k in range(1, depth + 1)]
    power = x
    for m in range(1, depth + 1):
        coef = (1.0 if m % 2 == 1 else -1.0) / m
        for k in range(depth + 1):
            out[k] = out[k] + coef * power[k]
        power = tensor_mul(power, x, depth)
    return out


def word_index(word: Sequence[int], d: int) -> int:
    """Flat index of a word in its level, row-major: sum_m w_m d^(k-1-m)."""
    idx = 0
    for w in word:
        if not 0 <= w < d:
            raise ValueError(f"letter {w} outside alphabet of size {d}")
        idx = idx * d + int(w)
    return idx


def sig_term(sig: Sig, word: Sequence[int]) -> np.ndarray:
    """S^{word} for every path in the batch."""
    if len(word) == 0:
        return sig[0][:, 0]
    d = sig[1].shape[1]
    return sig[len(word)][:, word_index(word, d)]


def flatten(sig: Sig, skip_level0: bool = True) -> np.ndarray:
    parts = sig[1:] if skip_level0 else sig
    return np.concatenate(parts, axis=1)


def levy_area(sig: Sig, i: int, j: int) -> np.ndarray:
    """
        A^{ij} = ( S^{ij} - S^{ji} ) / 2

    The signed area enclosed between the (i, j) projection of the path and the
    chord joining its endpoints. Antisymmetric, zero for any path that moves
    along a straight line, and the lowest-order signature term that depends on
    the ORDER of moves rather than their totals.
    """
    return 0.5 * (sig_term(sig, (i, j)) - sig_term(sig, (j, i)))


# ===========================================================================
# 2. PATH TRANSFORMS
# ===========================================================================

def add_basepoint(X: np.ndarray) -> np.ndarray:
    """
    Prepend the origin. The signature is translation invariant: it sees
    increments, never levels. Starting every path at 0 makes the initial level
    visible as a first increment, when the level itself is informative.
    """
    X = np.asarray(X, dtype=float)
    z = np.zeros_like(X[..., :1, :])
    return np.concatenate([z, X], axis=-2)


def time_augment(X: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    Append t as a channel.

    Without time, the signature is invariant to reparametrisation AND blind to
    tree-like excursions: a path that goes out and retraces its steps has the
    signature of the constant path. With a strictly increasing time channel,
    no excursion can retrace, and the signature determines the path uniquely
    (Hambly & Lyons). For market data, where "how fast" is information,
    time augmentation is usually required rather than optional.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[-2]
    t = np.linspace(0.0, T, n)
    t = np.broadcast_to(t.reshape((1,) * (X.ndim - 2) + (n, 1)), X.shape[:-1] + (1,))
    return np.concatenate([t, X], axis=-1)


def lead_lag(X: np.ndarray) -> np.ndarray:
    """
    Lead-lag embedding of a d-dimensional sampled path into R^{2d}.

    From X_0, ..., X_n build 2n + 1 points in which the LEAD copy moves first
    and the LAG copy catches up:

        (X_0, X_0), (X_1, X_0), (X_1, X_1), (X_2, X_1), (X_2, X_2), ...
         lead  lag

    Layout: channels 0..d-1 are the lead copy, d..2d-1 the lag copy.

    The point of the construction is an exact identity. For each coordinate,
    the Levy area between its lead and lag copies is half the realised
    quadratic variation:

        A( X^lead, X^lag ) = (1/2) sum_i ( X_{i+1} - X_i )^2

    and for two coordinates the lead-lag areas split cleanly,

        A( X^lead, Y^lag ) + A( Y^lead, X^lag ) = sum_i dX_i dY_i
        A( X^lead, Y^lag ) - A( Y^lead, X^lag ) = 2 A( X, Y )

    i.e. the SYMMETRIC combination is the realised covariation -- a
    second-moment, order-blind statistic -- and the ANTISYMMETRIC one is the
    Levy area of the underlying pair, which carries who-moved-first. Both are
    verified exactly in the self-tests.

    This is why the lead-lag transform exists: a piecewise-linear path has zero
    quadratic variation, so an Ito-type quantity is invisible to the signature
    of the raw interpolant. The lead-lag path puts it back, at level 2.
    """
    X = np.asarray(X, dtype=float)
    single = X.ndim == 2
    if single:
        X = X[None]
    B, n, d = X.shape
    out = np.empty((B, 2 * n - 1, 2 * d), dtype=float)
    out[:, 0::2, :d] = X
    out[:, 0::2, d:] = X
    out[:, 1::2, :d] = X[:, 1:]
    out[:, 1::2, d:] = X[:, :-1]
    return out[0] if single else out


def realised_covariation_from_leadlag(sig_ll: Sig, i: int, j: int, d: int) -> np.ndarray:
    """
    Realised covariation of channels i and j, read off the lead-lag signature:

        sum dX^i dX^j = A( X^i lead, X^j lag ) + A( X^j lead, X^i lag )

    For i == j this is the realised variance, 2 A( X^lead, X^lag ).
    """
    return levy_area(sig_ll, i, d + j) + levy_area(sig_ll, j, d + i)


# ===========================================================================
# 3. ROUGHNESS
# ===========================================================================

def fgn_davies_harte(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """
    Exact fractional Gaussian noise by circulant embedding.

    Autocovariance of unit fGN:

        gamma(k) = (1/2) ( |k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H} )

    Embed in a 2n circulant, diagonalise by FFT, colour complex white noise:

        Y = FFT( sqrt( lambda / m ) * ( Z1 + i Z2 ) ),   m = 2n

    Re(Y) then has covariance exactly equal to the circulant, whose leading
    n x n block is the fGN covariance. O(n log n) and exact, against O(n^3) for
    a Cholesky factorisation -- 2^16 points in milliseconds rather than never.
    Valid whenever every circulant eigenvalue is non-negative, which holds for
    all H in (0, 1).
    """
    if not 0.0 < H < 1.0:
        raise ValueError("H must lie in (0, 1)")
    k = np.arange(n + 1, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2.0 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    row = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(row).real
    if lam.min() < -1e-10:
        raise RuntimeError(f"circulant embedding not PSD (min eigenvalue {lam.min():.2e})")
    lam = np.maximum(lam, 0.0)
    m = row.size
    W = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(np.sqrt(lam / m) * W).real[:n]


def hurst_from_leadlag(path: np.ndarray, scales: Sequence[int] = (1, 2, 4, 8, 16, 32)
                       ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Hurst exponent from the scaling of the lead-lag Levy area.

    For fBM, E[ (X_{t+h} - X_t)^2 ] = sigma^2 h^{2H}. Sampled at step h over a
    horizon of n points there are n/h increments, so the realised variance --
    which is EXACTLY twice the lead-lag area -- scales as

        RV_h ~ n h^{2H - 1}     =>     H = ( slope of log RV_h on log h + 1 ) / 2

    H = 1/2 is Brownian (flat RV across scales). H < 1/2 is ROUGH: RV grows as
    the sampling gets finer, which is the empirical signature of log-volatility
    reported by Gatheral, Jaisson and Rosenbaum, with H around 0.1.

    Returns (H_hat, scales, realised variances).
    """
    x = np.asarray(path, dtype=float).ravel()
    rv = []
    for h in scales:
        sub = x[::h]
        sig = signature(lead_lag(sub[:, None]), depth=2)
        rv.append(float(2.0 * levy_area(sig, 0, 1)[0]))
    rv = np.asarray(rv)
    sc = np.asarray(scales, dtype=float)
    slope = float(np.polyfit(np.log(sc), np.log(rv), 1)[0])
    return (slope + 1.0) / 2.0, sc, rv


# ===========================================================================
# 4. SIMULATORS
# ===========================================================================

def simulate_gbm(
    n: int, T: float, mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray,
    s0: np.ndarray, rng: np.random.Generator, n_paths: int = 1,
) -> np.ndarray:
    """Correlated multivariate GBM, exact in distribution at the grid. (B, n+1, d)."""
    mu, sigma, s0 = map(lambda a: np.asarray(a, dtype=float), (mu, sigma, s0))
    d = mu.size
    dt = T / n
    L = np.linalg.cholesky(np.asarray(corr, dtype=float))
    Z = rng.standard_normal((n_paths, n, d)) @ L.T
    logret = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    lp = np.concatenate([np.zeros((n_paths, 1, d)), np.cumsum(logret, axis=1)], axis=1)
    return s0 * np.exp(lp)


def simulate_order_task(
    n_paths: int, n_steps: int = 60, rng: Optional[np.random.Generator] = None,
    vol_jump: float = 1.0, px_drop: float = 1.0, noise: float = 0.35,
    lag_range: Tuple[int, int] = (4, 16),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paths over (log price, log volume, spread) with one volume shift and one
    price drop. The ONLY difference between classes is their order.

        label 1  volume shifts up, then price drops after a random lag
        label 0  price drops, then volume shifts up after a random lag

    Jump sizes, noise, event timing distribution and lag distribution are
    identical across classes. Every order-blind statistic -- totals, moments,
    extremes, contemporaneous correlation -- therefore has the same
    distribution under both labels, and the task is solvable only from the
    temporal ordering. The lag is random so that no fixed-lag cross-correlation
    shortcut exists either.

    Moves are spread over three steps so the events are not single-point
    spikes that a detector could find by position alone.
    """
    rng = rng or np.random.default_rng(0)
    X = np.cumsum(rng.standard_normal((n_paths, n_steps, 3)) * noise * 0.25, axis=1)
    X[:, :, 2] = 1.0 + 0.1 * rng.standard_normal((n_paths, n_steps))
    y = rng.integers(0, 2, n_paths)
    ramp = np.array([1 / 3, 2 / 3, 1.0])
    for b in range(n_paths):
        lag = int(rng.integers(*lag_range))
        t1 = int(rng.integers(5, n_steps - lag - 8))
        t2 = t1 + lag
        tv, tp = (t1, t2) if y[b] == 1 else (t2, t1)
        for k, r in enumerate(ramp):
            X[b, tv + k, 1] += vol_jump * r
            X[b, tp + k, 0] -= px_drop * r
        X[b, tv + 3:, 1] += vol_jump
        X[b, tp + 3:, 0] -= px_drop
    return X, y


# ===========================================================================
# 5. CLASSIFIERS FOR THE ORDER EXPERIMENT
# ===========================================================================

def _logistic_fit_predict(Xtr, ytr, Xte, l2: float = 1e-2) -> np.ndarray:
    from scipy.optimize import minimize
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
    A = np.hstack([A, np.ones((A.shape[0], 1))])
    Bm = np.hstack([Bm, np.ones((Bm.shape[0], 1))])

    def f(w):
        z = A @ w
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        loss = -np.mean(ytr * np.log(p + eps) + (1 - ytr) * np.log(1 - p + eps))
        g = A.T @ (p - ytr) / len(ytr)
        return loss + 0.5 * l2 * w[:-1] @ w[:-1], g + np.r_[l2 * w[:-1], 0.0]

    w = minimize(f, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 500}).x
    return (Bm @ w > 0).astype(int)


def order_blind_features(X: np.ndarray) -> np.ndarray:
    """Totals, moments, extremes and contemporaneous correlations of increments."""
    dX = np.diff(X, axis=1)
    f = [X[:, -1] - X[:, 0], dX.std(1), dX.min(1), dX.max(1),
         np.abs(dX).sum(1), (dX**3).mean(1), (dX**4).mean(1)]
    c = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = dX[:, :, i], dX[:, :, j]
        c.append(((a - a.mean(1, keepdims=True)) * (b - b.mean(1, keepdims=True))).mean(1))
    return np.hstack(f + [np.stack(c, 1)])


def signature_features(X: np.ndarray, depth: int) -> np.ndarray:
    return flatten(signature(time_augment(add_basepoint(X)), depth))


def gru_accuracy(Xtr, ytr, Xte, yte, epochs: int = 60, seed: int = 0) -> float:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu = Xtr.reshape(-1, 3).mean(0)
    sd = Xtr.reshape(-1, 3).std(0) + 1e-9
    to = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    A, Bt = to(Xtr), to(Xte)
    ya = torch.tensor(ytr, dtype=torch.float32)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.GRU(3, 32, batch_first=True)
            self.h = nn.Linear(32, 1)

        def forward(self, x):
            return self.h(self.g(x)[0][:, -1]).squeeze(-1)

    m = M()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    bs = min(64, len(ytr))
    for _ in range(epochs):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.binary_cross_entropy_with_logits(m(A[idx]), ya[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = (m(Bt) > 0).int().numpy()
    return float(np.mean(pred == yte))


# ===========================================================================
# SELF-TESTS
# ===========================================================================

def _rand_path(n: int, d: int, seed: int) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal((n, d)), axis=0)


def _test_level1_is_the_increment() -> None:
    X = _rand_path(50, 4, 0)
    s = signature(X, 3)
    assert np.allclose(s[1][0], X[-1] - X[0], atol=1e-12)
    print("[ok] level 1 = X_T - X_0 exactly")


def _test_single_segment_is_the_tensor_exponential() -> None:
    D = np.array([0.3, -1.2, 0.7])
    X = np.stack([np.zeros(3), D])
    s = signature(X, 4)
    P = D.copy()
    for k in range(1, 5):
        assert np.allclose(s[k][0], P / math.factorial(k), atol=1e-14)
        P = np.outer(P, D).ravel()
    print("[ok] one linear segment has signature exp(D) = (1, D, D(x)D/2!, ...) exactly")


def _test_chen_identity() -> None:
    X, Y = _rand_path(30, 3, 1), _rand_path(25, 3, 2)
    Y = Y - Y[0] + X[-1]
    joined = np.concatenate([X, Y[1:]])
    lhs = signature(joined, 4)
    rhs = tensor_mul(signature(X, 4), signature(Y, 4), 4)
    err = max(float(np.max(np.abs(a - b))) for a, b in zip(lhs, rhs))
    assert err < 1e-9, err
    print(f"[ok] Chen: S(X * Y) = S(X) (x) S(Y) to {err:.1e} through level 4")


def _test_retracing_is_invisible() -> None:
    """
    A path followed by its own reversal has the signature of the constant path:
    the signature quotients out tree-like excursions. Time augmentation removes
    this blind spot.
    """
    X = _rand_path(40, 3, 3)
    loop = np.concatenate([X, X[-2::-1]])
    s = signature(loop, 4)
    worst = max(float(np.max(np.abs(s[k]))) for k in range(1, 5))
    assert worst < 1e-9, worst
    st = signature(time_augment(loop), 3)
    assert float(np.max(np.abs(st[2]))) > 1e-3
    print(f"[ok] out-and-back path has trivial signature (max term {worst:.1e}); "
          f"time augmentation makes it visible again")


def _test_reparametrisation_invariance() -> None:
    X = _rand_path(20, 3, 4)
    fine = [X[0]]
    for a, b in zip(X[:-1], X[1:]):
        for t in np.linspace(0, 1, 7)[1:]:
            fine.append(a + t * (b - a))
    fine = np.asarray(fine)
    dup = np.repeat(X, 3, axis=0)
    s0, s1, s2 = signature(X, 4), signature(fine, 4), signature(dup, 4)
    e1 = max(float(np.max(np.abs(a - b))) for a, b in zip(s0, s1))
    e2 = max(float(np.max(np.abs(a - b))) for a, b in zip(s0, s2))
    assert e1 < 1e-9 and e2 < 1e-12, (e1, e2)
    print(f"[ok] reparametrisation invariant: subdividing segments ({e1:.1e}) and "
          f"repeating points ({e2:.1e}) leave the signature unchanged")


def _test_shuffle_identities() -> None:
    """
    Products of signature terms are LINEAR in higher-level terms:

        S^i S^j      = S^{ij} + S^{ji}
        S^i S^{jk}   = S^{ijk} + S^{jik} + S^{jki}

    This is the algebraic reason a linear model on the signature can represent
    polynomial functionals of the path.
    """
    X = _rand_path(35, 3, 5)
    s = signature(X, 3)
    worst = 0.0
    for i, j in itertools.product(range(3), repeat=2):
        lhs = sig_term(s, (i,)) * sig_term(s, (j,))
        rhs = sig_term(s, (i, j)) + sig_term(s, (j, i))
        worst = max(worst, float(np.abs(lhs - rhs).max()))
    for i, j, k in itertools.product(range(3), repeat=3):
        lhs = sig_term(s, (i,)) * sig_term(s, (j, k))
        rhs = sig_term(s, (i, j, k)) + sig_term(s, (j, i, k)) + sig_term(s, (j, k, i))
        worst = max(worst, float(np.abs(lhs - rhs).max()))
    assert worst < 1e-8, worst
    print(f"[ok] shuffle identities hold at levels 1x1 and 1x2 over all 36 words "
          f"(max dev {worst:.1e})")


def _test_factorial_decay() -> None:
    """|S^k|_1 <= L^k / k! with L the l1 length of the path: truncation error
    decays factorially, which is what justifies stopping at small N."""
    X = _rand_path(60, 3, 6) * 0.2
    L = float(np.abs(np.diff(X, axis=0)).sum())
    s = signature(X, 6)
    rows = []
    for k in range(1, 7):
        norm = float(np.abs(s[k]).sum())
        bound = L**k / math.factorial(k)
        assert norm <= bound * (1 + 1e-9), (k, norm, bound)
        rows.append((k, norm, bound))
    print("[ok] |S^k|_1 <= L^k/k!: " + ", ".join(f"k={k}: {n:.2e}<={b:.2e}" for k, n, b in rows))


def _test_expected_signature_of_brownian_motion() -> None:
    """
    The exact expected signature of the PIECEWISE-LINEAR interpolant of
    Brownian motion on n steps of size h = T/n, derived from Chen and Gaussian
    moments:

        E[S^{ii}]   = T/2
        E[S^{iiii}] = T^2 / 8
        E[S^{iijj}] = T^2/8 - T^2/(12 n)       (i != j)
        E[S^{ijij}] = T^2 / (24 n)             (i != j)

    As n -> infinity these converge to the Stratonovich expected signature
    exp( (T/2) sum_i e_i (x) e_i ), where ijij vanishes. At n = 4 the discrete
    corrections are large enough to distinguish -- which verifies that this
    code computes the exact interpolant signature, not an approximation to the
    Stratonovich one.
    """
    rng = np.random.default_rng(7)
    n, T, B = 4, 1.0, 400_000
    h = T / n
    inc = rng.standard_normal((B, n, 2)) * math.sqrt(h)
    X = np.concatenate([np.zeros((B, 1, 2)), np.cumsum(inc, axis=1)], axis=1)
    s = signature(X, 4)
    got = {
        "ii": float(sig_term(s, (0, 0)).mean()),
        "ij": float(sig_term(s, (0, 1)).mean()),
        "iiii": float(sig_term(s, (0, 0, 0, 0)).mean()),
        "iijj": float(sig_term(s, (0, 0, 1, 1)).mean()),
        "ijij": float(sig_term(s, (0, 1, 0, 1)).mean()),
    }
    exact = {"ii": T / 2, "ij": 0.0, "iiii": T**2 / 8,
             "iijj": T**2 / 8 - T**2 / (12 * n), "ijij": T**2 / (24 * n)}
    strat = {"iijj": T**2 / 8, "ijij": 0.0}
    for k, v in exact.items():
        assert abs(got[k] - v) < 0.004, f"{k}: {got[k]:.5f} vs exact {v:.5f}"
    assert abs(got["ijij"] - strat["ijij"]) > 0.006, "cannot distinguish from Stratonovich"
    print(f"[ok] expected signature of piecewise-linear BM (n={n}, {B:,} paths):")
    for k in exact:
        extra = f"   Stratonovich limit {strat[k]:.5f}" if k in strat else ""
        print(f"       E[S^{k:<4}] = {got[k]:+.5f}   exact {exact[k]:+.5f}{extra}")


# -- Levy area and lead-lag ------------------------------------------------

def _test_levy_area_is_signed_area() -> None:
    """Unit square traversed anticlockwise from the origin: area +1/2 against
    the chord... checked against the shoelace formula for random polygons."""
    rng = np.random.default_rng(8)
    worst = 0.0
    for _ in range(50):
        P = rng.standard_normal((12, 2))
        s = signature(P, 2)
        # shoelace for the closed polygon path + chord back to start
        closed = np.vstack([P, P[:1]])
        shoelace = 0.5 * float(np.sum(closed[:-1, 0] * closed[1:, 1]
                                      - closed[1:, 0] * closed[:-1, 1]))
        worst = max(worst, abs(float(levy_area(s, 0, 1)[0]) - shoelace))
    assert worst < 1e-10, worst
    print(f"[ok] Levy area equals the shoelace signed area of path + closing chord "
          f"(50 random polygons, max dev {worst:.1e})")


def _test_order_flips_the_levy_area() -> None:
    """
    The central fact for order flow. Same two moves, opposite order: level 1
    identical, symmetric level 2 identical, Levy area differs by exactly
    dV * dP.
    """
    dV, dP = 1.3, -0.8
    vol_first = np.array([[0.0, 0.0], [dV, 0.0], [dV, dP]])      # (V, P)
    px_first = np.array([[0.0, 0.0], [0.0, dP], [dV, dP]])
    a, b = signature(vol_first, 3), signature(px_first, 3)

    assert np.allclose(a[1], b[1])
    sym_a = sig_term(a, (0, 1)) + sig_term(a, (1, 0))
    sym_b = sig_term(b, (0, 1)) + sig_term(b, (1, 0))
    assert np.allclose(sym_a, sym_b) and np.allclose(sym_a, dV * dP)
    diff = float(levy_area(a, 0, 1)[0] - levy_area(b, 0, 1)[0])
    assert abs(diff - dV * dP) < 1e-12, (diff, dV * dP)
    print(f"[ok] volume-then-price vs price-then-volume: identical level 1 and "
          f"symmetric level 2 ({float(sym_a[0]):+.3f}); Levy areas "
          f"{float(levy_area(a,0,1)[0]):+.3f} vs {float(levy_area(b,0,1)[0]):+.3f}, "
          f"difference {diff:+.4f} = dV*dP")


def _test_leadlag_area_is_half_quadratic_variation() -> None:
    worst = 0.0
    for seed in range(20):
        x = _rand_path(200, 1, 10 + seed).ravel()
        s = signature(lead_lag(x[:, None]), 2)
        qv = float(np.sum(np.diff(x) ** 2))
        worst = max(worst, abs(2 * float(levy_area(s, 0, 1)[0]) - qv))
    assert worst < 1e-8, worst
    print(f"[ok] A(X^lead, X^lag) = QV/2 exactly (20 paths, max dev {worst:.1e})")


def _test_leadlag_splits_covariation_and_area() -> None:
    """
        A(X^lead, Y^lag) + A(Y^lead, X^lag) = sum dX dY      realised covariation
        A(X^lead, Y^lag) - A(Y^lead, X^lag) = 2 A(X, Y)      Levy area of (X, Y)
    """
    worst_c, worst_a = 0.0, 0.0
    for seed in range(20):
        Z = _rand_path(150, 2, 40 + seed)
        s = signature(lead_lag(Z), 2)
        s_raw = signature(Z, 2)
        cov = float(np.sum(np.diff(Z[:, 0]) * np.diff(Z[:, 1])))
        a_xy = float(levy_area(s, 0, 2 + 1)[0])
        a_yx = float(levy_area(s, 1, 2 + 0)[0])
        worst_c = max(worst_c, abs((a_xy + a_yx) - cov))
        worst_a = max(worst_a, abs((a_xy - a_yx) - 2 * float(levy_area(s_raw, 0, 1)[0])))
    assert worst_c < 1e-8 and worst_a < 1e-8, (worst_c, worst_a)
    print(f"[ok] lead-lag level 2 splits exactly: symmetric part = realised "
          f"covariation ({worst_c:.1e}), antisymmetric part = 2 x Levy area "
          f"({worst_a:.1e})")


def _test_log_signature_level2_is_levy_area() -> None:
    X = _rand_path(40, 3, 60)
    s = signature(X, 3)
    ls = tensor_log(s, 3)
    assert np.allclose(ls[1], s[1])
    d = 3
    L2 = ls[2][0].reshape(d, d)
    assert np.allclose(L2, -L2.T, atol=1e-10), "level-2 log-signature must be antisymmetric"
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert abs(L2[i, j] - float(levy_area(s, i, j)[0])) < 1e-10
    print("[ok] log-signature: level 1 = increment, level 2 = Levy area and is "
          "antisymmetric -- the shuffle redundancy is gone")


# -- roughness -------------------------------------------------------------

def _test_fgn_covariance() -> None:
    rng = np.random.default_rng(11)
    for H in (0.1, 0.5, 0.8):
        z = np.stack([fgn_davies_harte(4096, H, rng) for _ in range(200)])
        var = float(z.var())
        acf1 = float(np.mean(z[:, 1:] * z[:, :-1]))
        g1 = 0.5 * (2 ** (2 * H) - 2)
        assert abs(var - 1.0) < 0.02, (H, var)
        assert abs(acf1 - g1) < 0.02, (H, acf1, g1)
    print("[ok] Davies-Harte fGN: unit variance and lag-1 autocovariance "
          "0.5(2^{2H}-2) reproduced for H in {0.1, 0.5, 0.8}")


def _test_hurst_recovery_via_leadlag_area() -> None:
    rng = np.random.default_rng(12)
    rows = []
    for H in (0.1, 0.3, 0.5, 0.7):
        ests = [hurst_from_leadlag(np.cumsum(fgn_davies_harte(2**14, H, rng)))[0]
                for _ in range(6)]
        Hh = float(np.mean(ests))
        assert abs(Hh - H) < 0.04, (H, Hh)
        rows.append((H, Hh, float(np.std(ests))))
    print("[ok] Hurst recovered from lead-lag area scaling: "
          + ", ".join(f"H={h}: {e:.3f}+/-{s:.3f}" for h, e, s in rows))


# -- the order experiment --------------------------------------------------

def _test_order_task_needs_order() -> None:
    """
    Order-blind statistics and level 1 sit at chance: the task is solvable only
    from timing.

    The Levy area -- the single feature the theory identifies as carrying order
    -- reaches ~83%, and a noise sweep shows that is its INFORMATION CEILING,
    not a defect: it separates perfectly at low noise and degrades as the
    background random walk's own area swamps the signal.

    Higher signature levels then recover order information the level-2 area
    cannot: accuracy climbs with depth. With time augmentation, level-3 and
    level-4 terms such as int dt dV dP localise WHEN each move happened, which
    the whole-path area averages away.
    """
    rng = np.random.default_rng(13)
    Xtr, ytr = simulate_order_task(4000, rng=rng)
    Xte, yte = simulate_order_task(2000, rng=np.random.default_rng(99))

    def acc(ftr, fte):
        return float(np.mean(_logistic_fit_predict(ftr, ytr, fte) == yte))

    def area(X):
        return levy_area(signature(X[:, :, :2], 2), 0, 1)[:, None]

    a_ob = acc(order_blind_features(Xtr), order_blind_features(Xte))
    a_s1 = acc(signature_features(Xtr, 1), signature_features(Xte, 1))
    a_ar = acc(area(Xtr), area(Xte))
    ladder = {dp: acc(signature_features(Xtr, dp), signature_features(Xte, dp))
              for dp in (2, 3, 4)}

    assert a_ob < 0.58 and a_s1 < 0.58, (a_ob, a_s1)
    assert a_ar > 0.75, f"area-only {a_ar:.3f}"
    assert ladder[2] < ladder[3] < ladder[4], f"depth ladder not increasing: {ladder}"
    assert ladder[4] > 0.88, ladder

    noise_rows = []
    for nz in (0.10, 0.35, 0.50):
        a, ya = simulate_order_task(3000, rng=np.random.default_rng(13), noise=nz)
        b, yb = simulate_order_task(1500, rng=np.random.default_rng(99), noise=nz)
        noise_rows.append((nz, float(np.mean(
            _logistic_fit_predict(area(a), ya, area(b)) == yb))))
    assert noise_rows[0][1] > 0.98 and noise_rows[0][1] > noise_rows[-1][1] + 0.15

    print(f"[ok] same moves, different order (test n={len(yte)}, linear classifier):")
    print(f"       order-blind statistics                 {100*a_ob:5.1f}%   chance")
    print(f"       signature level 1                      {100*a_s1:5.1f}%   chance")
    print(f"       Levy area of (price, volume) alone     {100*a_ar:5.1f}%")
    for dp, a in ladder.items():
        print(f"       signature levels 1-{dp}, time-augmented   {100*a:5.1f}%")
    print("     Levy-area-only ceiling vs noise: "
          + ", ".join(f"noise {nz:.2f} -> {100*a:.1f}%" for nz, a in noise_rows)
          + " -- 83% is its information limit; higher levels recover what it averages away")


def _test_signature_versus_gru_is_a_crossover() -> None:
    """
    The claim that signatures "capture interactions RNNs miss" is FALSE as
    stated, and this test shows it. A GRU is a universal sequence model; given
    enough data it learns the order rule and OUTPERFORMS a linear classifier on
    a truncated signature, which has a ceiling set by its depth.

    What is true is a crossover in sample size:

      small n   the signature wins decisively. The single Levy-area feature is
                near its ceiling from ~20 examples, where the GRU is at chance.
      large n   the GRU wins. It fits a nonlinear rule the truncated signature
                plus a linear readout cannot represent.

    Two further effects are measured. Richer signatures OVERFIT small samples --
    more features, same 20 labels -- so the one theory-chosen feature beats the
    full depth-3 signature there. And the GRU is unreliable in the transition:
    its accuracy is non-monotone in n, a training failure rather than a
    capacity limit.

    The practical reading for LOB work: labelled regimes are scarce, which is
    exactly where the signature's closed-form order information pays off.
    """
    Xte, yte = simulate_order_task(1500, rng=np.random.default_rng(98))

    def area(X):
        return levy_area(signature(X[:, :, :2], 2), 0, 1)[:, None]

    fa_te, s3_te = area(Xte), signature_features(Xte, 3)
    rows = []
    for n in (20, 200, 3000):
        Xtr, ytr = simulate_order_task(n, rng=np.random.default_rng(200 + n))
        f = lambda a, b: float(np.mean(_logistic_fit_predict(a, ytr, b) == yte))
        a_ar = f(area(Xtr), fa_te)
        a_s3 = f(signature_features(Xtr, 3), s3_te)
        a_gr = gru_accuracy(Xtr, ytr, Xte, yte, epochs=max(25, 5000 // n), seed=1)
        rows.append((n, a_ar, a_s3, a_gr))

    small, _, large = rows
    assert small[1] > 0.75 and small[3] < 0.62, (
        f"n=20: area {small[1]:.3f} should be high, GRU {small[3]:.3f} near chance"
    )
    assert small[1] > small[2], "one theory-chosen feature should beat the full signature at n=20"
    assert large[3] > large[2] + 0.03, (
        f"n=3000: GRU {large[3]:.3f} should beat signature {large[2]:.3f}"
    )
    print("[ok] test accuracy vs training set size (linear readout on signature features):")
    print(f"       {'n_train':>8} {'Levy area (1 feat)':>19} {'signature d3':>13} {'GRU(32)':>9}")
    for n, a, s3, g in rows:
        print(f"       {n:>8} {100*a:>18.1f}% {100*s3:>12.1f}% {100*g:>8.1f}%")
    print("     small n: the signature wins -- the GRU cannot learn order from 20 "
          "labelled paths")
    print("     large n: the GRU wins -- it fits a nonlinear rule a truncated "
          "signature + linear readout cannot")
    print("     'signatures capture what RNNs miss' is a sample-efficiency claim, "
          "not a capability claim")


def _run(title: str, tests) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    for f in tests:
        f()
    print()


# ===========================================================================
# MAIN: TOPOLOGICAL FEATURES OF A GBM PATH
# ===========================================================================

def _gbm_report() -> None:
    """
    What the signature extracts from a (price, volume) path, term by term.

    The pair is simulated as correlated GBM with KNOWN volatilities and
    correlation, so every extracted quantity can be scored against the truth.
    Then a copy of the path is modified so that volume LEADS price by a few
    steps -- same marginal behaviour, same correlation, different timing -- and
    the report shows which signature term moves and which do not.
    """
    print("=" * 78)
    print("MAIN: SIGNATURE FEATURES OF A GEOMETRIC BROWNIAN MOTION PATH")
    print("=" * 78)

    rng = np.random.default_rng(2024)
    n, T = 2000, 1.0
    sig_true = np.array([0.25, 0.60])            # price vol, volume vol
    rho_true = -0.45
    corr = np.array([[1.0, rho_true], [rho_true, 1.0]])
    S = simulate_gbm(n, T, mu=np.array([0.05, 0.0]), sigma=sig_true, corr=corr,
                     s0=np.array([100.0, 1e6]), rng=rng)[0]
    L = np.log(S)                                 # work in log coordinates

    # ------------------------------------------------------------------
    # Step 1. Level 1 of the raw path: the total log change of each channel.
    #         Translation-invariant, blind to everything that happened between
    #         the endpoints.
    # ------------------------------------------------------------------
    s_raw = signature(L, 3)
    print("\n[level 1]  total log change -- the only thing level 1 can see")
    print(f"    price  {float(s_raw[1][0,0]):+.5f}   (check: log S_T/S_0 = "
          f"{float(L[-1,0]-L[0,0]):+.5f})")
    print(f"    volume {float(s_raw[1][0,1]):+.5f}")

    # ------------------------------------------------------------------
    # Step 2. Lead-lag embedding. The raw piecewise-linear path has ZERO
    #         quadratic variation, so volatility is invisible to its
    #         signature. The lead-lag path restores it: the area between each
    #         channel's lead and lag copies is exactly half the realised
    #         variance.
    # ------------------------------------------------------------------
    s_ll = signature(lead_lag(L), 2)
    d = 2
    rv_p = float(realised_covariation_from_leadlag(s_ll, 0, 0, d)[0])
    rv_v = float(realised_covariation_from_leadlag(s_ll, 1, 1, d)[0])
    print("\n[level 2, lead-lag diagonal]  realised variance -> volatility")
    print(f"    price  sigma_hat = {math.sqrt(rv_p/T):.4f}   true {sig_true[0]:.4f}")
    print(f"    volume sigma_hat = {math.sqrt(rv_v/T):.4f}   true {sig_true[1]:.4f}")

    # ------------------------------------------------------------------
    # Step 3. Lead-lag cross terms. The SYMMETRIC combination of the two cross
    #         areas is the realised covariation, which gives the correlation.
    #         It is order-blind: a second-moment statistic.
    # ------------------------------------------------------------------
    rc = float(realised_covariation_from_leadlag(s_ll, 0, 1, d)[0])
    rho_hat = rc / math.sqrt(rv_p * rv_v)
    print("\n[level 2, lead-lag symmetric cross]  realised covariation -> correlation")
    print(f"    rho_hat = {rho_hat:+.4f}   true {rho_true:+.4f}")

    # ------------------------------------------------------------------
    # Step 4. The ANTISYMMETRIC cross term: the Levy area of (price, volume).
    #         This is the timing information. Make volume LEAD price by k
    #         steps -- same marginals, same correlation, only the timing moves.
    #
    #         Theory gives its expectation exactly. With volume leading, only
    #         pairs (s < t) with volume first carry covariance, so
    #
    #             E[ A(P, V) ] = - (1/2) rho sigma_P sigma_V T
    #
    #         -- a clean, signed, first-moment effect. But the Levy area of a
    #         single Brownian path is itself a stochastic integral with O(1)
    #         sampling noise, and here that noise is about twice the signal.
    # ------------------------------------------------------------------
    k = 15

    def shifted(Lp: np.ndarray) -> np.ndarray:
        out = Lp.copy()
        out[:-k, 1] = Lp[k:, 1]
        out[-k:, 1] = Lp[-1, 1]
        return out

    area_sync_1 = float(levy_area(s_raw, 0, 1)[0])
    area_lead_1 = float(levy_area(signature(shifted(L), 2), 0, 1)[0])

    def draws(lead: bool, n_draws: int, seed0: int) -> np.ndarray:
        out = np.empty(n_draws)
        for b in range(n_draws):
            Lb = np.log(simulate_gbm(n, T, mu=np.array([0.05, 0.0]), sigma=sig_true,
                                     corr=corr, s0=np.array([100.0, 1e6]),
                                     rng=np.random.default_rng(seed0 + b))[0])
            out[b] = float(levy_area(signature(shifted(Lb) if lead else Lb, 2), 0, 1)[0])
        return out

    null = draws(False, 300, 5000)          # synchronous: the no-lead distribution
    alt = draws(True, 300, 9000)            # volume leads by k
    theory = -0.5 * rho_true * sig_true[0] * sig_true[1] * T
    shift = float(alt.mean() - null.mean())
    d_eff = shift / float(null.std())
    M_star = int(math.ceil((3.0 / d_eff) ** 2))

    M = 64
    pooled = draws(True, M, 20000)
    z_pool = (float(pooled.mean()) - float(null.mean())) / (float(null.std()) / math.sqrt(M))

    print("\n[level 2, raw antisymmetric]  Levy area of (price, volume) -- WHO MOVED FIRST")
    print(f"    one path, synchronous    area = {area_sync_1:+.5f}")
    print(f"    one path, volume leads   area = {area_lead_1:+.5f}   "
          f"(the shift is lost in single-path noise)")
    print(f"    300 draws each: E[area] sync {null.mean():+.5f}, lead {alt.mean():+.5f}, "
          f"sd {null.std():.5f}")
    print(f"    measured shift {shift:+.5f}   theory -1/2 rho sP sV T = {theory:+.5f}")
    print(f"    per-path effect size d = {d_eff:.3f} sd  ->  3-sigma detection needs "
          f"M* = {M_star} pooled windows")
    print(f"    pooled over M = {M} windows: z = {z_pool:+.2f}")
    print(f"    the timing information is in the Levy area exactly as theory says; "
          f"extracting it is a POOLED statistic, not a per-path one")

    rc_lead = float(realised_covariation_from_leadlag(
        signature(lead_lag(shifted(L)), 2), 0, 1, d)[0])
    print(f"    realised covariation, synchronous vs lead: {rc:+.5f} vs {rc_lead:+.5f}  "
          f"(order-blind: it sees contemporaneous correlation vanish, not which way time ran)")

    # ------------------------------------------------------------------
    # Step 5. Roughness. The same lead-lag area, computed at coarser sampling,
    #         gives the scaling exponent. GBM log-price is Brownian, so H = 1/2.
    # ------------------------------------------------------------------
    H, sc, rv = hurst_from_leadlag(L[:, 0], scales=(1, 2, 4, 8, 16))
    print("\n[roughness]  Hurst exponent from lead-lag area scaling")
    print(f"    GBM log-price  H_hat = {H:.3f}   (Brownian: 0.5)")
    rough = np.cumsum(fgn_davies_harte(2**13, 0.1, np.random.default_rng(1)))
    Hr, _, _ = hurst_from_leadlag(rough)
    print(f"    fBM H=0.1      H_hat = {Hr:.3f}   (rough-volatility regime)")

    assert abs(math.sqrt(rv_p / T) - sig_true[0]) < 0.02
    assert abs(rho_hat - rho_true) < 0.08
    assert abs(shift - theory) < 0.25 * abs(theory), f"shift {shift:.4f} vs theory {theory:.4f}"
    assert 0.3 < d_eff < 0.7, f"per-path effect size {d_eff:.3f}"
    assert z_pool > 3.0, f"pooled z = {z_pool:.2f}"
    assert abs(H - 0.5) < 0.08 and abs(Hr - 0.1) < 0.05
    print("\nall extracted quantities agree with the simulation truth")


if __name__ == "__main__":
    t0 = time.time()
    _run("TENSOR ALGEBRA", [
        _test_level1_is_the_increment,
        _test_single_segment_is_the_tensor_exponential,
        _test_chen_identity,
        _test_retracing_is_invisible,
        _test_reparametrisation_invariance,
        _test_shuffle_identities,
        _test_factorial_decay,
        _test_expected_signature_of_brownian_motion,
    ])
    _run("LEVY AREA AND LEAD-LAG", [
        _test_levy_area_is_signed_area,
        _test_order_flips_the_levy_area,
        _test_leadlag_area_is_half_quadratic_variation,
        _test_leadlag_splits_covariation_and_area,
        _test_log_signature_level2_is_levy_area,
    ])
    _run("ROUGHNESS", [
        _test_fgn_covariance,
        _test_hurst_recovery_via_leadlag_area,
    ])
    _run("ORDER-DEPENDENT INTERACTIONS", [
        _test_order_task_needs_order,
        _test_signature_versus_gru_is_a_crossover,
    ])
    _gbm_report()
    print(f"\nall checks passed in {time.time()-t0:.0f}s")

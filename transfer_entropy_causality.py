"""
transfer_entropy_causality.py
=============================

Phase 24: Directed information flow by transfer entropy.

DEFINITION (Schreiber 2000)
    TE_{X->Y} = sum p( y_{t+1}, y_t^(k), x_s^(l) ) log [ p( y_{t+1} | y_t^(k), x_s^(l) )
                                                        / p( y_{t+1} | y_t^(k) ) ]

with y_t^(k) the last k values of Y and x_s^(l) the l values of X ending at the
source time s = t + 1 - lag. Three equivalent forms, all implemented and
required to agree:

    CONDITIONAL MUTUAL INFORMATION    TE = I( Y_{t+1} ; X_s^(l) | Y_t^(k) )
    ENTROPIES                         TE = H(Y+, Yk) - H(Yk) - H(Y+, Yk, Xl) + H(Yk, Xl)
    EXPECTED KL DIVERGENCE            TE = E_{Yk, Xl}[ D_KL( p(y+ | yk, xl) || p(y+ | yk) ) ]

The last is the one that says what TE means: how far the transition law of Y
moves when X's past is revealed, averaged over the states actually visited.
TE is asymmetric by construction, which Pearson correlation is not.

WHAT TE IS NOT, AND HOW THIS MODULE HANDLES IT
----------------------------------------------
1. NOT CAUSATION. TE is predictive improvement -- Granger causality in
   information-theoretic form, and for Gaussian processes exactly half the
   Granger log-variance ratio (Barnett, Barrett & Seth 2009). A hidden common
   driver Z produces TE_{X->Y} > 0 with no X->Y link at all. The remedy is
   CONDITIONAL transfer entropy, conditioning on Z's past. Demonstrated.

2. THE PLUG-IN ESTIMATOR IS BIASED UPWARD. Each empirical entropy is biased low
   by roughly (occupied cells - 1) / (2N ln 2) bits, and the four biases in TE
   do not cancel: the joint space grows as bins^(k+l+1). Two INDEPENDENT series
   give TE > 0, larger with more bins and deeper embeddings. "TE > 0" is
   therefore not evidence. Significance comes from SURROGATES: circularly
   time-shifted copies of the source, which keep every property of X except its
   alignment with Y.

3. COARSE-GRAINING FAKES REVERSE FLOW. In the coupled logistic maps y_t contains
   eps x_t -- the driver's current state at FULL precision. Once x is binned,
   y's past refines x's present below its bin and so predicts x's next value:
   TE in the WRONG direction comes out significant. A longer target embedding
   does NOT fix this, because the logistic map is expanding -- x's past bins
   spread over its present instead of narrowing it. What SHRINKS it is
   RESOLUTION: from 8 to 32 bins it falls from ~2% to ~0.1% of the forward flow.
   It vanishes only in the continuous limit, so at any finite resolution a
   powerful enough surrogate test can still detect it. Measured.

4. SYNCHRONISATION ERASES IT. Driven hard enough, a chaotic response locks onto
   its driver, becomes a function of it, and TE falls to zero in both
   directions -- at maximal coupling. For the logistic maps below the threshold
   follows from the transverse Lyapunov exponent: ln 2 + ln(1 - eps) < 0, i.e.
   eps > 1/2.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "discretize", "entropy", "transfer_entropy", "transfer_entropy_kl",
    "conditional_transfer_entropy", "te_test", "linear_granger",
    "coupled_logistic", "logistic",
]

LOG2 = math.log(2.0)


# ===========================================================================
# 1. DISCRETISATION AND ENTROPY
# ===========================================================================

def discretize(x: np.ndarray, bins: int = 8, method: str = "quantile") -> np.ndarray:
    """
    Map a series to symbols 0..bins-1.

    'quantile' (default) cuts at empirical quantiles, so every symbol is
    equally likely and the marginal entropy is maximal, log2(bins). This makes
    the estimator invariant to any monotone transform of the data -- TE is a
    property of the dependence, not of the units. 'uniform' uses equal-width
    bins, which leaves sparse cells in heavy-tailed data.
    """
    x = np.asarray(x, dtype=float)
    if method == "quantile":
        edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    elif method == "uniform":
        edges = np.linspace(x.min(), x.max(), bins + 1)[1:-1]
    else:
        raise ValueError(method)
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def entropy(labels: np.ndarray) -> float:
    """Plug-in Shannon entropy in bits of a sequence of integer state labels."""
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def _encode(cols: Sequence[np.ndarray], base: int) -> np.ndarray:
    """Pack several symbol columns into one integer label, base `base`."""
    out = np.zeros(cols[0].size, dtype=np.int64)
    for c in cols:
        out = out * base + c
    return out


def _embed(src: np.ndarray, tgt: np.ndarray, k: int, l: int, lag: int,
           cond: Optional[np.ndarray] = None, m: int = 1, cond_lag: int = 1
           ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Aligned symbol columns for predicting Y_{t+1}:

        future   Y_{t+1}
        past     Y_t .. Y_{t-k+1}
        source   X_s .. X_{s-l+1},   s = t + 1 - lag
        cond     Z_c .. Z_{c-m+1},   c = t + 1 - cond_lag

    lag = 1 means the source is X_t, one step before the predicted value.
    """
    n = tgt.size
    first = max(k - 1, lag + l - 2, (cond_lag + m - 2) if cond is not None else 0)
    t = np.arange(first, n - 1)
    fut = tgt[t + 1]
    past = [tgt[t - i] for i in range(k)]
    source = [src[t + 1 - lag - i] for i in range(l)]
    cnd = [cond[t + 1 - cond_lag - i] for i in range(m)] if cond is not None else []
    return fut, past, source, cnd


# ===========================================================================
# 2. TRANSFER ENTROPY
# ===========================================================================

def transfer_entropy(x_d: np.ndarray, y_d: np.ndarray, k: int = 1, l: int = 1,
                     lag: int = 1, bins: Optional[int] = None) -> float:
    """
    TE_{X->Y} in bits from already-discretised series, via four joint entropies:

        TE = H(Y+, Yk) - H(Yk) - H(Y+, Yk, Xl) + H(Yk, Xl)

    This is the plug-in estimator of I(Y+ ; Xl | Yk). As an average of
    empirical KL divergences it is non-negative for every sample -- and
    therefore biased upward whenever the true value is zero.
    """
    B = int(max(x_d.max(), y_d.max()) + 1) if bins is None else bins
    fut, past, src, _ = _embed(x_d, y_d, k, l, lag)
    yk = _encode(past, B)
    xl = _encode(src, B)
    fut_yk = _encode([fut, yk], B**k)          # fut * B^k + yk, unique since yk < B^k
    return (entropy(fut_yk)
            - entropy(yk)
            - entropy(_encode([fut_yk, xl], B**l))
            + entropy(_encode([yk, xl], B**l)))


def transfer_entropy_kl(x_d: np.ndarray, y_d: np.ndarray, k: int = 1, l: int = 1,
                        lag: int = 1) -> float:
    """
    The same quantity computed the long way round, as requirement 2 asks: build
    the two transition laws p(y+ | yk, xl) and p(y+ | yk) explicitly from counts
    and average their KL divergence over the visited (yk, xl) states.

    Exists to be checked against `transfer_entropy`. The two must agree to
    floating-point precision, because they are the same sum reordered.
    """
    B = int(max(x_d.max(), y_d.max()) + 1)
    fut, past, src, _ = _embed(x_d, y_d, k, l, lag)
    yk = _encode(past, B)
    xl = _encode(src, B)
    state = _encode([yk, xl], B**l)
    n = fut.size

    joint_full: Dict[Tuple[int, int], np.ndarray] = {}
    for s_, f_ in zip(state, fut):
        joint_full.setdefault(int(s_), np.zeros(B))[int(f_)] += 1
    joint_red: Dict[int, np.ndarray] = {}
    for y_, f_ in zip(yk, fut):
        joint_red.setdefault(int(y_), np.zeros(B))[int(f_)] += 1

    te = 0.0
    for s_, cnt in joint_full.items():
        y_state = s_ // (B**l)
        p_full = cnt / cnt.sum()
        red = joint_red[y_state]
        p_red = red / red.sum()
        nz = p_full > 0
        kl = float(np.sum(p_full[nz] * np.log2(p_full[nz] / p_red[nz])))
        te += (cnt.sum() / n) * kl
    return te


def conditional_transfer_entropy(x_d: np.ndarray, y_d: np.ndarray, z_d: np.ndarray,
                                 k: int = 1, l: int = 1, lag: int = 1,
                                 m: int = 1, cond_lag: int = 1) -> float:
    """
    TE_{X->Y | Z} = I( Y+ ; Xl | Yk, Zm ): the information X's past adds about
    Y's next value once BOTH Y's past and Z's past are known. If Z is a common
    driver, conditioning on it removes the spurious flow it induces.
    """
    B = int(max(x_d.max(), y_d.max(), z_d.max()) + 1)
    fut, past, src, cnd = _embed(x_d, y_d, k, l, lag, cond=z_d, m=m, cond_lag=cond_lag)
    ctx = _encode(past + cnd, B)
    xl = _encode(src, B)
    Bc = B ** (k + m)
    return (entropy(_encode([fut, ctx], Bc))
            - entropy(ctx)
            - entropy(_encode([_encode([fut, ctx], Bc), xl], B**l))
            + entropy(_encode([ctx, xl], B**l)))


# ===========================================================================
# 3. SIGNIFICANCE
# ===========================================================================

@dataclass
class TEResult:
    te: float                 # raw plug-in TE, bits
    null_mean: float
    null_sd: float
    p_value: float
    effective: float          # te - null_mean: the bias-corrected flow

    @property
    def z(self) -> float:
        return (self.te - self.null_mean) / self.null_sd if self.null_sd > 0 else float("inf")


def te_test(x: np.ndarray, y: np.ndarray, bins: int = 8, k: int = 1, l: int = 1,
            lag: int = 1, n_surr: int = 100, seed: int = 0,
            z: Optional[np.ndarray] = None, m: int = 1, cond_lag: int = 1) -> TEResult:
    """
    Transfer entropy with a surrogate null.

    SURROGATES ARE CIRCULAR TIME SHIFTS OF THE SOURCE, by a random offset between
    10% and 90% of the series. A shift preserves X's marginal distribution,
    autocorrelation and nonlinear dynamics exactly and destroys only its
    alignment with Y -- which is precisely the hypothesis under test. Random
    shuffling would destroy X's own dynamics too and give an anti-conservative
    null for autocorrelated sources.

    The surrogate mean is also the finite-sample BIAS of the plug-in estimator
    at this (bins, k, l, N): `effective` = te - null_mean is the bias-corrected
    information flow.
    """
    rng = np.random.default_rng(seed)
    xd, yd = discretize(x, bins), discretize(y, bins)
    zd = discretize(z, bins) if z is not None else None

    def te_of(xs):
        if zd is None:
            return transfer_entropy(xs, yd, k, l, lag, bins)
        return conditional_transfer_entropy(xs, yd, zd, k, l, lag, m, cond_lag)

    te = te_of(xd)
    n = xd.size
    null = np.array([te_of(np.roll(xd, int(rng.integers(n // 10, 9 * n // 10))))
                     for _ in range(n_surr)])
    p = (1 + int(np.sum(null >= te))) / (n_surr + 1)
    return TEResult(te, float(null.mean()), float(null.std(ddof=1)), p, te - float(null.mean()))


# ===========================================================================
# 4. LINEAR BASELINES
# ===========================================================================

def linear_granger(x: np.ndarray, y: np.ndarray, p: int = 2) -> Tuple[float, float]:
    """
    Linear Granger causality X -> Y with p lags: F-test of adding X's past to an
    autoregression of Y. Returns (log variance ratio ln(RSS_r/RSS_f), p-value).
    The honest linear competitor to TE -- directional, unlike Pearson, and blind
    to any coupling it cannot write as a linear combination of lags.
    """
    n = y.size
    rows = np.arange(p, n)
    Y = y[rows]
    lag_y = np.column_stack([y[rows - i] for i in range(1, p + 1)])
    lag_x = np.column_stack([x[rows - i] for i in range(1, p + 1)])
    one = np.ones((rows.size, 1))
    Xr = np.hstack([one, lag_y])
    Xf = np.hstack([one, lag_y, lag_x])
    rss_r = float(np.sum((Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]) ** 2))
    rss_f = float(np.sum((Y - Xf @ np.linalg.lstsq(Xf, Y, rcond=None)[0]) ** 2))
    dfn, dfd = p, rows.size - Xf.shape[1]
    F = ((rss_r - rss_f) / dfn) / (rss_f / dfd)
    return math.log(rss_r / rss_f), float(1 - stats.f.cdf(F, dfn, dfd))


# ===========================================================================
# 5. THE COUPLED LOGISTIC SYSTEM
# ===========================================================================

def logistic(z: np.ndarray, r: float = 4.0) -> np.ndarray:
    return r * z * (1.0 - z)


def coupled_logistic(n: int, eps: float, delay: int = 0, r: float = 4.0,
                     noise: float = 1e-5, seed: int = 0, burn: int = 1000
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unidirectionally coupled logistic maps, X drives Y:

        x_{t+1} = f(x_t)
        y_{t+1} = (1 - eps) f(y_t) + eps f(x_{t - delay})

    With r = 4 each map is fully chaotic with Lyapunov exponent ln 2. A small
    dynamical noise keeps the orbit off the measure-zero set of periodic points
    that floating-point iteration of f can collapse onto.

    SYNCHRONISATION. On the manifold y = x (delay 0) a transverse perturbation d
    obeys d_{t+1} = (1 - eps) f'(x_t) d_t, so it grows at the transverse
    Lyapunov exponent lambda_perp = ln 2 + ln(1 - eps). The response locks onto
    the driver when lambda_perp < 0, i.e. eps > 1/2.

    The source lag of the coupling is delay + 1: y_{t+1} depends on x_{t-delay}.
    """
    rng = np.random.default_rng(seed)
    N = n + burn + delay + 1
    x = np.empty(N)
    y = np.empty(N)
    x[0], y[0] = rng.uniform(0.1, 0.9, 2)
    for t in range(delay):
        x[t + 1] = logistic(x[t], r)
        y[t + 1] = logistic(y[t], r)
    for t in range(delay, N - 1):
        x[t + 1] = logistic(x[t], r)
        y[t + 1] = (1 - eps) * logistic(y[t], r) + eps * logistic(x[t - delay], r)
        if noise:
            x[t + 1] = min(max(x[t + 1] + noise * rng.standard_normal(), 1e-9), 1 - 1e-9)
            y[t + 1] = min(max(y[t + 1] + noise * rng.standard_normal(), 1e-9), 1 - 1e-9)
    return x[burn + delay + 1:], y[burn + delay + 1:]


# ===========================================================================
# SELF-TESTS
# ===========================================================================

def _test_entropy_and_te_identities() -> None:
    """
    Entropy of known distributions; the entropy form and the explicit
    KL-of-transition-probabilities form of TE agree exactly; plug-in TE is never
    negative.
    """
    rng = np.random.default_rng(0)
    u = rng.integers(0, 8, 400_000)
    assert abs(entropy(u) - 3.0) < 1e-3
    assert entropy(np.zeros(100, dtype=int)) == 0.0
    worst, neg = 0.0, 0
    for s in range(30):
        r2 = np.random.default_rng(100 + s)
        xd = r2.integers(0, 5, 3000)
        yd = np.clip(np.roll(xd, 1) + r2.integers(-1, 2, 3000), 0, 4)
        for k, l, lag in ((1, 1, 1), (2, 1, 1), (1, 2, 2)):
            a = transfer_entropy(xd, yd, k, l, lag, 5)
            b = transfer_entropy_kl(xd, yd, k, l, lag)
            worst = max(worst, abs(a - b))
            neg += int(a < -1e-12)
    assert worst < 1e-10 and neg == 0, (worst, neg)
    print(f"[ok] H(uniform on 8) = {entropy(u):.4f} bits; entropy form and explicit "
          f"KL-of-transition-laws form of TE agree to {worst:.1e} over 90 cases; plug-in "
          f"TE never negative")


def _test_plugin_bias_on_independent_series() -> None:
    """
    Two INDEPENDENT series. True TE is zero; the plug-in estimate is not, and it
    grows with bins and embedding depth. The surrogate mean tracks that bias,
    and the surrogate test does not reject.
    """
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    rows = []
    for bins, k in ((4, 1), (8, 1), (8, 2), (16, 2)):
        r = te_test(x, y, bins=bins, k=k, n_surr=60, seed=2)
        cells = bins ** (k + 2)
        mm = (cells - bins ** (k + 1) - bins ** k * bins + bins ** k) / (2 * n * LOG2)
        rows.append((bins, k, r.te, r.null_mean, r.p_value, cells, mm))
    assert rows[-1][2] > 10 * rows[0][2], "bias should grow with the state space"
    assert all(abs(te - nm) < 5 * max(nm, 1e-3) for _, _, te, nm, _, _, _ in rows)
    print(f"[ok] independent series, n = {n:,}: the plug-in TE is not zero")
    print(f"       {'bins':>5} {'k':>3} {'cells':>7} {'raw TE':>9} {'surrogate mean':>15} "
          f"{'p-value':>8}")
    for b, k, te, nm, p, c, _ in rows:
        print(f"       {b:>5} {k:>3} {c:>7,} {te:>9.4f} {nm:>15.4f} {p:>8.3f}")
    print("     raw TE grows with the state space; the surrogate mean is that bias, and "
          "the test does not reject")


def _test_nonlinear_coupling_invisible_to_linear_methods() -> None:
    """
    Y is driven by X through an EVEN function of X's lagged value. With X
    symmetric about zero, every linear statistic -- Pearson correlation at every
    lag, linear Granger causality -- is identically zero in expectation. The
    coupling is strong. TE finds it; the linear methods cannot.
    """
    rng = np.random.default_rng(3)
    n, lag = 20_000, 3
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(lag, n):
        y[t] = 0.4 * y[t - 1] + 0.8 * (x[t - lag] ** 2 - 1.0) + 0.5 * rng.standard_normal()
    pear = max(abs(float(np.corrcoef(x[:n - j], y[j:])[0, 1])) for j in range(0, 8))
    gc, gc_p = linear_granger(x, y, p=5)
    r = te_test(x, y, bins=6, k=1, lag=lag, n_surr=100, seed=4)
    rev = te_test(y, x, bins=6, k=1, lag=lag, n_surr=100, seed=5)
    assert pear < 0.03 and gc_p > 0.01, (pear, gc_p)
    assert r.p_value < 0.02 and rev.p_value > 0.05, (r.p_value, rev.p_value)
    print(f"[ok] y_t = 0.4 y_(t-1) + 0.8 (x_(t-{lag})^2 - 1) + noise, n = {n:,}:")
    print(f"       max |Pearson corr(x_t, y_t+j)|, j = 0..7   {pear:.4f}")
    print(f"       linear Granger X->Y (5 lags)             p = {gc_p:.3f}   (not detected)")
    print(f"       transfer entropy X->Y                    {r.effective:.4f} bits, "
          f"z = {r.z:.1f}, p = {r.p_value:.3f}")
    print(f"       transfer entropy Y->X                    {rev.effective:+.4f} bits, "
          f"p = {rev.p_value:.3f}")


def _lagged_even_system(n: int, lag: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """White-noise driver, even coupling at a single lag: y_{t+1} depends on
    x_{t+1-lag} and on nothing else of X."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(lag, n):
        y[t] = 0.4 * y[t - 1] + 0.8 * (x[t - lag] ** 2 - 1.0) + 0.5 * rng.standard_normal()
    return x, y


def _test_lag_is_identified() -> None:
    """
    Lag identification needs an UNAMBIGUOUS lag, which a deterministic driver
    does not have.

    With a white-noise driver, X at any lag other than the coupling lag carries
    nothing about Y's next value, so TE by source lag is a single spike.

    In the coupled logistic maps the coupling term eps f(x_{t-3}) IS the value
    x_{t-2}, because the driver obeys x_{t-2} = f(x_{t-3}) exactly. The term is
    most sharply represented by x_{t-2} -- recovering it from a binned x_{t-3}
    means pushing the bin through f, which stretches it by up to 4 -- so TE
    peaks at source lag 3, and neighbouring lags are elevated because a
    deterministic driver's adjacent values are functions of one another.
    """
    x, y = _lagged_even_system(20_000, 3, seed=6)
    clean = {L: te_test(x, y, bins=6, k=1, lag=L, n_surr=30, seed=7).effective
             for L in range(1, 8)}
    best = max(clean, key=clean.get)
    others = max(v for L, v in clean.items() if L != best)
    assert best == 3 and others < 0.05 * clean[best], clean

    xl, yl = coupled_logistic(20_000, eps=0.3, delay=3, seed=6)
    logi = {L: te_test(xl, yl, bins=8, k=1, lag=L, n_surr=30, seed=7).effective
            for L in range(1, 8)}
    lbest = max(logi, key=logi.get)
    assert lbest == 3, logi
    print("[ok] bias-corrected TE by source lag, coupling at source lag 3:")
    print("       white-noise driver   " + "  ".join(f"{L}:{v:.3f}" for L, v in clean.items())
          + f"   -> spike at {best}, others < {100*others/clean[best]:.1f}% of it")
    print("       logistic driver      " + "  ".join(f"{L}:{v:.3f}" for L, v in logi.items())
          + f"   -> peak at {lbest}, smeared")
    print("     in the logistic system eps f(x_(t-3)) is exactly x_(t-2): the lag of a "
          "deterministic")
    print("     driver is not unique, and TE reports the one where the coupling is least "
          "coarse-grained")


def _test_coarse_graining_fakes_reverse_flow() -> None:
    """
    X drives Y; nothing drives X. Yet the reverse TE Y->X is significant at
    coarse resolution, because y_t carries eps x_t at full precision and so
    refines x's binned state. Deepening X's embedding does not help -- the map
    is expanding, so x's past does not narrow down its present. Finer bins do.
    """
    x, y = coupled_logistic(40_000, eps=0.2, seed=8)
    rows = []
    for bins, k in ((8, 1), (8, 3), (16, 1), (32, 1)):
        f = te_test(x, y, bins=bins, k=k, n_surr=40, seed=9)
        b = te_test(y, x, bins=bins, k=k, n_surr=40, seed=10)
        rows.append((bins, k, f.effective, f.p_value, b.effective, b.p_value))
    by = {(b_, k_): (fe, fp, be, bp) for b_, k_, fe, fp, be, bp in rows}
    ratio = {key: max(v[2], 0.0) / v[0] for key, v in by.items()}
    assert by[(8, 1)][3] < 0.05, "expected spurious reverse flow at 8 bins"
    assert by[(8, 3)][3] < 0.05, "a deeper embedding should NOT remove it"
    assert ratio[(32, 1)] < 0.2 * ratio[(8, 1)], "finer bins should shrink it"
    print("[ok] X drives Y, nothing drives X; effective TE (bits):")
    print(f"       {'bins':>5} {'k':>3} {'X->Y':>9} {'p':>7} {'Y->X':>9} {'p':>7}")
    for b_, k_, fe, fp, be, bp in rows:
        flag = "   spurious" if bp < 0.05 else ""
        print(f"       {b_:>5} {k_:>3} {fe:>9.4f} {fp:>7.3f} {be:>9.4f} {bp:>7.3f}{flag}")
    print(f"     reverse/forward: {100*ratio[(8,1)]:.2f}% at 8 bins, {100*ratio[(8,3)]:.2f}% at "
          f"8 bins k=3, {100*ratio[(32,1)]:.3f}% at 32 bins.")
    print("     a deeper embedding leaves the artefact; finer bins shrink it. It vanishes only "
          "in the continuous limit")


def _test_synchronisation_erases_transfer_entropy() -> None:
    """
    TE is not monotone in coupling. It rises with eps, then collapses as the
    response synchronises to the driver at eps > 1/2 -- maximal coupling, zero
    measured flow.
    """
    rows = []
    for eps in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.9):
        x, y = coupled_logistic(20_000, eps=eps, seed=11)
        r = te_test(x, y, bins=8, k=1, n_surr=30, seed=12)
        sync = float(np.mean(np.abs(x - y)))
        lam = math.log(2) + math.log(1 - eps) if eps < 1 else -np.inf
        rows.append((eps, r.effective, sync, lam))
    peak = max(rows, key=lambda r: r[1])
    after = [r for r in rows if r[0] > 0.5]
    assert peak[0] < 0.5 and all(r[1] < 0.2 * peak[1] for r in after), rows
    assert all(r[2] < 1e-3 for r in after), "expected synchronisation above 1/2"
    print("[ok] TE X->Y against coupling strength (transverse Lyapunov exponent "
          "ln2 + ln(1-eps)):")
    print(f"       {'eps':>5} {'effective TE':>13} {'mean |x - y|':>13} {'lambda_perp':>12}")
    for eps, te, s, lam in rows:
        print(f"       {eps:>5} {te:>13.4f} {s:>13.2e} {lam:>12.3f}")
    print(f"     peak at eps = {peak[0]}; above 1/2 the response is a copy of the driver and "
          f"TE vanishes")


def _test_common_driver_and_conditioning() -> None:
    """
    Z drives X at lag 1 and Y at lag 2; X and Y are not coupled. X's past then
    predicts Y's future -- X carries Z's signal one step ahead of Y -- so
    TE_{X->Y} is significant. Conditioning on Z's past removes it.
    """
    rng = np.random.default_rng(13)
    n = 30_000
    zz = np.zeros(n)
    for t in range(1, n):
        zz[t] = 0.6 * zz[t - 1] + rng.standard_normal()
    x = np.zeros(n)
    y = np.zeros(n)
    x[1:] = np.tanh(zz[:-1]) + 0.3 * rng.standard_normal(n - 1)
    y[2:] = np.tanh(zz[:-2]) + 0.3 * rng.standard_normal(n - 2)
    plain = te_test(x, y, bins=6, k=1, n_surr=60, seed=14)
    cond = te_test(x, y, bins=6, k=1, n_surr=60, seed=15, z=zz, m=1, cond_lag=2)
    assert plain.p_value < 0.02 and cond.p_value > 0.05, (plain.p_value, cond.p_value)
    print(f"[ok] common driver Z -> X (lag 1), Z -> Y (lag 2), no X-Y link:")
    print(f"       TE X->Y                 {plain.effective:.4f} bits, p = {plain.p_value:.3f}"
          f"   SPURIOUS")
    print(f"       TE X->Y | Z_(t-1)        {cond.effective:+.4f} bits, p = {cond.p_value:.3f}")


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

def _main(path: str = "D:/Quant_Project/transfer_entropy.png") -> None:
    print("=" * 78)
    print("MAIN: COUPLED LOGISTIC MAPS, X DRIVES Y")
    print("=" * 78)
    eps, n = 0.2, 40_000
    x, y = coupled_logistic(n, eps=eps, seed=2024)
    print(f"  x_(t+1) = f(x_t),  y_(t+1) = {1-eps:.1f} f(y_t) + {eps:.1f} f(x_t),  "
          f"f(z) = 4 z (1 - z),  n = {n:,}\n")

    print("  linear view")
    c0 = float(np.corrcoef(x, y)[0, 1])
    cxy = float(np.corrcoef(x[:-1], y[1:])[0, 1])
    cyx = float(np.corrcoef(y[:-1], x[1:])[0, 1])
    g_xy, p_xy = linear_granger(x, y, p=3)
    g_yx, p_yx = linear_granger(y, x, p=3)
    print(f"    Pearson corr(x_t, y_t)         {c0:+.4f}   symmetric: says nothing about direction")
    print(f"    Pearson corr(x_t, y_t+1)       {cxy:+.4f}")
    print(f"    Pearson corr(y_t, x_t+1)       {cyx:+.4f}")
    print(f"    linear Granger X->Y            p = {p_xy:.3g}")
    print(f"    linear Granger Y->X            p = {p_yx:.3g}")
    print(f"    the coupling TERM f(x_t) is uncorrelated with x_t -- f is symmetric about 1/2 --")
    print(f"    but lagged correlation is not zero, because y_t already contains 0.2 x_t. And")
    print(f"    linear Granger gets the direction RIGHT in this system. TE's decisive edge over")
    print(f"    linear methods is purely nonlinear coupling, where Granger misses it entirely")
    print(f"    (see the even-coupling self-test); here the two agree on direction.\n")

    k, B = 1, 32
    fwd = te_test(x, y, bins=B, k=k, n_surr=200, seed=1)
    rev = te_test(y, x, bins=B, k=k, n_surr=200, seed=2)
    print(f"  transfer entropy ({B} quantile bins, target embedding k = {k}, 200 time-shift "
          f"surrogates)")
    print(f"    {'':8}{'raw TE':>9}{'null mean':>11}{'effective':>11}{'z':>8}{'p-value':>9}")
    for lab, r in (("X -> Y", fwd), ("Y -> X", rev)):
        print(f"    {lab:8}{r.te:>9.4f}{r.null_mean:>11.4f}{r.effective:>11.4f}"
              f"{r.z:>8.1f}{r.p_value:>9.3f}")
    net = fwd.effective - rev.effective
    print(f"\n    net flow X->Y minus Y->X: {net:+.4f} bits per step")
    c8 = te_test(y, x, bins=8, k=1, n_surr=200, seed=3)
    share = rev.effective / fwd.effective
    print(f"    Y->X at 32 bins is {100*share:.2f}% of the forward flow. It is the residual")
    print(f"    coarse-graining artefact, not coupling: at 8 bins it is {c8.effective:.4f} bits "
          f"(p = {c8.p_value:.3f}),")
    print(f"    and it falls as resolution rises. Judged on significance alone, both directions")
    print(f"    'pass' at this sample size; judged on magnitude, the flow is one-way.")
    one_way = fwd.p_value < 0.01 and fwd.effective > 100 * max(rev.effective, 0.0)
    verdict = ("X drives Y (forward flow over 100x the reverse)" if one_way
               else "direction NOT cleanly identified")
    print(f"    verdict: {verdict}")
    assert one_way and net > 0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        ax[0].scatter(x[:-1][:4000], y[1:][:4000], s=1, alpha=0.3, color="C0")
        ax[0].set_xlabel("x_t")
        ax[0].set_ylabel("y_(t+1)")
        ax[0].set_title(f"lagged scatter: Pearson {cxy:+.3f}")
        eps_grid = np.linspace(0.0, 0.9, 19)
        fte, rte = [], []
        for e in eps_grid:
            xx, yy = coupled_logistic(15_000, eps=float(e), seed=31)
            fte.append(te_test(xx, yy, bins=8, k=1, n_surr=15, seed=1).effective)
            rte.append(te_test(yy, xx, bins=8, k=1, n_surr=15, seed=2).effective)
        ax[1].plot(eps_grid, fte, "o-", color="C3", label="TE X->Y")
        ax[1].plot(eps_grid, rte, "s-", color="C0", label="TE Y->X")
        ax[1].axvline(0.5, color="0.5", ls="--", lw=1, label="sync threshold eps = 1/2")
        ax[1].set_xlabel("coupling eps")
        ax[1].set_ylabel("effective TE (bits)")
        ax[1].set_title("TE rises, then synchronisation erases it")
        ax[1].legend(fontsize=8)
        xs, ys = _lagged_even_system(20_000, 3, seed=6)
        lags = list(range(1, 9))
        tel = [te_test(xs, ys, bins=6, k=1, lag=L, n_surr=15, seed=7).effective for L in lags]
        ax[2].bar(lags, tel, color="C2")
        ax[2].set_xlabel("source lag")
        ax[2].set_ylabel("effective TE (bits)")
        ax[2].set_title("even coupling at lag 3, white-noise driver")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"\n  figure written to {path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n  figure skipped: {exc}")


if __name__ == "__main__":
    t0 = time.time()
    _run("ESTIMATOR", [
        _test_entropy_and_te_identities,
        _test_plugin_bias_on_independent_series,
    ])
    _run("WHAT TE DETECTS", [
        _test_nonlinear_coupling_invisible_to_linear_methods,
        _test_lag_is_identified,
    ])
    _run("WHERE TE MISLEADS", [
        _test_coarse_graining_fakes_reverse_flow,
        _test_synchronisation_erases_transfer_entropy,
        _test_common_driver_and_conditioning,
    ])
    _main()
    print(f"\ncompleted in {time.time()-t0:.0f}s")

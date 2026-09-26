"""
extreme_value_theory_risk.py
============================

Phase 23: Tail risk by Extreme Value Theory -- Peaks-over-Threshold with a
Generalized Pareto tail, fitted by maximum likelihood.

THE PICKANDS-BALKEMA-DE HAAN THEOREM
------------------------------------
For a wide class of loss distributions F, the excesses over a high threshold u,
Y = X - u | X > u, converge in distribution as u grows to the Generalized
Pareto Distribution

    G_{xi,beta}(y) = 1 - (1 + xi y / beta)^{-1/xi},     y >= 0

with shape xi (the tail index) and scale beta. xi > 0 is a power-law tail:
P(X > x) ~ x^{-1/xi}. For Student-t with nu degrees of freedom, xi = 1/nu.

TAIL ESTIMATORS (McNeil & Frey 2000)
------------------------------------
With N_u of n observations above u, and r = (n / N_u)(1 - p):

    VaR_p = u + (beta/xi) ( r^{-xi} - 1 )
    ES_p  = VaR_p / (1 - xi) + (beta - xi u) / (1 - xi)          (xi < 1)

ES is finite only if xi < 1 -- the tail must have a finite mean. At xi >= 1 the
expected shortfall does not exist, and any method that returns a finite number
there is wrong by construction.

WHAT THIS MODULE MEASURES RATHER THAN ASSERTS
---------------------------------------------
Student-t with nu = 2 has INFINITE VARIANCE. Gaussian VaR, sigma_hat * z_p, is
built on a sample standard deviation that does not converge: it grows with the
sample size and is set by the few largest losses. So the Gaussian error is not
simply "underestimates". It is a SHAPE error with a sign that flips: an inflated
sigma_hat over-covers moderate quantiles and a thin Gaussian tail under-covers
extreme ones.

At 99.99% with ten thousand observations there is about one loss beyond the
quantile. Every estimate there is an extrapolation, EVT's included, and the
honest claim for EVT is calibration -- confidence intervals that cover the
truth at their nominal rate -- not precision. Intervals are PROFILE LIKELIHOOD,
not the delta method: the sampling distribution of a far-tail quantile is
strongly right-skewed and a symmetric interval misplaces both ends.

The benchmark that matters is not the Gaussian, which nobody serious uses for
tail risk, but HISTORICAL SIMULATION, which most desks do. It is exact where
there is data and blind beyond it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, optimize, stats

__all__ = [
    "t_var", "t_es", "gpd_nll", "fit_gpd", "POTModel",
    "gaussian_var_es", "historical_var_es", "kupiec_pof", "mean_excess",
]

CHI2_95 = float(stats.chi2.ppf(0.95, 1))


# ===========================================================================
# 1. GROUND TRUTH: STUDENT-t
# ===========================================================================

def t_var(p: float, nu: float, scale: float = 1.0) -> float:
    return scale * float(stats.t.ppf(p, nu))


def t_es(p: float, nu: float, scale: float = 1.0) -> float:
    """
    Expected shortfall of a Student-t, closed form (valid for nu > 1):

        ES_p = f_nu(t_p) / (1 - p) * ( nu + t_p^2 ) / ( nu - 1 )

    with t_p the quantile and f_nu the density. Infinite for nu <= 1.
    """
    if nu <= 1:
        return float("inf")
    q = float(stats.t.ppf(p, nu))
    return scale * float(stats.t.pdf(q, nu)) / (1 - p) * (nu + q * q) / (nu - 1)


# ===========================================================================
# 2. GPD MAXIMUM LIKELIHOOD
# ===========================================================================

def gpd_nll(xi: float, beta: float, y: np.ndarray) -> float:
    """
    Negative log-likelihood of exceedances y under GPD(xi, beta):

        N log beta + (1 + 1/xi) sum log(1 + xi y / beta)        (xi != 0)
        N log beta + sum y / beta                               (xi  = 0)

    The xi -> 0 limit is taken smoothly through log1p(xi z)/xi -> z, so the
    likelihood has no discontinuity at the exponential boundary. Returns +inf
    outside the support 1 + xi y / beta > 0.
    """
    if beta <= 0:
        return float("inf")
    z = y / beta
    a = xi * z
    if np.any(a <= -1.0):
        return float("inf")
    l1p = np.log1p(a)
    if abs(xi) < 1e-8:
        s = float(np.sum(z - 0.5 * xi * z * z))
    else:
        s = float(np.sum(l1p)) / xi
    return y.size * math.log(beta) + float(np.sum(l1p)) + s


def _pwm_start(y: np.ndarray) -> Tuple[float, float]:
    """
    Probability-weighted-moment estimates (Hosking & Wallis 1987): closed form,
    consistent for xi < 1, and a starting point inside the likelihood's basin.
    """
    ys = np.sort(y)
    n = ys.size
    a0 = float(ys.mean())
    a1 = float(np.mean(ys * (n - np.arange(1, n + 1)) / (n - 1)))
    xi = 2.0 - a0 / (a0 - 2.0 * a1)
    beta = 2.0 * a0 * a1 / (a0 - 2.0 * a1)
    if not np.isfinite(xi) or beta <= 0:
        return 0.1, float(y.mean())
    return float(np.clip(xi, -0.45, 1.2)), float(beta)


@dataclass
class GPDFit:
    xi: float
    beta: float
    nll: float
    n: int
    cov: np.ndarray            # observed-information covariance of (xi, beta)

    @property
    def se(self) -> Tuple[float, float]:
        return float(math.sqrt(max(self.cov[0, 0], 0))), float(math.sqrt(max(self.cov[1, 1], 0)))


def fit_gpd(y: np.ndarray) -> GPDFit:
    """
    MLE over (xi, log beta), started from PWM, Nelder-Mead then L-BFGS-B polish.
    Regular -- asymptotically normal and efficient -- for xi > -1/2 (Smith 1985),
    which covers every financial loss tail.

    Covariance from the observed information: a finite-difference Hessian of the
    NLL in (xi, beta) at the optimum.
    """
    y = np.asarray(y, dtype=float)
    xi0, b0 = _pwm_start(y)
    f = lambda th: gpd_nll(th[0], math.exp(th[1]), y)
    r = optimize.minimize(f, [xi0, math.log(b0)], method="Nelder-Mead",
                          options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
    r = optimize.minimize(f, r.x, method="L-BFGS-B")
    xi, beta = float(r.x[0]), float(math.exp(r.x[1]))

    g = lambda a, b: gpd_nll(a, b, y)
    hx, hb = 1e-4, 1e-4 * beta
    H = np.empty((2, 2))
    H[0, 0] = (g(xi + hx, beta) - 2 * g(xi, beta) + g(xi - hx, beta)) / hx**2
    H[1, 1] = (g(xi, beta + hb) - 2 * g(xi, beta) + g(xi, beta - hb)) / hb**2
    H[0, 1] = H[1, 0] = (g(xi + hx, beta + hb) - g(xi + hx, beta - hb)
                         - g(xi - hx, beta + hb) + g(xi - hx, beta - hb)) / (4 * hx * hb)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    return GPDFit(xi, beta, float(r.fun), y.size, cov)


# ===========================================================================
# 3. THE PEAKS-OVER-THRESHOLD MODEL
# ===========================================================================

def _h(xi: float, r: float, measure: str) -> float:
    """
    Both tail measures are affine in beta for fixed xi:

        VaR_p = u + beta h_V(xi),   h_V = ( r^{-xi} - 1 ) / xi
        ES_p  = u + beta h_E(xi),   h_E = ( h_V + 1 ) / ( 1 - xi )

    which is what makes profiling them one-dimensional: fix the target, solve
    for beta, and maximise the likelihood over xi alone.
    """
    hv = -math.log(r) if abs(xi) < 1e-9 else math.expm1(-xi * math.log(r)) / xi
    if measure == "var":
        return hv
    return (hv + 1.0) / (1.0 - xi) if xi < 1.0 else float("inf")


class POTModel:
    """
    Peaks-over-threshold tail model on a loss sample.

    threshold_q sets u as an empirical quantile of the losses. That choice is a
    bias-variance trade-off with no free lunch: too low and the GPD is fitted to
    the body of the distribution, where the limit theorem does not yet hold;
    too high and there are too few exceedances to estimate anything. See
    `mean_excess` and the threshold-stability self-test.
    """

    def __init__(self, losses: np.ndarray, threshold_q: float = 0.95) -> None:
        self.x = np.asarray(losses, dtype=float)
        self.n = self.x.size
        self.u = float(np.quantile(self.x, threshold_q))
        self.y = self.x[self.x > self.u] - self.u
        self.Nu = self.y.size
        if self.Nu < 30:
            raise ValueError(f"only {self.Nu} exceedances; lower the threshold")
        self.fit = fit_gpd(self.y)

    def _r(self, p: float) -> float:
        r = (self.n / self.Nu) * (1.0 - p)
        if r >= 1.0:
            raise ValueError(f"p = {p} is below the threshold quantile; POT does not apply")
        return r

    def var(self, p: float) -> float:
        return self.u + self.fit.beta * _h(self.fit.xi, self._r(p), "var")

    def es(self, p: float) -> float:
        """Expected shortfall; +inf when the fitted tail has no finite mean."""
        if self.fit.xi >= 1.0:
            return float("inf")
        return self.u + self.fit.beta * _h(self.fit.xi, self._r(p), "es")

    def profile_ci(self, p: float, measure: str = "var", level: float = 0.95
                   ) -> Tuple[float, float]:
        """
        Profile-likelihood confidence interval for VaR_p or ES_p.

        For a candidate value T, set beta(xi) = (T - u) / h(xi) and maximise the
        likelihood over xi; the interval is { T : 2 (l_max - l_prof(T)) <= chi2_1 }.
        The upper end is +inf when the data cannot exclude an arbitrarily large
        value -- for ES, when they cannot rule out xi >= 1, i.e. an infinite
        expected shortfall.
        """
        crit = float(stats.chi2.ppf(level, 1))
        r = self._r(p)
        lo_xi, hi_xi = (-0.49, 0.999) if measure == "es" else (-0.49, 1.6)
        point = self.var(p) if measure == "var" else self.es(p)
        if not np.isfinite(point):
            return float("nan"), float("inf")
        nll0 = self.fit.nll

        def dev(T: float) -> float:
            if T <= self.u:
                return float("inf")

            def g(xi):
                h = _h(xi, r, measure)
                if not np.isfinite(h) or h <= 0:
                    return float("inf")
                return gpd_nll(xi, (T - self.u) / h, self.y)

            res = optimize.minimize_scalar(g, bounds=(lo_xi, hi_xi), method="bounded",
                                           options={"xatol": 1e-9})
            return 2.0 * (res.fun - nll0) - crit

        lo_T = self.u + 1e-9 * max(point - self.u, 1.0)
        if dev(lo_T) <= 0:
            lower = lo_T
        else:
            lower = optimize.brentq(dev, lo_T, point, xtol=1e-10 * point)
        hi_T = point
        upper = float("inf")
        for _ in range(60):
            hi_T = self.u + 1.6 * (hi_T - self.u)
            if dev(hi_T) > 0:
                upper = optimize.brentq(dev, point, hi_T, xtol=1e-10 * point)
                break
        return float(lower), float(upper)


# ===========================================================================
# 4. BASELINES, BACKTEST, DIAGNOSTICS
# ===========================================================================

def gaussian_var_es(losses: np.ndarray, p: float) -> Tuple[float, float]:
    """mu + sigma z_p and mu + sigma phi(z_p)/(1 - p): the textbook parametric VaR."""
    mu, sd = float(np.mean(losses)), float(np.std(losses, ddof=1))
    z = float(stats.norm.ppf(p))
    return mu + sd * z, mu + sd * float(stats.norm.pdf(z)) / (1 - p)


def historical_var_es(losses: np.ndarray, p: float) -> Tuple[float, float]:
    """
    Empirical quantile and the mean of losses beyond it. Exact where the sample
    reaches, and it cannot see past the sample maximum: at p above 1 - 1/n it
    returns the largest loss observed, however far the true quantile lies.
    """
    x = np.sort(np.asarray(losses, dtype=float))
    v = float(np.quantile(x, p))
    tail = x[x >= v]
    return v, float(tail.mean()) if tail.size else v


def kupiec_pof(exceed: int, n: int, p_tail: float) -> Tuple[float, float]:
    """
    Kupiec proportion-of-failures test: is the exceedance count consistent with
    the nominal tail probability? Returns (LR statistic, p-value), LR ~ chi2_1.
    """
    x = exceed
    pi = x / n
    if x == 0:
        lr = -2.0 * n * math.log(1 - p_tail)
    elif x == n:
        lr = -2.0 * n * math.log(p_tail)
    else:
        lr = -2.0 * ((n - x) * math.log(1 - p_tail) + x * math.log(p_tail)
                     - (n - x) * math.log(1 - pi) - x * math.log(pi))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def mean_excess(losses: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    e(u) = E[X - u | X > u]. For a GPD tail it is LINEAR in u with slope
    xi / (1 - xi): an upward line is the visual signature of a power law, a flat
    one of an exponential tail, and the threshold should sit where it turns
    linear.
    """
    x = np.asarray(losses, dtype=float)
    return np.array([float((x[x > u] - u).mean()) if np.any(x > u) else np.nan
                     for u in thresholds])


# ===========================================================================
# SELF-TESTS
# ===========================================================================

NU = 2.0
LEVELS = (0.99, 0.999, 0.9999)


def _t_losses(n: int, seed: int, nu: float = NU) -> np.ndarray:
    return stats.t.rvs(nu, size=n, random_state=np.random.default_rng(seed))


def _test_t_truth_formulas() -> None:
    """The closed-form Student-t ES against direct numerical integration."""
    worst = 0.0
    for nu in (2.0, 3.0, 5.0):
        for p in (0.99, 0.999, 0.9999):
            q = stats.t.ppf(p, nu)
            num, _ = integrate.quad(lambda x: x * stats.t.pdf(x, nu), q, np.inf,
                                    limit=500, epsabs=1e-13, epsrel=1e-12)
            worst = max(worst, abs(num / (1 - p) - t_es(p, nu)) / t_es(p, nu))
    assert worst < 1e-6, worst
    print(f"[ok] Student-t ES closed form matches numerical integration for "
          f"nu in {{2,3,5}}, p up to 0.9999 (max rel. error {worst:.1e})")


def _test_gpd_mle_recovers_parameters() -> None:
    """On pure GPD data the MLE must recover (xi, beta), agree with scipy, and
    its observed-information standard error must match the replicate spread."""
    xi_t, beta_t, N = 0.5, 1.3, 500
    ests = []
    for s in range(300):
        y = stats.genpareto.rvs(xi_t, scale=beta_t, size=N,
                                random_state=np.random.default_rng(1000 + s))
        f = fit_gpd(y)
        ests.append((f.xi, f.beta, f.se[0]))
    e = np.array(ests)
    y0 = stats.genpareto.rvs(xi_t, scale=beta_t, size=N, random_state=np.random.default_rng(7))
    ours = fit_gpd(y0)
    c, _, sc = stats.genpareto.fit(y0, floc=0.0)
    sd_emp = float(e[:, 0].std())
    se_theory = (1 + xi_t) / math.sqrt(N)
    assert abs(e[:, 0].mean() - xi_t) < 0.02 and abs(e[:, 1].mean() / beta_t - 1) < 0.03
    assert abs(ours.xi - c) < 1e-4 and abs(ours.beta / sc - 1) < 1e-4
    assert abs(sd_emp / se_theory - 1) < 0.15 and abs(e[:, 2].mean() / sd_emp - 1) < 0.15
    print(f"[ok] GPD MLE on 300 samples of {N}: mean xi {e[:,0].mean():.4f} (true {xi_t}), "
          f"mean beta {e[:,1].mean():.4f} (true {beta_t}); matches scipy to "
          f"{abs(ours.xi-c):.1e}")
    print(f"     sd(xi_hat) {sd_emp:.4f} vs asymptotic (1+xi)/sqrt(N) {se_theory:.4f} vs "
          f"mean observed-information SE {e[:,2].mean():.4f}")


def _test_pot_formulas_exact_on_gpd_tail() -> None:
    """If the tail above u IS exactly GPD, the POT VaR and ES formulas with the
    true parameters must reproduce the true quantile and shortfall."""
    xi, beta, u, zeta = 0.4, 2.0, 5.0, 0.05       # P(X > u) = zeta
    worst = 0.0
    for p in (0.99, 0.999, 0.9999):
        r = (1 - p) / zeta
        v = u + beta * _h(xi, r, "var")
        tail_p = 1 - (1 - p) / zeta
        v_true = u + float(stats.genpareto.ppf(tail_p, xi, scale=beta))
        num, _ = integrate.quad(lambda y: (u + y) * stats.genpareto.pdf(y, xi, scale=beta),
                                v - u, np.inf, limit=400)
        es_true = num / (1 - tail_p)
        es_f = u + beta * _h(xi, r, "es")
        worst = max(worst, abs(v / v_true - 1), abs(es_f / es_true - 1))
    assert worst < 1e-7, worst
    print(f"[ok] on an exactly-GPD tail the POT VaR and ES formulas reproduce the true "
          f"quantile and shortfall (max rel. error {worst:.1e})")


def _test_gaussian_error_changes_sign() -> None:
    """
    With infinite variance the sample standard deviation is inflated by the
    extremes. Gaussian VaR therefore OVERSTATES moderate quantiles and
    UNDERSTATES extreme ones: its error is one of shape, not of scale.
    """
    ratios = {p: [] for p in (0.95, 0.99, 0.999, 0.9999)}
    for s in range(200):
        x = _t_losses(10_000, 100 + s)
        for p in ratios:
            ratios[p].append(gaussian_var_es(x, p)[0] / t_var(p, NU))
    med = {p: float(np.median(v)) for p, v in ratios.items()}
    assert med[0.95] > 1.2 and med[0.9999] < 0.3, med
    print("[ok] Gaussian VaR / true VaR, t(nu=2), n = 10,000, median over 200 samples:")
    print("       " + "   ".join(f"{100*p:g}%: {m:.2f}x" for p, m in med.items()))
    print("     over at 95%, under beyond 99%: an inflated sigma_hat on a thin-tailed shape")


def _test_sample_sd_does_not_converge() -> None:
    """Infinite variance: the input to every Gaussian VaR grows with the sample."""
    rows = []
    for n in (1_000, 10_000, 100_000, 1_000_000):
        sds = [float(np.std(_t_losses(n, 500 + s), ddof=1)) for s in range(40 if n < 10**6 else 8)]
        rows.append((n, float(np.median(sds)), float(np.percentile(sds, 90))))
    assert rows[-1][1] > 1.5 * rows[0][1], rows
    print("[ok] sample standard deviation of t(nu=2) does not converge -- it grows with n:")
    print("       " + "   ".join(f"n={n:,}: median {m:.2f}, p90 {h:.2f}" for n, m, h in rows))


def _test_infinite_expected_shortfall_is_flagged() -> None:
    """
    For nu = 1 (Cauchy) the tail has no mean: ES does not exist. POT must say
    so -- an infinite point estimate once xi_hat >= 1, or a profile interval
    unbounded above -- while the Gaussian returns a finite number regardless.
    """
    x = _t_losses(20_000, 9, nu=1.0)
    m = POTModel(x, 0.95)
    es_hat = m.es(0.999)
    lo, hi = m.profile_ci(0.999, "var")
    g_es = gaussian_var_es(x, 0.999)[1]
    assert not np.isfinite(es_hat) or hi == float("inf"), (m.fit.xi, es_hat)
    xi_lo = m.fit.xi - 1.96 * m.fit.se[0]
    print(f"[ok] Cauchy (nu = 1, ES does not exist): xi_hat = {m.fit.xi:.3f} +/- "
          f"{m.fit.se[0]:.3f} (true 1.0); the data cannot exclude xi >= 1 "
          f"(95% lower bound {xi_lo:.3f}), so POT reports ES_99.9% = {es_hat}. The "
          f"Gaussian reports a finite {g_es:.1f}")


def _test_threshold_stability() -> None:
    """
    xi_hat across thresholds: biased where the threshold is too low for the
    limit theorem to hold, noisy where it is too high to leave enough data.
    """
    x = _t_losses(50_000, 21)
    rows = []
    for q in (0.80, 0.90, 0.95, 0.98, 0.99, 0.995):
        m = POTModel(x, q)
        rows.append((q, m.Nu, m.fit.xi, m.fit.se[0]))
    xs = {q: xi for q, _, xi, _ in rows}
    assert abs(xs[0.95] - 0.5) < abs(xs[0.80] - 0.5) + 0.05
    print(f"[ok] threshold stability, t(nu=2), n = 50,000 (true xi = 0.5):")
    print(f"       {'threshold':>10} {'N_u':>6} {'xi_hat':>8} {'se':>7}")
    for q, nu_, xi, se in rows:
        print(f"       {100*q:>9.1f}% {nu_:>6} {xi:>8.4f} {se:>7.4f}")


def _mc_study(n: int = 10_000, reps: int = 150, seed0: int = 3000) -> Dict:
    out = {p: {"evt_v": [], "evt_e": [], "g_v": [], "g_e": [], "h_v": [], "h_e": [],
               "cov_v": 0, "cov_e": 0, "w_v": [], "w_e": [],
               "abs": {"EVT": [], "Gaussian": [], "historical": []}} for p in (0.999, 0.9999)}
    xis = []
    for s in range(reps):
        x = _t_losses(n, seed0 + s)
        m = POTModel(x, 0.95)
        xis.append(m.fit.xi)
        for p, d in out.items():
            tv, te = t_var(p, NU), t_es(p, NU)
            d["evt_v"].append(m.var(p) / tv)
            d["evt_e"].append(m.es(p) / te)
            gv, ge = gaussian_var_es(x, p)
            hv, he = historical_var_es(x, p)
            d["g_v"].append(gv / tv)
            d["g_e"].append(ge / te)
            d["h_v"].append(hv / tv)
            d["h_e"].append(he / te)
            d["abs"]["EVT"].append(m.var(p))
            d["abs"]["Gaussian"].append(gv)
            d["abs"]["historical"].append(hv)
            lv, uv = m.profile_ci(p, "var")
            le, ue = m.profile_ci(p, "es")
            d["cov_v"] += int(lv <= tv <= uv)
            d["cov_e"] += int(le <= te <= ue)
            d["w_v"].append((uv - lv) / tv)
            d["w_e"].append((ue - le) / te if np.isfinite(ue) else np.inf)
    return {"out": out, "xis": np.array(xis), "reps": reps, "n": n}


def _test_monte_carlo_accuracy_and_coverage(study: Dict) -> None:
    """
    Across replicates: median and spread of each estimator relative to the
    truth, and the realised coverage of EVT's 95% profile intervals.
    """
    reps = study["reps"]
    print(f"[ok] {reps} samples of n = {study['n']:,} from t(nu=2); estimate / truth "
          f"(median, [5%, 95%]):")
    print(f"       {'':14}{'EVT (POT-GPD)':>24}{'Gaussian':>22}{'historical':>22}")
    for p, d in study["out"].items():
        for meas in ("v", "e"):
            lab = f"{'VaR' if meas == 'v' else 'ES '} {100*p:g}%"
            cells = []
            for k in (f"evt_{meas}", f"g_{meas}", f"h_{meas}"):
                a = np.asarray(d[k])
                cells.append(f"{np.median(a):6.3f} [{np.percentile(a,5):.2f},{np.percentile(a,95):.2f}]")
            print(f"       {lab:14}{cells[0]:>24}{cells[1]:>22}{cells[2]:>22}")
    print(f"     EVT 95% profile-likelihood interval coverage:")
    for p, d in study["out"].items():
        print(f"       {100*p:g}%:  VaR {100*d['cov_v']/reps:.1f}%  ES {100*d['cov_e']/reps:.1f}%   "
              f"median width  VaR {np.median(d['w_v']):.2f}x  ES {np.median(d['w_e']):.2f}x truth")
    xis = study["xis"]
    se_mean = xis.std() / math.sqrt(xis.size)
    print(f"     xi_hat across samples: mean {xis.mean():.4f}, sd {xis.std():.4f} (true 0.5) --")
    print(f"     biased LOW by {0.5 - xis.mean():.4f}, {(0.5 - xis.mean())/se_mean:.1f} standard "
          f"errors. Second-order bias, not noise: the t density")
    print(f"     is C x^-(nu+1) (1 - nu(nu+1)/(2x^2) + ...), lighter than pure Pareto at any "
          f"finite threshold,")
    print(f"     so the fitted tail is too thin and pulls ES_99.99% about "
          f"{100*(1-np.median(study['out'][0.9999]['evt_e'])):.0f}% low. The intervals still cover")
    print(f"     at nominal because their width dominates the bias.")
    d9 = study["out"][0.999]
    print(f"     at 99.9% historical simulation matches EVT (median {np.median(d9['h_v']):.3f} vs "
          f"{np.median(d9['evt_v']):.3f}): ten losses")
    print(f"     lie beyond it. EVT earns its keep at 99.99%, beyond the data.")
    d = study["out"][0.9999]
    assert abs(np.median(d["evt_v"]) - 1) < 0.2, np.median(d["evt_v"])
    assert np.median(d["g_v"]) < 0.3 and np.median(d["g_e"]) < 0.2
    assert d["cov_v"] / reps > 0.85, d["cov_v"] / reps


def _kupiec_rejection_prob(pi: float, m: int, p_tail: float, alpha: float = 0.05) -> float:
    """P(Kupiec rejects at level alpha) when the true breach probability is pi,
    over a backtest of m days -- exact, by summing the binomial."""
    crit = float(stats.chi2.ppf(1 - alpha, 1))
    xs = np.arange(m + 1)
    rej = np.array([kupiec_pof(int(x), m, p_tail)[0] > crit for x in xs])
    return float(np.sum(stats.binom.pmf(xs, m, pi)[rej]))


def _test_out_of_sample_backtest(study: Dict) -> None:
    """
    Out-of-sample calibration, done without the flaw a naive backtest has.

    A single fit tested on a million fresh losses tests nothing useful: with
    that many points Kupiec rejects any estimate a few percent off, and every
    estimate from 10,000 observations is -- so it tests one sample's luck.
    Instead, each replicate's fitted VaR is scored by its EXACT breach
    probability under the true distribution, and the spread across replicates
    is reported.

    Then the question that decides whether any of this can be checked in
    practice: with a realistic backtest length, can Kupiec even detect the
    Gaussian's failure?
    """
    rows = []
    for p, d in study["out"].items():
        for name, vals in d["abs"].items():
            ratio = stats.t.sf(np.asarray(vals), NU) / (1 - p)
            rows.append((p, name, float(np.median(ratio)), float(np.percentile(ratio, 5)),
                         float(np.percentile(ratio, 95))))
    by = {(p, nm): med for p, nm, med, _, _ in rows}
    assert by[(0.9999, "Gaussian")] > 10 and 0.6 < by[(0.9999, "EVT")] < 1.6
    print("[ok] out-of-sample breach frequency / nominal, exact under the true law, "
          f"{study['reps']} fits of n = {study['n']:,}:")
    print(f"       {'level':>7} {'method':<11}{'median':>9}{'[5%, 95%]':>18}")
    for p, nm, med, lo, hi in rows:
        print(f"       {100*p:>6g}% {nm:<11}{med:>8.2f}x{f'[{lo:.2f}, {hi:.2f}]':>18}")

    pi_g = by[(0.9999, "Gaussian")] * 1e-4
    print(f"     can a real backtest catch the Gaussian's {by[(0.9999,'Gaussian')]:.0f}x excess "
          f"at 99.99%? Kupiec rejection probability:")
    for m in (250, 1000, 2500):
        pr_g = _kupiec_rejection_prob(pi_g, m, 1e-4)
        pr_0 = _kupiec_rejection_prob(1e-4, m, 1e-4)
        print(f"       {m:>5} days: Gaussian rejected {100*pr_g:5.1f}% of the time  "
              f"(a correct VaR: {100*pr_0:.1f}%)  expected breaches {m*pi_g:.2f} vs {m*1e-4:.3f}")
    miss_1y = 1.0 - _kupiec_rejection_prob(pi_g, 250, 1e-4)
    print(f"     a one-year backtest still MISSES a {by[(0.9999,'Gaussian')]:.0f}x understatement "
          f"of 99.99% risk {100*miss_1y:.0f}% of the time --")
    print(f"     with under one expected breach a year, zero breaches is common. Catching it")
    print(f"     reliably takes a decade of data. Extreme VaR is validated by the model, not")
    print(f"     by backtesting it.")
    assert 0.2 < miss_1y < 0.6, miss_1y


def _mc_tests() -> None:
    study = _mc_study()
    _test_monte_carlo_accuracy_and_coverage(study)
    _test_out_of_sample_backtest(study)


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

def _main(path: str = "D:/Quant_Project/evt_tail_risk.png") -> None:
    print("=" * 78)
    print("MAIN: ONE 10,000-DAY SAMPLE OF t(nu=2) LOSSES")
    print("=" * 78)
    x = _t_losses(10_000, 2024)
    m = POTModel(x, 0.95)
    print(f"  threshold u = {m.u:.4f} (95th percentile), {m.Nu} exceedances")
    print(f"  GPD fit: xi = {m.fit.xi:.4f} +/- {m.fit.se[0]:.4f} (true 1/nu = 0.5), "
          f"beta = {m.fit.beta:.4f}")
    print(f"  sample sd = {np.std(x, ddof=1):.3f} -- for a distribution whose variance "
          f"is infinite\n")
    print(f"  {'':12}{'truth':>9}{'EVT':>9}{'EVT 95% profile CI':>22}{'Gaussian':>10}{'historical':>12}")
    for p in LEVELS:
        for meas in ("VaR", "ES"):
            tr = t_var(p, NU) if meas == "VaR" else t_es(p, NU)
            ev = m.var(p) if meas == "VaR" else m.es(p)
            lo, hi = m.profile_ci(p, "var" if meas == "VaR" else "es")
            g = gaussian_var_es(x, p)[0 if meas == "VaR" else 1]
            h = historical_var_es(x, p)[0 if meas == "VaR" else 1]
            ci = f"[{lo:.2f}, {hi:.2f}]"
            print(f"  {meas} {100*p:<7g}{tr:>9.2f}{ev:>9.2f}{ci:>22}{g:>10.2f}{h:>12.2f}")
    p = 0.9999
    print(f"\n  at 99.99% the Gaussian understates VaR {t_var(p,NU)/gaussian_var_es(x,p)[0]:.1f}x "
          f"and ES {t_es(p,NU)/gaussian_var_es(x,p)[1]:.1f}x.")
    print(f"  ES / VaR at 99.99%: truth {t_es(p,NU)/t_var(p,NU):.2f} (-> 1/(1-xi) = 2 in the "
          f"limit), EVT {m.es(p)/m.var(p):.2f},")
    print(f"  Gaussian {gaussian_var_es(x,p)[1]/gaussian_var_es(x,p)[0]:.2f} -- a thin tail has "
          f"almost nothing beyond its quantile")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        xs = np.sort(x)
        emp = 1 - (np.arange(1, xs.size + 1) - 0.5) / xs.size
        grid = np.geomspace(max(m.u, 1.0), 400, 300)
        ax[0].loglog(xs[xs > 1], emp[xs > 1], ".", ms=3, color="0.5", label="empirical")
        ax[0].loglog(grid, stats.t.sf(grid, NU), "k--", lw=1.5, label="true t(2)")
        gp = (m.Nu / m.n) * stats.genpareto.sf(grid - m.u, m.fit.xi, scale=m.fit.beta)
        ax[0].loglog(grid, gp, "C3", lw=2, label="POT-GPD fit")
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        ax[0].loglog(grid, stats.norm.sf(grid, mu, sd), "C0", lw=1.5, label="Gaussian fit")
        for lvl in LEVELS:
            ax[0].axhline(1 - lvl, color="0.85", lw=0.8)
        ax[0].set_ylim(1e-7, 1)
        ax[0].set_xlabel("loss")
        ax[0].set_ylabel("P(L > loss)")
        ax[0].set_title("tail probability: the Gaussian falls off a cliff")
        ax[0].legend(fontsize=8)
        us = np.quantile(x, np.linspace(0.80, 0.995, 40))
        ax[1].plot(us, mean_excess(x, us), "o-", ms=3, color="C3")
        ax[1].set_xlabel("threshold u")
        ax[1].set_ylabel("mean excess e(u)")
        ax[1].set_title("mean excess: linear rise = power-law tail")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"\n  figure written to {path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n  figure skipped: {exc}")


if __name__ == "__main__":
    t0 = time.time()
    _run("GROUND TRUTH AND ESTIMATION", [
        _test_t_truth_formulas,
        _test_gpd_mle_recovers_parameters,
        _test_pot_formulas_exact_on_gpd_tail,
    ])
    _run("WHY THE GAUSSIAN FAILS", [
        _test_sample_sd_does_not_converge,
        _test_gaussian_error_changes_sign,
    ])
    _run("TAIL ESTIMATION", [
        _test_threshold_stability,
        _test_infinite_expected_shortfall_is_flagged,
        _mc_tests,
    ])
    _main()
    print(f"\ncompleted in {time.time()-t0:.0f}s")

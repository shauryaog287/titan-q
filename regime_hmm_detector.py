"""
regime_hmm_detector.py
======================

Phase 11: Regime Detection Engine.

Gaussian HMM over (return, log realised variance, normalised log volume) with
Baum-Welch calibration, Viterbi decoding, transition analytics, and a causal
online filter.

    z_t in {1..K},  P(z_t = j | z_{t-1} = i) = A_ij,  z_1 ~ pi
    x_t | z_t = k   ~  N( mu_k, Sigma_k ),   x_t in R^d

CAUSALITY
---------
Viterbi and the smoothed posterior gamma_t(k) = P(z_t = k | x_1:T) both
condition on the ENTIRE sample, including the future. They are estimation
tools. The live filter conditions on x_1:t only, via the forward recursion,
and its labels differ from the smoothed labels on a material fraction of bars
(`filtered_vs_smoothed_disagreement` measures it on your own data). Backtesting
on Viterbi labels and trading on filtered labels is not the same strategy.

Everything is computed in log space. A 3-state model on 10 years of daily data
underflows float64 in the linear recursion within a few hundred observations.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, IntEnum
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
from scipy.special import logsumexp

_EPS = 1e-300
_LOG_EPS = -700.0

__all__ = [
    "Regime",
    "GaussianHMM",
    "HMMFitResult",
    "canonical_permutation",
    "FeaturePipeline",
    "RegimeModel",
    "TransitionAnalytics",
    "RegimeState",
    "MarketRegimeFilter",
    "select_n_states",
]


# ===========================================================================
# LABELS
# ===========================================================================

class Regime(IntEnum):
    CRASH = 0
    BEAR = 1
    SIDEWAYS = 2
    BULL = 3
    HIGH_VOL = 4
    UNKNOWN = -1


# ===========================================================================
# CORE HMM
# ===========================================================================

@dataclass
class HMMFitResult:
    log_likelihood: float
    n_iter: int
    converged: bool
    aic: float
    bic: float
    n_params: int
    restart_log_likelihoods: List[float] = field(default_factory=list)

    @property
    def restart_spread(self) -> float:
        """
        Spread of terminal log-likelihood across random restarts. Baum-Welch is
        EM: it converges to a local optimum. A wide spread means the objective
        surface is multi-modal and the selected fit is a draw, not a solution.
        """
        if len(self.restart_log_likelihoods) < 2:
            return 0.0
        ll = np.asarray(self.restart_log_likelihoods)
        return float(ll.max() - ll.min())


class GaussianHMM:
    """
    Multivariate Gaussian emissions, full or diagonal covariance, fitted by
    Baum-Welch (EM) with random restarts.
    """

    def __init__(
        self,
        n_states: int,
        covariance_type: Literal["full", "diag"] = "full",
        n_iter: int = 300,
        tol: float = 1e-6,
        covar_prior: float = 1e-6,
        min_covar: float = 1e-8,
        transition_prior: float = 0.0,
        random_state: int = 0,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be at least 2")
        self.K = int(n_states)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.covar_prior = float(covar_prior)
        self.min_covar = float(min_covar)
        self.transition_prior = float(transition_prior)
        self.random_state = int(random_state)

        self.startprob_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covars_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.fit_result_: Optional[HMMFitResult] = None

    # -- emissions --------------------------------------------------------- #

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        assert self.means_ is not None and self.covars_ is not None
        T, d = X.shape
        out = np.empty((T, self.K), dtype=float)

        for k in range(self.K):
            S = self.covars_[k]
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                S = S + (self.min_covar * 10.0) * np.eye(d)
                L = np.linalg.cholesky(S)
            diff = (X - self.means_[k]).T                     # (d, T)
            sol = linalg.solve_triangular(L, diff, lower=True)
            maha = np.sum(sol**2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, k] = -0.5 * (d * math.log(2.0 * math.pi) + log_det + maha)
        return out

    def mahalanobis(self, x: np.ndarray, state: int) -> float:
        assert self.means_ is not None and self.covars_ is not None
        L = np.linalg.cholesky(self.covars_[state])
        sol = linalg.solve_triangular(L, (x - self.means_[state]), lower=True)
        return float(np.sqrt(np.sum(sol**2)))

    def mahalanobis_all(self, x: np.ndarray) -> np.ndarray:
        assert self.means_ is not None
        return np.array(
            [self.mahalanobis(x, k) for k in range(self.K)], dtype=float
        )

    # -- recursions -------------------------------------------------------- #

    def _emission(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        log_b = self._log_emission(X)
        m = log_b.max(axis=1)
        return np.exp(log_b - m[:, None]), m

    def _forward(self, b: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        assert self.startprob_ is not None and self.transmat_ is not None
        T = b.shape[0]
        A = self.transmat_
        alpha = np.empty((T, self.K), dtype=float)
        c = np.empty(T, dtype=float)

        a = self.startprob_ * b[0]
        c[0] = max(a.sum(), _EPS)
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * b[t]
            c[t] = max(a.sum(), _EPS)
            alpha[t] = a / c[t]

        ll = float(np.sum(np.log(c)) + np.sum(m))
        return alpha, c, ll

    def _backward(self, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        assert self.transmat_ is not None
        T = b.shape[0]
        A = self.transmat_
        beta = np.empty((T, self.K), dtype=float)
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def score(self, X: np.ndarray) -> float:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        b, m = self._emission(X)
        return self._forward(b, m)[2]

    def filtered_posterior(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        b, m = self._emission(X)
        return self._forward(b, m)[0]

    def smoothed_posterior(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        b, m = self._emission(X)
        alpha, c, _ = self._forward(b, m)
        beta = self._backward(b, c)
        g = alpha * beta
        return g / np.maximum(g.sum(axis=1, keepdims=True), _EPS)

    def predict_next_state_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.transmat_ is not None
        return self.filtered_posterior(X)[-1] @ self.transmat_

    # -- Viterbi ----------------------------------------------------------- #

    def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        assert self.startprob_ is not None and self.transmat_ is not None
        log_b = self._log_emission(X)
        T = log_b.shape[0]
        log_A = np.log(np.maximum(self.transmat_, _EPS))

        delta = np.empty((T, self.K), dtype=float)
        psi = np.zeros((T, self.K), dtype=np.int64)

        delta[0] = np.log(np.maximum(self.startprob_, _EPS)) + log_b[0]
        for t in range(1, T):
            trans = delta[t - 1][:, None] + log_A          
            psi[t] = np.argmax(trans, axis=0)
            delta[t] = log_b[t] + trans[psi[t], np.arange(self.K)]

        path = np.empty(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(delta[-1, path[-1]])

    def joint_log_prob(self, X: np.ndarray, path: Sequence[int]) -> float:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        assert self.startprob_ is not None and self.transmat_ is not None
        log_b = self._log_emission(X)
        z = np.asarray(path, dtype=int)
        lp = math.log(max(self.startprob_[z[0]], _EPS)) + log_b[0, z[0]]
        for t in range(1, len(z)):
            lp += math.log(max(self.transmat_[z[t - 1], z[t]], _EPS)) + log_b[t, z[t]]
        return float(lp)

    # -- fitting ----------------------------------------------------------- #

    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        T, d = X.shape
        self.n_features_ = d

        idx = [int(rng.integers(T))]
        for _ in range(self.K - 1):
            dist = np.min(
                np.stack([np.sum((X - X[i]) ** 2, axis=1) for i in idx]), axis=0
            )
            tot = dist.sum()
            p = dist / tot if tot > _EPS else np.full(T, 1.0 / T)
            idx.append(int(rng.choice(T, p=p)))
        centres = X[idx]

        for _ in range(25):
            lab = np.argmin(
                np.stack([np.sum((X - c) ** 2, axis=1) for c in centres]), axis=0
            )
            for k in range(self.K):
                m = lab == k
                if m.sum() > d:
                    centres[k] = X[m].mean(axis=0)

        self.means_ = centres.copy()
        self.covars_ = np.empty((self.K, d, d), dtype=float)
        global_cov = np.cov(X.T) + self.min_covar * np.eye(d)
        for k in range(self.K):
            m = lab == k
            if m.sum() > d + 1:
                c = np.cov(X[m].T)
                c = np.atleast_2d(c)
            else:
                c = global_cov
            self.covars_[k] = self._regularise(c, d)

        stay = 0.90
        A = np.full((self.K, self.K), (1.0 - stay) / (self.K - 1))
        np.fill_diagonal(A, stay)
        A += rng.uniform(0.0, 0.02, (self.K, self.K))
        self.transmat_ = A / A.sum(axis=1, keepdims=True)
        self.startprob_ = np.full(self.K, 1.0 / self.K)

    def _regularise(self, c: np.ndarray, d: int) -> np.ndarray:
        c = np.atleast_2d(np.asarray(c, dtype=float))
        if self.covariance_type == "diag":
            c = np.diag(np.diag(c))
        c = 0.5 * (c + c.T) + self.covar_prior * np.eye(d)
        w, V = np.linalg.eigh(c)
        w = np.maximum(w, self.min_covar)
        return (V * w) @ V.T

    def fit(self, X: np.ndarray, n_restarts: int = 8, verbose: bool = False) -> HMMFitResult:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values")
        T, d = X.shape
        if T < 10 * self.K:
            raise ValueError(f"need T >= 10*K = {10*self.K}, got {T}")

        best: Optional[Tuple[float, tuple]] = None
        lls: List[float] = []

        for r in range(max(n_restarts, 1)):
            rng = np.random.default_rng(self.random_state + 1000 * r)
            self._init_params(X, rng)
            ll_prev = -np.inf
            converged = False
            it = 0

            for it in range(1, self.n_iter + 1):
                b, mrow = self._emission(X)
                alpha, c, ll = self._forward(b, mrow)
                beta = self._backward(b, c)

                gamma = alpha * beta
                gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)

                fwd = (b[1:] * beta[1:]) / c[1:, None]
                xi_sum = np.einsum("ti,ij,tj->ij", alpha[:-1], self.transmat_, fwd)

                self.startprob_ = np.maximum(gamma[0], _EPS)
                self.startprob_ /= self.startprob_.sum()

                denom = gamma[:-1].sum(axis=0)[:, None]
                A = (xi_sum + self.transition_prior) / np.maximum(
                    denom + self.transition_prior * self.K, _EPS
                )
                self.transmat_ = A / A.sum(axis=1, keepdims=True)

                w = gamma.sum(axis=0)
                self.means_ = (gamma.T @ X) / np.maximum(w[:, None], _EPS)
                for k in range(self.K):
                    diff = X - self.means_[k]
                    cvk = (gamma[:, k][:, None] * diff).T @ diff / max(w[k], _EPS)
                    self.covars_[k] = self._regularise(cvk, d)

                if ll - ll_prev < self.tol * abs(max(ll_prev, 1.0)) and it > 3:
                    converged = True
                    ll_prev = ll
                    break
                ll_prev = ll

            lls.append(ll_prev)
            if verbose:
                print(f"  restart {r}: ll={ll_prev:.4f} iters={it} conv={converged}")
            if best is None or ll_prev > best[0]:
                best = (
                    ll_prev,
                    (
                        self.startprob_.copy(),
                        self.transmat_.copy(),
                        self.means_.copy(),
                        self.covars_.copy(),
                        it,
                        converged,
                    ),
                )

        assert best is not None
        ll, (sp, A, mu, cv, it, conv) = best
        self.startprob_, self.transmat_, self.means_, self.covars_ = sp, A, mu, cv

        cov_params = self.K * (d if self.covariance_type == "diag" else d * (d + 1) // 2)
        n_params = (self.K - 1) + self.K * (self.K - 1) + self.K * d + cov_params

        self.fit_result_ = HMMFitResult(
            log_likelihood=ll,
            n_iter=it,
            converged=conv,
            aic=2.0 * n_params - 2.0 * ll,
            bic=n_params * math.log(T) - 2.0 * ll,
            n_params=n_params,
            restart_log_likelihoods=lls,
        )
        return self.fit_result_

    # -- generation -------------------------------------------------------- #

    def sample(self, T: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        assert self.startprob_ is not None and self.transmat_ is not None
        assert self.means_ is not None and self.covars_ is not None
        rng = np.random.default_rng(seed)
        d = self.means_.shape[1]

        cum = np.cumsum(self.transmat_, axis=1)
        u = rng.random(T)
        z = np.empty(T, dtype=np.int64)
        z[0] = int(np.searchsorted(np.cumsum(self.startprob_), u[0]))
        for t in range(1, T):
            z[t] = int(np.searchsorted(cum[z[t - 1]], u[t]))

        eps = rng.standard_normal((T, d))
        X = np.empty((T, d), dtype=float)
        for k in range(self.K):
            mk = z == k
            if not mk.any():
                continue
            L = np.linalg.cholesky(self.covars_[k])
            X[mk] = self.means_[k] + eps[mk] @ L.T
        return X, z

    # -- relabelling ------------------------------------------------------- #

    def permute(self, perm: np.ndarray) -> "GaussianHMM":
        assert self.transmat_ is not None
        p = np.asarray(perm, dtype=int)
        self.startprob_ = self.startprob_[p]
        self.transmat_ = self.transmat_[np.ix_(p, p)]
        self.means_ = self.means_[p]
        self.covars_ = self.covars_[p]
        return self


def canonical_permutation(
    means: np.ndarray, covars: np.ndarray, key: Literal["return", "vol"] = "return"
) -> np.ndarray:
    if key == "return":
        return np.argsort(-means[:, 0], kind="stable")
    return np.argsort(np.sqrt(covars[:, 0, 0]), kind="stable")


# ===========================================================================
# FEATURES
# ===========================================================================

@dataclass
class FeaturePipeline:
    rv_window: int = 21
    volume_window: int = 63
    annualisation: int = 252
    _mean: Optional[np.ndarray] = None
    _std: Optional[np.ndarray] = None
    _fitted: bool = False

    feature_names: Tuple[str, ...] = ("return", "log_rv", "rel_log_volume")

    def build(
        self, prices: np.ndarray, volumes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = np.asarray(prices, dtype=float)
        if np.any(p <= 0):
            raise ValueError("prices must be positive")
        r = np.diff(np.log(p))                                   

        T = r.size
        w = self.rv_window
        if T < w + self.volume_window + 5:
            raise ValueError(f"series too short: {T} returns")

        rv = np.full(T, np.nan)
        r2 = r**2
        csum = np.concatenate(([0.0], np.cumsum(r2)))
        rv[w - 1:] = (csum[w:] - csum[:-w + 1 or None][: T - w + 1]) / w
        rv = rv * self.annualisation
        log_rv = np.log(np.maximum(rv, 1e-12))

        if volumes is None:
            rel_v = np.zeros(T)
        else:
            v = np.asarray(volumes, dtype=float)[1:]
            if v.size != T:
                raise ValueError("volumes must align with prices")
            vw = self.volume_window
            rel_v = np.full(T, np.nan)
            lv = np.log(np.maximum(v, 1.0))
            for i in range(vw, T):
                med = np.median(lv[i - vw:i])                    
                rel_v[i] = lv[i] - med

        F = np.column_stack([r, log_rv, rel_v])
        valid = np.flatnonzero(np.all(np.isfinite(F), axis=1))
        return F[valid], valid + 1

    def fit_scaler(self, F: np.ndarray) -> "FeaturePipeline":
        self._mean = F.mean(axis=0)
        self._std = np.maximum(F.std(axis=0, ddof=1), 1e-12)
        self._fitted = True
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("scaler not fitted")
        return (F - self._mean) / self._std

    def fit_transform(self, F: np.ndarray) -> np.ndarray:
        return self.fit_scaler(F).transform(F)

    @property
    def scaler(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("scaler not fitted")
        return self._mean.copy(), self._std.copy()


# ===========================================================================
# TRANSITION ANALYTICS
# ===========================================================================

class TransitionAnalytics:
    def __init__(self, transmat: np.ndarray) -> None:
        A = np.asarray(transmat, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("transmat must be square")
        if not np.allclose(A.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("rows of transmat must sum to 1")
        self.A = A
        self.K = A.shape[0]

    def n_step(self, n: int) -> np.ndarray:
        return np.linalg.matrix_power(self.A, int(n))

    def stationary(self) -> np.ndarray:
        w, V = np.linalg.eig(self.A.T)
        i = int(np.argmin(np.abs(w - 1.0)))
        v = np.real(V[:, i])
        v = np.abs(v)
        return v / v.sum()

    def expected_duration(self) -> np.ndarray:
        return 1.0 / np.maximum(1.0 - np.diag(self.A), 1e-12)

    def first_passage_proba(self, source: int, target: int, n_steps: int) -> float:
        if source == target:
            return 1.0
        Aa = self.A.copy()
        Aa[target] = 0.0
        Aa[target, target] = 1.0
        return float(np.linalg.matrix_power(Aa, int(n_steps))[source, target])

    def first_passage_curve(self, source: int, target: int, max_steps: int) -> np.ndarray:
        Aa = self.A.copy()
        Aa[target] = 0.0
        Aa[target, target] = 1.0
        out = np.empty(max_steps + 1)
        P = np.eye(self.K)
        out[0] = 1.0 if source == target else 0.0
        for n in range(1, max_steps + 1):
            P = P @ Aa
            out[n] = P[source, target]
        return out

    def expected_hitting_time(self, target: int) -> np.ndarray:
        idx = [i for i in range(self.K) if i != target]
        Q = self.A[np.ix_(idx, idx)]
        try:
            h = np.linalg.solve(np.eye(len(idx)) - Q, np.ones(len(idx)))
        except np.linalg.LinAlgError:
            h = np.full(len(idx), np.inf)
        out = np.zeros(self.K)
        for pos, i in enumerate(idx):
            out[i] = h[pos]
        return out

    def report(self, labels: Optional[Sequence[str]] = None) -> str:
        lab = list(labels) if labels else [f"S{i}" for i in range(self.K)]
        wid = max(len(x) for x in lab) + 2
        lines = ["Transition matrix P(row -> col):", ""]
        lines.append(" " * wid + "".join(f"{x:>10}" for x in lab) + "   dwell   stat")
        pi = self.stationary()
        dur = self.expected_duration()
        for i in range(self.K):
            row = "".join(f"{self.A[i, j]:>10.4f}" for j in range(self.K))
            lines.append(f"{lab[i]:<{wid}}{row}{dur[i]:>8.1f}{pi[i]:>7.3f}")
        return "\n".join(lines)


# ===========================================================================
# REGIME MODEL
# ===========================================================================

def select_n_states(
    X: np.ndarray,
    candidates: Sequence[int] = (2, 3, 4, 5),
    covariance_type: Literal["full", "diag"] = "full",
    n_restarts: int = 6,
    random_state: int = 0,
    min_dwell: float = 3.0,
) -> Tuple[int, "Dict[int, Dict[str, float]]"]:
    scores: Dict[int, Dict[str, float]] = {}
    best_k, best_bic = None, np.inf

    for k in candidates:
        try:
            m = GaussianHMM(k, covariance_type=covariance_type, random_state=random_state)
            fr = m.fit(X, n_restarts=n_restarts)
        except Exception as exc:
            warnings.warn(f"K={k} failed: {exc}")
            continue
        dwell = TransitionAnalytics(m.transmat_).expected_duration()
        ok = bool(dwell.min() >= min_dwell)
        scores[k] = {
            "bic": fr.bic, "aic": fr.aic, "log_likelihood": fr.log_likelihood,
            "min_dwell": float(dwell.min()), "n_params": float(fr.n_params),
            "admissible": float(ok), "restart_spread": fr.restart_spread,
        }
        if ok and fr.bic < best_bic:
            best_k, best_bic = k, fr.bic

    if best_k is None:
        if not scores:
            raise RuntimeError("no candidate fitted")
        best_k = min(scores, key=lambda k: scores[k]["bic"])
        warnings.warn("no candidate met the dwell-time guard; falling back to raw BIC")
    return int(best_k), scores


class RegimeModel:
    def __init__(
        self,
        n_states: int = 4,
        covariance_type: Literal["full", "diag"] = "full",
        random_state: int = 0,
        annualisation: int = 252,
    ) -> None:
        self.hmm = GaussianHMM(n_states, covariance_type, random_state=random_state)
        self.n_states = int(n_states)
        self.annualisation = int(annualisation)
        self.labels_: Optional[List[Regime]] = None
        self.pipeline: Optional[FeaturePipeline] = None

    def fit(
        self, X: np.ndarray, n_restarts: int = 8, pipeline: Optional[FeaturePipeline] = None
    ) -> HMMFitResult:
        fr = self.hmm.fit(X, n_restarts=n_restarts)
        perm = canonical_permutation(self.hmm.means_, self.hmm.covars_, key="return")
        self.hmm.permute(perm)
        self.pipeline = pipeline
        self.labels_ = self._assign_labels()
        return fr

    def raw_means(self) -> np.ndarray:
        """State means in the ORIGINAL feature units. The HMM is fitted on
        standardised features, where 0 means "average", not "zero"."""
        mu = self.hmm.means_
        if self.pipeline is not None and getattr(self.pipeline, "_fitted", False):
            m, s = self.pipeline.scaler
            d = mu.shape[1]
            return mu * s[:d] + m[:d]
        return mu

    def _assign_labels(self) -> List[Regime]:
        # Labels are decided on raw means. On standardised features the old
        # test "mean return < 0" meant "below the sample average", which
        # labelled a +7%/yr high-volatility state as CRASH.
        mu = self.raw_means()
        K = self.n_states
        vol_rank = np.argsort(-mu[:, 1])
        stress = int(vol_rank[0])

        out: List[Regime] = [Regime.UNKNOWN] * K
        out[stress] = Regime.CRASH if mu[stress, 0] < 0 else Regime.HIGH_VOL

        rest = [k for k in range(K) if k != stress]
        rest.sort(key=lambda k: -mu[k, 0])

        if len(rest) == 1:
            out[rest[0]] = Regime.BULL
        elif len(rest) == 2:
            out[rest[0]], out[rest[1]] = Regime.BULL, Regime.BEAR
        elif len(rest) >= 3:
            out[rest[0]] = Regime.BULL
            out[rest[-1]] = Regime.BEAR
            for k in rest[1:-1]:
                out[k] = Regime.SIDEWAYS
        return out

    def state_of(self, regime: Regime) -> int:
        assert self.labels_ is not None
        for i, l in enumerate(self.labels_):
            if l is regime:
                return i
        raise KeyError(f"{regime.name} not present in this model")

    def label_names(self) -> List[str]:
        assert self.labels_ is not None
        return [l.name for l in self.labels_]

    def analytics(self) -> TransitionAnalytics:
        return TransitionAnalytics(self.hmm.transmat_)

    def describe(self) -> str:
        assert self.labels_ is not None
        mu, cv = self.hmm.means_, self.hmm.covars_
        rows = ["State      label       mu_ret     mu_logrv   mu_vol    sd_ret"]
        for k in range(self.n_states):
            rows.append(
                f"  {k:<8d} {self.labels_[k].name:<11s} {mu[k,0]:>+9.4f} "
                f"{mu[k,1]:>+10.4f} {mu[k,2]:>+9.4f} {math.sqrt(cv[k,0,0]):>9.4f}"
            )
        return "\n".join(rows)

    def filtered_vs_smoothed_disagreement(self, X: np.ndarray) -> Dict[str, float]:
        f = self.hmm.filtered_posterior(X).argmax(axis=1)
        s = self.hmm.smoothed_posterior(X).argmax(axis=1)
        v, _ = self.hmm.viterbi(X)
        return {
            "filtered_vs_smoothed": float(np.mean(f != s)),
            "filtered_vs_viterbi": float(np.mean(f != v)),
            "smoothed_vs_viterbi": float(np.mean(s != v)),
        }


# ===========================================================================
# LIVE FILTER
# ===========================================================================

@dataclass(frozen=True)
class RegimeState:
    state: int
    regime: Regime
    posterior: np.ndarray
    confidence: float
    entropy: float
    mahalanobis: float
    mahalanobis_min: float
    novel: bool
    raw_state: int
    n_observations: int
    switched: bool

    def __str__(self) -> str:
        flag = "  NOVEL" if self.novel else ""
        return (
            f"{self.regime.name:<9s} p={self.confidence:.3f} "
            f"H={self.entropy:.3f} maha={self.mahalanobis:.2f} "
            f"min={self.mahalanobis_min:.2f}{flag}"
        )


class MarketRegimeFilter:
    def __init__(
        self,
        model: RegimeModel,
        min_dwell: int = 3,
        switch_threshold: float = 0.60,
        novelty_quantile: float = 1e-3,
    ) -> None:
        if model.hmm.transmat_ is None:
            raise ValueError("model is not fitted")
        self.model = model
        self.hmm = model.hmm
        self.K = model.n_states
        self.min_dwell = int(min_dwell)
        self.switch_threshold = float(switch_threshold)
        self.novelty_quantile = float(novelty_quantile)
        d = int(model.hmm.n_features_ or model.hmm.means_.shape[1])
        self.novelty_threshold = float(
            np.sqrt(stats.chi2.ppf(1.0 - self.novelty_quantile, df=d))
        )

        self._log_A = np.log(np.maximum(self.hmm.transmat_, _EPS))
        self._log_pi = np.log(np.maximum(self.hmm.startprob_, _EPS))
        self.reset()

    def reset(self) -> None:
        self._log_alpha: Optional[np.ndarray] = None
        self._n = 0
        self._reported: Optional[int] = None
        self._candidate: Optional[int] = None
        self._candidate_run = 0
        self.history: List[RegimeState] = []

    def _log_emission_one(self, x: np.ndarray) -> np.ndarray:
        return self.hmm._log_emission(x.reshape(1, -1))[0]

    def update(self, x: np.ndarray, store: bool = True) -> RegimeState:
        x = np.asarray(x, dtype=float).ravel()
        if self.hmm.n_features_ is not None and x.size != self.hmm.n_features_:
            raise ValueError(f"expected {self.hmm.n_features_} features, got {x.size}")
        if not np.all(np.isfinite(x)):
            raise ValueError("observation contains non-finite values")

        log_b = self._log_emission_one(x)
        if self._log_alpha is None:
            self._log_alpha = self._log_pi + log_b
        else:
            self._log_alpha = log_b + logsumexp(
                self._log_alpha[:, None] + self._log_A, axis=0
            )

        self._log_alpha = self._log_alpha - self._log_alpha.max()
        self._n += 1

        post = np.exp(self._log_alpha - logsumexp(self._log_alpha))
        raw = int(np.argmax(post))
        conf = float(post[raw])
        ent = float(-np.sum(post * np.log(np.maximum(post, 1e-300))))

        switched = self._apply_hysteresis(raw, conf)
        reported = self._reported if self._reported is not None else raw

        dists = self.hmm.mahalanobis_all(x)
        maha = float(dists[reported])
        maha_min = float(dists.min())
        st = RegimeState(
            state=reported,
            regime=self.model.labels_[reported] if self.model.labels_ else Regime.UNKNOWN,
            posterior=post,
            confidence=conf,
            entropy=ent,
            mahalanobis=maha,
            mahalanobis_min=maha_min,
            novel=bool(maha_min > self.novelty_threshold),
            raw_state=raw,
            n_observations=self._n,
            switched=switched,
        )
        if store:
            self.history.append(st)
        return st

    def _apply_hysteresis(self, raw: int, conf: float) -> bool:
        if self._reported is None:
            self._reported = raw
            self._candidate, self._candidate_run = raw, 1
            return False
        if raw == self._reported:
            self._candidate, self._candidate_run = raw, 0
            return False
        if raw == self._candidate:
            self._candidate_run += 1
        else:
            self._candidate, self._candidate_run = raw, 1
        if self._candidate_run >= self.min_dwell and conf >= self.switch_threshold:
            self._reported = raw
            self._candidate_run = 0
            return True
        return False

    @property
    def posterior(self) -> Optional[np.ndarray]:
        if self._log_alpha is None:
            return None
        return np.exp(self._log_alpha - logsumexp(self._log_alpha))

    def next_state_proba(self) -> Optional[np.ndarray]:
        p = self.posterior
        return None if p is None else p @ self.hmm.transmat_

    def probability_of(self, regime: Regime) -> float:
        p = self.posterior
        if p is None:
            return float("nan")
        return float(p[self.model.state_of(regime)])

    def run_batch(self, X: np.ndarray) -> List[RegimeState]:
        return [self.update(row) for row in np.atleast_2d(X)]


if __name__ == "__main__":
    print("==========================================================================")
    print(" REGIME DETECTION ENGINE: HIDDEN MARKOV MODEL CALIBRATION & VITERBI DECODE")
    print("==========================================================================")
    
    rng = np.random.default_rng(2024)
    n = 4000
    # Simulate a market with 3 Regimes: Bull (0), Bear (1), Crash (2)
    A = np.array([[0.97, 0.025, 0.005],
                  [0.04, 0.93, 0.03],
                  [0.03, 0.17, 0.80]])
    drift = np.array([0.0007, -0.0002, -0.0025])
    vol = np.array([0.008, 0.013, 0.032])
    vol_mult = np.array([1.0, 1.4, 3.0])

    z = np.empty(n, dtype=int)
    z[0] = 0
    r = np.empty(n)
    vlm = np.empty(n)
    for t in range(n):
        if t > 0:
            z[t] = rng.choice(3, p=A[z[t - 1]])
        r[t] = rng.normal(drift[z[t]], vol[z[t]])
        vlm[t] = rng.lognormal(14.0, 0.3) * vol_mult[z[t]] * (1.0 + 1.5 * t / n)

    p = 100 * np.exp(np.cumsum(r))
    fp = FeaturePipeline(rv_window=21, volume_window=63)
    F, idx = fp.build(p, vlm)
    split = int(0.7 * len(F))
    Z = fp.fit_scaler(F[:split]).transform(F)

    print("\n[BOOT] Fitting Baum-Welch Expectation-Maximization (EM)...")
    model = RegimeModel(3, random_state=11)
    model.fit(Z[:split], n_restarts=3, pipeline=fp)
    names = model.label_names()
    
    print("\n[TEST 1] Viterbi Decoding & State Analysis")
    print(model.describe())
    
    print("\n[TEST 2] Transition Probabilities Matrix")
    ta = model.analytics()
    print(ta.report(names))
    
    calm = int(np.argmax(model.hmm.means_[:, 0]))
    stress = [i for i, l in enumerate(model.labels_)
              if l in (Regime.CRASH, Regime.HIGH_VOL)][0]
              
    twenty = ta.first_passage_proba(calm, stress, 20)
    print(f"\n-> Probability of transitioning {names[calm]} -> {names[stress]} within 20 bars: {twenty*100:.2f}%")
    
    print("\n[TEST 3] Out of Sample Causal Hysteresis Filter")
    filt = MarketRegimeFilter(model, min_dwell=3, switch_threshold=0.65)
    states = filt.run_batch(Z[split:])
    switches = int(np.sum(np.diff([s.state for s in states]) != 0))
    print(f"-> Processed {len(states)} live out-of-sample ticks.")
    print(f"-> Regime changes detected dynamically: {switches}")
    
    print("\n[SUCCESS] REGIME DETECTION ENGINE OPERATIONAL.")

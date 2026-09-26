"""
portfolio_hrp_allocator.py
==========================

Phase 12: Hierarchical Risk Parity (Lopez de Prado, 2016).

Pipeline
--------
    corr  ->  d_ij = sqrt( 0.5 (1 - rho_ij) )          metric distance
          ->  D_ij = || d_i - d_j ||_2                  correlation distance
          ->  linkage(squareform(D))                    hierarchical tree
          ->  quasi-diagonal leaf order                 similar assets adjacent
          ->  recursive bisection with inverse-variance allocation

HRP never inverts the covariance matrix. That is the entire point: the
minimum-variance solution w ~ Sigma^-1 1 amplifies the smallest eigenvalues of
Sigma, which are precisely the ones estimated worst, and on a sample covariance
with T not much larger than N those eigenvalues are noise. HRP touches only
diagonal blocks and pairwise cluster variances.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

_EPS = 1e-12

__all__ = [
    "correlation_to_distance",
    "correlation_distance_matrix",
    "build_linkage",
    "mst_edges",
    "single_linkage_equals_mst",
    "quasi_diagonalize",
    "tree_imbalance",
    "inverse_variance_weights",
    "cluster_variance",
    "recursive_bisection",
    "dendrogram_bisection",
    "HRPAllocator",
    "HRPResult",
    "minimum_variance_weights",
    "equal_weight",
    "cov_to_corr",
    "nearest_correlation",
]

LinkageMethod = Literal["single", "complete", "average", "weighted", "ward"]
Bisection = Literal["serial", "dendrogram"]


# ===========================================================================
# 1. DISTANCE
# ===========================================================================

def cov_to_corr(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (corr, stdev)."""
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        raise ValueError("covariance has a non-positive diagonal entry")
    corr = cov / np.outer(sd, sd)
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)
    return corr, sd


def nearest_correlation(corr: np.ndarray, eig_floor: float = 1e-10) -> np.ndarray:
    """Eigenvalue clip plus unit-diagonal rescale. Used only if corr is indefinite."""
    c = 0.5 * (corr + corr.T)
    w, V = np.linalg.eigh(c)
    if w.min() >= eig_floor:
        np.fill_diagonal(c, 1.0)
        return c
    w = np.maximum(w, eig_floor)
    c = (V * w) @ V.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    np.clip(c, -1.0, 1.0, out=c)
    np.fill_diagonal(c, 1.0)
    return c


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("corr must have unit diagonal")
    d = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def correlation_distance_matrix(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    n = d.shape[0]
    sq = np.sum(d**2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (d @ d.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# ===========================================================================
# 2. TREE AND MST
# ===========================================================================

def build_linkage(D: np.ndarray, method: LinkageMethod = "single") -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")
    if D.shape[0] < 2:
        raise ValueError("need at least 2 assets to cluster")
    cond = squareform(D, checks=False)
    return linkage(cond, method=method)


def mst_edges(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    T = minimum_spanning_tree(D)
    coo = T.tocoo()
    edges = np.column_stack([coo.row, coo.col, coo.data]).astype(float)

    if edges.shape[0] != n - 1:
        shift = float(D.max()) + 1.0
        M = D + shift
        np.fill_diagonal(M, 0.0)
        coo = minimum_spanning_tree(M).tocoo()
        edges = np.column_stack([coo.row, coo.col, coo.data - shift]).astype(float)

    if edges.shape[0] != n - 1:
        raise RuntimeError(f"MST has {edges.shape[0]} edges, expected {n-1}")
    return edges[np.argsort(edges[:, 2])]


def single_linkage_equals_mst(
    D: np.ndarray, atol: float = 1e-10
) -> Tuple[bool, np.ndarray, np.ndarray]:
    Z = build_linkage(D, method="single")
    heights = np.sort(Z[:, 2])
    w = np.sort(mst_edges(D)[:, 2])
    return bool(np.allclose(heights, w, atol=atol)), heights, w


# ===========================================================================
# 3. QUASI-DIAGONALIZATION
# ===========================================================================

def quasi_diagonalize(Z: np.ndarray, n_leaves: Optional[int] = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    n = int(n_leaves if n_leaves is not None else Z.shape[0] + 1)
    root = n + Z.shape[0] - 1

    order: List[int] = []
    stack: List[int] = [root]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(int(node))
        else:
            row = Z[node - n]
            stack.append(int(row[1]))   
            stack.append(int(row[0]))   
    if len(order) != n:
        raise RuntimeError(f"traversal produced {len(order)} leaves, expected {n}")
    return np.asarray(order, dtype=int)


def quasi_diagonal_concentration(corr: np.ndarray, order: Sequence[int]) -> float:
    P = np.asarray(corr, dtype=float)[np.ix_(order, order)]
    n = P.shape[0]
    idx = np.arange(n)
    w = 1.0 / (1.0 + np.abs(idx[:, None] - idx[None, :]))
    return float(np.sum(np.abs(P) * w) / np.sum(w))


# ===========================================================================
# 4. ALLOCATION
# ===========================================================================

def tree_imbalance(Z: np.ndarray) -> float:
    Z = np.asarray(Z, dtype=float)
    if Z.shape[0] < 2:
        return 0.0
    n = Z.shape[0] + 1
    size = np.ones(n + Z.shape[0], dtype=float)
    total = 0.0
    for r in range(Z.shape[0]):
        a, b = int(Z[r, 0]), int(Z[r, 1])
        size[n + r] = size[a] + size[b]
        total += abs(size[a] - size[b])
    denom = (n - 1) * (n - 2) / 2.0
    return float(total / denom) if denom > 0 else 0.0


def inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    v = np.diag(np.asarray(cov, dtype=float))
    if np.any(v <= 0):
        raise ValueError("covariance has a non-positive diagonal entry")
    w = 1.0 / v
    return w / w.sum()


def cluster_variance(cov: np.ndarray, items: Sequence[int]) -> float:
    items = np.asarray(items, dtype=int)
    sub = np.asarray(cov, dtype=float)[np.ix_(items, items)]
    w = inverse_variance_weights(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov: np.ndarray, sort_ix: Sequence[int]) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    sort_ix = np.asarray(sort_ix, dtype=int)
    n = sort_ix.size
    if n == 0:
        raise ValueError("empty sort order")
    if n == 1:
        out = np.zeros(cov.shape[0])
        out[sort_ix[0]] = 1.0
        return out

    w = np.ones(n, dtype=float)            
    clusters: List[np.ndarray] = [np.arange(n)]

    while clusters:
        nxt: List[np.ndarray] = []
        for c in clusters:
            if c.size > 1:
                mid = c.size // 2
                nxt.append(c[:mid])
                nxt.append(c[mid:])
        clusters = nxt
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0 = cluster_variance(cov, sort_ix[c0])
            v1 = cluster_variance(cov, sort_ix[c1])
            alpha = 1.0 - v0 / max(v0 + v1, _EPS)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha

    out = np.zeros(cov.shape[0], dtype=float)
    out[sort_ix] = w
    return out


def dendrogram_bisection(cov: np.ndarray, Z: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0] + 1
    if cov.shape[0] != n:
        raise ValueError(f"cov has {cov.shape[0]} assets, linkage implies {n}")

    members: List[np.ndarray] = [np.array([i]) for i in range(n)]
    for r in range(Z.shape[0]):
        a, b = int(Z[r, 0]), int(Z[r, 1])
        members.append(np.concatenate([members[a], members[b]]))

    w = np.zeros(n, dtype=float)
    stack: List[Tuple[int, float]] = [(n + Z.shape[0] - 1, 1.0)]
    while stack:
        node, scale = stack.pop()
        if node < n:
            w[node] = scale
            continue
        a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
        v0 = cluster_variance(cov, members[a])
        v1 = cluster_variance(cov, members[b])
        alpha = 1.0 - v0 / max(v0 + v1, _EPS)
        stack.append((a, scale * alpha))
        stack.append((b, scale * (1.0 - alpha)))
    return w


# ===========================================================================
# 5. BENCHMARKS
# ===========================================================================

def minimum_variance_weights(
    cov: np.ndarray, long_only: bool = False, ridge: float = 0.0
) -> np.ndarray:
    S = np.asarray(cov, dtype=float)
    n = S.shape[0]
    if ridge > 0:
        S = S + ridge * np.trace(S) / n * np.eye(n)
    one = np.ones(n)
    try:
        x = np.linalg.solve(S, one)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(S) @ one
    w = x / x.sum()
    if long_only:
        w = np.maximum(w, 0.0)
        s = w.sum()
        w = w / s if s > _EPS else np.full(n, 1.0 / n)
    return w


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


# ===========================================================================
# 6. ALLOCATOR
# ===========================================================================

@dataclass
class HRPResult:
    weights: np.ndarray
    order: np.ndarray
    linkage: np.ndarray
    distance: np.ndarray
    corr_distance: np.ndarray
    mst: np.ndarray
    method: str
    bisection: str
    labels: Optional[List[str]] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def leverage(self) -> float:
        return float(np.sum(np.abs(self.weights)))

    @property
    def effective_n(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def portfolio_variance(self, cov: np.ndarray) -> float:
        return float(self.weights @ np.asarray(cov, dtype=float) @ self.weights)

    def clusters(self, n_clusters: int) -> np.ndarray:
        return fcluster(self.linkage, n_clusters, criterion="maxclust")

    def report(self) -> str:
        lab = self.labels or [f"A{i}" for i in range(self.weights.size)]
        rows = [
            f"HRP({self.method}/{self.bisection})  "
            f"N={self.weights.size}  eff_N={self.effective_n:.2f}  "
            f"max={self.weights.max():.4f}  min={self.weights.min():.4f}",
            "",
            "  order   asset      weight",
        ]
        for p, i in enumerate(self.order):
            rows.append(f"  {p:>5d}   {lab[i]:<10s} {self.weights[i]:>8.4f}")
        return "\n".join(rows)


class HRPAllocator:
    def __init__(
        self,
        method: LinkageMethod = "single",
        bisection: Bisection = "dendrogram",
        repair_correlation: bool = True,
    ) -> None:
        self.method = method
        self.bisection = bisection
        self.repair_correlation = bool(repair_correlation)

    def allocate(
        self, cov: np.ndarray, labels: Optional[Sequence[str]] = None
    ) -> HRPResult:
        cov = np.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be square")
        if not np.all(np.isfinite(cov)):
            raise ValueError("cov contains non-finite values")
        n = cov.shape[0]

        if n == 1:
            return HRPResult(
                weights=np.array([1.0]), order=np.array([0]),
                linkage=np.empty((0, 4)), distance=np.zeros((1, 1)),
                corr_distance=np.zeros((1, 1)), mst=np.empty((0, 3)),
                method=self.method, bisection=self.bisection,
                labels=list(labels) if labels else None,
            )

        corr, _ = cov_to_corr(cov)
        if self.repair_correlation:
            corr = nearest_correlation(corr)

        d = correlation_to_distance(corr)
        D = correlation_distance_matrix(d)
        Z = build_linkage(D, method=self.method)
        order = quasi_diagonalize(Z, n)

        if self.bisection == "serial":
            w = recursive_bisection(cov, order)
        else:
            w = dendrogram_bisection(cov, Z)

        s = w.sum()
        if abs(s - 1.0) > 1e-9:
            w = w / s

        eq, heights, mstw = single_linkage_equals_mst(D)
        return HRPResult(
            weights=w, order=order, linkage=Z, distance=d, corr_distance=D,
            mst=mst_edges(D), method=self.method, bisection=self.bisection,
            labels=list(labels) if labels else None,
            diagnostics={
                "concentration_original": quasi_diagonal_concentration(corr, np.arange(n)),
                "concentration_sorted": quasi_diagonal_concentration(corr, order),
                "mst_matches_single_linkage": float(eq),
                "cophenetic_range": float(heights.max() - heights.min()),
                "max_weight": float(w.max()),
                "min_weight": float(w.min()),
                "tree_imbalance": tree_imbalance(Z),
            },
        )

    def allocate_from_returns(
        self, returns: np.ndarray, labels: Optional[Sequence[str]] = None
    ) -> HRPResult:
        R = np.asarray(returns, dtype=float)
        if R.ndim != 2:
            raise ValueError("returns must be (T, N)")
        if R.shape[0] < 2:
            raise ValueError("need at least 2 observations")
        return self.allocate(np.cov(R, rowvar=False), labels)


# ===========================================================================
# SELF-TESTS
# ===========================================================================

def _random_corr(n: int, seed: int, factor_dim: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, factor_dim))
    S = B @ B.T + np.diag(rng.uniform(0.2, 1.5, n))
    return cov_to_corr(S)[0]


def _block_cov(
    block_sizes: Sequence[int], rho_in: float, rho_out: float, sd: Optional[np.ndarray] = None
) -> np.ndarray:
    n = int(sum(block_sizes))
    C = np.full((n, n), rho_out, dtype=float)
    start = 0
    for b in block_sizes:
        C[start:start + b, start:start + b] = rho_in
        start += b
    np.fill_diagonal(C, 1.0)
    s = np.ones(n) if sd is None else np.asarray(sd, dtype=float)
    return C * np.outer(s, s)

def _test_end_to_end_report() -> None:
    labels = [f"BANK{i}" for i in range(3)] + [f"TECH{i}" for i in range(4)] + \
             [f"ENRG{i}" for i in range(3)]
    sd = np.array([0.18, 0.20, 0.19, 0.32, 0.35, 0.30, 0.33, 0.25, 0.27, 0.24])
    C = _block_cov([3, 4, 3], rho_in=0.75, rho_out=0.15)
    corr = cov_to_corr(C)[0]
    cov = corr * np.outer(sd, sd)

    rng = np.random.default_rng(1)
    X = rng.standard_normal((900, 10)) @ np.linalg.cholesky(cov).T
    r = HRPAllocator(method="single", bisection="dendrogram").allocate_from_returns(X, labels)

    print(r.report())
    print()
    d = r.diagnostics
    blocks = np.concatenate([np.full(s, b) for b, s in enumerate([3, 4, 3])])
    seq = blocks[r.order]
    runs = 1 + int(np.sum(np.diff(seq) != 0))
    print(f"     sector runs in leaf order: {runs} (3 sectors)")
    print(f"     quasi-diagonal concentration {d['concentration_original']:.4f} -> "
          f"{d['concentration_sorted']:.4f}")
    print(f"     MST == single linkage: {bool(d['mst_matches_single_linkage'])}; "
          f"MST edges {r.mst.shape[0]}")
    print(f"     sector weights: " + ", ".join(
        f"{nm}={r.weights[blocks == b].sum():.3f}"
        for b, nm in enumerate(("BANK", "TECH", "ENRG"))))
    print(f"     eff_N={r.effective_n:.2f}, portfolio vol={math.sqrt(r.portfolio_variance(cov)):.4f} "
          f"vs EW {math.sqrt(equal_weight(10) @ cov @ equal_weight(10)):.4f}")
    assert runs == 3
    assert abs(r.weights.sum() - 1.0) < 1e-12

if __name__ == "__main__":
    print("==========================================================================")
    print(" PORTFOLIO ALLOCATION: HIERARCHICAL RISK PARITY (HRP)")
    print("==========================================================================")
    _test_end_to_end_report()
    print("all checks passed")

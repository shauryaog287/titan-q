"""
graph_alpha_engine_kernels.py
=============================

Phase 8, Engine I — production implementations of the three compute
bottlenecks. Drop-in replacements for the corresponding skeleton classes.

Fast paths are numba-jitted; every kernel has a vectorised numpy fallback
that is exercised when numba is unavailable.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from numba import njit, prange

    _NUMBA = True
except ImportError:  # pragma: no cover
    _NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f

        return deco if not a else a[0]

    prange = range


_EPS = 1e-12


# ===========================================================================
# JIT KERNELS
# ===========================================================================

@njit(cache=True, fastmath=True, nogil=True)
def _hy_scan(
    ret_i: np.ndarray,
    bnd_i: np.ndarray,
    ret_j: np.ndarray,
    pref_j: np.ndarray,
    bnd_j: np.ndarray,
    shift: np.int64,
) -> Tuple[float, int]:
    """
    Single-pair Hayashi-Yoshida accumulation by monotone two-pointer merge.
    """
    n_i = ret_i.shape[0]
    n_j = ret_j.shape[0]

    total = 0.0
    count = 0
    lo = 0
    hi = 0

    for k in range(n_i):
        a0 = bnd_i[k] + shift
        a1 = bnd_i[k + 1] + shift

        if a1 <= a0:  
            continue

        while lo < n_j and bnd_j[lo + 1] <= a0:
            lo += 1

        if hi < lo:
            hi = lo

        while hi < n_j and bnd_j[hi] < a1:
            hi += 1

        if hi > lo:
            total += ret_i[k] * (pref_j[hi] - pref_j[lo])
            count += hi - lo

    return total, count


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _hy_pairs_kernel(
    ret_flat: np.ndarray,
    ret_off: np.ndarray,
    bnd_flat: np.ndarray,
    bnd_off: np.ndarray,
    pref_flat: np.ndarray,
    pref_off: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ragged-batched HY over all upper-triangle pairs, parallel over pairs."""
    n_pairs = pair_i.shape[0]
    cov = np.zeros(n_pairs, dtype=np.float64)
    cnt = np.zeros(n_pairs, dtype=np.int64)

    for p in prange(n_pairs):
        i = pair_i[p]
        j = pair_j[p]

        ri = ret_flat[ret_off[i] : ret_off[i + 1]]
        bi = bnd_flat[bnd_off[i] : bnd_off[i + 1]]
        rj = ret_flat[ret_off[j] : ret_off[j + 1]]
        bj = bnd_flat[bnd_off[j] : bnd_off[j + 1]]
        pj = pref_flat[pref_off[j] : pref_off[j + 1]]

        n_i = ri.shape[0]
        n_j = rj.shape[0]

        total = 0.0
        count = 0
        lo = 0
        hi = 0

        for k in range(n_i):
            a0 = bi[k]
            a1 = bi[k + 1]
            if a1 <= a0:
                continue

            while lo < n_j and bj[lo + 1] <= a0:
                lo += 1
            if hi < lo:
                hi = lo
            while hi < n_j and bj[hi] < a1:
                hi += 1

            if hi > lo:
                total += ri[k] * (pj[hi] - pj[lo])
                count += hi - lo

        cov[p] = total
        cnt[p] = count

    return cov, cnt


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _hy_lag_kernel(
    ret_i: np.ndarray,
    bnd_i: np.ndarray,
    ret_j: np.ndarray,
    pref_j: np.ndarray,
    bnd_j: np.ndarray,
    lags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Huth-Abergel surface: one independent merge-scan per candidate lag."""
    n_lags = lags.shape[0]
    n_i = ret_i.shape[0]
    n_j = ret_j.shape[0]

    cov = np.zeros(n_lags, dtype=np.float64)
    cnt = np.zeros(n_lags, dtype=np.int64)

    for p in prange(n_lags):
        d = lags[p]
        total = 0.0
        count = 0
        lo = 0
        hi = 0

        for k in range(n_i):
            a0 = bnd_i[k] + d
            a1 = bnd_i[k + 1] + d
            if a1 <= a0:
                continue

            while lo < n_j and bnd_j[lo + 1] <= a0:
                lo += 1
            if hi < lo:
                hi = lo
            while hi < n_j and bnd_j[hi] < a1:
                hi += 1

            if hi > lo:
                total += ret_i[k] * (pref_j[hi] - pref_j[lo])
                count += hi - lo

        cov[p] = total
        cnt[p] = count

    return cov, cnt


def _hy_scan_numpy(
    ret_i: np.ndarray,
    bnd_i: np.ndarray,
    ret_j: np.ndarray,
    pref_j: np.ndarray,
    bnd_j: np.ndarray,
    shift: int = 0,
) -> Tuple[float, int]:
    a0 = bnd_i[:-1] + shift
    a1 = bnd_i[1:] + shift
    n_j = ret_j.shape[0]

    lo = np.searchsorted(bnd_j, a0, side="right") - 1
    hi = np.searchsorted(bnd_j, a1, side="left")

    np.clip(lo, 0, n_j, out=lo)
    np.clip(hi, 0, n_j, out=hi)
    np.maximum(hi, lo, out=hi)

    valid = a1 > a0
    block = pref_j[hi] - pref_j[lo]
    total = float(np.dot(ret_i[valid], block[valid]))
    count = int((hi - lo)[valid].sum())
    return total, count


def _nearest_psd_eigen(a: np.ndarray, eps: float = 0.0) -> np.ndarray:
    s = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(s)
    np.clip(w, eps, None, out=w)
    return (v * w) @ v.T


def _nearest_correlation_higham(
    a: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-9,
    eig_floor: float = 1e-10,
) -> Tuple[np.ndarray, int, bool]:
    n = a.shape[0]
    y = 0.5 * (a + a.T).astype(np.float64, copy=True)
    ds = np.zeros((n, n), dtype=np.float64)
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        r = y - ds
        x = _nearest_psd_eigen(r, eps=eig_floor)
        ds = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)

        denom = max(np.linalg.norm(y_new, "fro"), _EPS)
        if np.linalg.norm(y_new - y, "fro") / denom < tol:
            y = y_new
            converged = True
            break
        y = y_new

    np.fill_diagonal(y, 1.0)
    return y, it, converged


class HayashiYoshidaEstimator:
    __slots__ = (
        "min_overlapping_intervals",
        "use_numba",
        "last_symbols_",
        "last_counts_",
        "last_unreliable_",
        "last_psd_projection_distance_",
        "last_psd_iterations_",
        "last_psd_converged_",
        "last_min_eigenvalue_raw_",
    )

    def __init__(
        self,
        min_overlapping_intervals: int = 200,
        use_numba: bool = True,
    ) -> None:
        self.min_overlapping_intervals = int(min_overlapping_intervals)
        self.use_numba = bool(use_numba and _NUMBA)
        if use_numba and not _NUMBA:
            warnings.warn("numba unavailable; falling back to searchsorted path")

        self.last_symbols_: Tuple[str, ...] = ()
        self.last_counts_: Optional[np.ndarray] = None
        self.last_unreliable_: List[Tuple[str, str, int]] = []
        self.last_psd_projection_distance_: float = 0.0
        self.last_psd_iterations_: int = 0
        self.last_psd_converged_: bool = True
        self.last_min_eigenvalue_raw_: float = 0.0

    @staticmethod
    def _prepare(
        returns: np.ndarray, bounds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.ascontiguousarray(returns, dtype=np.float64)
        b = np.ascontiguousarray(bounds, dtype=np.int64)

        if b.shape[0] != r.shape[0] + 1:
            raise ValueError(
                f"bounds must have len(returns)+1 entries; got {b.shape[0]} vs {r.shape[0]}"
            )
        if r.shape[0] == 0:
            raise ValueError("empty return series")
        if np.any(np.diff(b) < 0):
            raise ValueError("interval bounds are not monotone non-decreasing")
        if not np.all(np.isfinite(r)):
            raise ValueError("non-finite returns present")

        pref = np.empty(r.shape[0] + 1, dtype=np.float64)
        pref[0] = 0.0
        np.cumsum(r, out=pref[1:])
        return r, b, pref

    def covariance(
        self,
        returns_i: np.ndarray,
        interval_bounds_i: np.ndarray,
        returns_j: np.ndarray,
        interval_bounds_j: np.ndarray,
    ) -> Tuple[float, int]:
        ri, bi, _ = self._prepare(returns_i, interval_bounds_i)
        rj, bj, pj = self._prepare(returns_j, interval_bounds_j)

        if self.use_numba:
            cov, cnt = _hy_scan(ri, bi, rj, pj, bj, np.int64(0))
        else:
            cov, cnt = _hy_scan_numpy(ri, bi, rj, pj, bj, 0)

        return float(cov), int(cnt)

    def correlation(
        self,
        returns_i: np.ndarray,
        interval_bounds_i: np.ndarray,
        returns_j: np.ndarray,
        interval_bounds_j: np.ndarray,
    ) -> Tuple[float, int]:
        cov, cnt = self.covariance(
            returns_i, interval_bounds_i, returns_j, interval_bounds_j
        )
        vi = float(np.dot(returns_i, returns_i))
        vj = float(np.dot(returns_j, returns_j))
        denom = math.sqrt(max(vi * vj, _EPS))
        return float(np.clip(cov / denom, -1.0, 1.0)), cnt

    def covariance_matrix(
        self,
        event_series: Mapping[str, Tuple[np.ndarray, np.ndarray]],
        project_psd: bool = True,
        shrink_unreliable_to_zero: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        symbols = tuple(event_series.keys())
        n = len(symbols)
        if n == 0:
            raise ValueError("no series supplied")

        rets: List[np.ndarray] = []
        bnds: List[np.ndarray] = []
        prefs: List[np.ndarray] = []
        for s in symbols:
            r, b, p = self._prepare(*event_series[s])
            rets.append(r)
            bnds.append(b)
            prefs.append(p)

        variances = np.array([float(np.dot(r, r)) for r in rets], dtype=np.float64)

        iu, ju = np.triu_indices(n, k=1)
        cov = np.zeros((n, n), dtype=np.float64)
        counts = np.zeros((n, n), dtype=np.int64)

        if self.use_numba and n > 1:
            ret_off = np.zeros(n + 1, dtype=np.int64)
            bnd_off = np.zeros(n + 1, dtype=np.int64)
            pref_off = np.zeros(n + 1, dtype=np.int64)
            for k in range(n):
                ret_off[k + 1] = ret_off[k] + rets[k].shape[0]
                bnd_off[k + 1] = bnd_off[k] + bnds[k].shape[0]
                pref_off[k + 1] = pref_off[k] + prefs[k].shape[0]

            ret_flat = np.concatenate(rets) if n else np.empty(0)
            bnd_flat = np.concatenate(bnds)
            pref_flat = np.concatenate(prefs)

            pcov, pcnt = _hy_pairs_kernel(
                ret_flat,
                ret_off,
                bnd_flat,
                bnd_off,
                pref_flat,
                pref_off,
                iu.astype(np.int64),
                ju.astype(np.int64),
            )
        else:
            pcov = np.empty(iu.shape[0], dtype=np.float64)
            pcnt = np.empty(iu.shape[0], dtype=np.int64)
            for p, (i, j) in enumerate(zip(iu, ju)):
                c, k = _hy_scan_numpy(rets[i], bnds[i], rets[j], prefs[j], bnds[j], 0)
                pcov[p] = c
                pcnt[p] = k

        cov[iu, ju] = pcov
        cov[ju, iu] = pcov  
        counts[iu, ju] = pcnt
        counts[ju, iu] = pcnt
        np.fill_diagonal(cov, variances)
        np.fill_diagonal(counts, np.array([r.shape[0] for r in rets]))

        self.last_unreliable_ = []
        if shrink_unreliable_to_zero:
            thin = pcnt < self.min_overlapping_intervals
            if thin.any():
                cov[iu[thin], ju[thin]] = 0.0
                cov[ju[thin], iu[thin]] = 0.0
                self.last_unreliable_ = [
                    (symbols[i], symbols[j], int(c))
                    for i, j, c in zip(iu[thin], ju[thin], pcnt[thin])
                ]

        self.last_min_eigenvalue_raw_ = float(np.linalg.eigvalsh(cov).min())

        if project_psd:
            cov = self._project(cov, variances)

        self.last_symbols_ = symbols
        self.last_counts_ = counts
        return cov, counts

    def _project(self, cov: np.ndarray, variances: np.ndarray) -> np.ndarray:
        sd = np.sqrt(np.maximum(variances, _EPS))
        outer = np.outer(sd, sd)
        corr = cov / outer
        np.clip(corr, -1.0, 1.0, out=corr)
        np.fill_diagonal(corr, 1.0)

        w_min = float(np.linalg.eigvalsh(corr).min())
        if w_min >= -1e-12:
            self.last_psd_projection_distance_ = 0.0
            self.last_psd_iterations_ = 0
            self.last_psd_converged_ = True
            return corr * outer

        fixed, iters, ok = _nearest_correlation_higham(corr)
        self.last_psd_projection_distance_ = float(
            np.linalg.norm(fixed - corr, "fro")
        )
        self.last_psd_iterations_ = iters
        self.last_psd_converged_ = ok
        if not ok:
            warnings.warn(
                f"Higham projection did not converge in {iters} iterations; "
                f"Frobenius distance {self.last_psd_projection_distance_:.4e}"
            )
        return fixed * outer


@dataclass(frozen=True)
class EdgeEstimate:
    source: str
    target: str
    relation: object
    weight: float
    optimal_lag_us: Optional[int]
    rho_at_zero_lag: Optional[float]
    bootstrap_pvalue: Optional[float]
    passed_fdr: bool
    n_overlapping_intervals: int

    @property
    def directional_content(self) -> Optional[float]:
        if self.rho_at_zero_lag is None:
            return None
        base = abs(self.rho_at_zero_lag)
        if base < _EPS:
            return float("inf") if abs(self.weight) > _EPS else None
        return abs(self.weight) / base


class LeadLagDetector:
    __slots__ = (
        "estimator",
        "lag_grid_us",
        "min_directional_content",
        "refine_peak",
    )

    def __init__(
        self,
        estimator: HayashiYoshidaEstimator,
        lag_grid_us: Sequence[int],
        min_directional_content: float = 1.05,
        refine_peak: bool = True,
    ) -> None:
        self.estimator = estimator
        grid = np.asarray(sorted(set(int(x) for x in lag_grid_us)), dtype=np.int64)
        if grid.size == 0:
            raise ValueError("empty lag grid")
        self.lag_grid_us = grid
        self.min_directional_content = float(min_directional_content)
        self.refine_peak = bool(refine_peak)

    def cross_correlation_surface(
        self,
        series_i: Tuple[np.ndarray, np.ndarray],
        series_j: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        rho, _ = self._surface_with_counts(series_i, series_j)
        return rho

    def _surface_with_counts(
        self,
        series_i: Tuple[np.ndarray, np.ndarray],
        series_j: Tuple[np.ndarray, np.ndarray],
        lags: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ri, bi, _ = self.estimator._prepare(*series_i)
        rj, bj, pj = self.estimator._prepare(*series_j)

        grid = self.lag_grid_us if lags is None else np.asarray(lags, dtype=np.int64)

        vi = float(np.dot(ri, ri))
        vj = float(np.dot(rj, rj))
        denom = math.sqrt(max(vi * vj, _EPS))

        if self.estimator.use_numba:
            cov, cnt = _hy_lag_kernel(ri, bi, rj, pj, bj, grid)
        else:
            cov = np.empty(grid.shape[0], dtype=np.float64)
            cnt = np.empty(grid.shape[0], dtype=np.int64)
            for p, d in enumerate(grid):
                c, k = _hy_scan_numpy(ri, bi, rj, pj, bj, int(d))
                cov[p] = c
                cnt[p] = k

        rho = np.clip(cov / denom, -1.0, 1.0)
        return rho, cnt

    def detect(
        self,
        series_i: Tuple[np.ndarray, np.ndarray],
        series_j: Tuple[np.ndarray, np.ndarray],
        symbol_i: str,
        symbol_j: str,
        relation: object = None,
    ) -> EdgeEstimate:
        rho, cnt = self._surface_with_counts(series_i, series_j)

        if np.any(self.lag_grid_us == 0):
            zero_idx = int(np.flatnonzero(self.lag_grid_us == 0)[0])
            rho_zero = float(rho[zero_idx])
        else:
            rz, _ = self._surface_with_counts(
                series_i, series_j, lags=np.array([0], dtype=np.int64)
            )
            rho_zero = float(rz[0])

        peak = int(np.argmax(np.abs(rho)))
        best_lag = int(self.lag_grid_us[peak])
        best_rho = float(rho[peak])
        best_cnt = int(cnt[peak])

        if self.refine_peak and 0 < peak < rho.shape[0] - 1:
            best_lag, best_rho = self._parabolic_refine(
                self.lag_grid_us, np.abs(rho), peak, best_rho
            )

        base = abs(rho_zero)
        content = abs(best_rho) / base if base > _EPS else float("inf")
        directional = content >= self.min_directional_content

        return EdgeEstimate(
            source=symbol_i,
            target=symbol_j,
            relation=relation,
            weight=best_rho if directional else 0.0,
            optimal_lag_us=best_lag if directional else None,
            rho_at_zero_lag=rho_zero,
            bootstrap_pvalue=None,  
            passed_fdr=False,       
            n_overlapping_intervals=best_cnt,
        )

    @staticmethod
    def _parabolic_refine(
        grid: np.ndarray, abs_rho: np.ndarray, peak: int, signed_peak: float
    ) -> Tuple[int, float]:
        y0, y1, y2 = abs_rho[peak - 1], abs_rho[peak], abs_rho[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < _EPS:
            return int(grid[peak]), signed_peak

        offset = 0.5 * (y0 - y2) / denom
        offset = float(np.clip(offset, -0.5, 0.5))

        spacing = float(grid[peak + 1] - grid[peak - 1]) * 0.5
        refined_lag = int(round(grid[peak] + offset * spacing))
        refined_abs = float(y1 - 0.25 * (y0 - y2) * offset)
        return refined_lag, math.copysign(refined_abs, signed_peak)


class BidirectionalDiffusionConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_relations: int,
        diffusion_steps: int = 3,
        bias: bool = True,
        check_asymmetry: bool = True,
        asymmetry_tol: float = 1e-6,
        renormalise_masked: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_relations = int(n_relations)
        self.K = int(diffusion_steps)
        self.check_asymmetry = bool(check_asymmetry)
        self.asymmetry_tol = float(asymmetry_tol)
        self.renormalise_masked = bool(renormalise_masked)

        self.theta = nn.Parameter(
            torch.empty(self.n_relations, 2, self.K + 1, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.in_channels * self.n_relations * 2 * (self.K + 1)
        bound = math.sqrt(6.0 / (fan_in + self.out_channels))
        nn.init.uniform_(self.theta, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _assert_not_symmetrised(self, p_out: Tensor, p_in: Tensor) -> None:
        with torch.no_grad():
            for r in range(p_out.shape[-1]):
                po = p_out[..., r]
                if po.abs().sum() == 0:
                    continue
                asym = (po - po.transpose(-1, -2)).abs().max()
                if asym < self.asymmetry_tol:
                    raise RuntimeError(
                        f"relation slice {r}: P_out is symmetric. "
                        "Adjacency was symmetrised upstream; directed signal destroyed."
                    )

    @staticmethod
    def _apply_mask(p: Tensor, mask: Tensor, renormalise: bool) -> Tensor:
        p = p * mask.unsqueeze(-2) 
        if renormalise:
            denom = p.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            p = p / denom
        return p * mask.unsqueeze(-1) 

    def forward(
        self,
        x: Tensor,
        p_out: Tensor,
        p_in: Tensor,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected x of rank 4 (B,N,T,C)")
        B, N, T, C = x.shape
        batched_adj = p_out.dim() == 4

        if self.check_asymmetry and self.training:
            self._assert_not_symmetrised(
                p_out[0] if batched_adj else p_out,
                p_in[0] if batched_adj else p_in,
            )

        h = x
        if node_mask is not None:
            m = node_mask.to(x.dtype)
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(B, -1)
            h = h * m.view(B, N, 1, 1)
        else:
            m = None

        acc = x.new_zeros(B * N * T, self.out_channels)
        einsum_adj = "bnm,bmtc->bntc" if batched_adj else "nm,bmtc->bntc"

        for r in range(self.n_relations):
            for d, p_src in ((0, p_out), (1, p_in)):
                p = p_src[..., r]
                if m is not None:
                    if not batched_adj:
                        p = p.unsqueeze(0).expand(B, -1, -1)
                    p = self._apply_mask(p, m, self.renormalise_masked)
                    step = "bnm,bmtc->bntc"
                else:
                    step = einsum_adj

                if p.abs().sum() == 0:
                    acc = acc + torch.mm(h.reshape(-1, C), self.theta[r, d, 0])
                    continue

                z = h
                acc = acc + torch.mm(z.reshape(-1, C), self.theta[r, d, 0])
                for k in range(1, self.K + 1):
                    z = torch.einsum(step, p, z)
                    acc = acc + torch.mm(z.reshape(-1, C), self.theta[r, d, k])

        out = acc.view(B, N, T, self.out_channels)
        if self.bias is not None:
            out = out + self.bias
        if m is not None:
            out = out * m.view(B, N, 1, 1)
        return out

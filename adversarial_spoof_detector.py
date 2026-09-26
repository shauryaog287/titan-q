"""
adversarial_spoof_detector.py
=============================

Phase 16: Unsupervised detection of spoofing and layering in the limit order
book via reconstruction error.

    LOB snapshot / window  ->  frozen standardisation  ->  autoencoder
        ->  reconstruction error  ->  lagged robust threshold  ->  flag

WHAT A BOOK-ONLY DETECTOR CAN AND CANNOT SEE
-------------------------------------------
Spoofing is defined by INTENT: an order placed with no intention of being
executed. The book shows orders, not intent. A spoof and a genuine institutional
block look identical while resting; the only observable difference is what
happens to them -- a genuine order trades, a spoof is cancelled. Two
consequences are built into this module and measured in the self-tests:

  1. A snapshot autoencoder detects ANOMALOUS BOOK SHAPE, not spoofing. It
     flags a genuine block exactly as readily as a spoof. Specificity requires
     transition features that separate cancellation from execution.

  2. If genuine large orders are common enough to appear in the training
     distribution -- and on a liquid name they are -- then a large resting order
     stops being anomalous. The remaining signature is the cancel-without-fill,
     which by construction arrives AFTER the manipulation has worked. Detection
     from the book alone therefore lags the manipulation by the order's
     lifetime. This is a surveillance tool, not a pre-trade shield.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

torch.set_default_dtype(torch.float32)

__all__ = [
    "Event",
    "LOBSimConfig",
    "simulate_lob",
    "LOBFeatureEncoder",
    "LOBAutoencoder",
    "train_autoencoder",
    "PCABaseline",
    "wasserstein_1d",
    "depth_wasserstein",
    "DynamicThreshold",
    "PerLevelRule",
    "SpoofDetector",
    "evaluate_events",
    "attribute",
    "removal_cancel_fraction",
]


# ===========================================================================
# 1. SIMULATOR
# ===========================================================================

class Event(IntEnum):
    NORMAL = 0
    SPOOF = 1
    LAYERING = 2
    LEGIT_BLOCK = 3


@dataclass(frozen=True)
class LOBSimConfig:
    n_levels: int = 10
    phi: float = 0.92             # AR(1) persistence of log size
    sigma: float = 0.12           # innovation scale of log size
    common: float = 0.70          # share of innovation variance from a book-wide factor
    side_common: float = 0.15     # share from a side-specific factor
    base_size: float = 200.0
    depth_slope: float = 0.12     # base size grows with distance from touch
    exec_rate: float = 0.35       # probability of a trade at the touch per step
    exec_mean: float = 60.0
    tick_prob: float = 0.04       # probability the mid moves one tick


def simulate_lob(
    n_steps: int,
    cfg: LOBSimConfig = LOBSimConfig(),
    events: Sequence[Tuple[Event, int, int, str, Sequence[int], float]] = (),
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    L = cfg.n_levels
    lvl = np.arange(L)
    m = np.log(cfg.base_size * (1.0 + cfg.depth_slope * lvl))

    stat_sd = cfg.sigma / math.sqrt(1.0 - cfg.phi**2)
    ls = np.empty((n_steps, 2, L))
    ls[0] = m + rng.standard_normal((2, L)) * stat_sd

    a_c = math.sqrt(cfg.common)
    a_s = math.sqrt(cfg.side_common)
    a_i = math.sqrt(max(1.0 - cfg.common - cfg.side_common, 0.0))
    for t in range(1, n_steps):
        f = rng.standard_normal()
        fs = rng.standard_normal(2)
        e = rng.standard_normal((2, L))
        eps = a_c * f + a_s * fs[:, None] + a_i * e
        ls[t] = m + cfg.phi * (ls[t - 1] - m) + cfg.sigma * eps

    base = np.exp(ls)
    extra = np.zeros_like(base)
    execd = np.zeros_like(base)
    label = np.zeros(n_steps, dtype=np.int64)

    hit = rng.random((n_steps, 2)) < cfg.exec_rate
    execd[:, :, 0] = np.where(hit, rng.exponential(cfg.exec_mean, (n_steps, 2)), 0.0)

    ref = cfg.base_size * (1.0 + cfg.depth_slope * lvl)
    for (etype, t0, t1, side, levels, mag) in events:
        s = 0 if side == "bid" else 1
        levels = list(levels)
        t0, t1 = int(t0), int(min(t1, n_steps))
        label[t0:t1] = int(etype)
        for k in levels:
            amt = mag * ref[k]
            if etype == Event.LEGIT_BLOCK:
                fill_start = t0 + 2 * (t1 - t0) // 3
                extra[t0:fill_start, s, k] += amt
                n_fill = max(t1 - fill_start, 1)
                path = amt * (1.0 - np.arange(1, n_fill + 1) / n_fill)
                extra[fill_start:t1, s, k] += path
                prev = np.concatenate(([amt], path[:-1]))
                execd[fill_start:t1, s, k] += prev - path
            else:
                extra[t0:t1, s, k] += amt

    size = base + extra
    drop = np.zeros_like(size)
    drop[1:] = size[:-1] - size[1:]
    cancelled = np.maximum(drop - execd, 0.0)
    cancelled[0] = 0.0

    mid = 100.0 + 0.01 * np.cumsum(
        np.where(rng.random(n_steps) < cfg.tick_prob, rng.choice([-1, 1], n_steps), 0)
    )
    spread = 1.0 + (rng.random(n_steps) < 0.15).astype(float)

    return {
        "bid_sz": size[:, 0], "ask_sz": size[:, 1],
        "bid_exec": execd[:, 0], "ask_exec": execd[:, 1],
        "bid_cxl": cancelled[:, 0], "ask_cxl": cancelled[:, 1],
        "spread": spread, "mid": mid, "label": label,
        "events": list(events),
    }


# ===========================================================================
# 2. FEATURES
# ===========================================================================

class LOBFeatureEncoder:
    def __init__(self, window: int = 5) -> None:
        self.window = int(window)
        self._mu_s: Optional[np.ndarray] = None
        self._sd_s: Optional[np.ndarray] = None
        self._mu_w: Optional[np.ndarray] = None
        self._sd_w: Optional[np.ndarray] = None

    @staticmethod
    def snapshot_raw(d: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.log1p(d["bid_sz"]), np.log1p(d["ask_sz"]), d["spread"][:, None]], axis=1
        )

    @staticmethod
    def transition_raw(d: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.log1p(d["bid_cxl"]), np.log1p(d["ask_cxl"]),
             np.log1p(d["bid_exec"]), np.log1p(d["ask_exec"])], axis=1
        )

    def _windows(self, per_step: np.ndarray) -> np.ndarray:
        W = self.window
        T, D = per_step.shape
        out = np.empty((T - W + 1, W * D), dtype=per_step.dtype)
        for i in range(W):
            out[:, i * D:(i + 1) * D] = per_step[i:T - W + 1 + i]
        return out

    def fit(self, d: Dict[str, np.ndarray]) -> "LOBFeatureEncoder":
        s = self.snapshot_raw(d)
        self._mu_s, self._sd_s = s.mean(0), np.maximum(s.std(0), 1e-6)
        w = self._windows(np.concatenate([s, self.transition_raw(d)], axis=1))
        self._mu_w, self._sd_w = w.mean(0), np.maximum(w.std(0), 1e-6)
        return self

    def snapshots(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        if self._mu_s is None:
            raise RuntimeError("fit on a training window first")
        return ((self.snapshot_raw(d) - self._mu_s) / self._sd_s).astype(np.float32)

    def group_indices(self, n_levels: int) -> Dict[str, np.ndarray]:
        L = n_levels
        per = 2 * L + 1 + 4 * L
        out: Dict[str, List[int]] = {"size": [], "cancel": [], "exec": []}
        for w in range(self.window):
            o = w * per
            out["size"] += list(range(o, o + 2 * L + 1))
            out["cancel"] += list(range(o + 2 * L + 1, o + 4 * L + 1))
            out["exec"] += list(range(o + 4 * L + 1, o + 6 * L + 1))
        return {k: np.asarray(v) for k, v in out.items()}

    def windows(self, d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self._mu_w is None:
            raise RuntimeError("fit on a training window first")
        per = np.concatenate([self.snapshot_raw(d), self.transition_raw(d)], axis=1)
        w = self._windows(per)
        idx = np.arange(self.window - 1, per.shape[0])
        return ((w - self._mu_w) / self._sd_w).astype(np.float32), idx


# ===========================================================================
# 3. MODELS
# ===========================================================================

class LOBAutoencoder(nn.Module):
    def __init__(self, d_in: int, code: int = 6, hidden: Sequence[int] = (64, 32),
                 seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        enc: List[nn.Module] = []
        prev = d_in
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        enc.append(nn.Linear(prev, code))
        dec: List[nn.Module] = []
        prev = code
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        dec.append(nn.Linear(prev, d_in))
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        self.d_in, self.code = d_in, code

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        self.eval()
        out = [self(torch.from_numpy(X[i:i + batch])).numpy()
               for i in range(0, X.shape[0], batch)]
        return np.concatenate(out, axis=0)

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.mean((self.reconstruct(X) - X) ** 2, axis=1)


def train_autoencoder(
    X: np.ndarray, code: int = 6, hidden: Sequence[int] = (64, 32),
    epochs: int = 40, batch: int = 512, lr: float = 2e-3, seed: int = 0,
    verbose: bool = False,
) -> LOBAutoencoder:
    torch.manual_seed(seed)
    model = LOBAutoencoder(X.shape[1], code, hidden, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.05)
    Xt = torch.from_numpy(X)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xt.shape[0], generator=g)
        tot = 0.0
        for i in range(0, Xt.shape[0], batch):
            xb = Xt[perm[i:i + batch]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * xb.shape[0]
        sched.step()
        if verbose and ep % 10 == 0:
            print(f"      epoch {ep:3d}  mse={tot / Xt.shape[0]:.4f}")
    return model


class PCABaseline:
    def __init__(self, k: int) -> None:
        self.k = int(k)

    def fit(self, X: np.ndarray) -> "PCABaseline":
        self.mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = Vt[: self.k].T
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.mu
        R = Z - (Z @ self.V) @ self.V.T
        return np.mean(R**2, axis=1)


# ===========================================================================
# 4. DISTANCES
# ===========================================================================

def wasserstein_1d(p: np.ndarray, q: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-12)
    return spacing * np.sum(np.abs(np.cumsum(p, -1) - np.cumsum(q, -1)), axis=-1)


def depth_wasserstein(X_true: np.ndarray, X_rec: np.ndarray,
                      enc: LOBFeatureEncoder, n_levels: int) -> np.ndarray:
    mu, sd = enc._mu_s, enc._sd_s
    L = n_levels
    true_sz = np.expm1(X_true[:, : 2 * L] * sd[: 2 * L] + mu[: 2 * L])
    rec_sz = np.expm1(X_rec[:, : 2 * L] * sd[: 2 * L] + mu[: 2 * L])
    true_sz, rec_sz = np.maximum(true_sz, 0.0), np.maximum(rec_sz, 0.0)
    return (wasserstein_1d(true_sz[:, :L], rec_sz[:, :L])
            + wasserstein_1d(true_sz[:, L:], rec_sz[:, L:]))


def attribute(X: np.ndarray, X_rec: np.ndarray,
              groups: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    e = (X_rec - X) ** 2
    tot = np.maximum(e.sum(axis=1), 1e-12)
    return {k: e[:, idx].sum(axis=1) / tot for k, idx in groups.items()}


def removal_cancel_fraction(
    d: Dict[str, np.ndarray], side: str, level: int, t_from: int, t_to: int
) -> Tuple[float, float, float]:
    s = "bid" if side == "bid" else "ask"
    c = float(d[f"{s}_cxl"][t_from:t_to, level].sum())
    e = float(d[f"{s}_exec"][t_from:t_to, level].sum())
    return c / max(c + e, 1e-12), c, e


# ===========================================================================
# 5. THRESHOLDING
# ===========================================================================

class DynamicThreshold:
    def __init__(
        self, window: int = 500, z: float = 4.0,
        method: Literal["meanstd", "mad"] = "mad",
        transform: Literal["none", "log"] = "log",
        lag: int = 1, exclude_flagged: bool = True, min_history: int = 100,
    ) -> None:
        if lag < 0:
            raise ValueError("lag must be non-negative")
        self.window = int(window)
        self.z = float(z)
        self.method = method
        self.transform = transform
        self.lag = int(lag)
        self.exclude_flagged = bool(exclude_flagged)
        self.min_history = int(min_history)

    def nominal_fpr(self) -> float:
        return float(1.0 - stats.norm.cdf(self.z))

    def _t(self, s: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(s, 1e-12)) if self.transform == "log" else s

    def run(
        self, scores: np.ndarray, warm: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        s = self._t(np.asarray(scores, dtype=float))
        idx_hist: List[int] = []
        val_hist: List[float] = []
        if warm is not None:
            w = self._t(np.asarray(warm, dtype=float))
            n_w = w.size
            idx_hist = list(range(-n_w, 0))
            val_hist = [float(v) for v in w]

        thr = np.full(s.size, np.nan)
        flag = np.zeros(s.size, dtype=bool)

        for t in range(s.size):
            cutoff = t - self.lag
            k = len(idx_hist)
            while k > 0 and idx_hist[k - 1] > cutoff:
                k -= 1
            ref_list = val_hist[max(0, k - self.window):k]
            if self.lag == 0:
                ref_list = ref_list[-(self.window - 1):] + [float(s[t])] \
                    if self.window > 1 else [float(s[t])]
            ref = np.asarray(ref_list, dtype=float)

            if ref.size >= self.min_history:
                if self.method == "mad":
                    c = float(np.median(ref))
                    sc = 1.4826 * float(np.median(np.abs(ref - c)))
                else:
                    c = float(ref.mean())
                    sc = float(ref.std(ddof=1))
                thr[t] = c + self.z * max(sc, 1e-12)
                flag[t] = s[t] > thr[t]

            if (not self.exclude_flagged) or (not flag[t]):
                idx_hist.append(t)
                val_hist.append(float(s[t]))

        thr_out = np.exp(thr) if self.transform == "log" else thr
        return thr_out, flag


# ===========================================================================
# 6. BASELINE RULE
# ===========================================================================

class PerLevelRule:
    def __init__(self, z: float = 4.0, n_levels: int = 10) -> None:
        self.z = float(z)
        self.L = int(n_levels)

    def score(self, S: np.ndarray) -> np.ndarray:
        return np.max(np.abs(S[:, : 2 * self.L]), axis=1)

    def flags(self, S: np.ndarray) -> np.ndarray:
        return self.score(S) > self.z


# ===========================================================================
# 7. ORCHESTRATION AND EVALUATION
# ===========================================================================

@dataclass
class DetectionOutcome:
    etype: Event
    t_start: int
    t_end: int
    detected: bool
    first_flag: Optional[int]
    latency: Optional[int]
    detected_while_resting: bool
    detected_at_cancel: bool
    peak_ratio: float


def evaluate_events(
    flags: np.ndarray, scores: np.ndarray, thresholds: np.ndarray,
    events: Sequence, step_index: np.ndarray, cancel_slack: int = 6,
) -> List[DetectionOutcome]:
    pos = {int(t): i for i, t in enumerate(step_index)}
    out: List[DetectionOutcome] = []
    for (etype, t0, t1, side, levels, mag) in events:
        rest = [pos[t] for t in range(t0, t1) if t in pos]
        canc = [pos[t] for t in range(t1, t1 + cancel_slack) if t in pos]
        rf = [i for i in rest if flags[i]]
        cf = [i for i in canc if flags[i]]
        allf = sorted(rf + cf)
        first = int(step_index[allf[0]]) if allf else None
        win = rest + canc
        ratios = scores[win] / np.where(np.isfinite(thresholds[win]), thresholds[win], np.inf)
        out.append(DetectionOutcome(
            etype=Event(etype), t_start=int(t0), t_end=int(t1),
            detected=bool(allf), first_flag=first,
            latency=(first - int(t0)) if first is not None else None,
            detected_while_resting=bool(rf), detected_at_cancel=bool(cf),
            peak_ratio=float(np.nanmax(ratios)) if len(win) else float("nan"),
        ))
    return out


class SpoofDetector:
    def __init__(self, mode: Literal["snapshot", "window"] = "window",
                 window: int = 5, code: int = 8, z: float = 4.0,
                 threshold_window: int = 600) -> None:
        self.mode = mode
        self.enc = LOBFeatureEncoder(window=window)
        self.code = code
        self.thresh = DynamicThreshold(window=threshold_window, z=z,
                                       method="mad", transform="log",
                                       lag=1, exclude_flagged=True)
        self.model: Optional[LOBAutoencoder] = None
        self._warm: Optional[np.ndarray] = None

    def _features(self, d) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == "snapshot":
            X = self.enc.snapshots(d)
            return X, np.arange(X.shape[0])
        return self.enc.windows(d)

    def fit(self, d_train: Dict[str, np.ndarray], epochs: int = 40, seed: int = 0) -> "SpoofDetector":
        self.enc.fit(d_train)
        X, _ = self._features(d_train)
        self.model = train_autoencoder(X, code=self.code, epochs=epochs, seed=seed)
        self._warm = self.model.score(X)[-self.thresh.window:]
        return self

    def detect(self, d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        assert self.model is not None
        X, idx = self._features(d)
        s = self.model.score(X)
        thr, flag = self.thresh.run(s, warm=self._warm)
        return {"score": s, "threshold": thr, "flag": flag, "index": idx, "X": X}


if __name__ == "__main__":
    print("Spoof Detector Core Module Loaded.")

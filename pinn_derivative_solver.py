"""
pinn_derivative_solver.py
=========================

Phase 15: Deep Galerkin / Physics-Informed Neural Network solver for the
Black-Scholes PDE.

    dV/dt + (1/2) sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0
    V(S, T) = payoff(S)

Two changes of variable are applied before any network sees the problem, and
both are load-bearing rather than cosmetic.

TIME REVERSAL.  tau = T - t turns the backward-parabolic terminal-value
problem into a forward-parabolic initial-value problem:

    dV/dtau = (1/2) sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V ,  V(S,0) = payoff

LOG-MONEYNESS.  x = log(S/K),  V = K v(x, tau).  Black-Scholes is homogeneous
of degree one in (S, K), so dividing by K is EXACT, not an approximation, and
the PDE becomes constant-coefficient:

    dv/dtau = (1/2) sigma^2 v_xx + ( r - q - (1/2) sigma^2 ) v_x - r v

In S-space the diffusion coefficient (1/2) sigma^2 S^2 varies by two orders of
magnitude across a typical domain; in x-space it is a constant. The S-space
derivatives the user actually wants -- Delta and Gamma -- are recovered exactly
by the chain rule:

    dV/dS   = v_x e^{-x}
    d2V/dS2 = ( v_xx - v_x ) e^{-2x} / K
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

torch.set_default_dtype(torch.float64)

_SQRT2 = math.sqrt(2.0)

__all__ = [
    "BSParams",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_price_torch",
    "PINN",
    "CollocationSampler",
    "pde_residual",
    "PINNTrainer",
    "TrainConfig",
    "SolverResult",
    "crank_nicolson",
]


# ===========================================================================
# 1. CLOSED FORM (reference)
# ===========================================================================

@dataclass(frozen=True)
class BSParams:
    sigma: float
    r: float
    q: float = 0.0
    K: float = 100.0
    T: float = 1.0
    option: Literal["call", "put"] = "call"

    def __post_init__(self) -> None:
        if self.sigma <= 0 or self.K <= 0 or self.T <= 0:
            raise ValueError("sigma, K, T must be positive")
        if self.option not in ("call", "put"):
            raise ValueError("option must be 'call' or 'put'")

    @property
    def x_max(self) -> float:
        """Domain half-width in log-moneyness: 4 standard deviations."""
        return max(4.0 * self.sigma * math.sqrt(self.T), 0.5)

    def payoff_norm(self, x: np.ndarray | torch.Tensor):
        """payoff / K as a function of x = log(S/K)."""
        if isinstance(x, torch.Tensor):
            e = torch.exp(x)
            return torch.clamp(e - 1.0, min=0.0) if self.option == "call" \
                else torch.clamp(1.0 - e, min=0.0)
        e = np.exp(x)
        return np.maximum(e - 1.0, 0.0) if self.option == "call" \
            else np.maximum(1.0 - e, 0.0)


def _d1d2(S, K, sigma, r, q, tau):
    tau = np.maximum(tau, 1e-14)
    vt = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / vt
    return d1, d1 - vt


def _ncdf(z):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / _SQRT2))


def bs_price(S, tau, p: BSParams):
    """Exact price at time-to-maturity tau. Handles tau = 0 as the payoff."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    S, tau = np.broadcast_arrays(S, tau)
    out = np.empty_like(S, dtype=float)

    at_exp = tau <= 1e-14
    if at_exp.any():
        intrinsic = np.maximum(S - p.K, 0.0) if p.option == "call" \
            else np.maximum(p.K - S, 0.0)
        out[at_exp] = intrinsic[at_exp]

    m = ~at_exp
    if m.any():
        d1, d2 = _d1d2(S[m], p.K, p.sigma, p.r, p.q, tau[m])
        df_r = np.exp(-p.r * tau[m])
        df_q = np.exp(-p.q * tau[m])
        if p.option == "call":
            out[m] = S[m] * df_q * _ncdf(d1) - p.K * df_r * _ncdf(d2)
        else:
            out[m] = p.K * df_r * _ncdf(-d2) - S[m] * df_q * _ncdf(-d1)
    return out


def bs_delta(S, tau, p: BSParams):
    S = np.atleast_1d(np.asarray(S, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    S, tau = np.broadcast_arrays(S, tau)
    d1, _ = _d1d2(S, p.K, p.sigma, p.r, p.q, tau)
    df_q = np.exp(-p.q * tau)
    return df_q * _ncdf(d1) if p.option == "call" else df_q * (_ncdf(d1) - 1.0)


def bs_gamma(S, tau, p: BSParams):
    S = np.atleast_1d(np.asarray(S, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    S, tau = np.broadcast_arrays(S, tau)
    d1, _ = _d1d2(S, p.K, p.sigma, p.r, p.q, tau)
    npdf = np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return np.exp(-p.q * tau) * npdf / (S * p.sigma * np.sqrt(np.maximum(tau, 1e-14)))


def bs_price_torch(x: torch.Tensor, tau: torch.Tensor, p: BSParams) -> torch.Tensor:
    tau_s = torch.clamp(tau, min=1e-12)
    vt = p.sigma * torch.sqrt(tau_s)
    d1 = (x + (p.r - p.q + 0.5 * p.sigma**2) * tau_s) / vt
    d2 = d1 - vt
    ncdf = lambda z: 0.5 * (1.0 + torch.erf(z / _SQRT2))
    df_r = torch.exp(-p.r * tau_s)
    df_q = torch.exp(-p.q * tau_s)
    if p.option == "call":
        return torch.exp(x) * df_q * ncdf(d1) - df_r * ncdf(d2)
    return df_r * ncdf(-d2) - torch.exp(x) * df_q * ncdf(-d1)


# ===========================================================================
# 2. NETWORK
# ===========================================================================

class PINN(nn.Module):
    def __init__(
        self,
        p: BSParams,
        width: int = 64,
        depth: int = 4,
        activation: Literal["tanh", "softplus", "sin"] = "tanh",
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.p = p
        self.x_scale = p.x_max
        self.t_scale = p.T
        self.activation_name = activation

        act = {"tanh": nn.Tanh, "softplus": lambda: nn.Softplus(beta=5.0)}.get(activation)
        self._sin = activation == "sin"

        layers: List[nn.Module] = []
        dims = [2] + [width] * depth + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 and act is not None:
                layers.append(act())
        self.net = nn.ModuleList([l for l in layers if isinstance(l, nn.Linear)])
        self.acts = [l for l in layers if not isinstance(l, nn.Linear)]

        for l in self.net:
            nn.init.xavier_normal_(l.weight, gain=1.0)
            nn.init.zeros_(l.bias)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x / self.x_scale, tau / self.t_scale], dim=1)
        for i, l in enumerate(self.net):
            h = l(h)
            if i < len(self.net) - 1:
                h = torch.sin(h) if self._sin else self.acts[i](h)
        return h

    def derivatives(
        self, x: torch.Tensor, tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.requires_grad_(True)
        tau = tau.requires_grad_(True)
        v = self.forward(x, tau)
        ones = torch.ones_like(v)
        v_x = grad(v, x, grad_outputs=ones, create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_tau = grad(v, tau, grad_outputs=ones, create_graph=True)[0]
        return v, v_x, v_xx, v_tau

    @torch.no_grad()
    def price(self, S: np.ndarray, tau: np.ndarray) -> np.ndarray:
        x, t = self._to_xt(S, tau)
        return (self.p.K * self.forward(x, t)).squeeze(-1).cpu().numpy()

    def greeks(self, S: np.ndarray, tau: np.ndarray) -> Dict[str, np.ndarray]:
        x, t = self._to_xt(S, tau)
        v, v_x, v_xx, v_tau = self.derivatives(x, t)
        e = torch.exp(x)
        delta = (v_x / e).detach().squeeze(-1).cpu().numpy()
        gamma = ((v_xx - v_x) / (e**2 * self.p.K)).detach().squeeze(-1).cpu().numpy()
        theta = (-self.p.K * v_tau).detach().squeeze(-1).cpu().numpy()
        return {
            "price": (self.p.K * v).detach().squeeze(-1).cpu().numpy(),
            "delta": delta, "gamma": gamma, "theta": theta,
        }

    def _to_xt(self, S, tau) -> Tuple[torch.Tensor, torch.Tensor]:
        S = np.atleast_1d(np.asarray(S, dtype=float))
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        S, tau = np.broadcast_arrays(S, tau)
        x = torch.tensor(np.log(S / self.p.K), dtype=torch.float64).reshape(-1, 1)
        t = torch.tensor(np.asarray(tau, dtype=float), dtype=torch.float64).reshape(-1, 1)
        return x, t


# ===========================================================================
# 3. PHYSICS
# ===========================================================================

def pde_residual(
    v: torch.Tensor, v_x: torch.Tensor, v_xx: torch.Tensor,
    v_tau: torch.Tensor, p: BSParams,
) -> torch.Tensor:
    s2 = p.sigma**2
    return v_tau - 0.5 * s2 * v_xx - (p.r - p.q - 0.5 * s2) * v_x + p.r * v


# ===========================================================================
# 4. SAMPLING
# ===========================================================================

class CollocationSampler:
    def __init__(
        self, p: BSParams, kink_frac: float = 0.15, tau_beta: float = 0.7,
        kink_width: float = 0.5, seed: int = 0,
    ) -> None:
        self.p = p
        self.kink_frac = float(kink_frac)
        self.tau_beta = float(tau_beta)
        self.kink_width = float(kink_width)
        self.g = torch.Generator().manual_seed(seed)

    def _u(self, n: int) -> torch.Tensor:
        return torch.rand(n, 1, generator=self.g, dtype=torch.float64)

    def interior(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.p
        tau = p.T * self._u(n).pow(1.0 / self.tau_beta)
        n_k = int(self.kink_frac * n)
        x_u = (2.0 * self._u(n - n_k) - 1.0) * p.x_max
        w = self.kink_width * p.sigma * torch.sqrt(torch.clamp(tau[:n_k], min=1e-6))
        x_k = torch.clamp(
            torch.randn(n_k, 1, generator=self.g, dtype=torch.float64) * w,
            -p.x_max, p.x_max,
        )
        x = torch.cat([x_k, x_u], dim=0)
        return x, tau

    def terminal(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.p
        n_k = int(0.15 * n)
        x_u = (2.0 * self._u(n - n_k) - 1.0) * p.x_max
        x_k = torch.clamp(
            torch.randn(n_k, 1, generator=self.g, dtype=torch.float64) * 0.05,
            -p.x_max, p.x_max,
        )
        x = torch.cat([x_k, x_u], dim=0)
        return x, torch.zeros_like(x)

    def boundary(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.p
        tau = p.T * self._u(n)
        lo = torch.full_like(tau, -p.x_max)
        hi = torch.full_like(tau, p.x_max)
        return lo, hi, tau, tau

    def boundary_values(self, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.p
        e_r = torch.exp(-p.r * tau)
        e_q = torch.exp(-p.q * tau)
        if p.option == "call":
            return torch.zeros_like(tau), math.exp(p.x_max) * e_q - e_r
        return e_r - math.exp(-p.x_max) * e_q, torch.zeros_like(tau)


# ===========================================================================
# 5. TRAINING
# ===========================================================================

@dataclass
class TrainConfig:
    n_interior: int = 1024
    n_terminal: int = 1024
    n_boundary: int = 256
    adam_iters: int = 6000
    adam_lr: float = 2e-3
    lbfgs_iters: int = 100
    lbfgs_batch: int = 6144
    reweight_every: int = 200
    w_terminal: float = 50.0
    w_boundary: float = 20.0
    adaptive_weights: bool = False
    log_every: int = 1500
    eval_every: int = 500
    seed: int = 0


@dataclass
class SolverResult:
    model: PINN
    history: Dict[str, List[float]]
    weights: Dict[str, float]
    seconds: float
    adam_final: float
    lbfgs_final: float


class PINNTrainer:
    def __init__(self, model: PINN, sampler: CollocationSampler, cfg: TrainConfig) -> None:
        self.model = model
        self.sampler = sampler
        self.p = model.p
        self.cfg = cfg
        self.w = {"pde": 1.0, "terminal": cfg.w_terminal, "boundary": cfg.w_boundary}
        self.history: Dict[str, List[float]] = {
            "total": [], "pde": [], "terminal": [], "boundary": [], "linf": []
        }

    def loss_terms(self, batch: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        c = self.cfg
        if batch is None:
            xi, ti = self.sampler.interior(c.n_interior)
            xt, tt = self.sampler.terminal(c.n_terminal)
            xlo, xhi, tlo, thi = self.sampler.boundary(c.n_boundary)
        else:
            xi, ti, xt, tt, xlo, xhi, tlo, thi = (
                batch[k] for k in ("xi", "ti", "xt", "tt", "xlo", "xhi", "tlo", "thi")
            )

        v, v_x, v_xx, v_tau = self.model.derivatives(xi.clone(), ti.clone())
        l_pde = pde_residual(v, v_x, v_xx, v_tau, self.p).pow(2).mean()

        v_t = self.model(xt, tt)
        l_term = (v_t - self.p.payoff_norm(xt)).pow(2).mean()

        blo, bhi = self.sampler.boundary_values(tlo)
        l_bnd = ((self.model(xlo, tlo) - blo).pow(2).mean()
                 + (self.model(xhi, thi) - bhi).pow(2).mean())

        return {"pde": l_pde, "terminal": l_term, "boundary": l_bnd}

    def total(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.w[k] * terms[k] for k in terms)

    def _rebalance(self, terms: Dict[str, torch.Tensor], alpha: float = 0.9) -> None:
        params = [q for q in self.model.parameters() if q.requires_grad]
        norms: Dict[str, float] = {}
        for k, L in terms.items():
            g = grad(L, params, retain_graph=True, allow_unused=True)
            flat = torch.cat([t.reshape(-1) for t in g if t is not None])
            norms[k] = float(flat.abs().mean())
        ref = max(norms["pde"], 1e-16)
        for k in ("terminal", "boundary"):
            if norms[k] > 1e-16:
                target = ref / norms[k] * self.w["pde"]
                self.w[k] = alpha * self.w[k] + (1 - alpha) * float(np.clip(target, 1e-3, 1e5))

    def train(self, verbose: bool = True) -> SolverResult:
        c = self.cfg
        t0 = time.time()

        opt = torch.optim.Adam(self.model.parameters(), lr=c.adam_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c.adam_iters,
                                                           eta_min=c.adam_lr * 0.02)
        for it in range(c.adam_iters):
            terms = self.loss_terms()
            if c.adaptive_weights and it > 0 and it % c.reweight_every == 0:
                self._rebalance(terms)
            loss = self.total(terms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            opt.step()
            sched.step()

            if it % c.eval_every == 0:
                self.history["total"].append(float(loss.detach()))
                for k in terms:
                    self.history[k].append(float(terms[k].detach()))
                self.history["linf"].append(self.max_error())
            if verbose and it % c.log_every == 0:
                print(f"    adam {it:5d}  total={float(loss.detach()):.3e}  "
                      f"pde={float(terms['pde'].detach()):.3e}  term={float(terms['terminal'].detach()):.3e}  "
                      f"bnd={float(terms['boundary'].detach()):.3e}  Linf={self.max_error():.4e}")
        adam_final = self.max_error()

        xi, ti = self.sampler.interior(c.lbfgs_batch)
        xt, tt = self.sampler.terminal(c.lbfgs_batch // 4)
        xlo, xhi, tlo, thi = self.sampler.boundary(c.lbfgs_batch // 8)
        batch = dict(xi=xi, ti=ti, xt=xt, tt=tt, xlo=xlo, xhi=xhi, tlo=tlo, thi=thi)

        lb = torch.optim.LBFGS(
            self.model.parameters(), lr=1.0, max_iter=c.lbfgs_iters,
            history_size=50, tolerance_grad=1e-12, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lb.zero_grad(set_to_none=True)
            L = self.total(self.loss_terms(batch))
            L.backward()
            return L

        if c.lbfgs_iters > 0:
            lb.step(closure)
        lbfgs_final = self.max_error()

        if verbose:
            print(f"    L-BFGS: Linf {adam_final:.4e} -> {lbfgs_final:.4e}  "
                  f"({adam_final/max(lbfgs_final,1e-16):.1f}x)")

        return SolverResult(
            model=self.model, history=self.history, weights=dict(self.w),
            seconds=time.time() - t0, adam_final=adam_final, lbfgs_final=lbfgs_final,
        )

    def max_error(self, n: int = 60) -> float:
        p = self.p
        x = np.linspace(-p.x_max * 0.9, p.x_max * 0.9, n)
        tau = np.linspace(0.02 * p.T, p.T, n)
        X, TT = np.meshgrid(x, tau, indexing="ij")
        S = p.K * np.exp(X)
        with torch.no_grad():
            pred = self.model.price(S.ravel(), TT.ravel())
        return float(np.max(np.abs(pred - bs_price(S.ravel(), TT.ravel(), p))))



# ===========================================================================
# 6. FINITE-DIFFERENCE REFERENCE
# ===========================================================================

def crank_nicolson(p: BSParams, n_x: int = 400, n_t: int = 400) -> Callable:
    import scipy.linalg as sla

    x = np.linspace(-p.x_max, p.x_max, n_x)
    dx = x[1] - x[0]
    dt = p.T / n_t
    a = 0.5 * p.sigma**2
    b = p.r - p.q - 0.5 * p.sigma**2

    lo = a / dx**2 - b / (2 * dx)
    di = -2.0 * a / dx**2 - p.r
    up = a / dx**2 + b / (2 * dx)

    n_i = n_x - 2
    th = 0.5
    A_l = np.full(n_i - 1, -th * dt * lo)
    A_d = np.full(n_i, 1.0 - th * dt * di)
    A_u = np.full(n_i - 1, -th * dt * up)
    ab = np.zeros((3, n_i))
    ab[0, 1:] = A_u
    ab[1, :] = A_d
    ab[2, :-1] = A_l

    v = p.payoff_norm(x)
    for k in range(n_t):
        tau_new = (k + 1) * dt
        blo, bhi = _fd_bcs(p, tau_new)
        vi = v[1:-1]
        rhs = vi + (1 - th) * dt * (lo * v[:-2] + di * vi + up * v[2:])
        rhs[0] += th * dt * lo * blo
        rhs[-1] += th * dt * up * bhi
        vi_new = sla.solve_banded((1, 1), ab, rhs)
        v = np.concatenate(([blo], vi_new, [bhi]))

    from scipy.interpolate import interp1d
    terminal = interp1d(x, p.payoff_norm(x), kind="linear", bounds_error=False,
                        fill_value="extrapolate")
    final = interp1d(x, v, kind="cubic", bounds_error=False, fill_value="extrapolate")

    def solve(S, tau):
        S = np.atleast_1d(np.asarray(S, dtype=float))
        xq = np.log(S / p.K)
        return p.K * final(xq)

    return solve


def _fd_bcs(p: BSParams, tau: float) -> Tuple[float, float]:
    e_r, e_q = math.exp(-p.r * tau), math.exp(-p.q * tau)
    if p.option == "call":
        return 0.0, math.exp(p.x_max) * e_q - e_r
    return e_r - math.exp(-p.x_max) * e_q, 0.0


# ===========================================================================
# SELF-TESTS
# ===========================================================================

_P = BSParams(sigma=0.2, r=0.03, q=0.01, K=100.0, T=1.0, option="call")
_CACHE: Dict[str, SolverResult] = {}


def _solve(p: BSParams = _P, key: str = "call", **kw) -> SolverResult:
    if key in _CACHE:
        return _CACHE[key]
    cfg = TrainConfig(**kw)
    r = PINNTrainer(PINN(p, seed=0), CollocationSampler(p, seed=1), cfg).train(verbose=True)
    _CACHE[key] = r
    return r


# -- closed form ------------------------------------------------------------

def _test_closed_form_is_self_consistent() -> None:
    p = _P
    S = np.linspace(60, 160, 40)
    tau = np.full_like(S, 0.5)
    c = bs_price(S, tau, p)
    put = BSParams(p.sigma, p.r, p.q, p.K, p.T, "put")
    pr = bs_price(S, tau, put)
    parity = c - pr - (S * np.exp(-p.q * tau) - p.K * np.exp(-p.r * tau))
    assert np.max(np.abs(parity)) < 1e-10, np.max(np.abs(parity))

    d = bs_delta(S, tau, p)
    assert np.all((d > 0) & (d < math.exp(-p.q * 0.5)))
    assert np.all(np.diff(d) > 0), "delta must increase in S"
    g = bs_gamma(S, tau, p)
    assert np.all(g > 0)
    fd = np.gradient(bs_price(S, tau, p), S)
    assert np.max(np.abs(fd - d)[2:-2]) < 2e-3
    print("[ok] closed form: put-call parity to 1e-10, Delta in (0, e^{-q tau}) "
          "and monotone, Gamma > 0, Delta == dV/dS by finite difference")


def _test_residual_of_exact_solution_is_zero() -> None:
    p = _P
    x = torch.linspace(-0.7, 0.7, 60, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    tau = torch.linspace(0.05, 1.0, 60, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    X, TT = torch.meshgrid(x.squeeze(), tau.squeeze(), indexing="ij")
    X = X.reshape(-1, 1).detach().requires_grad_(True)
    TT = TT.reshape(-1, 1).detach().requires_grad_(True)

    v = bs_price_torch(X, TT, p)
    ones = torch.ones_like(v)
    v_x = grad(v, X, ones, create_graph=True)[0]
    v_xx = grad(v_x, X, torch.ones_like(v_x), create_graph=True)[0]
    v_tau = grad(v, TT, ones, create_graph=True)[0]
    R = pde_residual(v, v_x, v_xx, v_tau, p)
    m = float(R.abs().max().detach())
    assert m < 1e-10, f"residual of the exact solution is {m:.3e}"

    wrong = BSParams(p.sigma * 1.05, p.r, p.q, p.K, p.T, p.option)
    Rw = pde_residual(v, v_x, v_xx, v_tau, wrong)
    mw = float(Rw.abs().max().detach())
    assert mw > 1e-4, "residual insensitive to sigma"
    print(f"[ok] PDE residual of the exact solution = {m:.2e} (machine zero); "
          f"perturbing sigma by 5% gives {mw:.2e}")


def _test_log_transform_conditioning() -> None:
    rows = []
    for sig, T in ((0.2, 1.0), (0.4, 2.0), (0.6, 5.0)):
        p = BSParams(sigma=sig, r=0.03, q=0.01, K=100.0, T=T)
        S = p.K * np.exp(np.linspace(-p.x_max, p.x_max, 400))
        coef = 0.5 * p.sigma**2 * S**2
        spread = float(coef.max() / coef.min())
        assert abs(spread - math.exp(4 * p.x_max)) / spread < 1e-6
        rows.append((sig, T, p.x_max, spread))
    assert rows[0][3] > 20.0
    assert rows[-1][3] > 1e4, rows[-1][3]
    print("[ok] diffusion coefficient spread = e^{4 x_max}, constant in x-space: "
          + ", ".join(f"sigma={s},T={t}: {r:,.0f}x" for s, t, _, r in rows))


# -- autograd machinery -----------------------------------------------------

def _test_relu_destroys_the_diffusion_term() -> None:
    p = _P
    relu = PINN(p, width=32, depth=3, seed=0)
    for i, l in enumerate(relu.net):
        pass
    relu.acts = [nn.ReLU() for _ in relu.acts]
    relu._sin = False

    x = torch.linspace(-0.5, 0.5, 200, dtype=torch.float64).reshape(-1, 1)
    tau = torch.full_like(x, 0.5)
    _, _, v_xx_relu, _ = relu.derivatives(x.clone(), tau.clone())
    assert float(v_xx_relu.abs().max()) < 1e-12, (
        f"ReLU second derivative is {float(v_xx_relu.abs().max()):.3e}, expected 0"
    )

    tanh_net = PINN(p, width=32, depth=3, activation="tanh", seed=0)
    _, _, v_xx_tanh, _ = tanh_net.derivatives(x.clone(), tau.clone())
    assert float(v_xx_tanh.abs().max()) > 1e-4
    print(f"[ok] ReLU gives v_xx == 0 (max {float(v_xx_relu.abs().max()):.1e}) so the "
          f"diffusion term vanishes; tanh gives {float(v_xx_tanh.abs().max()):.3e}")


def _test_autograd_matches_finite_differences() -> None:
    p = _P
    net = PINN(p, width=32, depth=3, seed=3)
    x = torch.linspace(-0.4, 0.4, 50, dtype=torch.float64).reshape(-1, 1)
    tau = torch.full_like(x, 0.4)
    v, v_x, v_xx, v_tau = net.derivatives(x.clone(), tau.clone())

    h = 1e-5
    with torch.no_grad():
        f = lambda a, b: net(a, b)
        fd_x = (f(x + h, tau) - f(x - h, tau)) / (2 * h)
        fd_xx = (f(x + h, tau) - 2 * f(x, tau) + f(x - h, tau)) / h**2
        fd_t = (f(x, tau + h) - f(x, tau - h)) / (2 * h)
    assert float((v_x - fd_x).abs().max()) < 1e-7
    assert float((v_xx - fd_xx).abs().max()) < 1e-4
    assert float((v_tau - fd_t).abs().max()) < 1e-7
    print("[ok] autograd v_x, v_xx, v_tau match central differences")


def _test_chain_rule_to_S_space() -> None:
    p = _P
    net = PINN(p, width=32, depth=3, seed=5)
    S = np.linspace(80.0, 130.0, 40)
    tau = np.full_like(S, 0.5)
    g = net.greeks(S, tau)

    h = 1e-4 * p.K
    with torch.no_grad():
        up = net.price(S + h, tau)
        dn = net.price(S - h, tau)
        mid = net.price(S, tau)
    fd_d = (up - dn) / (2 * h)
    fd_g = (up - 2 * mid + dn) / h**2
    assert np.max(np.abs(g["delta"] - fd_d)) < 1e-6, np.max(np.abs(g["delta"] - fd_d))
    assert np.max(np.abs(g["gamma"] - fd_g)) < 1e-4, np.max(np.abs(g["gamma"] - fd_g))
    print("[ok] chain rule to S-space matches finite differences of the price in S")


# -- sampling ---------------------------------------------------------------

def _test_sampler_concentrates_where_regularity_fails() -> None:
    p = _P
    s = CollocationSampler(p, kink_frac=0.35, tau_beta=0.6, seed=2)
    x, tau = s.interior(40_000)
    x, tau = x.numpy().ravel(), tau.numpy().ravel()
    assert x.min() >= -p.x_max - 1e-9 and x.max() <= p.x_max + 1e-9
    assert tau.min() >= 0 and tau.max() <= p.T + 1e-9

    near_kink = np.mean(np.abs(x) < 0.05)
    uniform_rate = 0.05 / p.x_max
    assert near_kink > 3 * uniform_rate, f"{near_kink:.4f} vs uniform {uniform_rate:.4f}"
    early = np.mean(tau < 0.05 * p.T)
    assert early > 0.05, early
    print(f"[ok] sampler concentration: {100*near_kink:.1f}% of points within "
          f"|x|<0.05 (uniform would give {100*uniform_rate:.1f}%); "
          f"{100*early:.1f}% within tau<0.05T")


def _test_boundary_truncation_error_is_quantified() -> None:
    p = _P
    tau = np.linspace(0.05, p.T, 50)
    S_lo = p.K * math.exp(-p.x_max)
    S_hi = p.K * math.exp(p.x_max)
    err_lo = np.max(np.abs(bs_price(np.full_like(tau, S_lo), tau, p) - 0.0))
    fwd = S_hi * np.exp(-p.q * tau) - p.K * np.exp(-p.r * tau)
    err_hi = np.max(np.abs(bs_price(np.full_like(tau, S_hi), tau, p) - fwd))
    floor = max(err_lo, err_hi)
    assert floor < 0.05 * p.K, floor
    print(f"[ok] truncation at x_max={p.x_max:.3f} (S in [{S_lo:.1f}, {S_hi:.1f}]): "
          f"lower BC error {err_lo:.3e}, upper {err_hi:.3e} -> accuracy floor "
          f"{floor:.3e} ({1e4*floor/p.K:.2f} bp of K)")


# -- training ---------------------------------------------------------------

def _test_training_converges_to_the_closed_form() -> None:
    r = _solve()
    p = _P
    S = p.K * np.exp(np.linspace(-0.6, 0.6, 80))
    for tau in (0.1, 0.25, 0.5, 1.0):
        tt = np.full_like(S, tau)
        pred = r.model.price(S, tt)
        exact = bs_price(S, tt, p)
        linf = np.max(np.abs(pred - exact))
        assert linf < 1.0, f"tau={tau}: Linf={linf:.4f}"
    rmse = float(np.sqrt(np.mean((r.model.price(S, np.full_like(S, 0.5))
                                  - bs_price(S, np.full_like(S, 0.5), p))**2)))
    assert r.lbfgs_final < 1.0, r.lbfgs_final
    print(f"[ok] converged in {r.seconds:.1f}s: global Linf={r.lbfgs_final:.4e} "
          f"({1e4*r.lbfgs_final/p.K:.1f} bp of K), RMSE at tau=0.5 is {rmse:.4e}")


def _test_lbfgs_improves_on_adam() -> None:
    r = _solve()
    assert r.lbfgs_final <= r.adam_final, (
        f"L-BFGS worsened the solution: {r.adam_final:.3e} -> {r.lbfgs_final:.3e}"
    )
    print(f"[ok] Adam -> L-BFGS: Linf {r.adam_final:.4e} -> {r.lbfgs_final:.4e} "
          f"({r.adam_final/max(r.lbfgs_final,1e-16):.2f}x)")


def _test_error_concentrates_at_the_kink() -> None:
    r = _solve()
    p = _P
    S = p.K * np.exp(np.linspace(-0.6, 0.6, 120))

    def linf(tau, mask=None):
        tt = np.full_like(S, tau)
        e = np.abs(r.model.price(S, tt) - bs_price(S, tt, p))
        return float(np.max(e[mask] if mask is not None else e))

    atm = np.abs(np.log(S / p.K)) < 0.06
    otm = np.abs(np.log(S / p.K)) > 0.3
    e_short_atm = linf(0.01, atm)
    e_long_atm = linf(1.0, atm)
    e_short_otm = linf(0.01, otm)
    assert e_short_atm > e_long_atm, "short-dated ATM should be the hardest region"
    assert e_short_atm > e_short_otm
    print(f"[ok] error concentrates at the kink: ATM tau=0.01 -> {e_short_atm:.4e}, "
          f"ATM tau=1.0 -> {e_long_atm:.4e}, OTM tau=0.01 -> {e_short_otm:.4e}")


def _test_greeks_accuracy() -> None:
    r = _solve()
    p = _P
    S = p.K * np.exp(np.linspace(-0.45, 0.45, 60))
    tau = np.full_like(S, 0.5)
    g = r.model.greeks(S, tau)

    d_err = float(np.max(np.abs(g["delta"] - bs_delta(S, tau, p))))
    g_ex = bs_gamma(S, tau, p)
    g_err = float(np.max(np.abs(g["gamma"] - g_ex)) / np.max(g_ex))
    p_err = float(np.max(np.abs(g["price"] - bs_price(S, tau, p))) / p.K)

    assert d_err < 0.15, f"delta Linf {d_err:.4f}"
    assert g_err < 0.80, f"relative gamma error {g_err:.4f}"
    assert p_err < d_err, "price should be more accurate than its derivative"
    print(f"[ok] at tau=0.5: price rel err {p_err:.2e}, Delta Linf {d_err:.2e}, "
          f"Gamma rel err {g_err:.2e} -- each differentiation loses roughly a decade")


def _test_solution_is_arbitrage_free() -> None:
    r = _solve()
    p = _P
    x = np.linspace(-0.5, 0.5, 200)
    S = p.K * np.exp(x)
    rows = []
    for tau in (0.25, 0.5, 1.0):
        tt = np.full_like(S, tau)
        v = r.model.price(S, tt)
        assert np.all(np.diff(v) > -1e-2), f"tau={tau}: price not monotone in S"

        g = r.model.greeks(S, tt)
        gam, d = g["gamma"], g["delta"]
        peak = float(gam.max())
        neg = gam < 0.0
        frac = float(np.mean(neg))

        assert gam.min() > -0.40 * peak, (
            f"tau={tau}: Gamma {gam.min():.4e} vs peak {peak:.4e}"
        )
        core = np.abs(x) < 0.15
        assert not neg[core].any(), f"tau={tau}: Gamma negative inside |x|<0.15"
        assert gam.min() > -0.15 * peak, (
            f"tau={tau}: Gamma {gam.min():.3e} is {100*abs(gam.min())/peak:.1f}% of peak"
        )
        assert d.min() > -0.10 and d.max() < 1.10, f"delta range [{d.min():.3f},{d.max():.3f}]"
        true_g = bs_gamma(S, tt, p)
        reach = float(np.max(true_g[neg]) / true_g.max()) if neg.any() else 0.0
        span = (float(np.abs(x[neg]).min()), float(np.abs(x[neg]).max())) if neg.any() else (0.0, 0.0)
        rows.append((tau, frac, gam.min(), peak, span, reach))

    print("[ok] monotone in S and Delta in [0,1] at every maturity; convexity is "
          "violated away from the money:")
    for tau, frac, gmin, peak, span, reach in rows:
        print(f"     tau={tau:.2f}: Gamma < 0 on {100*frac:4.1f}% of the grid, worst "
              f"{gmin:+.3e} = {100*abs(gmin)/peak:4.1f}% of peak, at |x| in "
              f"[{span[0]:.3f}, {span[1]:.3f}] where true Gamma still reaches "
              f"{100*reach:.0f}% of its own peak")


def _test_residual_small_on_trained_network() -> None:
    r = _solve()
    p = _P
    x = torch.linspace(-0.5, 0.5, 50, dtype=torch.float64).reshape(-1, 1)
    tau = torch.linspace(0.1, 1.0, 50, dtype=torch.float64).reshape(-1, 1)
    X, TT = torch.meshgrid(x.squeeze(), tau.squeeze(), indexing="ij")
    X = X.reshape(-1, 1).clone()
    TT = TT.reshape(-1, 1).clone()
    v, v_x, v_xx, v_tau = r.model.derivatives(X, TT)
    R = pde_residual(v, v_x, v_xx, v_tau, p).abs()
    assert float(R.mean().detach()) < 5e-2, float(R.mean().detach())
    print(f"[ok] trained residual on the interior: mean {float(R.mean().detach()):.3e}, "
          f"max {float(R.max().detach()):.3e}")


def _test_put_via_parity() -> None:
    p_put = BSParams(_P.sigma, _P.r, _P.q, _P.K, _P.T, "put")
    cfg = TrainConfig(adam_iters=1200, lbfgs_iters=60, log_every=10**9, eval_every=10**9)
    rp = PINNTrainer(PINN(p_put, seed=7), CollocationSampler(p_put, seed=8), cfg).train(
        verbose=False)
    rc = _solve()

    S = _P.K * np.exp(np.linspace(-0.4, 0.4, 60))
    tau = np.full_like(S, 0.5)
    c = rc.model.price(S, tau)
    pu = rp.model.price(S, tau)
    fwd = S * np.exp(-_P.q * tau) - _P.K * np.exp(-_P.r * tau)
    viol = float(np.max(np.abs(c - pu - fwd)))
    exact_p = float(np.max(np.abs(pu - bs_price(S, tau, p_put))))
    assert viol < 2.0, f"parity violation {viol:.4f}"
    print(f"[ok] independently trained put: Linf vs closed form {exact_p:.4e}; "
          f"put-call parity across the two networks violated by at most {viol:.4e} "
          f"({1e4*viol/_P.K:.1f} bp of K), never imposed in either loss")


def _test_adaptive_weights_move() -> None:
    r = _solve()
    w = r.weights
    assert all(np.isfinite(list(w.values())))
    p = _P
    cfg = TrainConfig(adam_iters=300, lbfgs_iters=0, adaptive_weights=True,
                      w_terminal=1.0, w_boundary=1.0, log_every=10**9, eval_every=10**9)
    ra = PINNTrainer(PINN(p, seed=11), CollocationSampler(p, seed=12), cfg).train(verbose=False)
    wa = ra.weights
    assert wa["terminal"] != 1.0 or wa["boundary"] != 1.0, "adaptive weights never moved"
    print(f"[ok] default fixed weights pde={w['pde']:.3g}, terminal={w['terminal']:.3g}, "
          f"boundary={w['boundary']:.3g}; adaptive path moves them to "
          f"terminal={wa['terminal']:.3g}, boundary={wa['boundary']:.3g} -- "
          f"down-weighting the Dirichlet data, which is why it is off by default")


def _test_finite_difference_benchmark() -> None:
    p = _P
    t0 = time.time()
    fd = crank_nicolson(p, n_x=800, n_t=800)
    fd_time = time.time() - t0

    S = p.K * np.exp(np.linspace(-0.7, 0.7, 300))
    tau_T = np.full_like(S, p.T)
    exact = bs_price(S, tau_T, p)
    fd_err = float(np.max(np.abs(fd(S, tau_T) - exact)))

    r = _solve()
    pinn_err = float(np.max(np.abs(r.model.price(S, tau_T) - exact)))

    assert fd_err < pinn_err, f"FD {fd_err:.2e} vs PINN {pinn_err:.2e}"
    assert fd_time < r.seconds, f"FD {fd_time:.2f}s vs PINN {r.seconds:.1f}s"
    print(f"[ok] at tau=T on the same domain: Crank-Nicolson Linf={fd_err:.3e} in "
          f"{fd_time:.2f}s, PINN Linf={pinn_err:.3e} in {r.seconds:.0f}s")
    print(f"     FD is {pinn_err/max(fd_err,1e-16):.0f}x more accurate and "
          f"{r.seconds/max(fd_time,1e-6):.0f}x faster -- in 1-D there is no case "
          f"for the PINN; its argument is dimensional, not numerical")


def _run(title: str, tests) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    for f in tests:
        f()
    print()


if __name__ == "__main__":
    torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
    _run("CLOSED FORM AND PHYSICS", [
        _test_closed_form_is_self_consistent,
        _test_residual_of_exact_solution_is_zero,
        _test_log_transform_conditioning,
    ])
    _run("AUTOGRAD", [
        _test_relu_destroys_the_diffusion_term,
        _test_autograd_matches_finite_differences,
        _test_chain_rule_to_S_space,
    ])
    _run("SAMPLING", [
        _test_sampler_concentrates_where_regularity_fails,
        _test_boundary_truncation_error_is_quantified,
    ])
    print("=" * 78)
    print("TRAINING")
    print("=" * 78)
    _solve()
    print()
    _run("SOLUTION QUALITY", [
        _test_training_converges_to_the_closed_form,
        _test_lbfgs_improves_on_adam,
        _test_error_concentrates_at_the_kink,
        _test_greeks_accuracy,
        _test_solution_is_arbitrage_free,
        _test_residual_small_on_trained_network,
        _test_adaptive_weights_move,
        _test_put_via_parity,
        _test_finite_difference_benchmark,
    ])
    print("all checks passed")

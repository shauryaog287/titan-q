"""
pinn_options_pricer.py
======================

Phase 15b: Deep Galerkin Method / PINN for the Heston stochastic volatility PDE.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^1_t
    dv_t = kappa ( theta - v_t ) dt + sigma sqrt(v_t) dW^2_t
    d<W^1, W^2>_t = rho dt

The pricing PDE for U(S, v, t):

    U_t + (1/2) v S^2 U_SS + rho sigma v S U_Sv + (1/2) sigma^2 v U_vv
        + (r - q) S U_S + kappa ( theta - v ) U_v - r U = 0
    U(S, v, T) = payoff(S)

Two transforms are applied before the network sees anything.

TIME REVERSAL  tau = T - t, turning a terminal-value problem into an
initial-value problem.

LOG-MONEYNESS  x = log(S/K), U = K u(x, v, tau). Heston is homogeneous of
degree one in (S, K), so dividing by K is exact. The PDE becomes

    u_tau = (1/2) v u_xx + rho sigma v u_xv + (1/2) sigma^2 v u_vv
          + ( r - q - (1/2) v ) u_x + kappa ( theta - v ) u_v - r u

The S-space diffusion coefficient (1/2) v S^2 spans orders of magnitude across
the domain; in x-space it is (1/2) v, independent of x.

WHY A MESH-FREE METHOD IS EVEN ARGUABLE HERE. In 1-D Black-Scholes it is not:
a tridiagonal Crank-Nicolson solve beats a PINN by four orders of magnitude in
accuracy and two in runtime. Heston is 2+1 dimensional, so an FD grid is
O(N_x N_v N_tau) and still entirely tractable -- ADI schemes solve this in
milliseconds. The honest position is that Heston is the LAST problem where the
grid still wins comfortably; the mesh-free argument only begins to bite at
three or more stochastic factors, where the grid is O(N^d). This module is
built so the comparison can be made rather than asserted: every result is
scored against the semi-analytic Lewis integral.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import integrate
from torch.autograd import grad

torch.set_default_dtype(torch.float64)

__all__ = [
    "HestonParams",
    "heston_cf",
    "heston_price",
    "bs_price",
    "DGMNet",
    "MLPNet",
    "HestonSampler",
    "heston_residual",
    "derivatives",
    "TrainConfig",
    "HestonPINNTrainer",
]


# ===========================================================================
# 1. MODEL PARAMETERS
# ===========================================================================

@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    r: float = 0.0
    q: float = 0.0
    K: float = 100.0
    T: float = 1.0
    option: Literal["call", "put"] = "call"

    def __post_init__(self) -> None:
        for n in ("v0", "kappa", "theta", "sigma", "T", "K"):
            if getattr(self, n) <= 0:
                raise ValueError(f"{n} must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    @property
    def feller(self) -> float:
        return 2.0 * self.kappa * self.theta - self.sigma**2

    @property
    def feller_ok(self) -> bool:
        return self.feller > 0.0

    @property
    def x_max(self) -> float:
        return max(4.0 * math.sqrt(self.theta * self.T), 0.5)

    @property
    def v_max(self) -> float:
        sd = math.sqrt(self.sigma**2 * self.theta / (2.0 * self.kappa))
        return max(self.theta + 6.0 * sd, 3.0 * self.v0, 2.0 * self.theta)

    def payoff_norm(self, x):
        if isinstance(x, torch.Tensor):
            e = torch.exp(x)
            return torch.clamp(e - 1.0, min=0.0) if self.option == "call" \
                else torch.clamp(1.0 - e, min=0.0)
        e = np.exp(x)
        return np.maximum(e - 1.0, 0.0) if self.option == "call" \
            else np.maximum(1.0 - e, 0.0)

    def describe(self) -> str:
        return (
            f"Heston(v0={self.v0:.4f}, kappa={self.kappa:.3f}, theta={self.theta:.4f}, "
            f"sigma={self.sigma:.3f}, rho={self.rho:+.3f}, r={self.r:.4f}, q={self.q:.4f})\n"
            f"  K={self.K:.1f} T={self.T:.2f} {self.option}  "
            f"Feller 2*k*th - s^2 = {self.feller:+.5f} "
            f"({'satisfied' if self.feller_ok else 'VIOLATED: v can reach 0'})\n"
            f"  domain: x in [{-self.x_max:.3f}, {self.x_max:.3f}] "
            f"(S in [{self.K*math.exp(-self.x_max):.1f}, {self.K*math.exp(self.x_max):.1f}]), "
            f"v in [0, {self.v_max:.4f}] (vol up to {100*math.sqrt(self.v_max):.1f}%)"
        )


# ===========================================================================
# 2. SEMI-ANALYTIC REFERENCE (Lewis)
# ===========================================================================

def heston_cf(u: complex, tau: float, v: float, p: HestonParams) -> complex:
    k, th, s, rho = p.kappa, p.theta, p.sigma, p.rho
    iu = 1j * u
    a = rho * s * iu - k
    d = np.sqrt(a * a + s * s * (iu + u * u))
    num = k - rho * s * iu - d
    den = k - rho * s * iu + d
    g = num / den
    e = np.exp(-d * tau)
    C = (k * th / (s * s)) * (num * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (num / (s * s)) * ((1.0 - e) / (1.0 - g * e))
    return np.exp(C + D * v)


def heston_price(S, tau, v, p: HestonParams, limit: int = 300) -> np.ndarray:
    S = np.atleast_1d(np.asarray(S, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    S, tau, v = np.broadcast_arrays(S, tau, v)
    out = np.empty(S.shape, dtype=float)

    flat = [S.ravel(), tau.ravel(), v.ravel()]
    res = np.empty(flat[0].size)
    for i in range(flat[0].size):
        Si, ti, vi = float(flat[0][i]), float(flat[1][i]), float(flat[2][i])
        if ti <= 1e-12:
            res[i] = max(Si - p.K, 0.0) if p.option == "call" else max(p.K - Si, 0.0)
            continue
        F = Si * math.exp((p.r - p.q) * ti)
        kk = math.log(F / p.K)

        def integrand(uu: float) -> float:
            phi = heston_cf(uu - 0.5j, ti, vi, p)
            return float(np.real(np.exp(1j * uu * kk) * phi) / (uu * uu + 0.25))

        val, _ = integrate.quad(integrand, 0.0, np.inf, limit=limit,
                                epsabs=1e-12, epsrel=1e-10)
        call = math.exp(-p.r * ti) * (F - math.sqrt(F * p.K) / math.pi * val)
        if p.option == "call":
            res[i] = call
        else:
            res[i] = call - Si * math.exp(-p.q * ti) + p.K * math.exp(-p.r * ti)
    out[...] = res.reshape(S.shape)
    return out


def bs_price(S, tau, vol, K, r, q, option="call") -> np.ndarray:
    from scipy.special import erf
    S = np.atleast_1d(np.asarray(S, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    S, tau = np.broadcast_arrays(S, tau)
    vt = vol * np.sqrt(np.maximum(tau, 1e-14))
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * tau) / vt
    d2 = d1 - vt
    N = lambda z: 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    c = S * np.exp(-q * tau) * N(d1) - K * np.exp(-r * tau) * N(d2)
    return c if option == "call" else c - S * np.exp(-q * tau) + K * np.exp(-r * tau)


# ===========================================================================
# 3. NETWORKS
# ===========================================================================

class DGMLayer(nn.Module):
    def __init__(self, d_in: int, width: int, act: Callable) -> None:
        super().__init__()
        self.act = act
        self.Uz, self.Ug, self.Ur, self.Uh = (nn.Linear(d_in, width, bias=False)
                                              for _ in range(4))
        self.Wz, self.Wg, self.Wr, self.Wh = (nn.Linear(width, width) for _ in range(4))
        for m in (self.Uz, self.Ug, self.Ur, self.Uh,
                  self.Wz, self.Wg, self.Wr, self.Wh):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, e: torch.Tensor, S: torch.Tensor, S1: torch.Tensor) -> torch.Tensor:
        Z = self.act(self.Uz(e) + self.Wz(S))
        G = self.act(self.Ug(e) + self.Wg(S1))
        R = self.act(self.Ur(e) + self.Wr(S))
        H = self.act(self.Uh(e) + self.Wh(S * R))
        return (1.0 - G) * H + Z * S


class _Base(nn.Module):
    def __init__(self, p: HestonParams) -> None:
        super().__init__()
        self.p = p
        self.x_scale = p.x_max
        self.v_scale = p.v_max
        self.t_scale = p.T

    def features(self, x, v, tau) -> torch.Tensor:
        return torch.cat([x / self.x_scale, v / self.v_scale, tau / self.t_scale], dim=1)


class DGMNet(_Base):
    def __init__(self, p: HestonParams, width: int = 64, n_layers: int = 3,
                 activation: str = "tanh", seed: int = 0) -> None:
        super().__init__(p)
        torch.manual_seed(seed)
        self.act = {"tanh": torch.tanh,
                    "softplus": lambda z: nn.functional.softplus(z, beta=5.0),
                    "relu": torch.relu}[activation]
        self.activation_name = activation
        self.inp = nn.Linear(3, width)
        self.layers = nn.ModuleList([DGMLayer(3, width, self.act) for _ in range(n_layers)])
        self.out = nn.Linear(width, 1)
        nn.init.xavier_normal_(self.inp.weight); nn.init.zeros_(self.inp.bias)
        nn.init.xavier_normal_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, v, tau) -> torch.Tensor:
        e = self.features(x, v, tau)
        S = self.act(self.inp(e))
        S1 = S
        for l in self.layers:
            S = l(e, S, S1)
        return self.out(S)


class MLPNet(_Base):
    def __init__(self, p: HestonParams, width: int = 64, n_layers: int = 4,
                 activation: str = "tanh", seed: int = 0) -> None:
        super().__init__(p)
        torch.manual_seed(seed)
        self.act = {"tanh": torch.tanh,
                    "softplus": lambda z: nn.functional.softplus(z, beta=5.0),
                    "relu": torch.relu}[activation]
        self.activation_name = activation
        dims = [3] + [width] * n_layers + [1]
        self.lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        for m in self.lin:
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, v, tau) -> torch.Tensor:
        h = self.features(x, v, tau)
        for i, l in enumerate(self.lin):
            h = l(h)
            if i < len(self.lin) - 1:
                h = self.act(h)
        return h


# ===========================================================================
# 4. DERIVATIVES AND RESIDUAL
# ===========================================================================

def derivatives(net: nn.Module, x, v, tau) -> Dict[str, torch.Tensor]:
    x = x.requires_grad_(True)
    v = v.requires_grad_(True)
    tau = tau.requires_grad_(True)
    u = net(x, v, tau)
    o = torch.ones_like(u)

    u_x, u_v, u_tau = grad(u, [x, v, tau], grad_outputs=o, create_graph=True)
    ox = torch.ones_like(u_x)
    u_xx, u_xv = grad(u_x, [x, v], grad_outputs=ox, create_graph=True)
    u_vv = grad(u_v, v, grad_outputs=torch.ones_like(u_v), create_graph=True)[0]

    return {"u": u, "u_x": u_x, "u_v": u_v, "u_tau": u_tau,
            "u_xx": u_xx, "u_xv": u_xv, "u_vv": u_vv}


def heston_residual(d: Dict[str, torch.Tensor], v: torch.Tensor,
                    p: HestonParams) -> torch.Tensor:
    return d["u_tau"] - (
        0.5 * v * d["u_xx"]
        + p.rho * p.sigma * v * d["u_xv"]
        + 0.5 * p.sigma**2 * v * d["u_vv"]
        + (p.r - p.q - 0.5 * v) * d["u_x"]
        + p.kappa * (p.theta - v) * d["u_v"]
        - p.r * d["u"]
    )


# ===========================================================================
# 5. SAMPLING
# ===========================================================================

class HestonSampler:
    def __init__(self, p: HestonParams, tau_beta: float = 0.7,
                 kink_frac: float = 0.12, seed: int = 0) -> None:
        self.p = p
        self.tau_beta = float(tau_beta)
        self.kink_frac = float(kink_frac)
        self.g = torch.Generator().manual_seed(seed)

    def _u(self, n: int, d: int = 1) -> torch.Tensor:
        return torch.rand(n, d, generator=self.g, dtype=torch.float64)

    def _x(self, n: int) -> torch.Tensor:
        p = self.p
        n_k = int(self.kink_frac * n)
        xu = (2.0 * self._u(n - n_k) - 1.0) * p.x_max
        xk = torch.clamp(torch.randn(n_k, 1, generator=self.g, dtype=torch.float64) * 0.06,
                         -p.x_max, p.x_max)
        return torch.cat([xk, xu], dim=0)

    def _v(self, n: int) -> torch.Tensor:
        p = self.p
        z = self._u(n).pow(1.6)
        return z * p.v_max

    def interior(self, n: int):
        tau = self.p.T * self._u(n).pow(1.0 / self.tau_beta)
        return self._x(n), self._v(n), tau

    def terminal(self, n: int):
        x = self._x(n)
        return x, self._v(n), torch.zeros_like(x)

    def x_boundary(self, n: int):
        p = self.p
        tau = p.T * self._u(n)
        v = self._v(n)
        lo = torch.full_like(tau, -p.x_max)
        hi = torch.full_like(tau, p.x_max)
        return lo, hi, v, tau

    def v_boundary(self, n: int):
        p = self.p
        tau = p.T * self._u(n)
        x = self._x(n)
        return x, torch.full_like(tau, p.v_max), tau

    def degenerate(self, n: int):
        p = self.p
        tau = p.T * self._u(n).pow(1.0 / self.tau_beta)
        x = self._x(n)
        return x, torch.zeros_like(tau), tau

    def boundary_values(self, tau: torch.Tensor):
        p = self.p
        e_r, e_q = torch.exp(-p.r * tau), torch.exp(-p.q * tau)
        if p.option == "call":
            return torch.zeros_like(tau), math.exp(p.x_max) * e_q - e_r
        return e_r - math.exp(-p.x_max) * e_q, torch.zeros_like(tau)


# ===========================================================================
# 6. TRAINING
# ===========================================================================

@dataclass
class TrainConfig:
    n_interior: int = 1024
    n_terminal: int = 1024
    n_xbound: int = 256
    n_vbound: int = 128
    n_degen: int = 256
    adam_iters: int = 6000
    adam_lr: float = 2e-3
    lbfgs_iters: int = 80
    lbfgs_batch: int = 4096
    w_terminal: float = 50.0
    w_xbound: float = 20.0
    w_vbound: float = 1.0
    w_degen: float = 1.0
    log_every: int = 1000
    eval_every: int = 1000


@dataclass
class Result:
    model: nn.Module
    seconds: float
    adam_linf: float
    final_linf: float
    history: Dict[str, List[float]] = field(default_factory=dict)


class HestonPINNTrainer:
    def __init__(self, net: nn.Module, sampler: HestonSampler, cfg: TrainConfig) -> None:
        self.net = net
        self.s = sampler
        self.p = sampler.p
        self.cfg = cfg
        self.history = {"total": [], "pde": [], "terminal": [], "xbnd": [],
                        "vbnd": [], "degen": [], "linf": []}
        self._ref: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    def loss_terms(self, batch: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        c, p = self.cfg, self.p
        b = batch or {}

        xi, vi, ti = b.get("int") or self.s.interior(c.n_interior)
        d = derivatives(self.net, xi.clone(), vi.clone(), ti.clone())
        l_pde = heston_residual(d, vi, p).pow(2).mean()

        xt, vt, tt = b.get("term") or self.s.terminal(c.n_terminal)
        l_term = (self.net(xt, vt, tt) - p.payoff_norm(xt)).pow(2).mean()

        xlo, xhi, vb, tb = b.get("xb") or self.s.x_boundary(c.n_xbound)
        blo, bhi = self.s.boundary_values(tb)
        l_xb = ((self.net(xlo, vb, tb) - blo).pow(2).mean()
                + (self.net(xhi, vb, tb) - bhi).pow(2).mean())

        xv, vv, tv = b.get("vb") or self.s.v_boundary(c.n_vbound)
        dv = derivatives(self.net, xv.clone(), vv.clone(), tv.clone())
        l_vb = dv["u_v"].pow(2).mean()

        xg, vg, tg = b.get("dg") or self.s.degenerate(c.n_degen)
        dg = derivatives(self.net, xg.clone(), vg.clone(), tg.clone())
        l_dg = heston_residual(dg, vg, p).pow(2).mean()

        return {"pde": l_pde, "terminal": l_term, "xbnd": l_xb,
                "vbnd": l_vb, "degen": l_dg}

    def total(self, t: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.cfg
        return (t["pde"] + c.w_terminal * t["terminal"] + c.w_xbound * t["xbnd"]
                + c.w_vbound * t["vbnd"] + c.w_degen * t["degen"])

    def _reference(self):
        if self._ref is None:
            p = self.p
            x = np.linspace(-0.75 * p.x_max, 0.75 * p.x_max, 13)
            v = np.linspace(0.3 * p.theta, min(2.5 * p.theta, p.v_max), 5)
            tau = np.array([0.25 * p.T, 0.6 * p.T, p.T])
            X, V, TT = np.meshgrid(x, v, tau, indexing="ij")
            S = p.K * np.exp(X)
            ref = heston_price(S.ravel(), TT.ravel(), V.ravel(), p)
            self._ref = (X.ravel(), V.ravel(), TT.ravel(), ref)
        return self._ref

    @torch.no_grad()
    def max_error(self) -> float:
        X, V, TT, ref = self._reference()
        pred = self.p.K * self.net(
            torch.tensor(X).reshape(-1, 1),
            torch.tensor(V).reshape(-1, 1),
            torch.tensor(TT).reshape(-1, 1),
        ).squeeze(-1).numpy()
        return float(np.max(np.abs(pred - ref)))

    def train(self, verbose: bool = True) -> Result:
        c = self.cfg
        t0 = time.time()
        opt = torch.optim.Adam(self.net.parameters(), lr=c.adam_lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.adam_iters,
                                                         eta_min=c.adam_lr * 0.02)
        for it in range(c.adam_iters):
            terms = self.loss_terms()
            loss = self.total(terms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            opt.step(); sch.step()

            if it % c.eval_every == 0:
                self.history["total"].append(float(loss.detach()))
                for k in terms:
                    self.history[k].append(float(terms[k].detach()))
                le = self.max_error()
                self.history["linf"].append(le)
                if verbose and it % c.log_every == 0:
                    print(f"    adam {it:5d}  total={float(loss.detach()):.3e}  "
                          f"pde={float(terms['pde'].detach()):.2e}  "
                          f"term={float(terms['terminal'].detach()):.2e}  "
                          f"xb={float(terms['xbnd'].detach()):.2e}  "
                          f"dg={float(terms['degen'].detach()):.2e}  Linf={le:.4f}")
        adam_linf = self.max_error()

        if c.lbfgs_iters > 0:
            batch = {
                "int": self.s.interior(c.lbfgs_batch),
                "term": self.s.terminal(c.lbfgs_batch // 2),
                "xb": self.s.x_boundary(c.lbfgs_batch // 8),
                "vb": self.s.v_boundary(c.lbfgs_batch // 16),
                "dg": self.s.degenerate(c.lbfgs_batch // 8),
            }
            lb = torch.optim.LBFGS(self.net.parameters(), lr=1.0,
                                   max_iter=c.lbfgs_iters, history_size=50,
                                   tolerance_grad=1e-12, tolerance_change=1e-14,
                                   line_search_fn="strong_wolfe")

            def closure():
                lb.zero_grad(set_to_none=True)
                L = self.total(self.loss_terms(batch))
                L.backward()
                return L

            lb.step(closure)

        final = self.max_error()
        if verbose:
            print(f"    L-BFGS: Linf {adam_linf:.4f} -> {final:.4f} ")
        return Result(self.net, time.time() - t0, adam_linf, final, self.history)


# ===========================================================================
# 7. INFERENCE
# ===========================================================================

@torch.no_grad()
def price_surface(net: nn.Module, S, tau, v, p: HestonParams) -> np.ndarray:
    S = np.atleast_1d(np.asarray(S, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    S, tau, v = np.broadcast_arrays(S, tau, v)
    x = torch.tensor(np.log(S / p.K), dtype=torch.float64).reshape(-1, 1)
    vv = torch.tensor(np.asarray(v, dtype=float), dtype=torch.float64).reshape(-1, 1)
    tt = torch.tensor(np.asarray(tau, dtype=float), dtype=torch.float64).reshape(-1, 1)
    return (p.K * net(x, vv, tt)).squeeze(-1).numpy().reshape(S.shape)


def greeks(net: nn.Module, S, tau, v, p: HestonParams) -> Dict[str, np.ndarray]:
    S = np.atleast_1d(np.asarray(S, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    S, tau, v = np.broadcast_arrays(S, tau, v)
    x = torch.tensor(np.log(S / p.K), dtype=torch.float64).reshape(-1, 1)
    vv = torch.tensor(np.asarray(v, dtype=float), dtype=torch.float64).reshape(-1, 1)
    tt = torch.tensor(np.asarray(tau, dtype=float), dtype=torch.float64).reshape(-1, 1)
    d = derivatives(net, x, vv, tt)
    e = torch.exp(x)
    f = lambda t: t.detach().squeeze(-1).numpy().reshape(S.shape)
    return {
        "price": f(p.K * d["u"]),
        "delta": f(d["u_x"] / e),
        "gamma": f((d["u_xx"] - d["u_x"]) / (e**2 * p.K)),
        "vega_v": f(p.K * d["u_v"]),
        "theta": f(-p.K * d["u_tau"]),
    }

if __name__ == "__main__":
    print("Heston PINN Options Pricer Core Module Loaded.")

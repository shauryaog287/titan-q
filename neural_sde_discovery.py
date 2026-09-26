"""
neural_sde_discovery.py
=======================

Phase 21: Neural stochastic differential equations for discovering market
drift and diffusion.

    dX_t = mu_phi(X_t, t) dt + sigma_psi(X_t, t) dW_t

with mu_phi and sigma_psi deep networks, trained on discretely observed
trajectories.

TWO WAYS TO TRAIN THIS WRONGLY, AND WHAT IS DONE INSTEAD
--------------------------------------------------------
1. PATHWISE MSE KILLS THE DIFFUSION. Simulating paths with Euler-Maruyama and
   minimising squared distance to the observed paths is the obvious reading of
   "match the observed trajectories", and it is wrong. The simulated Brownian
   increments are independent of the ones that produced the data, so

       E[ (Xhat - X)^2 ] = Var(Xhat) + Var(X) + ( E Xhat - E X )^2

   is minimised by Var(Xhat) = 0: the loss drives sigma to zero and the model
   collapses to the ODE of the conditional mean. Measured in the self-tests.

   Two correct objectives are built:

     TRANSITION LIKELIHOOD. Euler-Maruyama gives a Gaussian transition,
         X_{t+D} | X_t ~ N( X_t + mu(X_t) D,  sigma(X_t)^2 D ),
     so the negative log-likelihood trains drift and diffusion jointly, with no
     simulation at all. The default.

     DISTRIBUTIONAL MATCHING. Simulate forward from observed states with
     Euler-Maruyama, backpropagate through the solver, and minimise the MMD
     between simulated and observed (X_t, X_{t+kD}) pairs. Compares
     DISTRIBUTIONS, so it cannot be satisfied by suppressing noise.

2. EVERY DISCRETISED OBJECTIVE INHERITS THE DISCRETISATION. The Euler
   likelihood for an OU process is maximised at theta_EM = (1 - e^{-theta D})/D,
   not at theta. Whatever the network, it converges to the biased value. At
   fine sampling the bias is negligible; at coarse sampling it dominates.

WHAT CAN BE DISCOVERED AT ALL
-----------------------------
Diffusion is estimated from quadratic variation and sharpens as sampling gets
finer. Drift is not: the precision of a drift estimate is set by the SPAN of
the data, not its frequency -- Merton's observation, the same one that
governed the Kelly sizer in Phase 5. A neural SDE on market data will
therefore learn sigma(x) well and mu(x) poorly, and nothing about the
architecture changes that. Measured below.

GRADIENTS. Gradients are taken by backpropagation through the Euler-Maruyama
solver -- discretise-then-optimise -- which is exact for the discretised
objective and checked against finite differences. The stochastic adjoint
(Li et al. 2020) trades that exactness for O(1) memory in the number of steps;
at the horizons used here memory is not the constraint, so it is not used.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# NOTE: this module used to call torch.set_default_dtype(torch.float64) at import,
# which silently turned every model built afterwards (the Phase 18 transformer
# included) into float64. NeuralSDE is now float64 on its own; the global default
# is only set when the self-tests run (see __main__ below).

__all__ = [
    "simulate_ou", "simulate_sde", "ou_exact_mle", "ou_euler_mle",
    "NeuralSDE", "fit_transition_nll", "fit_simulation",
    "extract_ou", "transition_nll",
]


# ===========================================================================
# 1. GROUND TRUTH
# ===========================================================================

def simulate_ou(theta: float, m: float, sigma: float, dt: float, n: int,
                x0: Optional[float] = None, seed: int = 0) -> np.ndarray:
    """
    Exact discretisation of dX = theta (m - X) dt + sigma dW:

        X_{t+D} = m + (X_t - m) a + eps,   a = e^{-theta D},
        eps ~ N( 0, sigma^2 (1 - a^2) / (2 theta) )

    Exact in distribution at any step size -- no discretisation error enters
    the ground truth, so every bias measured downstream is the estimator's.
    """
    rng = np.random.default_rng(seed)
    a = math.exp(-theta * dt)
    s = sigma * math.sqrt((1.0 - a * a) / (2.0 * theta))
    x = np.empty(n + 1)
    x[0] = m + sigma / math.sqrt(2 * theta) * rng.standard_normal() if x0 is None else x0
    z = rng.standard_normal(n)
    for i in range(n):
        x[i + 1] = m + (x[i] - m) * a + s * z[i]
    return x


def simulate_sde(mu: Callable, sigma: Callable, x0: float, dt: float, n: int,
                 substeps: int = 20, seed: int = 0) -> np.ndarray:
    """Fine Euler-Maruyama for general dynamics, sampled every `substeps`
    internal steps. Used for nonlinear ground truth."""
    rng = np.random.default_rng(seed)
    h = dt / substeps
    x = np.empty(n + 1)
    x[0] = x0
    cur = float(x0)
    for i in range(n):
        for _ in range(substeps):
            cur = cur + mu(cur) * h + sigma(cur) * math.sqrt(h) * rng.standard_normal()
        x[i + 1] = cur
    return x


# ===========================================================================
# 2. CLASSICAL ESTIMATORS
# ===========================================================================

def ou_exact_mle(x: np.ndarray, dt: float) -> Dict[str, float]:
    """
    Exact conditional MLE for OU. Regress X_{t+1} = alpha + beta X_t + eps:

        theta = -ln(beta)/D,   m = alpha/(1 - beta),
        sigma^2 = 2 theta s^2 / (1 - beta^2)

    The estimator every neural SDE on OU data must be scored against.
    """
    x0, x1 = x[:-1], x[1:]
    A = np.column_stack([np.ones_like(x0), x0])
    (alpha, beta), *_ = np.linalg.lstsq(A, x1, rcond=None)
    resid = x1 - (alpha + beta * x0)
    s2 = float(resid.var())
    beta = float(np.clip(beta, 1e-9, 1 - 1e-12))
    theta = -math.log(beta) / dt
    return {"theta": theta, "m": float(alpha / (1 - beta)),
            "sigma": math.sqrt(2 * theta * s2 / (1 - beta * beta))}


def ou_euler_mle(x: np.ndarray, dt: float) -> Dict[str, float]:
    """
    Maximiser of the EULER pseudo-likelihood for OU, in closed form:

        theta_EM = (1 - beta)/D,   sigma_EM^2 = s^2 / D

    Converges to (1 - e^{-theta D})/D rather than theta. This is what any
    network trained on the Euler transition likelihood is converging to.
    """
    x0, x1 = x[:-1], x[1:]
    A = np.column_stack([np.ones_like(x0), x0])
    (alpha, beta), *_ = np.linalg.lstsq(A, x1, rcond=None)
    s2 = float((x1 - (alpha + beta * x0)).var())
    theta = (1.0 - float(beta)) / dt
    return {"theta": theta, "m": float(alpha / (1 - beta)), "sigma": math.sqrt(s2 / dt)}


# ===========================================================================
# 3. THE NEURAL SDE
# ===========================================================================

def _mlp(d_in: int, hidden: int, depth: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class NeuralSDE(nn.Module):
    """
    Drift and diffusion as separate networks of (x, t), inputs standardised by
    the data. sigma = softplus(.) + eps keeps the diffusion strictly positive,
    so the transition variance never degenerates.

    `time_dependent=False` drops t from the inputs. With a single observed
    path each t is visited ONCE, so a time-dependent drift is identified only
    through the network's smoothness prior; on stationary data it buys nothing
    and fits noise. Measured in the self-tests.
    """

    def __init__(self, hidden: int = 32, depth: int = 2, time_dependent: bool = False,
                 x_loc: float = 0.0, x_scale: float = 1.0, t_scale: float = 1.0,
                 seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.td = time_dependent
        d_in = 2 if time_dependent else 1
        self.f = _mlp(d_in, hidden, depth)
        self.g = _mlp(d_in, hidden, depth)
        self.x_loc, self.x_scale, self.t_scale = float(x_loc), float(x_scale), float(t_scale)
        # Always float64, independent of torch's global default dtype. Importing
        # this module no longer changes that default for everyone else.
        self.double()

    def _inp(self, x: torch.Tensor, t: Optional[torch.Tensor]) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float64)
        z = ((x - self.x_loc) / self.x_scale).reshape(-1, 1)
        if self.td:
            tt = (torch.zeros_like(z) if t is None
                  else (torch.as_tensor(t, dtype=torch.float64) / self.t_scale).reshape(-1, 1))
            z = torch.cat([z, tt], dim=1)
        return z

    def mu(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.f(self._inp(x, t)).reshape(x.shape)

    def sigma(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return (nn.functional.softplus(self.g(self._inp(x, t))) + 1e-4).reshape(x.shape)

    def simulate(self, x0: torch.Tensor, dt: float, n_steps: int,
                 t0: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None,
                 dW: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Euler-Maruyama from x0 (batch), differentiable end to end:

            X_{k+1} = X_k + mu(X_k, t_k) h + sigma(X_k, t_k) dW_k,  dW_k ~ N(0, h)

        Returns the path, shape (n_steps + 1, batch). Pass `dW` to fix the noise
        (for gradient checks) or a generator to make it reproducible.
        """
        x = x0
        t = torch.zeros_like(x0) if t0 is None else t0
        out = [x]
        for k in range(n_steps):
            if dW is not None:
                w = dW[k]
            else:
                w = torch.randn(x.shape, generator=generator) * math.sqrt(dt)
            x = x + self.mu(x, t) * dt + self.sigma(x, t) * w
            t = t + dt
            out.append(x)
        return torch.stack(out)


# ===========================================================================
# 4. OBJECTIVES AND TRAINING
# ===========================================================================

def transition_nll(model: NeuralSDE, x0: torch.Tensor, x1: torch.Tensor, dt: float,
                   t0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean negative Euler-Maruyama transition log-likelihood:

        0.5 log( 2 pi sigma^2 D ) + ( x1 - x0 - mu D )^2 / ( 2 sigma^2 D )

    Drift enters only through the mean, diffusion through the variance, so the
    two are identified by different features of the data: the drift by the
    conditional mean of the increment, the diffusion by its conditional spread.
    """
    mu = model.mu(x0, t0)
    var = model.sigma(x0, t0) ** 2 * dt
    r = x1 - x0 - mu * dt
    return (0.5 * torch.log(2 * math.pi * var) + r * r / (2 * var)).mean()


def fit_transition_nll(x: np.ndarray, dt: float, steps: int = 1500, lr: float = 1e-2,
                       hidden: int = 32, time_dependent: bool = False, seed: int = 0,
                       t: Optional[np.ndarray] = None, batch: int = 2048) -> NeuralSDE:
    """
    Minimise the Euler transition NLL by mini-batch Adam. The loss is a mean over
    transitions, so a random mini-batch gives an unbiased gradient; full-batch
    steps on 20-30k transitions cost ~35 ms each on one CPU core, mini-batches
    of 2,048 about a tenth of that, and the cosine decay to a small learning rate
    removes the mini-batch noise by the end of training.
    """
    xt = torch.from_numpy(np.asarray(x, dtype=float))
    x0, x1 = xt[:-1], xt[1:]
    tt = None
    if time_dependent:
        tv = np.arange(x.size - 1) * dt if t is None else t[:-1]
        tt = torch.from_numpy(np.asarray(tv, dtype=float))
    model = NeuralSDE(hidden=hidden, time_dependent=time_dependent,
                      x_loc=float(x.mean()), x_scale=float(x.std()),
                      t_scale=float(max(x.size * dt, 1.0)), seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    g = torch.Generator().manual_seed(seed)
    n = x0.numel()
    b = min(batch, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (b,), generator=g)
        loss = transition_nll(model, x0[idx], x1[idx], dt,
                              None if tt is None else tt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    return model


def _mmd2(a: torch.Tensor, b: torch.Tensor, bandwidths: Sequence[float]) -> torch.Tensor:
    """Unbiased MMD^2 with a mixture of Gaussian kernels."""
    def k(u, v):
        d2 = torch.cdist(u, v).pow(2)
        return sum(torch.exp(-d2 / (2 * h * h)) for h in bandwidths)
    n, m = a.shape[0], b.shape[0]
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    return ((kaa.sum() - kaa.diagonal().sum()) / (n * (n - 1))
            + (kbb.sum() - kbb.diagonal().sum()) / (m * (m - 1))
            - 2 * kab.mean())


def fit_simulation(x: np.ndarray, dt: float, loss: str = "mmd", horizon: int = 4,
                   substeps: int = 2, iters: int = 500, batch: int = 384,
                   lr: float = 1e-2, seed: int = 0) -> NeuralSDE:
    """
    Train by SIMULATING: start from observed states X_t, integrate `horizon`
    observation steps forward with Euler-Maruyama (`substeps` internal steps
    each), backpropagate through the solver, and compare with the observed
    X_{t+horizon}.

      loss = "mse"  pathwise squared error -- collapses sigma (see module doc)
      loss = "mmd"  MMD between simulated and observed (X_t, X_{t+horizon}) pairs,
                    which compares distributions and so must reproduce the spread
    """
    xt = torch.from_numpy(np.asarray(x, dtype=float))
    n = xt.numel() - horizon
    model = NeuralSDE(x_loc=float(x.mean()), x_scale=float(x.std()), seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    g = torch.Generator().manual_seed(seed)
    scale = float(x.std())
    bw = [0.25 * scale, 0.5 * scale, 1.0 * scale, 2.0 * scale]
    h = dt / substeps
    for _ in range(iters):
        idx = torch.randint(0, n, (batch,), generator=g)
        start, target = xt[idx], xt[idx + horizon]
        end = model.simulate(start, h, horizon * substeps, generator=g)[-1]
        if loss == "mse":
            L = ((end - target) ** 2).mean()
        else:
            L = _mmd2(torch.stack([start, end], 1), torch.stack([start, target], 1), bw)
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()
        sch.step()
    return model


# ===========================================================================
# 5. READING PARAMETERS BACK OUT
# ===========================================================================

@torch.no_grad()
def extract_ou(model: NeuralSDE, x: np.ndarray, q: Tuple[float, float] = (0.05, 0.95)
               ) -> Dict[str, float]:
    """
    Recover (theta, m, sigma) from a trained network by projecting it onto the
    OU family over the data's central range: least-squares fit of
    mu_net(x) = theta (m - x), and the mean of sigma_net over the observed
    states. Restricting to the data's support matters -- outside it the network
    is extrapolating and its values carry no information.
    """
    lo, hi = np.quantile(x, q)
    grid = torch.linspace(float(lo), float(hi), 200)
    mu = model.mu(grid).numpy()
    A = np.column_stack([np.ones(200), grid.numpy()])
    (a, b), *_ = np.linalg.lstsq(A, mu, rcond=None)
    theta = -float(b)
    sig = float(model.sigma(torch.from_numpy(np.asarray(x, dtype=float))).mean())
    return {"theta": theta, "m": float(a) / theta if theta != 0 else float("nan"),
            "sigma": sig}


# ===========================================================================
# SELF-TESTS
# ===========================================================================

TRUE = {"theta": 2.0, "m": 0.5, "sigma": 0.3}


def _test_ou_simulator_is_exact() -> None:
    th, m, s, dt = TRUE["theta"], TRUE["m"], TRUE["sigma"], 0.05
    xs = np.stack([simulate_ou(th, m, s, dt, 4000, seed=k) for k in range(40)])
    var = float(xs.var())
    x0, x1 = xs[:, :-1].ravel() - m, xs[:, 1:].ravel() - m
    rho = float(np.dot(x0, x1) / np.dot(x0, x0))
    assert abs(var / (s * s / (2 * th)) - 1) < 0.03, var
    assert abs(rho - math.exp(-th * dt)) < 0.005, rho
    print(f"[ok] exact OU simulator: stationary variance {var:.5f} vs sigma^2/2theta "
          f"{s*s/(2*th):.5f}; lag-1 autocorrelation {rho:.4f} vs e^(-theta D) "
          f"{math.exp(-th*dt):.4f}")


def _test_gradient_through_euler_maruyama() -> None:
    """Backprop through the solver is exact for the discretised objective:
    autograd must match central finite differences with the noise held fixed."""
    torch.manual_seed(0)
    model = NeuralSDE(hidden=8, seed=3)
    x0 = torch.linspace(-1, 1, 16)
    dW = torch.randn(12, 16) * math.sqrt(0.05)
    loss = lambda: (model.simulate(x0, 0.05, 12, dW=dW)[-1] ** 2).mean()
    L = loss()
    L.backward()
    worst = 0.0
    for p in list(model.parameters())[:4]:
        flat = p.data.view(-1)
        for i in range(min(4, flat.numel())):
            h = 1e-6
            old = float(flat[i])
            flat[i] = old + h
            up = float(loss().detach())
            flat[i] = old - h
            dn = float(loss().detach())
            flat[i] = old
            fd = (up - dn) / (2 * h)
            ag = float(p.grad.view(-1)[i])
            worst = max(worst, abs(fd - ag) / max(abs(fd), 1e-8))
    assert worst < 1e-5, worst
    print(f"[ok] gradient through 12 Euler-Maruyama steps matches finite differences "
          f"(max relative error {worst:.1e})")


def _test_pathwise_mse_collapses_the_diffusion() -> None:
    """
    Same simulator, same data, same gradient path -- only the loss differs.
    Pathwise MSE drives sigma toward zero; MMD recovers it.
    """
    th, m, s, dt = TRUE["theta"], TRUE["m"], TRUE["sigma"], 0.05
    x = simulate_ou(th, m, s, dt, 8000, seed=1)
    mse = fit_simulation(x, dt, loss="mse", iters=400, seed=0)
    mmd = fit_simulation(x, dt, loss="mmd", iters=400, seed=0)
    s_mse = extract_ou(mse, x)["sigma"]
    s_mmd = extract_ou(mmd, x)["sigma"]
    assert s_mse < 0.25 * s, f"MSE sigma {s_mse:.4f} did not collapse"
    assert abs(s_mmd / s - 1) < 0.25, f"MMD sigma {s_mmd:.4f} vs {s}"
    print(f"[ok] Euler-Maruyama + autograd, true sigma = {s}: pathwise MSE learns "
          f"sigma = {s_mse:.4f} ({100*s_mse/s:.0f}% of truth -- collapsed to the "
          f"mean ODE); MMD learns {s_mmd:.4f}")


def _test_recovers_ou_from_transition_likelihood() -> None:
    th, m, s, dt = TRUE["theta"], TRUE["m"], TRUE["sigma"], 0.02
    x = simulate_ou(th, m, s, dt, 20_000, seed=2)
    net = extract_ou(fit_transition_nll(x, dt, seed=0), x)
    mle, eu = ou_exact_mle(x, dt), ou_euler_mle(x, dt)
    for k in ("theta", "m", "sigma"):
        assert abs(net[k] - mle[k]) < 0.12 * abs(mle[k]) + 0.02, (k, net[k], mle[k])
    print(f"[ok] neural SDE on the Euler likelihood, OU with theta={th}, m={m}, "
          f"sigma={s}, D={dt}, T={20_000*dt:.0f}:")
    print(f"       {'':8}{'truth':>9}{'exact MLE':>11}{'Euler MLE':>11}{'neural':>9}")
    for k in ("theta", "m", "sigma"):
        print(f"       {k:8}{TRUE[k]:>9.4f}{mle[k]:>11.4f}{eu[k]:>11.4f}{net[k]:>9.4f}")
    print(f"     diffusion recovered to {100*abs(net['sigma']/mle['sigma']-1):.1f}% of the "
          f"MLE; drift shrunk {100*(1-net['theta']/eu['theta']):.1f}% below the Euler "
          f"optimum it should reach -- see the overfitting test")


def _test_discretisation_bias_is_inherited() -> None:
    """
    At coarse sampling the Euler likelihood's optimum is theta_EM =
    (1 - e^{-theta D})/D, and the network converges there -- not to theta.
    """
    th, m, s, dt = TRUE["theta"], TRUE["m"], TRUE["sigma"], 0.2
    x = simulate_ou(th, m, s, dt, 30_000, seed=3)
    net = extract_ou(fit_transition_nll(x, dt, seed=0), x)
    exact, euler = ou_exact_mle(x, dt), ou_euler_mle(x, dt)
    limit = (1 - math.exp(-th * dt)) / dt
    assert abs(net["theta"] - euler["theta"]) < abs(net["theta"] - exact["theta"])
    assert abs(euler["theta"] - limit) < 0.06
    print(f"[ok] coarse sampling D={dt}: exact MLE theta = {exact['theta']:.3f} "
          f"(truth {th}); Euler pseudo-MLE {euler['theta']:.3f} (limit "
          f"(1-e^-thD)/D = {limit:.3f}); neural {net['theta']:.3f} -- the network "
          f"inherits the {100*(1-limit/th):.0f}% Euler bias")


def _test_drift_needs_span_diffusion_needs_frequency() -> None:
    """
    Monte Carlo of the EXACT MLE -- the best-case estimator, so if it cannot
    pin the drift no network can. Refining the grid at fixed span sharpens sigma
    and leaves theta alone; lengthening the span sharpens theta.
    """
    th, m, s = TRUE["theta"], TRUE["m"], TRUE["sigma"]

    def spread(T, dt, reps=60):
        ests = [ou_exact_mle(simulate_ou(th, m, s, dt, int(T / dt), seed=100 + r), dt)
                for r in range(reps)]
        return (float(np.std([e["theta"] for e in ests]) / th),
                float(np.std([e["sigma"] for e in ests]) / s))

    rows = {("T=20", "D=0.1"): spread(20, 0.1), ("T=20", "D=0.01"): spread(20, 0.01),
            ("T=200", "D=0.1"): spread(200, 0.1)}
    th_c, s_c = rows[("T=20", "D=0.1")]
    th_f, s_f = rows[("T=20", "D=0.01")]
    th_l, s_l = rows[("T=200", "D=0.1")]
    assert s_f < 0.5 * s_c, "finer sampling should sharpen sigma"
    assert th_f > 0.6 * th_c, "finer sampling should NOT materially sharpen theta"
    assert th_l < 0.5 * th_c, "longer span should sharpen theta"
    print("[ok] relative standard error of the exact MLE, 60 replicates:")
    print(f"       {'span':>6} {'step':>7} {'theta':>8} {'sigma':>8}")
    for (T, D), (a, b) in rows.items():
        print(f"       {T:>6} {D:>7} {100*a:>7.1f}% {100*b:>7.1f}%")
    print(f"     10x finer sampling: sigma error / {s_c/s_f:.1f}, theta error / "
          f"{th_c/th_f:.1f}. 10x longer span: theta error / {th_c/th_l:.1f}")


def _test_flexible_drift_overfits_weak_signal() -> None:
    """
    The drift network does NOT converge to the correct linear drift. Per step the drift contributes
    theta (m - x) D ~ 0.006 against a noise of sigma sqrt(D) ~ 0.04, so the
    transition likelihood is dominated by the diffusion and barely registers
    the drift. A flexible network spends that slack fitting noise as curvature
    in mu(x): it beats the correctly specified linear model on the TRAINING
    likelihood and loses to it OUT OF SAMPLE, while the extracted theta is
    dragged below the Euler optimum. Whether the damage grows monotonically
    with training steps is not asserted: under mini-batch noise it was
    visible at 8,000 steps but not reliably at 6,000.

    The size of the likelihood differences -- a few 1e-4 nats per transition --
    is the finding, not a detail: that is how little the data says about drift.
    """
    dt = 0.02
    x = simulate_ou(TRUE["theta"], TRUE["m"], TRUE["sigma"], dt, 25_000, seed=2)
    tr, te = x[:20_000], x[20_000:]
    eu = ou_euler_mle(tr, dt)

    def lin_nll(d):
        mu = eu["theta"] * (eu["m"] - d[:-1])
        v = eu["sigma"] ** 2 * dt
        r = d[1:] - d[:-1] - mu * dt
        return float(np.mean(0.5 * np.log(2 * math.pi * v) + r * r / (2 * v)))

    def nll(model, d):
        t = torch.from_numpy(d)
        with torch.no_grad():
            return float(transition_nll(model, t[:-1], t[1:], dt))

    lo, hi = np.quantile(tr, [0.05, 0.95])
    g = torch.linspace(float(lo), float(hi), 200)
    A = np.column_stack([np.ones(200), g.numpy()])
    rows = []
    for steps in (1500, 6000):
        mdl = fit_transition_nll(tr, dt, steps=steps, seed=0)
        with torch.no_grad():
            mu = mdl.mu(g).numpy()
        c, *_ = np.linalg.lstsq(A, mu, rcond=None)
        rows.append((steps, nll(mdl, tr), nll(mdl, te), float(np.std(mu - A @ c)),
                     extract_ou(mdl, tr)["theta"]))
    lt, lh = lin_nll(tr), lin_nll(te)
    (s1, tr1, te1, cu1, th1), (s2, tr2, te2, cu2, th2) = rows
    assert tr2 < lt, "network should fit the training data better than the linear model"
    assert te2 > lh, "network should generalise worse than the linear model"
    assert min(cu1, cu2) > 0.005, "network should show spurious curvature where the truth has none"
    print("[ok] flexible drift overfits a weak signal (true drift is exactly linear):")
    print(f"       {'model':<24}{'train NLL':>12}{'held-out NLL':>14}{'curvature':>11}{'theta':>8}")
    print(f"       {'linear Euler MLE':<24}{lt:>12.6f}{lh:>14.6f}{0.0:>11.4f}{eu['theta']:>8.3f}")
    for st, a, b, c_, t_ in rows:
        print(f"       {f'neural, {st} steps':<24}{a:>12.6f}{b:>14.6f}{c_:>11.4f}{t_:>8.3f}")
    print(f"     the whole drift signal is worth {1e4*abs(lh - te2):.1f}e-4 nats per "
          f"transition out of sample: validate on held-out data, and prefer a "
          f"structured drift where the physics supplies one")


def _test_extrapolation_outside_the_support() -> None:
    """The drift is identified only where the process has been. Outside the
    visited range the network extrapolates and its error grows."""
    th, m, s, dt = TRUE["theta"], TRUE["m"], TRUE["sigma"], 0.02
    x = simulate_ou(th, m, s, dt, 20_000, seed=4)
    model = fit_transition_nll(x, dt, seed=0)
    lo, hi = np.quantile(x, [0.01, 0.99])
    width = hi - lo
    inside = torch.linspace(float(lo), float(hi), 200)
    outside = torch.cat([torch.linspace(float(lo) - 3 * width, float(lo) - width, 100),
                         torch.linspace(float(hi) + width, float(hi) + 3 * width, 100)])
    with torch.no_grad():
        e_in = float((model.mu(inside) - th * (m - inside)).abs().mean())
        e_out = float((model.mu(outside) - th * (m - outside)).abs().mean())
    scale_in = float((th * (m - inside)).abs().mean())
    scale_out = float((th * (m - outside)).abs().mean())
    assert e_out / scale_out > 2 * e_in / scale_in
    print(f"[ok] drift relative error inside the 1-99% data range "
          f"{100*e_in/scale_in:.1f}%, outside it {100*e_out/scale_out:.1f}% -- the "
          f"network knows the dynamics only where the path has been")


def _test_neural_earns_its_keep_on_nonlinear_dynamics() -> None:
    """
    A double-well drift with a leverage-style diffusion that rises as the
    state falls. OU cannot represent either feature; the neural SDE recovers
    both and wins on held-out likelihood. On TRUE OU data it would not.
    """
    mu_true = lambda z: 1.5 * (z - z**3)
    sg_true = lambda z: 0.40 - 0.18 * math.tanh(z)
    dt = 0.02
    x = simulate_sde(mu_true, sg_true, 1.0, dt, 30_000, substeps=10, seed=5)
    tr, te = x[:24_000], x[24_000:]
    model = fit_transition_nll(tr, dt, steps=2000, seed=0)
    ou = ou_exact_mle(tr, dt)

    xt = torch.from_numpy(te)
    with torch.no_grad():
        nll_net = float(transition_nll(model, xt[:-1], xt[1:], dt))
    a = math.exp(-ou["theta"] * dt)
    v = ou["sigma"] ** 2 * (1 - a * a) / (2 * ou["theta"])
    pred = ou["m"] + (te[:-1] - ou["m"]) * a
    nll_ou = float(np.mean(0.5 * np.log(2 * math.pi * v) + (te[1:] - pred) ** 2 / (2 * v)))

    lo, hi = np.quantile(tr, [0.05, 0.95])
    grid = np.linspace(lo, hi, 200)
    with torch.no_grad():
        g = torch.from_numpy(grid)
        mu_n, sg_n = model.mu(g).numpy(), model.sigma(g).numpy()
    mu_t = np.array([mu_true(z) for z in grid])
    sg_t = np.array([sg_true(z) for z in grid])
    corr_mu = float(np.corrcoef(mu_n, mu_t)[0, 1])
    corr_sg = float(np.corrcoef(sg_n, sg_t)[0, 1])
    assert nll_net < nll_ou, (nll_net, nll_ou)
    assert corr_sg > 0.9, corr_sg
    print(f"[ok] double-well drift + leverage diffusion, held-out transition NLL: "
          f"neural {nll_net:.4f} vs OU {nll_ou:.4f} (lower is better)")
    print(f"     correlation with the true functions on the 5-95% range: "
          f"drift {corr_mu:.3f}, diffusion {corr_sg:.3f}")


def _test_time_dependence_overfits_stationary_data() -> None:
    th, m, s, dt = TRUE["theta"], TRUE["m"], TRUE["sigma"], 0.02
    x = simulate_ou(th, m, s, dt, 15_000, seed=6)
    tr, te = x[:12_000], x[12_000:]
    t_all = np.arange(x.size) * dt
    homo = fit_transition_nll(tr, dt, seed=0)
    td = fit_transition_nll(tr, dt, seed=0, time_dependent=True, t=t_all[:12_000])
    xt = torch.from_numpy(te)
    tt = torch.from_numpy(t_all[12_000:-1])
    with torch.no_grad():
        n_h = float(transition_nll(homo, xt[:-1], xt[1:], dt))
        n_t = float(transition_nll(td, xt[:-1], xt[1:], dt, tt))
    assert n_t >= n_h - 1e-3, (n_h, n_t)
    print(f"[ok] stationary OU, held-out NLL: time-homogeneous {n_h:.4f}, "
          f"time-dependent {n_t:.4f} -- with one path each t is seen once, and "
          f"the extra input only fits noise")


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

def _main(path: str = "D:/Quant_Project/neural_sde_ou.png") -> None:
    """
    Recover a hidden Ornstein-Uhlenbeck process from raw scatter data.

    The network is never told the process is OU. It sees the scatter of
    increments dX against X and must discover that the drift is linear and
    mean-reverting and the diffusion constant.
    """
    print("=" * 78)
    print("MAIN: DISCOVERING A HIDDEN ORNSTEIN-UHLENBECK PROCESS")
    print("=" * 78)
    th, m, s, dt = TRUE["theta"], TRUE["m"], TRUE["sigma"], 0.02
    n = 20_000
    x = simulate_ou(th, m, s, dt, n, seed=7)
    print(f"  hidden:  dX = {th} ({m} - X) dt + {s} dW")
    print(f"  data:    {n:,} observations at D = {dt}, span T = {n*dt:.0f} "
          f"({th*n*dt:.0f} mean-reversion times)\n")

    t0 = time.time()
    model = fit_transition_nll(x, dt, seed=0)
    t_fit = time.time() - t0
    net = extract_ou(model, x)
    mle = ou_exact_mle(x, dt)

    lo, hi = np.quantile(x, [0.05, 0.95])
    g = torch.linspace(float(lo), float(hi), 200)
    with torch.no_grad():
        mu_g, sg_g = model.mu(g).numpy(), model.sigma(g).numpy()
    lin_resid = float(np.std(mu_g - (net["theta"] * (net["m"] - g.numpy()))))
    sig_cv = float(np.std(sg_g) / np.mean(sg_g))

    print(f"  {'parameter':<10}{'truth':>9}{'exact MLE':>12}{'neural SDE':>12}"
          f"{'net vs MLE':>12}{'MLE vs truth':>14}")
    for k in ("theta", "m", "sigma"):
        print(f"  {k:<10}{TRUE[k]:>9.4f}{mle[k]:>12.4f}{net[k]:>12.4f}"
              f"{100*(net[k]/mle[k]-1):>11.2f}%{100*(mle[k]/TRUE[k]-1):>13.1f}%")
    print(f"  'MLE vs truth' is sampling error in this one path, shared by the efficient")
    print(f"  estimator; 'net vs MLE' is the error attributable to the network.")
    print(f"\n  trained in {t_fit:.1f}s. What the network discovered, unprompted:")
    print(f"    drift is LINEAR:     residual from the best line {lin_resid:.4f} against a "
          f"drift range of {float(np.ptp(mu_g)):.3f}")
    print(f"    diffusion CONSTANT:  coefficient of variation of sigma(x) across the "
          f"data range {100*sig_cv:.1f}%")
    print(f"\n  The neural SDE matches the exact OU MLE; it does not beat it, and on truly")
    print(f"  OU data it cannot -- the MLE is the efficient estimator for this family. Its")
    print(f"  value is on dynamics OU cannot represent (see the nonlinear self-test).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        dx = np.diff(x) / dt
        sub = np.random.default_rng(0).choice(dx.size, 4000, replace=False)
        ax[0].scatter(x[:-1][sub], dx[sub], s=3, alpha=0.25, color="0.4", label="dX/dt (raw)")
        gg = g.numpy()
        ax[0].plot(gg, th * (m - gg), "k--", lw=1.5, label="true drift")
        ax[0].plot(gg, mu_g, "C3", lw=2, label="learned drift")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("dX/dt")
        ax[0].set_title("drift from raw scatter")
        ax[0].legend(fontsize=8)
        ax[1].axhline(s, color="k", ls="--", lw=1.5, label="true sigma")
        ax[1].plot(gg, sg_g, "C0", lw=2, label="learned sigma(x)")
        ax[1].set_ylim(0, 2 * s)
        ax[1].set_xlabel("X")
        ax[1].set_title("diffusion")
        ax[1].legend(fontsize=8)
        with torch.no_grad():
            sim = model.simulate(torch.full((5,), float(x[0])), dt, 1500,
                                 generator=torch.Generator().manual_seed(1)).numpy()
        tt = np.arange(1501) * dt
        ax[2].plot(tt, x[:1501], color="0.3", lw=0.8, label="observed")
        for k in range(5):
            ax[2].plot(tt, sim[:, k], lw=0.6, alpha=0.8)
        ax[2].axhline(m, color="k", ls=":", lw=1)
        ax[2].set_xlabel("t")
        ax[2].set_title("observed vs 5 paths from the learned SDE")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"\n  figure written to {path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n  figure skipped: {exc}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
    t0 = time.time()
    _run("GROUND TRUTH AND GRADIENTS", [
        _test_ou_simulator_is_exact,
        _test_gradient_through_euler_maruyama,
    ])
    _run("TRAINING OBJECTIVES", [
        _test_pathwise_mse_collapses_the_diffusion,
        _test_recovers_ou_from_transition_likelihood,
        _test_discretisation_bias_is_inherited,
    ])
    _run("WHAT CAN BE DISCOVERED", [
        _test_drift_needs_span_diffusion_needs_frequency,
        _test_flexible_drift_overfits_weak_signal,
        _test_extrapolation_outside_the_support,
        _test_time_dependence_overfits_stationary_data,
        _test_neural_earns_its_keep_on_nonlinear_dynamics,
    ])
    _main()
    print(f"\ncompleted in {time.time()-t0:.0f}s")

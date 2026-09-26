# Titan-Q: Architecture, Mathematics and Implementation Status of an Autonomous Microstructure Trading Stack

**Technical Whitepaper, version 1.1**
**Date:** 24 September 2026
**Scope:** the code in `D:\Quant_Project` and the dashboard at `C:\Users\jiten\titan-q-dashboard\index.html`, as read on the date above
**Classification:** Internal

> **How to read this paper.** Each phase is presented in three layers. The first is the financial and mathematical theory. The second is the Python implementation in the phase module, with code excerpts copied verbatim from the files (indentation removed). The third is an **integration status** note stating what the live orchestrator `main_trading_loop.py` actually does with that module today. The research modules are considerably more mature than their integration: most ship with self-tests against known ground truth, but several are represented in the live loop by placeholders. Where this paper and the code disagree, the code is authoritative. Section 6 re-examines the historical backtest, Section 7 tabulates the status of every phase, and Section 8 is the roadmap.

---

## Abstract

Titan-Q is a research and paper-trading system for NSE equities built around a single asynchronous Python process. It combines five layers. The first holds signal models that work on limit order book (LOB) state and on continuous-time stochastic dynamics. The second is a combinatorial asset-selection layer, formulated as a QUBO/Ising problem and solved by classical simulated annealing. The third weights positions by Hierarchical Risk Parity and sizes them with a Kelly sizer corrected for parameter uncertainty. The fourth is a set of defensive shields: Extreme Value Theory (EVT) tail risk, Hawkes/VPIN order-flow toxicity, a reconstruction-error book-anomaly detector and a Gaussian hidden Markov regime model. The fifth covers execution: a state machine, a paper broker, a synthetic multivariate GBM market feed and a React telemetry terminal.

This paper derives the mathematics of each component, shows how the code implements it and states which components are connected to live decision-making. There are three principal findings:

1. The component libraries are mathematically sound and unusually well tested.
2. The live loop currently makes its trade decisions with untrained or placeholder models.
3. The backtest statistics circulated for the system (a win rate of 53.3 to 58.7 percent and a Sharpe ratio of 11.46) are fixed by two constants of a random-number generator in the backtest harness, not by any model.

Section 8 sets out the work needed to close that gap.

---

## Contents

1. Introduction and Core Philosophy
2. Signal Generation (the Brain): Phases 18, 21, 24 and the decision layer
3. Portfolio Optimization (the Allocation): Phases 20, 12 and 4
4. Defensive Shields (Risk Management): Phases 23, 14, 16 and 11
5. Execution and Infrastructure: Phase 22, the engine, the broker, the feed, telemetry and operations
6. Validation: the Historical Backtest Re-examined
7. Integration Status Matrix
8. Roadmap to a Defensible Live System
9. Conclusion
10. References

Appendices: A. Notation; B. File map; C. Reproducing the numbers in this paper

---

## 1. Introduction and Core Philosophy

### 1.1 Why classical technical analysis fails at high frequency

Classical technical indicators such as RSI, MACD, Bollinger bands and moving-average crossovers share one structure: each is a causal filter applied to the series of last-traded prices, sampled on a fixed clock. At the timescales where an automated system competes (milliseconds to minutes), that structure fails for five separate reasons.

**(a) The information set is wrong.** The last-traded price is a lossy projection of the state of the limit order book. It discards the size resting at each price level, the rate at which orders arrive and are cancelled, and the queue position of every participant. At short horizons, these discarded quantities are what predict the next move of the mid-price. Cont, Kukanov and Stoikov (2014) show that mid-price changes over short intervals are explained, to first order, by a linear function of **order flow imbalance** (OFI), with a slope that is inversely proportional to market depth:

$$\Delta P_k = \beta \, \mathrm{OFI}_k + \varepsilon_k, \qquad \beta \propto \frac{1}{\text{depth}}.$$

No indicator computed from trade prices can see OFI.

**(b) Microstructure noise dominates the signal.** An observed trade price is the efficient price plus a bid-ask bounce. In Roll's (1984) model, with half-spread $c$ and trade sign $q_t \in \{-1, +1\}$,

$$p_t = m_t + c\, q_t \quad \Longrightarrow \quad \mathrm{Cov}(\Delta p_t, \Delta p_{t-1}) = -c^2 .$$

At tick scale, "momentum" and "mean reversion" read from trade prices largely measure this bounce. The same noise corrupts volatility. The expected realised variance at sampling interval $\Delta$ over a horizon $T$ is

$$\mathbb{E}\left[\mathrm{RV}_\Delta\right] = \sigma^2 T + 2 \frac{T}{\Delta} \mathbb{E}[\varepsilon^2],$$

which diverges as sampling becomes finer (Zhang, Mykland and Ait-Sahalia, 2005). Bollinger bands built from tick returns therefore measure noise.

**(c) Lag is structural.** An exponential moving average $E_t = a\, p_t + (1-a) E_{t-1}$ has impulse response $a(1-a)^j$ and mean lag

$$\bar{L} = \sum_{j \ge 0} j \, a (1-a)^j = \frac{1-a}{a}.$$

With $a = 2/(n+1)$ this is $(n-1)/2$ bars, so MACD(12, 26) carries lags of 5.5 and 12.5 bars. Wilder's RSI(14) uses $a = 1/14$ and lags 13 bars. An indicator is a low-pass filter: it reports what the price has already done.

**(d) The price process is discrete.** On an exchange tick grid the mid-price moves in integer ticks, and a move happens when a queue at the touch is depleted. That makes it a point process driven by order arrivals and cancellations. Continuous-valued indicators of a discrete, queue-driven process model the wrong object.

**(e) Parameter search manufactures edge.** Indicator periods and thresholds are chosen by searching over history. With enough trials, the best in-sample Sharpe ratio is large even when every true Sharpe ratio is zero. Bailey and Lopez de Prado (2014) give the deflation required. Section 6 returns to this point.

### 1.2 The paradigm shift: microstructure, continuous time and statistical physics

Titan-Q is organised around three alternatives to price-chart analysis.

**Limit order book microstructure.** The system state is the book itself: prices and sizes at the top $L$ levels on each side, together with the flows that change them. The simplest informative statistic is the depth imbalance

$$I_t = \frac{V^b_t - V^a_t}{V^b_t + V^a_t} \in [-1, 1],$$

and Phases 18 (the transformer) and 16 (the anomaly detector) consume the full ten-level book.

**Continuous-time stochastic modelling.** Prices are modelled as diffusions, $dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t$ (Phase 21). Order arrivals are self-exciting point processes (Phase 14). Quoting is a stochastic optimal control problem solved through a Hamilton-Jacobi-Bellman equation (Phase 22). Macro state is a hidden Markov chain (Phase 11). EVT (Phase 23) provides the limit theory for the tails.

**Statistical physics and quantum-inspired optimization.** Portfolio selection is written as an Ising problem: binary spins with pairwise couplings. This is the form accepted by quantum annealers, which evolve under a time-dependent Hamiltonian

$$\hat H(s) = -A(s) \sum_i \hat\sigma^x_i + B(s)\, \hat H_P, \qquad s: 0 \to 1,$$

moving from a transverse-field ground state into the ground state of the problem Hamiltonian $\hat H_P$ (Kadowaki and Nishimori, 1998). **Titan-Q uses the formalism but not the hardware.** Its Phase 20 solver is classical Metropolis simulated annealing, and the module also contains simulated bifurcation, a classical algorithm inspired by quantum adiabatic dynamics. Throughout this paper, "quantum" names the problem formulation and the hardware it would be compatible with. It does not describe how the problem is computed.

### 1.3 Engineering philosophy visible in the code

The phase modules share an unusual discipline. Every estimator ships with self-tests against a known ground truth, comparisons against strong simple baselines, and a docstring section stating what the module does *not* claim. For example:

- The transformer forecaster enforces a purge gap between training and test data.
- The neural SDE module demonstrates that the obvious training loss destroys the diffusion.
- The annealer states that annealing does not escape NP-hardness.
- The spoofing detector describes itself as a surveillance tool, not a pre-trade shield.

This paper holds the integrated system to the same standard.

### 1.4 System overview

At the process level, Titan-Q is three programs launched by one batch file:

```
Windows Scheduled Task "TitanQ_Live_Trading"  (daily, 09:10 IST)
  -> D:\Quant_Project\start_titan_q.bat
       |
       +-- nse_whole_market_feed.py   500-asset correlated GBM -> live_state.json, data lake
       +-- main_trading_loop.py       asyncio: ExecutionEngine.run() || live_trading_orchestrator()
       |                                -> PaperTradingBroker (fills at yfinance last price)
       |                                -> dashboard_state.json
       +-- dashboard_server.py        http.server on :5000 -> index.html (React 18), polls every 2 s
```

The orchestrator runs one cycle roughly every five seconds. Its steps, in code order, are:

```
 A     Hawkes + VPIN toxicity score        score > 0.90   -> skip cycle
 A.2   spoofing autoencoder                flag           -> skip cycle
 B     HMM macro regime                    CRASH          -> skip cycle
 C.1   rough-path signature features       logged
 C.2   Temporal Fusion Transformer         logged
 C.2b  transfer entropy                    logged
 C.3   PPO policy                          BUY / SELL / HOLD
 D.1   annealer                            choose 4 of 50 assets
 D.2   neural SDE drift and diffusion -> Kelly fraction; HRP weights over the 4
 D.3   EVT expected shortfall              ES > 12%       -> halve the order size
 E     route one OrderIntent per selected asset to the ExecutionEngine
       export state to dashboard_state.json
```

Two process-level facts frame the rest of the paper. First, the execution engine and the orchestrator share one event loop through `asyncio.gather`:

```python
async def main():
    broker = PaperTradingBroker(initial_capital=500000.0)
    config = EngineConfig(algo_id="TITAN-Q-FUND", warmup_bars_required=0)
    engine = ExecutionEngine(broker=broker, config=config)
    
    await asyncio.gather(
        engine.run(),
        live_trading_orchestrator(engine)
    )
```

Second, the market feed runs as a separate process and writes its own files. **The orchestrator reads neither of them.** Every model input in the live loop is generated inside the loop itself (Section 5.4).

---

## 2. Signal Generation (the Brain)

### 2.1 Phase 18: Temporal Fusion Transformer on the ten-level book

**Problem.** The task is to forecast the direction of the average mid-price move over the next $H$ ticks from the last $W$ snapshots of an $L$-level book. The input is a tensor $X \in \mathbb{R}^{W \times 4 \times L}$ whose four channels are ask price, ask size, bid price and bid size. The code uses $W = 50$, $H = 50$ and $L = 10$.

**Label.** The module follows the FI-2010 / DeepLOB convention (Ntakaris et al., 2018; Zhang, Zohren and Roberts, 2019):

$$r_t = \frac{1}{\text{tick}} \left( \frac{1}{H} \sum_{h=1}^{H} m_{t+h} - m_t \right), \qquad
y_t = \begin{cases} \text{Up} & r_t > \alpha_{hi} \\ \text{Down} & r_t < \alpha_{lo} \\ \text{Stationary} & \text{otherwise} \end{cases}$$

The cut points $\alpha_{lo}$ and $\alpha_{hi}$ are the terciles of $r_t$ **on training data only**, so the classes are balanced in training and chance-level macro-F1 is about 0.33.

**Normalisation.** Prices are expressed in ticks relative to the mid, because absolute price is non-stationary and would let the network memorise the training period's price level. Sizes are transformed with `log1p`. Both are z-scored with training statistics only.

**Architecture.** The model has four stages, taken from the Temporal Fusion Transformer of Lim et al. (2021) and adapted to the book:

1. *Spatial convolution across price levels.* Two width-3 `Conv1d` layers slide along the level axis, independently for each snapshot, so each output sees a five-level neighbourhood. This is the local depth structure (imbalance at the touch and the shape of the first queues) that drives short-horizon moves.
2. *Gated Residual Network (GRN)*:

$$\eta_2 = \mathrm{ELU}(W_2 a + b_2), \quad \eta_1 = W_1 \eta_2 + b_1, \quad
\mathrm{GRN}(a) = \mathrm{LayerNorm}\left( a + \sigma(W_4 \eta_1) \odot W_5 \eta_1 \right).$$

   The sigmoid gate lets a block collapse to the identity when its nonlinearity is not needed, which is the right default on small data.
3. *Interpretable multi-head causal attention.* Queries and keys are head-specific and the value projection is shared:

$$A_h = \mathrm{softmax}\left( \frac{Q_h K_h^\top}{\sqrt{d_k}} + M \right), \qquad
\mathrm{out} = W_o \left( \frac{1}{n_h} \sum_h A_h \right) V, \qquad M_{ij} = -\infty \ \text{for } j > i.$$

   Because every head mixes the same values, the averaged matrix is the actual mixing applied to the input and can be read as a map of which past ticks were used. The causal mask means no position ever sees a later tick.
4. *Head.* A final GRN is applied and the last token is mapped to three logits.

The implementation is compact:

```python
def forward(self, a: torch.Tensor) -> torch.Tensor:
    e = self.drop(self.w1(F.elu(self.w2(a))))
    return self.norm(a + torch.sigmoid(self.gate(e)) * self.value(e))
```

```python
for h in range(self.h):
    s = self.q[h](x) @ self.k[h](x).transpose(1, 2) / math.sqrt(self.dk)
    if mask is not None:
        s = s + mask
    attn.append(torch.softmax(s, dim=-1))
A = torch.stack(attn, 1).mean(1)                    # (B, W, W)
return self.o(self.drop(A) @ V), A
```

**Evaluation protocol.** The module's evaluation design matters more than its architecture, because the deep-LOB literature contains many results that do not survive a proper evaluation. The protocol has four parts.

- *Purged chronological split.* Sample $t$ uses features from $[t-W+1, t]$ and a label from $[t+1, t+H]$. A gap of $H + W$ snapshots between the last training sample and the first test sample guarantees that no training label reads a test-period price.
- *Effective sample size.* Consecutive windows share $H-1$ of their $H$ label ticks, so $n$ windows carry roughly $n/H$ independent labels. A moving-block bootstrap with block length $H$ is used for confidence intervals, because an i.i.d. bootstrap would understate the variance by a factor of about $H$.
- *Baselines on the same test set.* The transformer is compared with a majority-class baseline, a multinomial logistic regression on one or two hand-built imbalance features, the same logistic averaged over the window, and an **oracle** logistic that is told the hidden order-flow regime. The oracle is an upper bound that no observable model can beat.
- *Synthetic book with a known hidden driver.* Mid-price moves follow $P(\text{up} \mid \text{move}) = \mathrm{sigmoid}(\beta I_t + \gamma z_t)$, where $z_t \in \{-1, 0, 1\}$ is a sticky hidden regime. Size noise is AR(1) with persistence $\phi$, so a window of $W$ ticks holds about $W_{\mathrm{eff}} = W (1-\phi)/(1+\phi)$ effective samples. The history of the book therefore genuinely helps, and attention has a real job to do.

**Measured performance.** An earlier project summary quotes a macro-F1 of about 63 percent. That number does not appear in any saved output in the project. The harness's own printed readout, which is hard-coded prose written after an earlier run, says the transformer "does NOT beat the baseline at either training size". The experiment was therefore re-run for this paper, unmodified (`_main_experiment()`, independent 40,000-snapshot test series):

| Model (test macro-F1; chance about 0.33) | Trained on 1,000 snapshots | Trained on 30,000 snapshots |
|---|---|---|
| Majority class | 0.199 | 0.166 |
| Logistic, last snapshot (2 features) | 0.520 | 0.543 |
| Logistic, windowed (5 features) | 0.515 | 0.565 |
| **Temporal Fusion Transformer** (32,403 parameters) | **0.494** | **0.551** |
| Oracle (told the hidden regime; not achievable) | 0.635 | 0.666 |
| TFT minus windowed logistic, 95% block-bootstrap CI | -0.021 [-0.032, -0.009] | -0.014 [-0.025, -0.004] |

Two conclusions follow. First, **the "about 63 percent" figure is the oracle's score**, 0.635 at 1,000 training snapshots. The oracle is a logistic regression that is handed the simulator's hidden regime. It is an upper bound that no model seeing only the book can reach, and it is not the transformer's performance. Second, the transformer scores 0.494 and 0.551 and is **significantly worse** than a five-feature logistic regression at both training sizes, because both confidence intervals exclude zero. At 30,000 snapshots its train/test gap is only 0.028, so the shortfall is representational rather than overfitting. The logistic is given the imbalance ratio directly, while the transformer must learn a ratio of exponentials from normalised log sizes. The module suggests the next experiment: add an explicit imbalance channel before drawing conclusions about the architecture. All of these numbers come from the synthetic book, not from NSE data. The full run took 118 seconds.

> **Integration status (live loop).** `TemporalFusionLOB(n_levels=10, window=50)` is constructed with freshly initialised weights. No `state_dict` is loaded. The model is fed Gaussian noise, `torch.randn(1, 50, 4, 10, dtype=torch.float64)`, and the resulting probabilities are logged but not consumed by any decision. With seed-0 weights on random input, the mean forecast is Down 51.5 percent, Stationary 25.2 percent and Up 23.3 percent (300 draws). The `float64` dtype is a workaround: `neural_sde_discovery.py` calls `torch.set_default_dtype(torch.float64)` at import time, so every module constructed afterwards, including this transformer, is built in double precision.

### 2.2 Phase 21: Neural stochastic differential equations

**Model.** The market state is modelled as a continuous-time stochastic particle,

$$dX_t = \mu_\phi(X_t)\, dt + \sigma_\psi(X_t)\, dW_t,$$

with drift $\mu_\phi$ and diffusion $\sigma_\psi$ given by separate multilayer perceptrons (SiLU activations). Positivity of the diffusion is enforced with $\sigma = \mathrm{softplus}(\cdot) + 10^{-4}$.

**Training objective.** The Euler-Maruyama scheme implies a Gaussian transition,

$$X_{t+\Delta} \mid X_t \sim \mathcal{N}\left( X_t + \mu(X_t)\Delta, \ \sigma^2(X_t)\Delta \right),$$

so the negative log-likelihood per transition is

$$\ell = \tfrac{1}{2} \log\left( 2\pi \sigma^2(x_0) \Delta \right) + \frac{\left( x_1 - x_0 - \mu(x_0)\Delta \right)^2}{2 \sigma^2(x_0) \Delta}.$$

Drift enters only through the conditional mean and diffusion only through the conditional variance, so the two are identified by different features of the data. The implementation mirrors the formula exactly:

```python
def transition_nll(model: NeuralSDE, x0: torch.Tensor, x1: torch.Tensor, dt: float,
                   t0: Optional[torch.Tensor] = None) -> torch.Tensor:
    mu = model.mu(x0, t0)
    var = model.sigma(x0, t0) ** 2 * dt
    r = x1 - x0 - mu * dt
    return (0.5 * torch.log(2 * math.pi * var) + r * r / (2 * var)).mean()
```

**Why the obvious loss is wrong.** One tempting approach is to simulate paths with Euler-Maruyama and minimise the squared distance to the observed paths. Because the simulated Brownian increments are independent of those that produced the data,

$$\mathbb{E}\left[ (\hat X - X)^2 \right] = \mathrm{Var}(\hat X) + \mathrm{Var}(X) + \left( \mathbb{E}\hat X - \mathbb{E}X \right)^2,$$

which is minimised by $\mathrm{Var}(\hat X) = 0$. Pathwise MSE drives $\sigma$ to zero and collapses the SDE to an ODE for the conditional mean. The module demonstrates this collapse in a self-test. It also provides a distribution-matching alternative, the maximum mean discrepancy

$$\mathrm{MMD}^2 = \mathbb{E}\,k(a, a') + \mathbb{E}\,k(b, b') - 2\,\mathbb{E}\,k(a, b),$$

computed between simulated and observed $(X_t, X_{t+k\Delta})$ pairs, with gradients back-propagated through the solver.

**What can be discovered at all.** Three results from the module bound what any neural SDE can learn from market data:

1. *Discretisation bias is inherited.* For an Ornstein-Uhlenbeck process the Euler likelihood is maximised at $\theta_{EM} = (1 - e^{-\theta\Delta})/\Delta$, not at $\theta$. Whatever the network, it converges to the biased value, and at coarse sampling the bias dominates.
2. *Diffusion needs frequency; drift needs span.* $\sigma$ is estimated from quadratic variation and sharpens as sampling gets finer. The standard error of the drift is governed by the calendar span $T$ of the data (Merton, 1980), $\mathrm{SE}(\hat\mu) \approx \sigma/\sqrt{T}$. A neural SDE on market data therefore learns $\sigma(x)$ well and $\mu(x)$ poorly, and no architecture changes that. This is the same fact that governs the Kelly shrinkage in Section 3.3.
3. *Flexible drifts overfit.* On data whose true drift is exactly linear, the network beats the linear model in-sample and loses to it out of sample. The whole drift signal is worth a few $10^{-4}$ nats per transition.

The neural SDE earns its place on non-linear dynamics that an OU process cannot represent, such as a double-well drift with leverage-style diffusion. On those, the module shows it winning on held-out likelihood.

> **Integration status (live loop).** `NeuralSDE(hidden=32, depth=2)` is constructed with seed-0 random weights and never trained or loaded. The loop evaluates it at a constant state, `idx_state = torch.tensor([[0.5]])`, so its outputs are the same number on every cycle: $\mu = -0.1012$ and $\sigma = 0.6412$ per unit time. The loop annualises these as daily quantities, giving a drift of $-2{,}550$ percent and a diffusion of $1{,}018$ percent per year. Section 3.3 follows those numbers into the Kelly sizer.

### 2.3 Phase 24: Transfer entropy causality engine

**Definition.** Pearson correlation is symmetric and linear, so it cannot say which series leads. Transfer entropy (Schreiber, 2000) measures how much the past of a source $X$ reduces uncertainty about the next value of a target $Y$, beyond what $Y$'s own past already provides:

$$\mathrm{TE}_{X \to Y} = \sum p\left( y_{t+1}, y_t^{(k)}, x_s^{(l)} \right) \log_2 \frac{p\left( y_{t+1} \mid y_t^{(k)}, x_s^{(l)} \right)}{p\left( y_{t+1} \mid y_t^{(k)} \right)}.$$

Three equivalent forms are implemented and required to agree to floating-point precision. The first is a conditional mutual information:

$$\mathrm{TE}_{X \to Y} = I\left( Y_{t+1} ; X^{(l)} \mid Y^{(k)} \right).$$

The second writes it as four joint entropies:

$$\mathrm{TE}_{X \to Y} = H(Y_+, Y_k) - H(Y_k) - H(Y_+, Y_k, X_l) + H(Y_k, X_l).$$

The third is an expected KL divergence between the two transition laws:

$$\mathrm{TE}_{X \to Y} = \mathbb{E}\left[ D_{KL}\left( p(y_+ \mid y_k, x_l) \,\|\, p(y_+ \mid y_k) \right) \right].$$

The KL form says what TE means: how far the transition law of $Y$ moves when $X$'s past is revealed. For Gaussian processes TE equals half the Granger log-variance ratio (Barnett, Barrett and Seth, 2009), $\mathrm{TE} = \tfrac{1}{2} \ln(\mathrm{RSS}_r / \mathrm{RSS}_f)$ nats. Its advantage over Granger causality is **non-linear** coupling. The module's self-test builds $y_t = 0.4 y_{t-1} + 0.8 (x_{t-3}^2 - 1) + \text{noise}$, a strong coupling that is invisible to every linear statistic, and TE detects it where Pearson correlation and linear Granger causality both fail.

The entropy form is the production estimator:

```python
B = int(max(x_d.max(), y_d.max()) + 1) if bins is None else bins
fut, past, src, _ = _embed(x_d, y_d, k, l, lag)
yk = _encode(past, B)
xl = _encode(src, B)
fut_yk = _encode([fut, yk], B**k)          # fut * B^k + yk, unique since yk < B^k
return (entropy(fut_yk)
        - entropy(yk)
        - entropy(_encode([fut_yk, xl], B**l))
        + entropy(_encode([yk, xl], B**l)))
```

**Where TE misleads, and the remedies in the module.**

1. *The plug-in estimator is biased upward.* Each empirical entropy is biased low by about $(M - 1)/(2N \ln 2)$ bits, where $M$ is the number of occupied cells, and in TE the four biases do not cancel. The joint space grows as $B^{k+l+1}$. Significance must therefore come from **surrogates**: circular time shifts of the source, which preserve everything about $X$ except its alignment with $Y$. The surrogate mean is also the finite-sample bias, so `effective = te - null_mean` is the bias-corrected flow. This is implemented in `te_test`.
2. *TE is not causation.* A hidden common driver $Z$ produces $\mathrm{TE}_{X \to Y} > 0$ with no link between $X$ and $Y$. The remedy is conditional transfer entropy, $I(Y_+ ; X_l \mid Y_k, Z_m)$, which the module implements and demonstrates.
3. *Coarse-graining fakes reverse flow.* Binning a driver lets the response's past refine the driver's present, which produces spurious flow in the wrong direction. Finer bins shrink the artefact; deeper embeddings do not.
4. *Synchronisation erases TE.* Under strong coupling a chaotic response locks onto its driver and TE falls to zero in both directions. For coupled logistic maps the threshold is $\ln 2 + \ln(1 - \epsilon) < 0$, that is $\epsilon > 1/2$.

> **Integration status (live loop).** Each cycle the loop builds two synthetic series of length 200 in which the second is a lagged copy of the first by construction. It then computes the **raw** plug-in TE with 8 bins and no surrogate correction, and logs the value. The value does not feed any decision, and no TE is computed across NIFTY 500 assets. The raw number is dominated by estimator bias. Over 300 draws, the coupled series give a mean of 1.95 bits, while **independent** series with the same settings give 1.30 bits, and every one of the 300 independent draws exceeded 1.0 bit. Section 6 shows why this matters for the backtest.

### 2.4 The decision layer: PPO policy and rough-path signatures

The trade direction is chosen by a reinforcement-learning policy trained with Proximal Policy Optimization (Schulman et al., 2017), which maximises the clipped surrogate objective

$$L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat A_t, \ \mathrm{clip}\left( r_t(\theta), 1-\epsilon, 1+\epsilon \right) \hat A_t \right) \right], \qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}.$$

The checkpoint name (`ppo_sortino_18000000_steps.zip`) indicates a policy trained for 18 million steps on a Sortino-type reward.

The Phase 17 feature engine computes depth-3 path signatures, the iterated integrals

$$S(X)^{i_1 \cdots i_k}_{s,t} = \int_{s < u_1 < \cdots < u_k < t} dX^{i_1}_{u_1} \cdots dX^{i_k}_{u_k},$$

which characterise a path up to reparametrisation (Lyons, 1998). It also computes the Levy area between price and volume, $A^{12} = \tfrac{1}{2}\left( S^{12} - S^{21} \right)$, whose sign indicates which of the two moved first.

> **Integration status (live loop).** This is the most important status note in the paper. The loop tries `PPO.load("checkpoints/ppo_sortino_18000000_steps.zip")`. The `checkpoints` directory does not exist in `D:\Quant_Project`, and the path is relative to whatever the working directory is when the scheduled task starts. The load fails, `ppo_model` becomes `None`, and the direction is drawn at random:
>
> ```python
> mock_obs = (np.random.rand(1, 65) * 2 - 1).astype(np.float32)
> if ppo_model:
>     action, _ = ppo_model.predict(mock_obs, deterministic=True)
>     direction = "BUY" if action[0] == 2 else "SELL" if action[0] == 0 else "HOLD"
> else:
>     direction = random.choice(["BUY", "BUY", "HOLD", "SELL"])
> ```
>
> Live directions are therefore BUY 50 percent, HOLD 25 percent and SELL 25 percent, independent of any market data. Even with a checkpoint present, the policy's observation is uniform noise (`mock_obs`), not the transformer's probabilities or the signature features. A 2 MB file, `master_ai_brain_sortino.zip`, sits at the project root. If that is the trained policy, it should be loaded by absolute path. The signature features are computed on a random-walk path and logged only.

---

## 3. Portfolio Optimization (the Allocation)

### 3.1 Phase 20: Cardinality-constrained selection by annealing

**Problem.** The task is to choose exactly $K$ of $N$ assets, equal-weighted, to maximise the Sharpe ratio:

$$\max_{x \in \{0,1\}^N} \ S(x) = \frac{\mu^\top x}{\sqrt{x^\top \Sigma x}} \quad \text{subject to} \quad \mathbf{1}^\top x = K.$$

With weights $w = x/K$ the factor $K$ cancels, so this is exactly the equal-weight Sharpe ratio of the selected subset.

**Why it is hard.** The feasible set is the $\binom{N}{K}$ vertices of a hypercube slice, which is non-convex and exponential in size. Binary quadratic optimization under a cardinality constraint is NP-hard in general; it contains problems such as densest-$k$-subgraph as special cases, and cardinality-constrained mean-variance selection is a classic hard mixed-integer program (Bienstock, 1996). The objective landscape is also rugged. At $N = 50$, $K = 10$, the module's self-test runs swap local search from 40 random starting subsets and reaches at least 30 **distinct** local optima. Any method that follows a single deterministic path settles into whichever basin it enters first.

**Scale check for the deployed configuration.** NP-hardness is a statement about asymptotic worst cases. The live loop selects $K = 4$ from $N = 50$, and $\binom{50}{4} = 230{,}300$. The module's vectorised `brute_force` enumerates every subset in **0.10 seconds**, as measured for this paper. At the deployed size the global optimum can be certified exactly, and annealing is useful only if $K$ or $N$ grows. (With 2,000 steps, `anneal_swap` found the exact optimum in 50 of 50 seeds on a test instance, so it is not wrong here, just unnecessary.)

**The QUBO route.** A QUBO minimises a single quadratic form $x^\top Q x$, and Sharpe is a ratio, so it is not directly a QUBO. It becomes one exactly through **Dinkelbach's method** (Dinkelbach, 1967) applied to $S^2$. For $\mu^\top x > 0$, solve a sequence of QUBOs:

$$x_k = \arg\min_x \ x^\top \left( \theta_k \Sigma - \mu\mu^\top \right) x + P\left( \mathbf{1}^\top x - K \right)^2, \qquad \theta_{k+1} = \frac{(\mu^\top x_k)^2}{x_k^\top \Sigma x_k}.$$

With an exact inner solver, $\theta_k$ rises monotonically to the optimal squared Sharpe. The matrix $\theta\Sigma - \mu\mu^\top$ is indefinite, which is what makes the landscape rugged. The cardinality penalty exists only because Ising hardware accepts unconstrained problems. Using $x_i^2 = x_i$, its linear term folds onto the diagonal. The spin substitution $x = (s + 1)/2$ with $s \in \{-1, +1\}^N$ gives the Ising form

$$E(s) = -\tfrac{1}{2} s^\top J s - h^\top s + \text{const}, \qquad J_{ij} = -\tfrac{1}{2} Q_{ij} \ (i \ne j), \qquad h_i = -\tfrac{1}{2} \sum_j Q_{ij}.$$

The module verifies this mapping exhaustively on all 1,024 states of a 10-spin problem.

**Simulated annealing, and why it escapes local optima.** At temperature $T$, the Metropolis rule accepts a move that changes the objective by $\Delta S$ with probability

$$P(\text{accept}) = \min\left( 1, \ e^{\Delta S / T} \right).$$

The chain's stationary distribution is the Boltzmann law $\pi_T(x) \propto e^{S(x)/T}$, which concentrates on the global maxima as $T \to 0$. A downhill move of size $|\Delta S|$ is accepted with probability $e^{-|\Delta S|/T}$, so at high temperature the chain crosses the barriers between basins that trap greedy search. Cooling then freezes it into the deepest basin it has found. Logarithmic cooling guarantees convergence (Hajek, 1988). The geometric schedule used in practice, $T_k = T_0 (T_1/T_0)^{k/(n-1)}$, is faster and carries no guarantee, which is why the module scores every result against brute force or a certified upper bound.

Titan-Q's production annealer works directly on Sharpe and moves by **swaps**, removing one selected asset and adding one unselected asset. The swap preserves $\mathbf{1}^\top x = K$ exactly, so the penalty disappears. With $r = \mu^\top x$, $v = x^\top\Sigma x$ and the running vector $s = \Sigma x$, a swap out of $i$ and into $j$ updates in $O(1)$:

$$r' = r - \mu_i + \mu_j, \qquad v' = v - 2s_i + \Sigma_{ii} + 2s_j - 2\Sigma_{ij} + \Sigma_{jj}.$$

```python
for k in range(n_steps):
    pa, pb = ai[k], bi[k]
    a, b = inside[pa], outside[pb]
    r2 = r - mu[a] + mu[b]
    v2 = v - 2 * s[a] + dSig[a] + 2 * s[b] - 2 * Sigma[a, b] + dSig[b]
    S2 = r2 / math.sqrt(v2)
    dS = S2 - S
    if dS >= 0.0 or u[k] < math.exp(dS / T):
        s += Sigma[:, b] - Sigma[:, a]
        r, v, S = r2, v2, S2
        inside[pa], outside[pb] = b, a
        x[a], x[b] = 0.0, 1.0
        if S > best_S:
            best_S, best_x = S, x.copy()
    T *= decay
```

The module also implements single-flip annealing on the penalised QUBO and **simulated bifurcation** (Goto, Tatsumura and Dixon, 2019). In simulated bifurcation each spin is an oscillator whose potential bifurcates as a pump parameter $a(t)$ ramps from 0 to $a_0$:

$$\dot y_i = -\left( a_0 - a(t) \right) x_i + c_0 \Big( \sum_j J_{ij} f(x_j) + h_i \Big), \qquad \dot x_i = a_0 y_i.$$

The module's recorded finding: on the full problem, the penalty-based methods come within a few percent of swap annealing **only at a tuned penalty**, and swing by about 30 percent across $P \in [1\times, 20\times]$. The penalty is a hardware limitation, not a feature of the problem.

> **Integration status (live loop).** `anneal_swap` runs for real, with `n_steps=2000`, on a **random** problem regenerated every cycle: `mock_mu = np.random.normal(0.05, 0.2, 50)` and a random three-factor covariance. The four selected indices are mapped to tickers by position in a hard-coded NIFTY 50 list. The chosen stocks are therefore unrelated to the actual returns or covariances of those companies. Note also that "Simulated Quantum Annealing" in the strict sense (path-integral Monte Carlo of a transverse-field Ising model) is not implemented. The solver is classical thermal annealing.

### 3.2 Phase 12: Hierarchical Risk Parity

**Why Markowitz fails in practice.** The minimum-variance portfolio is

$$w_{MV} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}, \qquad \Sigma^{-1} = \sum_k \frac{1}{\lambda_k} v_k v_k^\top.$$

Inversion amplifies the **smallest** eigenvalues, which are the ones estimated worst. For a sample correlation matrix of pure noise with $N$ assets and $T$ observations, the eigenvalues spread over the Marchenko-Pastur interval

$$\lambda_\pm = \left( 1 \pm \sqrt{N/T} \right)^2,$$

so small eigenvalues are noise even when $T$ is several times $N$. In a crash, correlations converge. For an equicorrelated matrix with correlation $\rho$, the eigenvalues are $1 + (N-1)\rho$ and $1 - \rho$, so the condition number is

$$\kappa = \frac{1 + (N-1)\rho}{1 - \rho}.$$

At $N = 50$ this is 22 at $\rho = 0.3$, 451 at $\rho = 0.9$ and 4,951 at $\rho = 0.99$. The inverse, and every portfolio built on it, becomes unstable precisely when stability matters most.

**The HRP algorithm** (Lopez de Prado, 2016) never inverts $\Sigma$. It has four steps:

1. *Metric distance.* $d_{ij} = \sqrt{\tfrac{1}{2}(1 - \rho_{ij})}$ is a proper metric on assets.
2. *Correlation distance.* $\tilde d_{ij} = \lVert d_{\cdot i} - d_{\cdot j} \rVert_2$ compares assets by their whole correlation profile.
3. *Tree.* Hierarchical clustering with single linkage on $\tilde d$. Single linkage is equivalent to the **minimum spanning tree** of the complete distance graph, so the merge heights equal the MST edge weights. The allocator checks this identity on every call (`mst_matches_single_linkage`).
4. *Quasi-diagonalisation and recursive bisection.* Leaves are reordered so similar assets are adjacent. Walking down the dendrogram, each split into clusters $c_0$ and $c_1$ divides weight by inverse cluster variance:

$$V_c = \tilde w_c^\top \Sigma_c \tilde w_c, \quad \tilde w_c \propto \mathrm{diag}(\Sigma_c)^{-1}, \qquad \alpha = 1 - \frac{V_0}{V_0 + V_1}, \qquad w_{c_0} \leftarrow \alpha\, w_{c_0}, \quad w_{c_1} \leftarrow (1-\alpha)\, w_{c_1}.$$

Only diagonal blocks and pairwise cluster variances are ever touched:

```python
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
```

> **Integration status (live loop).** HRP runs for real (`method="single", bisection="dendrogram"`) on the $4 \times 4$ block of the random covariance matrix for the four annealer-selected indices. The algorithm is correct, but its input is synthetic. With four assets HRP stays close to inverse-variance weighting, which the saved dashboard state reflects: weights of 0.21, 0.29, 0.26 and 0.24.

### 3.3 Phase 4: The Kelly criterion, with the corrections that make it usable

*(Some module docstrings refer to the sizer as part of Phase 5; this paper keeps the Phase 4 label used in `main_trading_loop.py`.)*

**Derivation.** Invest a fraction $f$ of wealth in an asset following $dS/S = \mu\,dt + \sigma\,dB$, with the remainder at the risk-free rate $r$. By Ito's lemma,

$$d\ln W_t = \left[ r + f(\mu - r) - \tfrac{1}{2} f^2 \sigma^2 \right] dt + f\sigma\, dB_t,$$

so the long-run growth rate is

$$g(f) = r + f(\mu - r) - \tfrac{1}{2} f^2\sigma^2,$$

which is maximised at the **Kelly fraction**

$$\boxed{ f^* = \frac{\mu - r}{\sigma^2} }, \qquad g(f^*) = r + \frac{(\mu - r)^2}{2\sigma^2}.$$

**Why full Kelly is not used.** A fraction $\lambda$ of Kelly earns

$$\frac{g(\lambda f^*) - r}{g(f^*) - r} = 2\lambda - \lambda^2$$

of the maximal excess growth at $\lambda$ times the volatility, and the probability of ever suffering a drawdown to a fraction $a$ of peak wealth is $a^{2/\lambda - 1}$. Growth is flat at the optimum while volatility is linear in $\lambda$, and that asymmetry is the whole argument for fractional Kelly. `KellySizer` applies five corrections in order.

1. *Posterior shrinkage of the edge.* Merton's result gives $\mathrm{SE}(\hat\mu) = \sigma/\sqrt{T}$, where $T$ is the calendar span in years. With a prior $\mu \sim \mathcal{N}(0, \tau^2)$,

$$\mu_{\mathrm{post}} = \hat\mu \, \frac{\tau^2}{\tau^2 + \mathrm{SE}^2}.$$

2. *Predictive variance.* The denominator becomes $\sigma^2 + \mathrm{SE}^2$.
3. *Cost fixed point.* The trade size depends on $f$, and $f$ depends on the edge net of the cost of trading that size. Round-trip cost uses square-root impact plus linear frictions,

$$c(Q) = 2\left[ Y \sigma_d \sqrt{Q / V_{\mathrm{ADV}}} + h + \frac{\text{fees} + \text{STT}}{10^4} \right],$$

   and the map $f \mapsto (\mu_{\mathrm{post}} - c(fE) - r)/(\sigma^2 + \mathrm{SE}^2)$ is iterated to its fixed point.
4. *Higher moments.* Expanding $\mathbb{E}[\ln(1 + fR)]$ to fourth order gives the first-order condition $m_1 - f m_2 + f^2 m_3 - f^3 m_4 = 0$. Negative skew and excess kurtosis both push $f$ down.
5. *Fractional scaling and caps.* Multiply by $\lambda$, then cap at a maximum position weight (default 10 percent) and maximum leverage.

```python
se = self.se_mu(sigma_annual, sample_years)
mu_post = self.posterior_mu(mu_hat, se)

# predictive variance
var_pred = sigma_annual**2 + se**2

f_naive = (mu_hat - self.rf) / sigma_annual**2
f_shrunk = (mu_post - self.rf) / var_pred

# cost recursion: Q depends on f, f depends on mu net of cost(Q)
f = f_shrunk
mu_net = mu_post
for _ in range(max_iter):
    notional = abs(f) * equity
    cost = cost_model.round_trip_cost(notional, adv_notional, daily_vol)
    mu_net = mu_post - cost
    if mu_net <= 0:
        f = 0.0
        break
    f_new = (mu_net - self.rf) / var_pred
    if abs(f_new - f) < 1e-6:
        f = f_new
        break
    f = f_new
```

**Worked example** (computed with the module, using the live loop's settings: $T = 10$ years, $\tau = 3\%$, $\lambda = 0.25$, equity Rs. 5,00,000, ADV Rs. 85,00,00,000, half-spread 4 bps). The example assumes an estimated edge of $\hat\mu = 10\%$ with $\sigma = 20\%$:

| Stage | Value | Comment |
|---|---|---|
| Naive Kelly $f^* = \mu/\sigma^2$ | 2.500 | 250 percent of equity |
| $\mathrm{SE}(\hat\mu) = 0.20/\sqrt{10}$ | 6.32% | larger than most real edges |
| Posterior edge $\mu_{\mathrm{post}}$ | 1.84% | multiplier $0.0009 / 0.0049 = 0.184$ |
| Shrunk $f$ with predictive variance | 0.417 | |
| After cost fixed point | 0.334 | net edge 1.47% |
| Final, $\times \lambda = 0.25$ | **0.0835** | 8.35 percent of equity |
| Growth retained / P(50% drawdown) | 43.75% / 0.78% | $2\lambda - \lambda^2$; $0.5^{7}$ |

The thirty-fold reduction from 2.5 to 0.0835 is not conservatism. It is what the posterior, the costs and the drawdown law jointly imply.

> **Integration status (live loop).** The sizer is real, but it receives the untrained neural SDE's outputs: an annualised drift of $-25.51$, clamped to $0.01$, and a diffusion of $10.18$. With $\sigma = 10.18$ over 10 years, $\mathrm{SE} = 3.22$, the shrinkage multiplier is $0.0009/(0.0009 + 10.36) \approx 8.7 \times 10^{-5}$, the posterior edge is essentially zero and the edge net of costs is negative. The sizer therefore returns $f = 0$ with binding constraint `negative_net_edge` on **every cycle**. The share count is then computed from hard-coded constants, not from the broker's equity or any live price:
>
> ```python
> shares = max(1, int((capital_alloc * 100000) / 3000))
> ```
>
> With $f = 0$ this gives 1 share, and each HRP-weighted child order is `max(1, int(shares * weight))` = 1 share. Every live signal therefore sends four 1-share orders.

---

## 4. Defensive Shields (Risk Management)

### 4.1 Phase 23: Extreme Value Theory and expected shortfall

**Limit theorem.** By the Pickands-Balkema-de Haan theorem, for a wide class of loss distributions the excesses over a high threshold $u$, $Y = X - u \mid X > u$, converge as $u$ grows to the Generalized Pareto Distribution (GPD)

$$G_{\xi,\beta}(y) = 1 - \left( 1 + \frac{\xi y}{\beta} \right)^{-1/\xi}, \qquad y \ge 0.$$

A shape $\xi > 0$ is a power-law tail, $P(X > x) \sim x^{-1/\xi}$. A Student-$t$ with $\nu$ degrees of freedom has $\xi = 1/\nu$.

**Tail estimators** (McNeil and Frey, 2000). With $N_u$ of $n$ observations above $u$ and $r = (n/N_u)(1-p)$:

$$\mathrm{VaR}_p = u + \frac{\beta}{\xi}\left( r^{-\xi} - 1 \right), \qquad
\mathrm{ES}_p = \frac{\mathrm{VaR}_p}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi} \quad (\xi < 1).$$

Expected shortfall exists only if $\xi < 1$. For heavier tails it is infinite, and a method that returns a finite number there is wrong by construction. The module writes both measures as affine functions of $\beta$ for fixed $\xi$, which makes profile-likelihood confidence intervals one-dimensional:

```python
def _h(xi: float, r: float, measure: str) -> float:
    hv = -math.log(r) if abs(xi) < 1e-9 else math.expm1(-xi * math.log(r)) / xi
    if measure == "var":
        return hv
    return (hv + 1.0) / (1.0 - xi) if xi < 1.0 else float("inf")
```

**Fitting.** The GPD is fitted by maximum likelihood on the exceedances, starting from probability-weighted-moment estimates. The negative log-likelihood is

$$-\ell(\xi, \beta) = N_u \log\beta + \left( 1 + \frac{1}{\xi} \right) \sum_i \log\left( 1 + \frac{\xi y_i}{\beta} \right).$$

Confidence intervals come from the profile likelihood, not the delta method, because the sampling distribution of a far-tail quantile is strongly right-skewed. The threshold is diagnosed with the mean-excess function

$$e(u) = \mathbb{E}[X - u \mid X > u] = \frac{\beta + \xi u}{1 - \xi},$$

which is linear in $u$ with slope $\xi/(1-\xi)$ once the GPD regime is reached.

**Why not Gaussian or historical VaR.** On Student-$t(2)$ losses, which have infinite variance, the sample standard deviation does not converge. The Gaussian VaR error is therefore a *shape* error whose sign flips: it overstates the 95 percent quantile and understates the 99.99 percent quantile several-fold. Historical simulation is exact within the data and blind beyond the sample maximum. EVT's defensible claim is calibration: profile intervals that cover the truth at their nominal rate. It does not claim precision. The module also shows that a one-year backtest cannot reliably detect even a large understatement of 99.99 percent risk, so extreme VaR has to be validated through the model rather than by backtesting.

**Policy.** When the short-horizon expected shortfall of the proposed position exceeds 12 percent, the order size is halved:

```python
evt_mock_es = random.uniform(0.01, 0.15)
if evt_mock_es > 0.12: # If short-term tail risk is > 12% loss potential
    logging.warning(f"[EVT RISK] Fat-Tail Risk elevated (ES={evt_mock_es:.1%}). Trimming order size by 50%.")
    shares = max(1, shares // 2)
```

> **Integration status (live loop).** As the excerpt shows, the ES in the loop is `random.uniform(0.01, 0.15)`, not `POTModel.es`. The trim therefore fires on $(0.15 - 0.12)/0.14 = 21.4$ percent of signals at random. Because the order is already 1 share, `max(1, 1 // 2)` leaves it at 1, so the trim has no effect. The policy also leaves unstated the confidence level and horizon of the ES it compares against 12 percent. The production version should specify both, for example the 99 percent one-day ES of the position's loss distribution.

### 4.2 Phase 14: Hawkes processes and order-flow toxicity

**Self-exciting arrivals.** Order and trade arrivals cluster: each event raises the short-term probability of further events. A Hawkes process (Hawkes, 1971) with exponential kernel has conditional intensity

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha\, e^{-\beta (t - t_i)}.$$

Its branching ratio $n = \alpha/\beta$ is the expected number of events triggered by each event, which is a direct measure of how *endogenous* the flow is. The process is stationary iff $n < 1$, with mean intensity $\mu/(1 - n)$. The exponential kernel admits an $O(1)$ Markov recursion,

$$A_i = e^{-\beta(t_i - t_{i-1})} (1 + A_{i-1}), \qquad \lambda(t_i^+) = \mu + \alpha A_i,$$

and a closed-form log-likelihood for maximum-likelihood fitting:

$$\ell(\mu, \alpha, \beta) = \sum_i \log \lambda(t_i) - \mu T - \frac{\alpha}{\beta} \sum_i \left( 1 - e^{-\beta (T - t_i)} \right).$$

Goodness of fit is checked by time rescaling: the compensator increments $\tau_i = \int_{t_{i-1}}^{t_i} \lambda(s)\,ds$ are i.i.d. Exp(1) under a correct model. The live filter is the recursion itself:

```python
self._A = self._A * math.exp(-self.params.beta * (t - self._t_last)) + 1.0
self._t_last = t
self._n += 1
lam = self.params.mu + self.params.alpha * self._A
```

**VPIN.** Volume-synchronised probability of informed trading (Easley, Lopez de Prado and O'Hara, 2012) measures order-flow imbalance in volume time, which is the adverse-selection risk a liquidity provider faces. Trades are grouped into buckets of equal volume $V$. Each bucket's volume is split into buy and sell parts by bulk volume classification, $V^B_\tau = V\,\Phi(\Delta P_\tau / \sigma_{\Delta P})$, and

$$\mathrm{VPIN} = \frac{\sum_{\tau=1}^{N} \left| V^S_\tau - V^B_\tau \right|}{N V}.$$

**Combined toxicity score.** Each channel is converted to a percentile against a reference window from a quiet period, using an empirical CDF $\hat F$. The percentiles are then combined geometrically:

$$\mathrm{Tox} = \hat F_\lambda(\lambda)^{w} \cdot \hat F_{\mathrm{VPIN}}(\mathrm{VPIN})^{1-w}, \qquad w = 0.5, \qquad \text{halt if } \mathrm{Tox} > 0.90.$$

The geometric mean requires both channels to be extreme at once: fast arrivals alone, or imbalance alone, do not trip the shield.

```python
def score(self, intensities: np.ndarray, vpins: np.ndarray) -> np.ndarray:
    if self._ref_lam is None or self._ref_vpin is None:
        raise RuntimeError("call fit_reference on a quiet window first")
    lam = np.atleast_1d(np.asarray(intensities, dtype=float))
    vp = np.atleast_1d(np.asarray(vpins, dtype=float))
    ph = self._ecdf(self._ref_lam, lam)
    pv = self._ecdf(self._ref_vpin, vp)
    out = self._combine(ph, pv)
    out[~np.isfinite(lam) | ~np.isfinite(vp)] = np.nan
    return np.clip(out, 0.0, 1.0)
```

The live parameters are $\mu = 4$, $\alpha = 8.5$ and $\beta = 10$, giving a branching ratio of 0.85: 85 percent of events in the configured model are triggered by earlier events. The kernel half-life is $\ln 2/\beta = 69$ ms.

> **Integration status (live loop).** The Hawkes filter is updated once per cycle at a synthetic time that advances by exactly 0.1 s, unrelated to any market event. Its intensity therefore converges to the deterministic fixed point
>
> $$\lambda^* = \mu + \frac{\alpha}{1 - e^{-\beta \cdot 0.1}} = 4 + \frac{8.5}{1 - e^{-1}} = 17.45,$$
>
> and stays there; this was confirmed numerically. VPIN is `random.uniform(0.3, 0.7)`, and the reference distributions are random draws (`np.random.gamma(4.0, 2.0, 100)` and `np.random.beta(5, 5, 100)`) taken at boot. The halt rate therefore depends on the boot's random reference sample: across five seeds it ranged from 0 to 16.5 percent of cycles. The `VPINEstimator` and `fit_hawkes_mle` are not called. The dashboard always shows a toxicity of 0.0, because the export reads a variable, `tox_values`, that is never defined.

### 4.3 Phase 16: Adversarial spoofing autoencoder

**Theory.** Spoofing and layering place large resting orders that the trader intends to cancel, in order to move other participants' beliefs about supply and demand. An autoencoder trained on normal book states learns a low-dimensional manifold of typical book *shapes*. A snapshot $x$ is compressed to a latent code and reconstructed, $\hat x = D(E(x))$, and its anomaly score is the reconstruction error

$$e(x) = \frac{1}{d} \lVert x - D(E(x)) \rVert_2^2.$$

States far from the learned manifold, such as an implausibly thick queue several levels deep, reconstruct badly. A linear autoencoder is equivalent to PCA, and the module includes a PCA baseline. The non-linear network can follow a curved manifold.

**Implementation.** The snapshot features are `log1p` sizes on both sides plus the spread, so $d = 2L + 1 = 21$ for ten levels. A window mode adds cancellation and execution flows. The encoder is $21 \to 64 \to 32 \to 6$ with GELU activations, and the decoder mirrors it. Scores are thresholded adaptively on a log scale, using a median plus $z$ scaled MADs computed over a lagged window that excludes already-flagged points:

$$\mathrm{thr}_t = \mathrm{med}\left( \log e \right) + z \cdot 1.4826 \cdot \mathrm{MAD}\left( \log e \right), \qquad z = 4 \ \Rightarrow \ \text{nominal FPR} = 1 - \Phi(4) = 3.2 \times 10^{-5}.$$

```python
if ref.size >= self.min_history:
    if self.method == "mad":
        c = float(np.median(ref))
        sc = 1.4826 * float(np.median(np.abs(ref - c)))
    else:
        c = float(ref.mean())
        sc = float(ref.std(ddof=1))
    thr[t] = c + self.z * max(sc, 1e-12)
    flag[t] = s[t] > thr[t]
```

**What the detector can and cannot see.** The module's own docstring is explicit, and the limits should shape how the shield is used:

1. Spoofing is defined by **intent**, and the book shows orders, not intent. While resting, a spoof and a genuine institutional block look identical. A snapshot autoencoder detects anomalous book *shape*, and it flags a genuine block exactly as readily as a spoof.
2. On a liquid name, large genuine orders are common enough to appear in the training distribution, so large resting size stops being anomalous. The remaining signature is cancellation without a fill, which by construction arrives *after* the manipulation has worked. The module calls itself "a surveillance tool, not a pre-trade shield".

The defensible use is therefore narrower than a pre-trade abort. The detector is a risk dampener: it widens thresholds or cuts size in anomalous book states. It is also a post-trade surveillance feed when used in window mode with cancel and execution features.

> **Integration status (live loop).** `SpoofDetector(mode="snapshot", code=6, z=4.0)` is constructed but never fitted, and `detect` is never called. The abort is:
>
> ```python
> is_spoofing = random.random() < 0.02
> ```
>
> The shield therefore skips 2 percent of cycles at random.

### 4.4 Phase 11: Hidden Markov regime model

**Model.** The latent regime $z_t \in \{1, \dots, K\}$ follows a Markov chain with transition matrix $A_{ij} = P(z_t = j \mid z_{t-1} = i)$. Observations are Gaussian given the regime:

$$x_t \mid z_t = k \ \sim \ \mathcal{N}(\mu_k, \Sigma_k), \qquad x_t = \left( \text{return}, \ \log \text{realised variance}, \ \text{normalised log volume} \right).$$

Parameters are fitted by Baum-Welch (EM) with random restarts, entirely in log space (a three-state model on ten years of daily data underflows float64 in the linear recursion). States are labelled by their means: the highest-volatility state is **CRASH** if its mean return is negative and HIGH_VOL otherwise, and the remaining states are ranked by mean return into BULL, SIDEWAYS and BEAR.

**Decoding: Viterbi versus filtering.** Viterbi finds the jointly most likely state path by max-product dynamic programming:

$$\delta_t(j) = \log b_j(x_t) + \max_i \left[ \delta_{t-1}(i) + \log A_{ij} \right], \qquad \psi_t(j) = \arg\max_i \left[ \delta_{t-1}(i) + \log A_{ij} \right],$$

followed by backtracking from $\arg\max_j \delta_T(j)$:

```python
delta[0] = np.log(np.maximum(self.startprob_, _EPS)) + log_b[0]
for t in range(1, T):
    trans = delta[t - 1][:, None] + log_A          
    psi[t] = np.argmax(trans, axis=0)
    delta[t] = log_b[t] + trans[psi[t], np.arange(self.K)]
```

This is an important correction to the phrase "continuous Viterbi decoding". Viterbi, like the smoothed posterior, conditions on the **entire** sample, including the future. It is an estimation tool: the label it assigns to bar $t$ can change when bar $t + 10$ arrives. A live system must use the **forward filter**, which conditions on $x_{1:t}$ only:

$$\log \alpha_t(j) = \log b_j(x_t) + \mathrm{logsumexp}_i\left( \log\alpha_{t-1}(i) + \log A_{ij} \right).$$

Backtesting on Viterbi labels and trading on filtered labels are different strategies, and the module measures their disagreement rate (`filtered_vs_smoothed_disagreement`). The live filter `MarketRegimeFilter` adds two safeguards. The first is **hysteresis**: a switch is reported only after the new state has been the most likely state for `min_dwell = 3` bars with posterior at least 0.60. The second is a **novelty test**: an observation whose smallest Mahalanobis distance to every state exceeds $\sqrt{\chi^2_{d}(0.999)}$ is flagged as belonging to no known regime.

```python
log_b = self._log_emission_one(x)
if self._log_alpha is None:
    self._log_alpha = self._log_pi + log_b
else:
    self._log_alpha = log_b + logsumexp(
        self._log_alpha[:, None] + self._log_A, axis=0
    )
```

`TransitionAnalytics` supplies the regime statistics a risk manager actually needs. The expected dwell in state $k$ is $1/(1 - A_{kk})$. The stationary distribution is the left eigenvector of $A$ for eigenvalue 1. The expected hitting times of the CRASH state are $h = (I - Q)^{-1}\mathbf{1}$, where $Q$ is $A$ restricted to the non-target states.

> **Integration status (live loop).** The loop uses a `MockRegimeFilter` whose `state` is always "BULL". The regime used for decisions is drawn at random on every cycle:
>
> ```python
> rand_state = random.random()
> current_regime = "BULL" if rand_state < 0.8 else "BEAR" if rand_state < 0.95 else "CRASH"
> ```
>
> A "CRASH" (5 percent of cycles) logs "Liquidating non-hedged positions" and skips the cycle, but **no liquidation code exists**: nothing is sold and no intent is sent. "BEAR" has no effect.

---

## 5. Execution and Infrastructure

### 5.1 Phase 22: Avellaneda-Stoikov market making via the HJB equation

**Model** (Avellaneda and Stoikov, 2008). The mid-price follows $dS_t = \sigma\,dW_t$. A market maker quotes a bid at $S - \delta^b$ and an ask at $S + \delta^a$, and fills arrive as Poisson processes with intensity $\lambda(\delta) = A e^{-k\delta}$ on each side. Inventory is bounded, $q \in [-Q, Q]$, and the maker maximises CARA utility of terminal wealth,

$$\max \ \mathbb{E}\left[ -\exp\left( -\gamma \left( X_T + q_T S_T - L(q_T) \right) \right) \right],$$

where $L$ is an optional liquidation penalty.

**The HJB equation and its reduction.** The value function $u(t, x, s, q)$ satisfies

$$0 = u_t + \tfrac{1}{2}\sigma^2 u_{ss} + \sup_{\delta^b} A e^{-k\delta^b}\left[ u(t, x - s + \delta^b, s, q+1) - u \right] + \sup_{\delta^a} A e^{-k\delta^a}\left[ u(t, x + s + \delta^a, s, q-1) - u \right].$$

CARA utility makes wealth and price separable. The ansatz $u = -\exp\left(-\gamma(x + qs + \theta_q(t))\right)$ reduces the four-variable PDE **exactly** to one ODE per inventory level:

$$\theta_q' = \tfrac{1}{2}\sigma^2\gamma q^2 - H(\theta_q - \theta_{q+1}) - H(\theta_q - \theta_{q-1}), \qquad H(p) = \frac{\eta}{k} e^{-kp}, \qquad \eta = A\left( 1 + \frac{\gamma}{k} \right)^{-(1 + k/\gamma)}.$$

The supremum is attained at the optimal quote distances

$$\delta^b_q = \theta_q - \theta_{q+1} + \frac{1}{\gamma}\ln\left( 1 + \frac{\gamma}{k} \right), \qquad \delta^a_q = \theta_q - \theta_{q-1} + \frac{1}{\gamma}\ln\left( 1 + \frac{\gamma}{k} \right).$$

The module solves this stiff non-linear system with an implicit Radau IIA integrator and an analytic Jacobian. Explicit RK4 overflows once a terminal penalty is present. It then checks that the solution satisfies the original four-variable HJB, and cross-checks it against the independent linearisation $w_q = e^{k\theta_q}$, $w' = Mw$ (Gueant, Lehalle and Fernandez-Tapia, 2013):

```python
def _rhs(tau: float, th: np.ndarray, P: MMParams) -> np.ndarray:
    q = P.qs
    bid = np.zeros_like(th)
    ask = np.zeros_like(th)
    bid[:-1] = np.exp(-P.k * (th[:-1] - th[1:]))      # no bid at q = +Q
    ask[1:] = np.exp(-P.k * (th[1:] - th[:-1]))       # no offer at q = -Q
    return -0.5 * P.sigma**2 * P.gamma * q**2 + (P.eta / P.k) * (bid + ask)
```

**The familiar closed form is a short-horizon approximation.** Avellaneda and Stoikov's reservation price and spread,

$$r = s - q\gamma\sigma^2(T - t), \qquad \delta^a + \delta^b = \gamma\sigma^2(T - t) + \frac{2}{\gamma}\ln\left( 1 + \frac{\gamma}{k} \right),$$

are the first-order expansion of the exact quotes in $T - t$, and they grow without bound as the horizon lengthens. The exact quotes converge to a stationary limit set by the ground state $v_0$ of the tridiagonal operator $M = \mathrm{diag}(\alpha q^2) - \eta(S_+ + S_-)$, with $\alpha = k\gamma\sigma^2/2$. This is a discrete Schrodinger operator: a harmonic potential with hopping. The Gueant-Lehalle-Fernandez-Tapia asymptotic

$$\delta^b \approx \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right) + \frac{2q + 1}{2}\,\omega, \qquad \omega = \sqrt{ \frac{\gamma\sigma^2}{2kA} \left( 1 + \frac{\gamma}{k} \right)^{1 + k/\gamma} }$$

is shown in the module to be the continuum approximation of that ground state. With the Avellaneda-Stoikov experimental parameters ($\sigma = 2$, $\gamma = 0.1$, $k = 1.5$, $A = 140$), the zero-inventory half-spread term is $\ln(1 + 0.1/1.5)/0.1 = 0.645$.

The module also recorded a real defect found this way. The Phase 9 solver applied $\gamma L(q)$ where the ansatz requires $L(q)$ in the terminal condition, which cut the intended liquidation penalty by a factor of ten. The two solvers agreed everywhere except near the close, the only place the bug could be seen. It is fixed in Phase 9 and guarded by a cross-phase test.

**Scope.** Avellaneda-Stoikov is a two-sided market-making model: it quotes both sides around the mid and manages inventory risk. It is not an execution algorithm for a directional parent order. For executing a directional decision at minimum cost, the analogous control problem is optimal liquidation (Almgren and Chriss, 2001; Gueant, Lehalle and Fernandez-Tapia, 2012), which uses the same Hamiltonian machinery with one side switched off.

> **Integration status (live loop).** `hjb_market_maker.py` is not imported by `main_trading_loop.py`. Step E creates `OrderIntent` objects whose default `order_type` is `"LIMIT"` with `limit_price=None`. The paper broker ignores both fields and fills the whole quantity at the yfinance last traded price. No limit order is placed, no spread is captured and no queue is modelled.

### 5.2 The asynchronous execution engine

`execution_engine.py` implements a disciplined order state machine whose governing rule is that **the state machine is the only component that emits orders**. Producers enqueue `OrderIntent` objects, and the engine moves through `BOOT -> AUTH -> RECONCILE -> WARMUP -> LIVE`, with `DEGRADED`, `HALTED` and `FLATTENING` for faults. Its controls follow industry practice:

- A token bucket caps order submission at 7 per second, below SEBI's 10 orders-per-second threshold.
- A timeout is treated as UNKNOWN, never as a rejection, and is never blindly retried, which avoids double-sizing on a lost acknowledgement.
- Any symbol with an UNKNOWN order is blocked from new intents.
- The broker is the source of truth for positions. On any discrepancy the engine adopts the broker's view and **halts**; it never trades to close the gap.
- Degraded states drop queued risk-opening intents, and recovery never returns directly to LIVE.
- "Halt is not flatten": halting cancels working orders without liquidating.

```python
async def _do_reconcile(self) -> None:
    agreed, mismatches = await self.reconciler.reconcile(dict(self.local_positions))
    if not agreed and mismatches:
        # Adopt the broker's view, then halt. Never trade to close the gap.
        self.local_positions = defaultdict(int, self.reconciler.positions)
        self._transition(EngineState.HALTED, f"position discrepancy {mismatches}")
        return
```

> **Integration notes.** `PaperTradingBroker` implements three of the eight methods of the `BrokerClient` protocol (`authenticate`, `place_order`, `get_quote`). It lacks `logout`, `cancel_order`, `get_order_by_client_id`, `get_positions` and `get_working_orders`, and the reconciler has a workaround that reads `broker.positions` directly. Fills are never reported back through `on_fill`, so the engine's local positions stay at zero while the broker's grow. On a reading of the code (not verified at runtime), the 45-second reconcile loop should therefore see a discrepancy after the first fill and move the engine to `HALTED`, by its own rule. *Update, 26 September 2026:* this was confirmed at runtime. The engine halted 45 seconds after its first fill, then logged about seven "cancel failed ... no attribute 'cancel_order'" warnings per second. It is now fixed: the paper broker reports fills through `fill_callback = engine.on_fill`, and `OrderManager.submit` no longer overwrites a fill that arrives before the acknowledgement. Separately, `yfinance` calls inside `async def place_order` and `get_quote` are synchronous network I/O. Each one blocks the single event loop, stalling both the engine and the orchestrator.

### 5.3 The paper broker ledger

The broker applies a cash ledger with a flat 3 bps fee $c$:

$$\text{BUY:} \quad C \leftarrow C - p\,q\,(1 + c), \qquad \text{SELL:} \quad C \leftarrow C + p\,q\,(1 - c).$$

```python
trade_value = live_price * qty
fee = trade_value * self.brokerage_rate

if side == "BUY":
    if self.capital < (trade_value + fee):
        # (insufficient-funds log line omitted)
        return None
    self.capital -= (trade_value + fee)
    self.positions[symbol] = self.positions.get(symbol, 0) + qty
    
elif side == "SELL":
    current_qty = self.positions.get(symbol, 0)
    if current_qty < qty:
        logging.error(f"Paper Broker: INSUFFICIENT SHARES. Trying to sell {qty}, own {current_qty}")
        return None
    self.capital += (trade_value - fee)
    self.positions[symbol] -= qty
```

Four properties of this ledger matter for interpreting any displayed result:

1. **Long-only.** A SELL without inventory is rejected, so short signals mostly fail.
2. **No mark-to-market.** Equity is $E = C + \sum_i q_i p_i$, but only the cash $C$ is exported, under the name `current_pnl`. The saved dashboard state shows Rs. 4,75,109.55. That is cash after purchases, not a loss of Rs. 24,890.45.
3. **Inconsistent costs.** The ledger charges 3 bps, while the Kelly sizer's `CostModel` assumes 13 bps of fees and STT plus half-spread plus square-root impact. Sizing and accounting should share one cost model.
4. **Encoding hazard.** The file contained the rupee glyph in three f-strings, which is the same class of CP1252 failure recorded in the project's bug log. *Fixed 26 September 2026:* they now use "Rs.". `historical_backtest.py` still contains the glyph (lines 69, 147 and 148).

### 5.4 The multivariate GBM synthetic market feed

**Model.** Multivariate geometric Brownian motion gives each asset the dynamics

$$dS_i = \mu_i S_i\, dt + \sigma_i S_i\, dW_i, \qquad d\langle W_i, W_j \rangle_t = \rho_{ij}\, dt.$$

It is simulated exactly by factoring the correlation matrix, $C = LL^\top$ (Cholesky), and setting

$$S_i(t + \Delta) = S_i(t) \exp\left( \left( \mu_i - \tfrac{1}{2}\sigma_i^2 \right)\Delta + \sigma_i \sqrt{\Delta}\, (L z)_i \right), \qquad z \sim \mathcal{N}(0, I).$$

The feed uses the first-order version with zero drift, $S \leftarrow S(1 + \varepsilon)$ with $\varepsilon = Lz$ and $z \sim \mathcal{N}(0, 0.0005^2 I)$. At this shock size the difference from the exact scheme is negligible per step. The multiplicative update makes prices a martingale, and the log price drifts down by about $\tfrac{1}{2}\sigma^2$ per step.

```python
corr_matrix = np.full((N_STOCKS, N_STOCKS), 0.3) # 0.3 base market correlation
np.fill_diagonal(corr_matrix, 1.0)
cholesky_lower = np.linalg.cholesky(corr_matrix)

state_file = "D:/Quant_Project/live_state.json"

tick_count = 0

while True:
    try:
        # Generate 500 correlated random shocks in microseconds
        uncorrelated_shocks = np.random.normal(0, 0.0005, N_STOCKS) # 0.05% volatility per tick
        correlated_shocks = cholesky_lower.dot(uncorrelated_shocks)
        
        # Apply shocks to all 500 equity prices simultaneously (Geometric Brownian Motion)
        base_prices = base_prices * (1 + correlated_shocks)
```

**Properties.**

- *Equicorrelation.* The matrix $C = (1-\rho)I + \rho\mathbf{1}\mathbf{1}^\top$ with $\rho = 0.3$ has one market eigenvalue $1 + 499 \times 0.3 = 150.7$ and 499 eigenvalues of 0.7. It is positive definite, so the Cholesky factor exists. There is a single market factor and **no sector structure**, despite the code comment.
- *Calibration.* At one step per second over a 6 h 15 min NSE session (22,500 steps), the per-step volatility of 0.05 percent compounds to $0.0005\sqrt{22{,}500} = 7.5$ percent daily per stock, roughly five times a typical NSE large cap. That makes the feed a stress scenario rather than a calibrated one.
- *Prices.* Base prices are lognormal with median $e^5 \approx$ Rs. 148, clipped to Rs. 5 to Rs. 25,000.
- *Dead code.* The option-pricing branch for `_CE` / `_PE` symbols never runs, because no generated symbol has those suffixes.

> **Integration status.** The feed writes ticks to the data lake and a display record to `live_state.json`. **Neither is read by `main_trading_loop.py`.** The brain and the feed are currently disconnected.

### 5.5 Telemetry: the dashboard server and the React terminal

The dashboard server is a standard-library `http.server` (Flask is not used) on port 5000 with two routes. `/` serves the HTML, and `/api/live_state` proxies the JSON state file:

```python
elif self.path == '/api/live_state':
    self.send_header('Content-type', 'application/json')
    self.end_headers()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'rb') as f:
            self.wfile.write(f.read())
    else:
        default_data = json.dumps({
            "current_pnl": 500000.0,
            "toxicity_score": 0.0,
            "quantum_assets": [],
            "recent_trades": []
        })
        self.wfile.write(default_data.encode('utf-8'))
```

The orchestrator writes that contract at the end of each cycle:

```python
dashboard_data = {
    "current_pnl": float(engine.broker.capital),
    "toxicity_score": float(np.mean(tox_values)) if 'tox_values' in locals() and len(tox_values) > 0 else 0.0,
    "quantum_assets": q_assets,
    "recent_trades": [] # The broker trade history parsing could go here, omitting for simplicity
}
```

`index.html` is a single-file React 18 terminal (UMD builds from a CDN, in-browser Babel, Tailwind and Chart.js). It is pure ASCII by design. Its header documents one behaviour that matters for anyone reading its screens:

```
  Data    polls GET http://localhost:5000/api/live_state every 2 s.
          AUTO mode falls back to a built-in simulator while the backend is down
          and re-probes it every 10 s.
```

**Reading the dashboard correctly.**

- *PnL panel:* shows broker cash (Section 5.3), not marked-to-market PnL.
- *Toxicity chart:* always 0.0 in live mode (Section 4.2).
- *Execution tape:* always empty, because `recent_trades` is hard-coded to `[]`.
- *Simulator fallback:* when the backend is unreachable, the terminal runs a built-in simulator and labels it ("backend unreachable, retrying" or "demo feed"). Screens captured in that mode show simulated data and must never be presented as system output.
- *Network exposure:* the server binds `0.0.0.0` and sends `Access-Control-Allow-Origin: *`, which exposes the state to the local network. It should bind `127.0.0.1`.
- *404 bug:* for unknown paths the handler sends a 200 status line before the 404.

### 5.6 Operations

- **Schedule.** The Windows task `TitanQ_Live_Trading` (state Ready, confirmed) runs `start_titan_q.bat` daily at 09:10 IST. The batch file starts three processes: the feed, the trading loop and the dashboard server.
- **Kill switch.** The loop halts on `now.hour == 22 and now.minute >= 40`, a condition that is true only from 22:40 to 22:59. **At 23:00 it becomes false and trading resumes overnight.** This is consistent with the overnight yfinance SSL errors in the project log, when the router's block page intercepts requests. The NSE cash session runs 09:15 to 15:30 IST, and orders sent outside it are paper-filled at stale last prices.
- **Dtype incident.** `neural_sde_discovery.py` sets the global PyTorch default dtype to float64 at import, which silently changes the dtype of every module constructed afterwards. The current fix casts the inputs to float64. The durable fix is to remove the global side effect from the module.
- **Phase 25.** The C++ transpiler's tests are blocked because `g++` is not on PATH. That module was not reviewed for this paper.

---

## 6. Validation: the Historical Backtest Re-examined

The figures circulated for Titan-Q are a win rate of 53.3 to 58.7 percent and a Sharpe ratio of 11.46 over "10,000 ticks". They come from `historical_backtest.py`, whose core loop is:

```python
with torch.no_grad():
    # Predict direction
    forecast_logits = lob_transformer(lob_window)
    forecast_probs = torch.softmax(forecast_logits, dim=-1)[0].numpy()
    
    # Discover Drift/Diffusion for this exact historical tick
    idx_state = torch.tensor([[0.5]], dtype=torch.float64)
    dyn_mu = sde_model.mu(idx_state).item() * 252.0
    dyn_sigma = sde_model.sigma(idx_state).item() * np.sqrt(252.0)
    
# --- Strategy Logic ---
# If AI predicts strong upside (prob > 0.4) and TE causality is high
if forecast_probs[2] > 0.4 and te_bits > 1.0:
    # ... Kelly sizing and EVT trim omitted ...
    trade_amount = capital * fraction
    
    # Simulate a historical PnL outcome based on rough drift
    pnl_pct = np.random.normal(0.0005, 0.002) # 5 bps mean return on winning trades
    capital += trade_amount * pnl_pct
    
elif forecast_probs[0] > 0.4 and te_bits > 1.0:
    # Short signal logic (similar but inverted)
    fraction = 0.05
    trade_amount = capital * fraction
    pnl_pct = np.random.normal(0.0002, 0.002) 
    capital += trade_amount * pnl_pct
```

and the Sharpe ratio is annualised as `np.sqrt(252 * 75) * (returns.mean() / (returns.std() + 1e-9))`. Re-reading the harness and recomputing from its saved output (`equity_curve_results.csv`) gives the following.

1. **Length.** `n_ticks = 2000` in both branches, and the CSV has 2,001 rows. The run covered 2,000 ticks, not 10,000.
2. **No historical data is read.** `DATA_LAKE_PATH = "G:\\ My Drive\\NIFTY_Data"` contains a stray space after the drive separator, so the path check fails. Even when it succeeds, no file is ever loaded.
3. **Every input is random.** The LOB window is `torch.randn`, the transfer-entropy series are random, and the transformer and SDE are untrained.
4. **The outcome is drawn independently of the signal.** Each trade's return is a fresh draw from $\mathcal{N}(+5\text{ bp}, 20\text{ bp})$ in the long branch and $\mathcal{N}(+2\text{ bp}, 20\text{ bp})$ in the short branch. The short branch *adds* a positive-mean draw, so shorts profit on average by construction.
5. **In practice only the short branch trades.** The untrained transformer on random input puts more than 0.4 on "Down" in 88 percent of draws and on "Up" in 1.7 percent. The TE gate always passes, because independent series exceed 1.0 bit in 100 percent of draws (Section 2.3). The long branch's Kelly fraction is 0 (`negative_net_edge`, Section 3.3), so its returns are exactly zero. The recomputed run has 1,695 active ticks out of 2,000.
6. **The win rate is a constant of the random generator.** A trade with return $\varepsilon \sim \mathcal{N}(m, s^2)$ wins with probability $\Phi(m/s)$:

$$\Phi\left( \frac{0.0002}{0.002} \right) = \Phi(0.10) = 54.0\% \ \text{(short branch)}, \qquad \Phi\left( \frac{0.0005}{0.002} \right) = \Phi(0.25) = 59.9\% \ \text{(long branch)}.$$

   The recomputed win rate is **53.33 percent**, within one standard error (1.2 points) of 54.0 percent. The quoted range of 53.3 to 58.7 percent lies between the two constants.
7. **The Sharpe ratio is a constant times an annualisation factor.** With a fixed fraction 0.05, each active tick returns $0.05\,\varepsilon$, so the per-trade Sharpe ratio is $m/s = 0.1$. With active fraction $p$, the per-tick Sharpe ratio is approximately $\sqrt{p}\, m/s$. Hence

$$\widehat{\mathrm{SR}} \approx \sqrt{252 \times 75} \times \sqrt{0.8475} \times 0.1 = 137.5 \times 0.921 \times 0.1 = 12.7,$$

   with sampling standard error $\approx 137.5/\sqrt{2000} = 3.1$. The recomputed value from the CSV is **11.46**, matching the reported figure exactly and consistent with 12.7 within half a standard error. The "75 ticks per day" annualisation is itself an assumption with no basis in the data.
8. **Sizing cannot "size up winners".** The Kelly fraction and the EVT trim are fixed before each trade's outcome is drawn, from quantities independent of it. They scale winners and losers alike, and the EVT trim is itself a random draw.

**Conclusion.** `historical_backtest.py` is a useful **pipeline smoke test**: it proves the modules import and run together end to end. It is not a performance estimate. Its win rate and Sharpe ratio were chosen, implicitly, when the constants 0.0005, 0.0002 and 0.002 were typed. Those figures should be withdrawn from any description of the system.

A valid backtest needs five things:

1. Historical NSE tick or book data replayed point-in-time.
2. Models trained on an earlier window and frozen before the test window.
3. A purged, walk-forward evaluation, following the protocol the Phase 18 module already implements.
4. The full Indian cost stack: brokerage, STT, exchange transaction charges, SEBI fees, stamp duty, GST and slippage.
5. A Sharpe ratio deflated for the number of configurations tried (Bailey and Lopez de Prado, 2014).

After that, the frozen code should run in paper trading before any capital is committed.

---

## 7. Integration Status Matrix

| Phase | Module | Library status | Called by live loop? | What the live loop does today |
|---|---|---|---|---|
| 4 Kelly | `fusion_and_sizing.py` | Complete, with shrinkage, costs and moments | Yes | Receives untrained SDE outputs; returns 0; orders fall back to 1 share |
| 5 PPO policy | `stable_baselines3` | Checkpoint path missing | Attempted | Random BUY/HOLD/SELL (50/25/25) on a random observation |
| 11 HMM | `regime_hmm_detector.py` | Complete: Baum-Welch, Viterbi, causal filter | No (mock) | Regime drawn at random 80/15/5; CRASH skips the cycle; no liquidation |
| 12 HRP | `portfolio_hrp_allocator.py` | Complete, MST check | Yes | Real HRP on a 4x4 block of a random covariance |
| 13 Copula | `stat_arb_copula.py` | Not reviewed | Constructed only | Unused |
| 14 Hawkes / VPIN | `hft_hawkes_toxicity.py` | Complete: MLE, filter, VPIN, score | Partly | Intensity pinned at 17.45 by a synthetic clock; VPIN uniform(0.3, 0.7) |
| 16 Spoof AE | `adversarial_spoof_detector.py` | Complete, with stated limits | No | Abort on `random.random() < 0.02` |
| 17 Signatures | `rough_volatility_signatures.py` | Not reviewed in depth | Yes | Computed on a random path; logged only |
| 18 TFT | `transformer_lob_forecaster.py` | Complete, purged evaluation | Yes, untrained | Random weights on Gaussian noise; logged only |
| 20 Annealer | `quantum_portfolio_annealer.py` | Complete, exact cross-checks | Yes | Real swap annealing on a random problem; tickers by list position |
| 21 Neural SDE | `neural_sde_discovery.py` | Complete | Yes, untrained | Constant outputs at state 0.5 |
| 22 HJB | `hjb_market_maker.py` | Complete, cross-validated | No | Orders fill at the last traded price |
| 23 EVT | `extreme_value_theory_risk.py` | Complete, profile CIs | No | ES drawn uniform(1%, 15%); trims 21.4% of signals |
| 24 TE | `transfer_entropy_causality.py` | Complete, surrogate test | Partly | Raw biased TE on synthetic series; logged only |
| 25 C++ transpiler | `alpha_cpp_transpiler.py` | Not reviewed | No | Tests blocked: no `g++` on PATH |
| Engine | `execution_engine.py` | Complete state machine | Yes | Paper broker implements 3 of 8 protocol methods; fills now reach the engine, so it no longer self-halts (fixed 26 Sep) |
| Broker | `paper_broker.py` | Working, long-only | Yes | Cash exported as "PnL"; no mark-to-market |
| Feed | `nse_whole_market_feed.py` | Working | Separate process | Output not consumed by the loop |
| Dashboard | `dashboard_server.py`, `index.html` | Working | Yes | Toxicity always 0; trade tape always empty |

---

## 8. Roadmap to a Defensible Live System

The items are ordered by priority. The first group fixes what is shown to people. The later groups make the system's decisions depend on the market.

**P0: accuracy of what is reported**

1. Withdraw the win-rate and Sharpe figures from all material, and label `historical_backtest.py` as a smoke test in its docstring and log output.
2. Export equity ($C + \sum q_i p_i$), the real toxicity score and a trade list to the dashboard, and rename `current_pnl` or compute it as equity minus initial capital.
3. Replace the kill switch with an NSE session gate (09:15 to 15:30 IST plus the exchange holiday calendar).
4. Replace the rupee glyph with "Rs." in `paper_broker.py` and `historical_backtest.py`.

**P1: connect the brain to data**

5. Make the orchestrator consume a market feed, either the synthetic feed or a real NSE feed, and derive *every* model input from it:
   - book windows for Phase 18;
   - return, realised-variance and volume features for Phase 11;
   - trade timestamps for Hawkes;
   - price and volume buckets for VPIN;
   - position loss distributions for EVT;
   - estimated means and covariances for the annealer and HRP.
6. Train, freeze and version every learned component (TFT, neural SDE, autoencoder, HMM, PPO). Load them by absolute path with `load_state_dict` or `PPO.load`, and **refuse to trade** if an artifact is missing instead of falling back to random choices.
7. Replace each placeholder with the real call: `MarketRegimeFilter.update`, `SpoofDetector.detect`, `VPINEstimator.compute`, `POTModel.es`, and `te_test` with surrogates, conditioned on the market factor.
8. Derive direction from calibrated model outputs, for example the TFT's probabilities after checking its expected calibration error. The PPO observation should be the real feature vector.
9. Remove the global `set_default_dtype` call from `neural_sde_discovery.py`.

**P2: execution realism**

10. Complete the paper broker against the `BrokerClient` protocol (positions, working orders, cancels, fills via `on_fill`), add limit orders with a fill model, and share one cost model with the Kelly sizer.
11. Use the HJB quoter for market making, or an optimal-liquidation schedule for directional orders.
12. Move blocking I/O (`yfinance`) into `asyncio.to_thread`.
13. At $K = 4$, select assets by exact enumeration (0.1 s) and keep annealing for larger $K$.
14. Bind the dashboard to `127.0.0.1`.

**P3: validation**

15. Build the real backtest described at the end of Section 6, then run a fixed-code paper-trading period before committing any capital.

---

## 9. Conclusion

Titan-Q's phase libraries are strong: in their mathematics, in their implementation and, above all, in their honesty about their own limits. Several examples show this:

- the purged, oracle-bounded evaluation of the LOB transformer;
- the demonstration that pathwise losses destroy a neural SDE's diffusion;
- the Dinkelbach reduction and exact cross-checks behind the annealer;
- the profile-likelihood EVT tail and the causal regime filter;
- the two independent HJB solvers that caught a real bug.

Each of these is work of institutional quality.

The integrated system is not yet a trading system in the sense its documentation implies. Today, the live loop draws its direction, regime, spoofing flag, tail-risk estimate and VPIN from random number generators. It evaluates untrained networks on noise and sends 1-share orders to a long-only paper ledger that reports cash as PnL. Its headline backtest statistics are consequences of two hard-coded constants. None of this is a criticism of the research; it is a statement of where the project stands. The path forward is concrete, and Section 8 lists it in order. The research layer is already sound. What remains is to feed it market data, train and freeze its models, and hold the result to the standard the modules already set for themselves.

---

## 10. References

- Almgren, R. and Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk* 3.
- Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3).
- Bailey, D. and Lopez de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management* 40(5).
- Balkema, A. and de Haan, L. (1974). Residual life time at great age. *Annals of Probability* 2(5).
- Barnett, L., Barrett, A. and Seth, A. (2009). Granger causality and transfer entropy are equivalent for Gaussian variables. *Physical Review Letters* 103.
- Bienstock, D. (1996). Computational study of a family of mixed-integer quadratic programming problems. *Mathematical Programming* 74.
- Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1).
- Dinkelbach, W. (1967). On nonlinear fractional programming. *Management Science* 13(7).
- Easley, D., Lopez de Prado, M. and O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies* 25(5).
- Goto, H., Tatsumura, K. and Dixon, A. (2019). Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems. *Science Advances* 5(4).
- Gueant, O., Lehalle, C.-A. and Fernandez-Tapia, J. (2012). Optimal portfolio liquidation with limit orders. *SIAM Journal on Financial Mathematics* 3.
- Gueant, O., Lehalle, C.-A. and Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics* 7(4).
- Hajek, B. (1988). Cooling schedules for optimal annealing. *Mathematics of Operations Research* 13(2).
- Hawkes, A. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1).
- Kadowaki, T. and Nishimori, H. (1998). Quantum annealing in the transverse Ising model. *Physical Review E* 58.
- Kelly, J. (1956). A new interpretation of information rate. *Bell System Technical Journal* 35(4).
- Kirkpatrick, S., Gelatt, C. and Vecchi, M. (1983). Optimization by simulated annealing. *Science* 220.
- Li, X., Wong, T.-K. L., Chen, R. T. Q. and Duvenaud, D. (2020). Scalable gradients for stochastic differential equations. *AISTATS*.
- Lim, B., Arik, S., Loeff, N. and Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting* 37(4).
- Lopez de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management* 42(4).
- Lucas, A. (2014). Ising formulations of many NP problems. *Frontiers in Physics* 2.
- Lyons, T. (1998). Differential equations driven by rough signals. *Revista Matematica Iberoamericana* 14(2).
- Marchenko, V. and Pastur, L. (1967). Distribution of eigenvalues for some sets of random matrices. *Mathematics of the USSR-Sbornik* 1(4).
- McNeil, A. and Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *Journal of Empirical Finance* 7.
- Merton, R. (1980). On estimating the expected return on the market. *Journal of Financial Economics* 8(4).
- Ntakaris, A. et al. (2018). Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods. *Journal of Forecasting* 37(8).
- Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics* 3(1).
- Rabiner, L. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE* 77(2).
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance* 39(4).
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters* 85(2).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Viterbi, A. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. *IEEE Transactions on Information Theory* 13(2).
- Zhang, L., Mykland, P. and Ait-Sahalia, Y. (2005). A tale of two time scales. *Journal of the American Statistical Association* 100.
- Zhang, Z., Zohren, S. and Roberts, S. (2019). DeepLOB: deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing* 67(11).

---

## Appendix A: Notation

| Symbol | Meaning |
|---|---|
| $m_t$, $S_t$ | mid-price / asset price |
| $L$, $W$, $H$ | book levels, lookback window, label horizon (ticks) |
| $I_t$, OFI | depth imbalance, order flow imbalance |
| $\mu$, $\sigma$ | drift and diffusion (per unit time unless annualised) |
| $\Delta$ | sampling interval |
| $\theta$, $m$ | OU mean-reversion speed and level; also the Dinkelbach ratio in 3.1 |
| $x$, $s$ | binary selection vector; Ising spins $s = 2x - 1$ |
| $Q$, $J$, $h$ | QUBO matrix; Ising couplings and fields |
| $T$ (Section 3.1) | annealing temperature |
| $f^*$, $\lambda$, $\tau$ | Kelly fraction, Kelly multiplier, prior edge volatility |
| $\xi$, $\beta$, $u$ | GPD shape, scale, threshold |
| $\lambda(t)$, $\alpha$, $\beta$, $n$ | Hawkes intensity, jump, decay, branching ratio |
| $A$, $b_k$, $\delta_t$, $\alpha_t$ | HMM transitions, emissions, Viterbi scores, forward variables |
| $\gamma$, $k$, $A$, $\theta_q$ | CARA risk aversion, fill decay, fill scale, HJB inventory value |

## Appendix B: File map

| File | Role |
|---|---|
| `main_trading_loop.py` | Master orchestrator and asyncio entry point |
| `execution_engine.py` | Order state machine, rate limits, reconciliation |
| `paper_broker.py` | Paper ledger, fills at yfinance last price |
| `nse_whole_market_feed.py` | 500-asset correlated GBM feed |
| `dashboard_server.py` | Standard-library HTTP server on :5000 |
| `C:\Users\jiten\titan-q-dashboard\index.html` | React 18 telemetry terminal |
| `start_titan_q.bat` | Launches feed, loop and dashboard |
| `historical_backtest.py` | End-to-end smoke test (Section 6) |
| `transformer_lob_forecaster.py` | Phase 18 |
| `neural_sde_discovery.py` | Phase 21 |
| `transfer_entropy_causality.py` | Phase 24 |
| `quantum_portfolio_annealer.py` | Phase 20 |
| `portfolio_hrp_allocator.py` | Phase 12 |
| `fusion_and_sizing.py` | Phase 4 Kelly sizer and cost model |
| `extreme_value_theory_risk.py` | Phase 23 |
| `hft_hawkes_toxicity.py` | Phase 14 |
| `adversarial_spoof_detector.py` | Phase 16 |
| `regime_hmm_detector.py` | Phase 11 |
| `hjb_market_maker.py` | Phase 22 |

## Appendix C: Reproducing the numbers in this paper

The backtest statistics in Section 6 can be recomputed from the saved equity curve with the project's own interpreter (`D:\trader_env\Scripts\python.exe`):

```python
import pandas as pd, numpy as np
from scipy.stats import norm
df = pd.read_csv(r"D:\Quant_Project\equity_curve_results.csv")
r = df["Equity"].pct_change().dropna()
act = r[r != 0]
print("ticks", len(r), "active", len(act))                              # 2000, 1695
print("win rate", (act > 0).mean())                                      # 0.5333
print("Sharpe", np.sqrt(252 * 75) * r.mean() / (r.std() + 1e-9))         # 11.46
print("Phi(0.10)", norm.cdf(0.10), "Phi(0.25)", norm.cdf(0.25))          # 0.540, 0.599
```

The live-loop measurements in Sections 2 to 4 were produced by instantiating each module exactly as `main_trading_loop.py` does (same constructor arguments, same import order, and hence the same float64 default dtype) and evaluating it on the same inputs:

- the neural SDE's outputs at state 0.5 and the Kelly sizer's result;
- the Hawkes fixed point and toxicity halt rates over 5,000 cycles for five boot seeds;
- the TFT's output distribution over 300 random windows;
- plug-in transfer entropy for coupled and independent series over 300 draws;
- brute force versus swap annealing at $N = 50$, $K = 4$.

The Phase 18 table in Section 2.1 is the unmodified output of `transformer_lob_forecaster._main_experiment()` (118 s on this machine).

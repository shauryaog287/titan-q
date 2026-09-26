"""
graph_alpha_engine.py
=====================

Phase 8, Engine I: Spatio-Temporal Graph Neural Network for cross-sectional
residual return forecasting on the Nifty 50.

ARCHITECTURAL SKELETON. Method bodies are intentionally unimplemented; the
contracts, type signatures and invariants are the deliverable.

Design horizon
--------------
5 seconds to 15 minutes. NOT milliseconds. A K=3 diffusion pass over 50 nodes
costs 0.3-2 ms before feature construction and order encode; a colocated
latency-arbitrage participant runs 50-100 us end to end with a fixed linear
rule. At millisecond resolution, inter-constituent lead-lag is dominated by
mechanical index-arb and ETF creation flow rather than information diffusion.
This engine competes on statistical quality, not on wire length.

Invariants enforced across the module
-------------------------------------
I1. Covariance and lead-lag are estimated by Hayashi-Yoshida on raw event
    times. No resampling to a common grid anywhere upstream of the adjacency
    (previous-tick interpolation reintroduces the Epps attenuation, which is
    differential across liquidity tiers and therefore encodes ADV rank into
    the topology).
I2. Directed slices are NEVER symmetrised. `(A + A.T)/2` annihilates the
    lead-lag asymmetry that is the entire signal. Spectral convolution is
    therefore structurally unavailable; message passing only.
I3. Targets and edge features are built from FACTOR RESIDUALS, with betas
    estimated on a strictly prior window and held fixed through the
    prediction period.
I4. Normalisation is cross-sectional within each bar, never per-symbol across
    time.
I5. Every tensor carries the vintage of every input that produced it,
    including index membership.

Dependencies: numpy, pandas, torch, torch_geometric (optional), scipy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "RelationType",
    "EdgeEstimate",
    "AdjacencyTensor",
    "GraphVintage",
    "HayashiYoshidaEstimator",
    "LeadLagDetector",
    "IntensityPreservingBootstrap",
    "FactorResidualiser",
    "NodeFeatureBuilder",
    "CrossSectionalNormaliser",
    "AdjacencyBuilder",
    "AdaptiveAdjacency",
    "BidirectionalDiffusionConv",
    "EdgeConditionedAttention",
    "GatedDilatedTCN",
    "SpatioTemporalBlock",
    "GraphAlphaNet",
    "GrangerSparsityPenalty",
    "AcyclicityConstraint",
    "CompositeLoss",
    "GraphAblationSuite",
    "EdgeStabilityMonitor",
    "GraphAlphaEngine",
]


# ===========================================================================
# 1. TYPES AND CONTRACTS
# ===========================================================================

class RelationType(Enum):
    """
    Slices of the adjacency tensor. Each has its own decay profile, update
    cadence and directedness; collapsing them to a scalar edge weight
    discards information the network can use.
    """

    LEAD_LAG = auto()      # directed, HY cross-correlation at optimal lag
    CONTEMPORANEOUS = auto()  # undirected, HY at zero lag, LW-shrunk
    SECTOR = auto()        # undirected, static, GICS sub-industry
    COINTEGRATION = auto()  # undirected, from Phase 6, weighted by |z|
    ORDER_FLOW = auto()    # directed, OFI transfer from Phase 3 Lee-Ready
    ADAPTIVE = auto()      # directed, learned, low-rank


@dataclass(frozen=True)
class EdgeEstimate:
    """
    A single directed edge with its provenance. `passed_fdr` is not optional
    metadata -- an edge that has not cleared the intensity-preserving null
    under FDR control must not enter the tensor.
    """

    source: str
    target: str
    relation: RelationType
    weight: float
    optimal_lag_us: Optional[int]
    rho_at_zero_lag: Optional[float]
    bootstrap_pvalue: Optional[float]
    passed_fdr: bool
    n_overlapping_intervals: int

    @property
    def directional_content(self) -> Optional[float]:
        """
        |rho(delta*)| / |rho(0)|. Near 1.0 means the linkage is contemporaneous
        and the edge carries no directional information -- prune it.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class GraphVintage:
    """
    Point-in-time provenance. Without index-membership vintage, training over
    a multi-year history using today's Nifty constituents is survivorship bias
    with a graph attached, and it is invisible in every diagnostic.
    """

    as_of: pd.Timestamp
    beta_estimation_end: pd.Timestamp
    corporate_action_vintage: pd.Timestamp
    index_membership_vintage: pd.Timestamp
    symbols: Tuple[str, ...]
    edge_window_start: pd.Timestamp
    edge_window_end: pd.Timestamp

    def assert_no_leakage(self, prediction_ts: pd.Timestamp) -> None:
        """Raise if any vintage post-dates the prediction timestamp."""
        raise NotImplementedError


@dataclass
class AdjacencyTensor:
    """
    Dense tensor (N, N, R) plus the dual row-normalised transition operators
    per slice. Holds `P_out` and `P_in` separately -- see invariant I2.
    """

    symbols: Tuple[str, ...]
    relations: Tuple[RelationType, ...]
    raw: Tensor                    # (N, N, R)
    p_out: Tensor                  # (N, N, R), D_O^-1 A
    p_in: Tensor                   # (N, N, R), D_I^-1 A^T
    vintage: GraphVintage
    node_mask: Tensor              # (N,) bool; stale nodes excluded from passing

    def density(self, relation: RelationType) -> float:
        raise NotImplementedError

    def assert_not_symmetrised(self, tol: float = 1e-6) -> None:
        """
        Guard on invariant I2. Raises if a directed slice has become
        symmetric, which is the signature of a library call that silently
        applied (A + A.T)/2.
        """
        raise NotImplementedError

    def top_k_edges(self, relation: RelationType, k: int) -> List[Tuple[int, int, float]]:
        raise NotImplementedError


class MarketDataSource(Protocol):
    """Event-time tick access. Implement against the NSE feed handler."""

    def trades(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...

    def quotes(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...

    def index_membership(self, as_of: pd.Timestamp) -> Tuple[str, ...]: ...

    def corporate_actions(self, as_of: pd.Timestamp) -> pd.DataFrame: ...


# ===========================================================================
# 2. EDGE ESTIMATION
# ===========================================================================

class HayashiYoshidaEstimator:
    """
    Integrated covariance under asynchronous observation.

        [X^i, X^j]_T = sum_k sum_l dX^i_k dX^j_l
                       * 1{ (t^i_{k-1}, t^i_k] cap (t^j_{l-1}, t^j_l] != empty }

    Accumulates a product for every pair of return intervals overlapping in
    time. No interpolation, no common grid, no Epps attenuation. This is the
    only covariance estimator permitted upstream of the adjacency.

    Complexity is O(n_i + n_j) with a merge-scan over sorted interval
    endpoints, NOT O(n_i * n_j) -- the naive double loop is intractable at
    tick frequency and the merge-scan implementation is the whole engineering
    content of this class.
    """

    def __init__(self, min_overlapping_intervals: int = 200) -> None: ...

    def covariance(
        self,
        returns_i: np.ndarray,
        interval_bounds_i: np.ndarray,
        returns_j: np.ndarray,
        interval_bounds_j: np.ndarray,
    ) -> Tuple[float, int]:
        """Returns (integrated covariance, n overlapping interval pairs)."""
        raise NotImplementedError

    def correlation(
        self,
        returns_i: np.ndarray,
        interval_bounds_i: np.ndarray,
        returns_j: np.ndarray,
        interval_bounds_j: np.ndarray,
    ) -> Tuple[float, int]:
        raise NotImplementedError

    def covariance_matrix(
        self, event_series: Mapping[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full N x N HY covariance plus the matching count matrix. The result is
        NOT guaranteed positive semi-definite -- HY is consistent pairwise but
        the assembled matrix can have negative eigenvalues in finite samples.
        Project to the nearest PSD matrix (Higham) before any use requiring
        it, and record the Frobenius distance of that projection as a data
        quality metric.
        """
        raise NotImplementedError


class LeadLagDetector:
    """
    Huth-Abergel lead-lag: shift asset j's intervals by delta and maximise the
    HY cross-correlation over a candidate lag grid.

        delta*_ij = argmax_delta | rho_ij(delta) |

    sign(delta*) gives edge direction: positive means i leads j.
    """

    def __init__(
        self,
        estimator: HayashiYoshidaEstimator,
        lag_grid_us: Sequence[int],
        min_directional_content: float = 1.05,
    ) -> None: ...

    def cross_correlation_surface(
        self,
        series_i: Tuple[np.ndarray, np.ndarray],
        series_j: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """rho(delta) over the lag grid."""
        raise NotImplementedError

    def detect(
        self,
        series_i: Tuple[np.ndarray, np.ndarray],
        series_j: Tuple[np.ndarray, np.ndarray],
        symbol_i: str,
        symbol_j: str,
    ) -> EdgeEstimate:
        raise NotImplementedError


class IntensityPreservingBootstrap:
    """
    Null distribution for lead-lag significance.

    Two mechanisms manufacture edges that do not exist, and both are defeated
    by the same resampling scheme:

    (a) INTENSITY ASYMMETRY. If i trades 10x more often than j, the HY overlap
        indicator fires asymmetrically across positive and negative delta and
        the cross-correlation function acquires a skew unrelated to any
        information flow. Untreated, the recovered topology reproduces the ADV
        ranking and predicts nothing.

    (b) MULTIPLE TESTING. 50 nodes = 2,450 directed pairs; at 40 candidate
        lags that is 98,000 hypotheses. ~4,900 clear at a nominal 5% level on
        noise alone.

    The scheme: circular block bootstrap over the RETURN MARKS while holding
    each asset's own ARRIVAL TIMESTAMPS fixed. This preserves the intensity
    process exactly while destroying genuine cross-asset timing, so the
    resulting null contains the artefact but not the signal.
    """

    def __init__(self, n_resamples: int = 1000, block_length_us: int = 5_000_000) -> None: ...

    def null_distribution(
        self,
        series_i: Tuple[np.ndarray, np.ndarray],
        series_j: Tuple[np.ndarray, np.ndarray],
        lag_grid_us: Sequence[int],
    ) -> np.ndarray:
        """(n_resamples, n_lags) array of |rho| under the null."""
        raise NotImplementedError

    def pvalue(self, observed_abs_rho: float, null_draws: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def benjamini_hochberg(
        pvalues: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, float]:
        """Returns (reject mask, adaptive threshold)."""
        raise NotImplementedError

    def diagnose_intensity_artefact(
        self, surviving: Sequence[EdgeEstimate], adv_rank: Mapping[str, int]
    ) -> Dict[str, float]:
        """
        Rank correlation between surviving edge direction and ADV rank. A
        strong positive value means the bootstrap is not preserving intensity
        correctly and the topology is an ADV ranking. Gate G1.
        """
        raise NotImplementedError


# ===========================================================================
# 3. FEATURE PIPELINE
# ===========================================================================

class FactorResidualiser:
    """
    Strip market, sector and style before targets are formed.

    Nifty constituents share 40-70% of their variance with the index. On raw
    returns the network learns beta, scores well on MSE, carries no
    cross-sectional information, and drives the graph toward a complete
    topology because every node loads on the same factor.

    Betas MUST be fitted on a strictly prior window and frozen through the
    prediction period. Fitting on a window containing t+s leaks the factor
    realisation into the residual; the leak is large enough to produce a
    spectacular backtest that dies immediately in production.
    """

    def __init__(
        self,
        factors: Sequence[str] = ("market", "sector", "size", "value"),
        estimation_window_days: int = 120,
        refit_frequency_days: int = 21,
        shrink_betas: bool = True,
    ) -> None: ...

    def fit(
        self, returns: pd.DataFrame, factor_returns: pd.DataFrame, as_of: pd.Timestamp
    ) -> "FactorResidualiser": ...

    def residualise(
        self, returns: pd.DataFrame, factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError

    def forward_residual_target(
        self, returns: pd.DataFrame, factor_returns: pd.DataFrame, horizons: Sequence[int]
    ) -> Dict[int, pd.DataFrame]:
        """y_{i,t}^{(h)} = sum_{s=1..h} residual_{i,t+s}, one frame per horizon."""
        raise NotImplementedError

    @property
    def beta_vintage(self) -> pd.Timestamp: ...


class NodeFeatureBuilder:
    """
    Per-node feature vector. Consumes the Phase 3 microstructure outputs
    rather than recomputing them.

        x_it = [ residual_return, realised_vol, OFI, VPIN, hawkes_intensity,
                 depth_imbalance, spread, sin/cos(time_of_day) ]

    Time-of-day is encoded circularly. Intraday seasonality in volatility and
    spread is strong; without the encoding the network spends capacity
    rediscovering the U-shape.
    """

    def __init__(
        self,
        include_microstructure: bool = True,
        realised_vol_windows: Sequence[int] = (12, 60, 240),
        sampling: Literal["volume_clock", "trade_clock", "tick_clock"] = "volume_clock",
    ) -> None: ...

    def build(
        self,
        residuals: pd.DataFrame,
        microstructure: Mapping[str, pd.DataFrame],
        as_of: pd.Timestamp,
        lookback_bars: int,
    ) -> Tuple[Tensor, Tuple[str, ...]]:
        """Returns (N, T, F) tensor and the feature names."""
        raise NotImplementedError

    def calibrate_volume_thresholds(
        self, volume_history: pd.DataFrame, target_bars_per_session: int
    ) -> Dict[str, float]:
        """
        Per-symbol volume-clock thresholds set so bar counts are comparable
        across the universe. Without this the liquid names dominate every
        batch and the illiquid nodes are effectively padding.
        """
        raise NotImplementedError


class CrossSectionalNormaliser:
    """
    Normalise WITHIN each bar, ACROSS symbols. Invariant I4.

        x~_it = (x_it - median_j x_jt) / (1.4826 * MAD_j x_jt)

    Per-symbol time-series normalisation uses each symbol's own future
    distribution unless carefully lagged, and destroys the cross-sectional
    comparability the graph depends on. Median/MAD rather than mean/std so a
    single limit-up name cannot set the scale for the entire bar.
    """

    def __init__(self, winsorise_at: float = 5.0, min_mad: float = 1e-8) -> None: ...

    def transform(self, x: Tensor) -> Tensor:
        """(N, T, F) -> (N, T, F), normalised over the N axis per (T, F)."""
        raise NotImplementedError

    def diagnose_scale_domination(self, x: Tensor) -> Dict[str, float]:
        """Fraction of bars where one symbol accounts for >50% of the MAD."""
        raise NotImplementedError


# ===========================================================================
# 4. ADJACENCY CONSTRUCTION
# ===========================================================================

class AdjacencyBuilder:
    """
    Assembles the multi-relational tensor and the dual transition operators.

    Runs on a SLOWER CADENCE than inference (1-5 minutes). A full lead-lag
    surface over 2,450 pairs x 40 lags is not a per-bar computation. Publish
    to shared memory with a TTL and let the inference path read the most
    recent valid tensor; topology changes far more slowly than node features.
    """

    def __init__(
        self,
        lead_lag: LeadLagDetector,
        bootstrap: IntensityPreservingBootstrap,
        relations: Sequence[RelationType],
        fdr_alpha: float = 0.05,
        shrinkage: Literal["ledoit_wolf", "oas", "none"] = "ledoit_wolf",
        staleness_threshold_s: float = 30.0,
    ) -> None: ...

    def build(
        self,
        source: MarketDataSource,
        symbols: Sequence[str],
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        cointegration_pairs: Optional[Mapping[Tuple[str, str], float]] = None,
        sector_map: Optional[Mapping[str, str]] = None,
    ) -> AdjacencyTensor:
        raise NotImplementedError

    @staticmethod
    def dual_normalise(a: Tensor) -> Tuple[Tensor, Tensor]:
        """
        P_out = D_O^-1 A  (forward along lead-lag)
        P_in  = D_I^-1 A^T (backward)

        Both are required. A node's future depends on who leads it AND on who
        it leads -- the second carries the reflexive information that its own
        move will be echoed back.

        DOES NOT SYMMETRISE. Invariant I2.
        """
        raise NotImplementedError

    def order_flow_slice(
        self, ofi: pd.DataFrame, transfer_lag_bars: int
    ) -> np.ndarray:
        """
        A_flow[i,j] = corr( OFI_i(t-tau:t), OFI_j(t:t+tau) )

        The most direct encoding of the rotation phenomenon: flow leaving one
        book and arriving in another is observed rather than inferred from the
        price response.
        """
        raise NotImplementedError


class AdaptiveAdjacency(nn.Module):
    """
    Learned low-rank directed adjacency.

        A_adp = softmax( ReLU( E_s @ E_t^T ) ),  E_s, E_t in R^{N x d}

    Asymmetric by construction since E_s != E_t, so it can represent lead-lag.
    2Nd parameters instead of N^2.

    WHY LOW RANK IS MANDATORY. A free directed adjacency over N=50 has 2,450
    parameters. Detecting an edge at true partial correlation rho=0.05 with
    5% significance and 80% power needs ~3,100 observations uncorrected, and
    ~9,600 under Bonferroni across 2,450 edges -- roughly two sessions at
    5-second bars. The edge is not stable for two sessions. That tension is
    unresolvable by collecting more history; it is only resolvable by
    shrinking the parameter count. At d=8 this is 800 parameters, and the
    factorisation imposes the correct prior: nodes influence each other
    through a few latent channels (sector, factor, liquidity tier), not
    through 2,450 independent bilateral relationships.
    """

    def __init__(self, n_nodes: int, embed_dim: int = 8, temperature: float = 1.0) -> None: ...

    def forward(self) -> Tensor:
        """(N, N), row-stochastic, asymmetric."""
        raise NotImplementedError

    def sparsity_penalty(self) -> Tensor: ...

    def effective_rank(self) -> float:
        """Entropy of the singular value spectrum. Collapse toward 1 means the
        learned graph has degenerated to a single factor."""
        raise NotImplementedError


# ===========================================================================
# 5. NETWORK
# ===========================================================================

class BidirectionalDiffusionConv(nn.Module):
    """
    Directed diffusion convolution over R relation slices.

        H' = sigma( sum_k sum_r [ (P_out^r)^k H Theta^r_{k,1}
                                + (P_in^r)^k  H Theta^r_{k,2} ] )

    NOT spectral convolution. The Kipf-Welling normalisation D^-1/2 A D^-1/2
    requires symmetric A, and library code will silently apply (A + A.T)/2 to
    satisfy it. If HDFC Bank leads Reliance at 0.4 and Reliance leads HDFC
    Bank at 0.1, symmetrisation writes 0.25 in both directions and the
    rotation signal is gone. Every Laplacian-eigendecomposition GNN has this
    property. Spatial message passing only.

    K=2 or 3. On a 50-node graph of realistic density the diameter is small
    and larger K over-smooths toward the graph mean.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_relations: int,
        diffusion_steps: int = 3,
        bias: bool = True,
    ) -> None: ...

    def forward(self, x: Tensor, p_out: Tensor, p_in: Tensor, node_mask: Tensor) -> Tensor:
        """x: (B, N, T, C_in) -> (B, N, T, C_out)."""
        raise NotImplementedError


class EdgeConditionedAttention(nn.Module):
    """
    Attention with explicit edge features, so the network modulates the
    structural prior by current state:

        e_ij = LeakyReLU( a^T [ W h_i || W h_j || W_e psi_ij ] )
        alpha_ij = softmax_j( e_ij )

    psi_ij = [rho(0), rho(delta*), delta*, sector, z_coint, dOFI].

    alpha_ij != alpha_ji because the softmax normalises over each node's own
    neighbourhood -- attention is directed for free, a further reason to
    prefer it over spectral convolution here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Returns (updated node states, attention weights for attribution)."""
        raise NotImplementedError


class GatedDilatedTCN(nn.Module):
    """
    Gated dilated causal convolution.

        h = tanh(W_f * X) (elementwise*) sigmoid(W_g * X)

    Dilation 2^l per layer gives receptive field 2^{L+1} - 1. At L=6 on
    5-second bars that is 127 bars ~ 10.5 minutes, matched to the signal
    horizon.

    Preferred over recurrence for three reasons: parallel over the time axis;
    receptive field is an explicit architectural parameter rather than an
    emergent property of gating; and CAUSALITY IS ENFORCED STRUCTURALLY by
    the kernel's left-padding rather than by discipline in the data pipeline.
    An LSTM will consume a future-leaking feature without complaint.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 2,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...

    @property
    def receptive_field(self) -> int: ...


class SpatioTemporalBlock(nn.Module):
    """Residual block: TCN( DiffConv( x ) ) + x, with a skip projection out."""

    def __init__(
        self,
        channels: int,
        n_relations: int,
        edge_dim: int,
        diffusion_steps: int,
        tcn_layers: int,
        use_attention: bool = True,
    ) -> None: ...

    def forward(
        self, x: Tensor, adj: AdjacencyTensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Returns (residual output, skip contribution)."""
        raise NotImplementedError


class GraphAlphaNet(nn.Module):
    """
    Full stack. Skip connections from every layer to the head are load-bearing
    here rather than decorative: useful signal lives at different diffusion
    depths in different regimes, and forcing everything through the deepest
    representation discards the shallow-propagation information that dominates
    in calm markets.
    """

    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        n_relations: int = 6,
        edge_dim: int = 6,
        horizons: Sequence[int] = (1, 3, 12, 60),
        adaptive_embed_dim: int = 8,
        dropout: float = 0.15,
    ) -> None: ...

    def forward(
        self, x: Tensor, adj: AdjacencyTensor, edge_attr: Tensor
    ) -> Dict[int, Tensor]:
        """Returns {horizon: (B, N) predicted residual return}."""
        raise NotImplementedError

    def attention_attribution(
        self, x: Tensor, adj: AdjacencyTensor, edge_attr: Tensor
    ) -> pd.DataFrame:
        """
        Per-edge attention mass for a given forward pass. This is the
        interpretability surface -- what the model claims is leading what.
        Treat as a hypothesis generator for research, NEVER as evidence of
        causation: attention weights are not identified.
        """
        raise NotImplementedError


# ===========================================================================
# 6. OBJECTIVES AND CONSTRAINTS
# ===========================================================================

class GrangerSparsityPenalty(nn.Module):
    """
    Group-lasso over each incoming input block of the first layer:

        L_GC = lambda * sum_i sum_{j != i} || W^(1)_{i <- j} ||_2

    The group norm drives entire input channels to exactly zero, so the
    surviving support IS the estimated Granger graph -- recovered as a
    by-product of training rather than estimated in a separate step.
    """

    def __init__(self, lambda_gc: float = 1e-3) -> None: ...

    def forward(self, first_layer_weight: Tensor, n_nodes: int) -> Tensor: ...

    def recovered_graph(self, first_layer_weight: Tensor, threshold: float) -> np.ndarray:
        raise NotImplementedError


class AcyclicityConstraint(nn.Module):
    """
    NOTEARS: h(A) = tr( exp(A o A) ) - N, exactly zero iff A is acyclic,
    smooth and differentiable, so it enters as an augmented-Lagrangian term
    rather than a combinatorial search.

    USE DELIBERATELY. Genuine reflexive feedback exists between index
    constituents -- arbitrage flow runs both directions -- and forcing a DAG
    misattributes it to one. Apply only to the slow structural slice, never
    to ORDER_FLOW.
    """

    def __init__(self, rho_init: float = 1.0, alpha_init: float = 0.0) -> None: ...

    def forward(self, a: Tensor) -> Tensor: ...

    def dual_ascent_step(self, h_value: float) -> None: ...


class CompositeLoss(nn.Module):
    """
        L = sum_h w_h Huber(yhat_h, y_h)          accuracy
            - gamma * IC(yhat_h0, y_h0)           cross-sectional rank quality
            + lambda1 ||A_adp||_1                 sparsity
            + lambda2 tr(Y^T L Y)                 Dirichlet smoothness
            + L_GC                                Granger sparsity

    The IC term is not redundant with the Huber term. A model can reduce MSE
    by shrinking every prediction toward zero while destroying the ranking
    that actually determines P&L. Huber rather than MSE because the residual
    return distribution has fat tails and squared loss lets one earnings gap
    dominate a batch.
    """

    def __init__(
        self,
        horizon_weights: Mapping[int, float],
        ic_weight: float = 0.3,
        ic_horizon: int = 12,
        l1_weight: float = 1e-4,
        dirichlet_weight: float = 1e-3,
        huber_delta: float = 1.0,
    ) -> None: ...

    def forward(
        self,
        predictions: Mapping[int, Tensor],
        targets: Mapping[int, Tensor],
        adaptive_adj: Tensor,
        graph_laplacian: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Returns (total loss, per-component breakdown for logging)."""
        raise NotImplementedError

    @staticmethod
    def cross_sectional_ic(pred: Tensor, target: Tensor) -> Tensor:
        """Spearman IC within each bar, averaged over the batch."""
        raise NotImplementedError


# ===========================================================================
# 7. VALIDATION
# ===========================================================================

class GraphAblationSuite:
    """
    Gate G3. Train against identity, density-matched random, and sector-only
    adjacency. If the full tensor does not beat all three by more than the
    across-path standard deviation of the CPCV distribution, the graph is
    decoration and the sparse-VAR baseline should ship instead.
    """

    def __init__(self, base_config: Mapping[str, Any], n_cpcv_paths: int = 12) -> None: ...

    def run(
        self, train_data: Any, test_data: Any
    ) -> pd.DataFrame:
        """One row per ablation with IC distribution across paths."""
        raise NotImplementedError

    def verdict(self, results: pd.DataFrame) -> Tuple[bool, str]: ...

    def sparse_var_baseline(self, train_data: Any, test_data: Any) -> Dict[str, float]:
        """
        Lasso-regularised VAR. At N=50 this is close enough to the break-even
        point that it is a legitimate competitor, and it must be run as one.
        """
        raise NotImplementedError


class EdgeStabilityMonitor:
    """
    Gate G4. Jaccard overlap of top-k edges across consecutive refits. Median
    below ~0.3 means the topology is being re-estimated from noise each
    period and the parameter budget is binding -- reduce the adaptive embed
    dimension or lengthen the estimation window.
    """

    def __init__(self, top_k: int = 100, history_length: int = 60) -> None: ...

    def record(self, adjacency: AdjacencyTensor) -> None: ...

    def jaccard_series(self, relation: RelationType) -> pd.Series: ...

    def is_stable(self, threshold: float = 0.3) -> Tuple[bool, float]: ...


class ResidualisationLeakTest:
    """
    Gate G2. Shuffle targets across nodes within each bar and retrain. A
    non-zero IC on shuffled targets means information is reaching the model
    through the normalisation or the beta estimation rather than through the
    features.
    """

    def __init__(self, n_shuffles: int = 20) -> None: ...

    def run(self, engine: "GraphAlphaEngine", data: Any) -> Tuple[bool, float]: ...


# ===========================================================================
# 8. ORCHESTRATION
# ===========================================================================

@dataclass
class GraphAlphaConfig:
    symbols: Tuple[str, ...]
    horizons: Tuple[int, ...] = (1, 3, 12, 60)
    bar_sampling: Literal["volume_clock", "trade_clock"] = "volume_clock"
    target_bars_per_session: int = 4500
    adjacency_refresh_seconds: float = 120.0
    adjacency_ttl_seconds: float = 300.0
    lookback_bars: int = 128
    lag_grid_us: Tuple[int, ...] = ()
    fdr_alpha: float = 0.05
    device: str = "cuda"


class GraphAlphaEngine:
    """
    Top-level orchestration. Emits forecasts of FACTOR-RESIDUAL return, which
    are consumed by the Phase 5 fusion layer as one signal among several --
    never routed directly to the execution engine.

    PROCESS TOPOLOGY. The adjacency refresh must not share an event loop with
    inference. A full HY lead-lag surface is seconds of compute; running it
    inline stalls the feature path and corrupts bar aggregation. Refresh in a
    separate process, publish with a TTL, read the latest valid tensor.
    """

    def __init__(
        self,
        config: GraphAlphaConfig,
        source: MarketDataSource,
        residualiser: FactorResidualiser,
        feature_builder: NodeFeatureBuilder,
        normaliser: CrossSectionalNormaliser,
        adjacency_builder: AdjacencyBuilder,
        model: GraphAlphaNet,
    ) -> None: ...

    # -- lifecycle -------------------------------------------------------- #

    async def refresh_adjacency(self) -> AdjacencyTensor:
        """Slow path. Runs on `adjacency_refresh_seconds` cadence."""
        raise NotImplementedError

    async def on_bar(self, bar_ts: pd.Timestamp) -> Dict[str, Dict[int, float]]:
        """
        Fast path. Returns {symbol: {horizon: predicted residual return}}.
        Raises if the cached adjacency has exceeded its TTL -- a stale graph
        must degrade the engine, not silently serve stale topology.
        """
        raise NotImplementedError

    def to_signal_frame(
        self, predictions: Mapping[str, Mapping[int, float]], horizon: int
    ) -> pd.Series:
        """
        Cross-sectionally standardised z-score for the Phase 5 combiner. The
        combiner's go/no-go test (IC_2 <= rho * IC_1) applies to this signal
        exactly as it does to sentiment.
        """
        raise NotImplementedError

    # -- diagnostics ------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        """
        Adjacency age, node mask coverage, adaptive effective rank, attention
        entropy, inference p99 latency. Adaptive effective rank collapsing
        toward 1 means the learned graph has degenerated to a single factor
        and the engine is reproducing beta.
        """
        raise NotImplementedError

    def assert_invariants(self) -> None:
        """I1-I5. Call in CI and on every adjacency refresh."""
        raise NotImplementedError

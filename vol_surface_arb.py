"""
vol_surface_arb.py
==================

Phase 8, Engine II: volatility surface dispersion on NIFTY 50 index options
against constituent options.

ARCHITECTURAL SKELETON. Method bodies are intentionally unimplemented.

What this engine actually trades
--------------------------------
    Pi ~ N_I * B * ( rho_implied - rho_realised )
    B  = sum_{i!=j} w~_i w~_j sigma_i sigma_j      (the correlation budget)

The implied-realised correlation gap is a RISK PREMIUM, not a mispricing.
Institutions hedge with index puts; yield programmes overwrite single-name
calls; no arbitrageur can close the gap without putting on this exact trade.
The carry is compensation for a short-convexity payoff: in a dislocation
rho_realised -> 1 while B expands (constituent vols rise), so both factors
move against the position simultaneously, and index skew steepens through
the vanna channel at the same time.

Consequence for Phase 5: the higher-moment Kelly root is the BINDING term
here, not a refinement. Size from the cubic m1 - f m2 + f^2 m3 - f^3 m4 = 0,
not from m1/m2.

NSE executability constraints (these shape every class below)
-------------------------------------------------------------
- Index options weekly / stock options monthly: no clean vega match.
- Stock options PHYSICALLY SETTLED: ITM legs deliver shares. Close or roll
  before the physical window, forfeiting the last and richest week of theta.
- MWPL ban at 95% of open interest, released at 80%: a leg becomes
  reduce-only mid-trade. This is a PRE-TRADE ELIGIBILITY FILTER, not an alert.
- Client position limit 10% of MWPL, prop 20%: caps capacity per name.
- Single-name OTM liquidity thin and worse near expiry: variance replication
  runs on a TRUNCATED strip, which is short a put spread versus a true
  variance swap -- and the truncation error is largest in exactly the
  down-move where the correlation leg is already losing.

Deployable instrument: monthly expiry, 8-12 most-liquid constituents by
option premium turnover, ATM +/- 1-2 strikes, index futures hedging residual
basket delta and beta. That is a PROXY for dispersion. The sub-basket
tracking error is a factor exposure requiring its own limit.

Dependencies: numpy, pandas, scipy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
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

__all__ = [
    "OptionQuote",
    "VolSlice",
    "VolSurface",
    "QuoteQualityReport",
    "CorrelationMeasure",
    "DispersionBasket",
    "DispersionSignal",
    "SurfaceFitter",
    "SVIParameterisation",
    "ArbitrageChecker",
    "DupireLocalVol",
    "ImpliedCorrelationCalculator",
    "LocalCorrelationModel",
    "AdmissibilityDiagnosis",
    "VarianceSwapReplicator",
    "BasketConstructor",
    "MWPLMonitor",
    "DispersionRiskEngine",
    "VolSurfaceArbEngine",
]


# ===========================================================================
# 1. MARKET DATA TYPES
# ===========================================================================

class SettlementType(Enum):
    CASH = auto()       # index options
    PHYSICAL = auto()   # NSE stock options


@dataclass(frozen=True)
class OptionQuote:
    """One two-sided option quote with its freshness provenance."""

    underlying: str
    expiry: pd.Timestamp
    strike: float
    right: Literal["C", "P"]
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    bid_size: int
    ask_size: int
    open_interest: int
    quote_ts: pd.Timestamp
    settlement: SettlementType

    @property
    def mid(self) -> Optional[float]: ...

    @property
    def is_two_sided(self) -> bool: ...

    @property
    def relative_spread(self) -> Optional[float]: ...

    def age_seconds(self, now: pd.Timestamp) -> float: ...


@dataclass(frozen=True)
class QuoteQualityReport:
    """
    Gate G8 artefact. In a thin single-name options market, stale data and a
    genuine dislocation produce IDENTICAL symptoms in the local-correlation
    solve. Build this gate first and the signal extraction second; inverted,
    the engine trades almost exclusively on bad data.
    """

    underlying: str
    expiry: pd.Timestamp
    n_strikes_quoted: int
    n_two_sided: int
    max_quote_age_s: float
    median_relative_spread: float
    strike_coverage_in_sigma: Tuple[float, float]  # (lower, upper) in std devs
    has_crossed_quotes: bool
    density_non_negative: bool

    @property
    def usable_for_local_vol(self) -> bool:
        """Conservative conjunction. Any failure disqualifies the surface."""
        raise NotImplementedError

    @property
    def wing_truncated(self) -> bool:
        """True if strike coverage does not reach +/- 2 sigma."""
        raise NotImplementedError


@dataclass
class VolSlice:
    """One expiry. Stored in total implied variance against log-moneyness."""

    underlying: str
    expiry: pd.Timestamp
    forward: float
    log_moneyness: np.ndarray       # k = log(K/F)
    total_variance: np.ndarray      # w(k) = sigma_imp^2 * T
    quality: QuoteQualityReport

    @property
    def tenor_years(self) -> float: ...

    def implied_vol(self, k: float) -> float: ...

    def risk_neutral_density(self, k: np.ndarray) -> np.ndarray:
        """
        Breeden-Litzenberger. MUST be checked non-negative before any
        downstream use -- a single crossed quote in a thin chain produces a
        negative density across an entire strike region, and the local vol
        derived from it is imaginary.
        """
        raise NotImplementedError


@dataclass
class VolSurface:
    underlying: str
    as_of: pd.Timestamp
    slices: Tuple[VolSlice, ...]
    spot: float

    def slice_at(self, expiry: pd.Timestamp) -> VolSlice: ...

    def atm_vol(self, expiry: pd.Timestamp) -> float: ...

    def interpolate(self, k: float, t: float) -> float:
        """Total-variance interpolation in t, never vol interpolation."""
        raise NotImplementedError


# ===========================================================================
# 2. SURFACE FITTING
# ===========================================================================

class SVIParameterisation:
    """
    Raw SVI:  w(k) = a + b [ rho (k - m) + sqrt( (k - m)^2 + sigma^2 ) ]

    Fit in TOTAL VARIANCE against LOG-MONEYNESS, never in price space. The
    Dupire denominator is proportional to the risk-neutral density; finite
    differencing it across sparse, wide-spread single-name quotes is
    numerically unstable and frequently negative.
    """

    def __init__(
        self,
        enforce_butterfly: bool = True,
        enforce_calendar: bool = True,
        param_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    ) -> None: ...

    def fit_slice(
        self, log_moneyness: np.ndarray, total_variance: np.ndarray, weights: np.ndarray
    ) -> Dict[str, float]: ...

    def evaluate(self, params: Mapping[str, float], k: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def gatheral_jacquier_butterfly(params: Mapping[str, float]) -> Tuple[bool, float]:
        """
        Durrleman condition. Returns (satisfied, minimum g(k)). A negative
        minimum means negative density somewhere on the slice.
        """
        raise NotImplementedError

    @staticmethod
    def calendar_spread_ok(
        earlier: Mapping[str, float], later: Mapping[str, float]
    ) -> bool:
        """Total variance must be non-decreasing in t at every k."""
        raise NotImplementedError


class SurfaceFitter:
    """
    Quote chain -> arbitrage-free surface. Weight the fit by quote quality
    (inverse relative spread x size), not uniformly -- a single wide,
    one-sided far-OTM quote will otherwise dominate the wing fit.
    """

    def __init__(
        self,
        parameterisation: Literal["svi", "ssvi", "spline"] = "ssvi",
        min_two_sided_strikes: int = 5,
        max_quote_age_s: float = 5.0,
    ) -> None: ...

    def assess_quotes(
        self, quotes: Sequence[OptionQuote], now: pd.Timestamp, atm_vol_prior: float
    ) -> QuoteQualityReport:
        """Runs BEFORE fitting. Gate G8 lives here."""
        raise NotImplementedError

    def fit(
        self, quotes: Sequence[OptionQuote], spot: float, now: pd.Timestamp
    ) -> Optional[VolSurface]:
        """Returns None if quote quality disqualifies the chain."""
        raise NotImplementedError


class ArbitrageChecker:
    """Gate G7. No violation may reach the local-vol computation."""

    def check_slice(self, vol_slice: VolSlice) -> Dict[str, bool]: ...

    def check_surface(self, surface: VolSurface) -> Dict[str, bool]: ...

    def repair(self, surface: VolSurface) -> VolSurface:
        """
        Repair at the FIT stage by re-fitting under constraints, never by
        patching local vol downstream.
        """
        raise NotImplementedError


# ===========================================================================
# 3. LOCAL VOLATILITY AND LOCAL CORRELATION
# ===========================================================================

class DupireLocalVol:
    """
        sigma_loc^2(K,T) = [ dC/dT + (r-q) K dC/dK + q C ]
                           / [ 0.5 K^2 d2C/dK2 ]

    Computed analytically from the fitted SVI/SSVI parameters rather than by
    finite-differencing a price grid. The implied-variance form removes the
    catastrophic cancellation in the denominator.
    """

    def __init__(self, min_local_var: float = 1e-6, max_local_vol: float = 3.0) -> None: ...

    def from_surface(
        self, surface: VolSurface, rate: float, dividend_yield: float
    ) -> Callable[[float, float], float]:
        """Returns sigma_loc(t, S)."""
        raise NotImplementedError

    def grid(
        self, surface: VolSurface, spot_grid: np.ndarray, time_grid: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def diagnose(self, local_vol_grid: np.ndarray) -> Dict[str, float]:
        """Fraction of grid clipped at floor/ceiling. High means a bad fit."""
        raise NotImplementedError


@dataclass(frozen=True)
class AdmissibilityDiagnosis:
    """
    The local-correlation solve either lands in [0,1] or it does not, and the
    direction of failure carries different information.

    lambda > 1 : requires correlation BELOW rho0 -- more dispersion than the
                 base matrix allows. Index vol cheap versus constituents, or
                 rho0 estimated in a high-correlation window.
    lambda < 0 : requires correlation ABOVE 1. Index vol rich beyond any
                 attainable correlation, or constituent quotes stale/crossed.
    in [0,1] but rho_bar not PSD : pairwise structure inconsistent; rho0 is
                 badly conditioned and needs shrinkage.
    """

    lambda_value: float
    is_admissible: bool
    failure_direction: Optional[Literal["above_unity", "below_base", "not_psd"]]
    min_eigenvalue: float
    quote_quality_explains: bool
    stale_underlyings: Tuple[str, ...]

    @property
    def is_tradable_signal(self) -> bool:
        """
        Inadmissible AND not explained by quote quality. The quote-quality
        conjunct is what separates a dislocation from stale data, and it is
        never optional.
        """
        raise NotImplementedError


class LocalCorrelationModel:
    """
    Langnau local correlation. Dupire gives sigma_i,loc per name and
    sigma_I,loc for the index; these are generally MUTUALLY INCONSISTENT.
    Close the gap with a state-dependent correlation.

        rho_bar_ij(t,I) = lambda(t,I) rho0_ij + (1 - lambda(t,I)),  i != j

        lambda(t,I) =
            [ sigma_I,loc^2 I^2 - sum_ij w_i w_j S_i S_j sigma_i,loc sigma_j,loc ]
            / [ sum_{i!=j} w_i w_j S_i S_j sigma_i,loc sigma_j,loc (rho0_ij - 1) ]

    For lambda in [0,1], rho_bar is a convex combination of two PSD matrices
    (rho0 and the rank-one all-ones matrix) and is automatically PSD. Outside
    that interval there is no guarantee.

    INADMISSIBILITY IS THE TRADING SIGNAL, and it is a better one than a raw
    implied-correlation spread. A spread tells you implied exceeds realised,
    which it almost always does -- that is the risk premium. Inadmissibility
    says the index and constituent surfaces cannot be reconciled by ANY
    correlation structure, which is an arbitrage-relevant statement.
    """

    def __init__(
        self,
        base_correlation_estimator: Literal["ledoit_wolf", "oas", "sample"] = "ledoit_wolf",
        psd_projection: bool = True,
    ) -> None: ...

    def estimate_base_correlation(
        self, returns: pd.DataFrame, half_life_days: float
    ) -> np.ndarray:
        """Shrunk rho0. Report the shrinkage intensity -- a high value means
        the sample matrix was badly conditioned and the solve is fragile."""
        raise NotImplementedError

    def solve_lambda(
        self,
        index_local_var: float,
        index_level: float,
        weights: np.ndarray,
        spots: np.ndarray,
        constituent_local_vols: np.ndarray,
        base_correlation: np.ndarray,
    ) -> float:
        raise NotImplementedError

    def diagnose(
        self,
        lambda_value: float,
        base_correlation: np.ndarray,
        quality: Mapping[str, QuoteQualityReport],
    ) -> AdmissibilityDiagnosis:
        raise NotImplementedError

    def correlation_surface(
        self,
        index_surface: VolSurface,
        constituent_surfaces: Mapping[str, VolSurface],
        weights: Mapping[str, float],
        index_grid: np.ndarray,
        time_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lambda grid, admissibility mask) over (t, I)."""
        raise NotImplementedError


# ===========================================================================
# 4. IMPLIED CORRELATION
# ===========================================================================

class CorrelationMeasure(Enum):
    CLEAN = auto()   # subtracts the idiosyncratic budget
    DIRTY = auto()   # does not; what most desks quote


class ImpliedCorrelationCalculator:
    """
    CLEAN:  ( sigma_I^2 - sum_i w~_i^2 sigma_i^2 ) / sum_{i!=j} w~_i w~_j sigma_i sigma_j
    DIRTY:  sigma_I^2 / ( sum_i w~_i sigma_i )^2

    rho_dirty >= rho_clean always, with the gap proportional to the weight
    Herfindahl. For the S&P 500 the gap is 1-2 correlation points and the
    distinction is cosmetic. FOR NIFTY, with sum w~^2 ~ 0.045 (HDFC Bank ~13%,
    Reliance ~9%), THE GAP RUNS TO 4-5 POINTS -- comparable to the entire
    implied-realised spread being harvested.

    Using DIRTY for signal generation on this index produces a persistent
    phantom edge that is pure weight arithmetic. Gate G9 tests exactly this.
    """

    def __init__(self, measure: CorrelationMeasure = CorrelationMeasure.CLEAN) -> None: ...

    def implied(
        self,
        index_vol: float,
        constituent_vols: Mapping[str, float],
        weights: Mapping[str, float],
    ) -> float:
        raise NotImplementedError

    def both_measures(
        self,
        index_vol: float,
        constituent_vols: Mapping[str, float],
        weights: Mapping[str, float],
    ) -> Tuple[float, float]:
        """Returns (clean, dirty). Log both; never mix them in one series."""
        raise NotImplementedError

    def realised(
        self, returns: pd.DataFrame, weights: Mapping[str, float], window: int
    ) -> float:
        """Weight-consistent realised correlation over the same budget."""
        raise NotImplementedError

    @staticmethod
    def weight_herfindahl(weights: Mapping[str, float]) -> float:
        """sum w~^2. Drives the clean/dirty gap and the concentration drift."""
        raise NotImplementedError

    def correlation_budget(
        self, constituent_vols: Mapping[str, float], weights: Mapping[str, float]
    ) -> float:
        """
        B = sum_{i!=j} w~_i w~_j sigma_i sigma_j

        Observable ex ante, so the trade's P&L per correlation point is known
        at inception. This is what makes the position sizeable against the
        Phase 5 Kelly machinery.
        """
        raise NotImplementedError

    def decompose_concentration_effect(
        self, weights_t0: Mapping[str, float], weights_t1: Mapping[str, float],
        constituent_vols: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Separate the change in implied correlation into (a) genuine pairwise
        correlation change and (b) pure weight arithmetic. In a rally led by
        a few large names, sum w~^2 rises, mechanically raising the
        idiosyncratic share and LOWERING implied correlation for unchanged
        pairwise correlations. That component is not information and must be
        removed before the spread is treated as a signal.
        """
        raise NotImplementedError


# ===========================================================================
# 5. REPLICATION
# ===========================================================================

class VarianceSwapReplicator:
    """
        E^Q[RV] = (2/T) [ int_0^F P(K)/K^2 dK + int_F^inf C(K)/K^2 dK ]

    The 1/K^2 weighting puts real value in the low-strike wing. In Indian
    single names that wing is frequently unquoted, so what is actually
    tradable is a TRUNCATED LOG CONTRACT -- short a put spread relative to a
    true variance swap.

    The truncation error is largest in exactly the down-move scenario where
    the correlation leg is already losing. Model it as a COST OF THE
    STRATEGY, not a rounding difference. Gate G10.
    """

    def __init__(self, n_strikes: int = 40, strike_range_sigma: float = 3.0) -> None: ...

    def replication_weights(
        self, forward: float, strikes: np.ndarray, tenor: float
    ) -> np.ndarray: ...

    def fair_variance(
        self, vol_slice: VolSlice, rate: float
    ) -> Tuple[float, float]:
        """Returns (fair variance strike, estimated truncation error)."""
        raise NotImplementedError

    def truncation_error_under_stress(
        self,
        vol_slice: VolSlice,
        spot_shock: float = -0.15,
        vol_shock: float = 0.5,
    ) -> float:
        """
        Replication error along a stressed down path. If this exceeds the
        expected carry over a typical holding period, you are selling a wing
        you cannot hedge. Gate G10 fail condition.
        """
        raise NotImplementedError


# ===========================================================================
# 6. BASKET CONSTRUCTION AND EXECUTION CONSTRAINTS
# ===========================================================================

class MWPLMonitor:
    """
    Market-wide position limit. Ban at 95% of aggregate open interest,
    released at 80%. In a ban period only position-reducing trades are
    permitted, with penalties for any increase.

    THIS IS A PRE-TRADE ELIGIBILITY FILTER. A leg that enters a ban mid-trade
    is unrollable and unhedgeable, and there is direct historical precedent
    for an F&O ban on a Nifty constituent breaking index basket trades.
    """

    def __init__(
        self,
        entry_threshold: float = 0.95,
        release_threshold: float = 0.80,
        headroom_required: float = 0.70,
    ) -> None: ...

    def current_utilisation(self, symbol: str) -> float: ...

    def is_banned(self, symbol: str) -> bool: ...

    def eligible_for_new_position(self, symbol: str, horizon_days: int) -> Tuple[bool, str]:
        """
        Eligibility over the INTENDED HOLDING PERIOD, not just today.
        Utilisation trending toward the threshold disqualifies a leg even if
        it is currently clear.
        """
        raise NotImplementedError

    def capacity_notional(
        self, symbol: str, participant_type: Literal["client", "prop", "fpi_cat_i"]
    ) -> float:
        """10% of MWPL for a client, 20% prop, 30% FPI Cat-I / MF."""
        raise NotImplementedError

    def backtest_ban_incidence(
        self, basket: Sequence[str], history: pd.DataFrame, holding_days: int
    ) -> pd.DataFrame:
        """Gate G11: how often would a leg have gone reduce-only mid-trade."""
        raise NotImplementedError


@dataclass
class DispersionBasket:
    """
    The traded sub-basket. 8-12 names, not 50. The residual against the full
    index is an explicit factor exposure requiring its own limit -- and is
    exactly the cross-sectional residual Engine I forecasts.
    """

    index_symbol: str
    expiry: pd.Timestamp
    constituents: Tuple[str, ...]
    index_weights: Mapping[str, float]        # true index weights
    basket_weights: Mapping[str, float]       # renormalised within the basket
    variance_notionals: Mapping[str, float]   # N_i = N_I * w~_i^2
    index_notional: float
    futures_hedge_lots: int

    @property
    def index_coverage(self) -> float:
        """Sum of true index weights represented. Below ~0.5 the trade is a
        sector bet with a correlation label."""
        raise NotImplementedError

    @property
    def uncovered_idio_variance(self) -> float:
        """
        Idiosyncratic budget of the EXCLUDED names. With a partial basket the
        cancellation is incomplete and this residual is a long position in
        variance you did not intend to own.
        """
        raise NotImplementedError

    def tracking_error(self, covariance: np.ndarray) -> float: ...


class BasketConstructor:
    """
    Selects the tradable sub-basket under the NSE constraint set.

    Selection criteria, jointly:
      - option premium turnover rank (liquidity)
      - MWPL headroom over the intended holding period
      - quote quality sufficient for local vol
      - strike coverage reaching at least +/- 2 sigma
      - index weight (coverage of the correlation budget)
      - pairwise correlation diversity (avoid an all-banks basket)
    """

    def __init__(
        self,
        mwpl: MWPLMonitor,
        min_names: int = 8,
        max_names: int = 12,
        min_index_coverage: float = 0.50,
        max_sector_concentration: float = 0.40,
        participant_type: Literal["client", "prop", "fpi_cat_i"] = "prop",
    ) -> None: ...

    def select(
        self,
        candidates: Sequence[str],
        index_weights: Mapping[str, float],
        quality: Mapping[str, QuoteQualityReport],
        premium_turnover: Mapping[str, float],
        sector_map: Mapping[str, str],
        holding_days: int,
    ) -> Optional[DispersionBasket]:
        """Returns None if no admissible basket exists today."""
        raise NotImplementedError

    def size_variance_notionals(
        self, basket_weights: Mapping[str, float], index_notional: float
    ) -> Dict[str, float]:
        """N_i = N_I * w~_i^2 -- the weighting that cancels the idiosyncratic
        budgets exactly in the full-basket limit."""
        raise NotImplementedError

    def futures_hedge(
        self, basket: DispersionBasket, betas: Mapping[str, float], lot_size: int
    ) -> int:
        """Index futures lots neutralising the sub-basket's residual beta."""
        raise NotImplementedError


# ===========================================================================
# 7. SIGNAL AND RISK
# ===========================================================================

@dataclass(frozen=True)
class DispersionSignal:
    as_of: pd.Timestamp
    expiry: pd.Timestamp
    rho_implied_clean: float
    rho_implied_dirty: float
    rho_realised: float
    rho_forecast: float
    correlation_budget: float
    spread: float                       # clean implied minus forecast
    admissibility: AdmissibilityDiagnosis
    expected_pnl_per_corr_point: float
    truncation_cost_estimate: float
    conviction: float                   # [0, 1]

    @property
    def net_expected_edge(self) -> float:
        """Spread x budget, net of truncation cost and expected slippage."""
        raise NotImplementedError

    @property
    def direction(self) -> Literal["short_correlation", "long_correlation", "flat"]:
        raise NotImplementedError


class DispersionRiskEngine:
    """
    Risk for a short-convexity carry position.

    Standard vega-neutral framing understates the risk in two places:

    (1) CORRELATION GAMMA. w~_i = w_i S_i / I drifts continuously. A position
        struck on w~(0) leaves variance-neutrality as the weights move, and
        the rehedge cadence is a design parameter, not an afterthought. The
        weight diffusion is d(w~_i)/w~_i = (sigma_i dW_i - sum_k w~_k sigma_k
        dW_k) + drift -- the constituent's return IN EXCESS OF THE INDEX -- so
        outperformers mechanically gain weight and index variance with them.

    (2) VANNA. Index skew is structurally steeper than the weighted average of
        constituent skews, for the same reason implied correlation exceeds
        realised: index put demand. A vega-neutral dispersion is NOT
        skew-neutral, and a sharp down move hurts through the correlation and
        skew channels simultaneously.
    """

    def __init__(
        self,
        rehedge_threshold_weight_drift: float = 0.05,
        max_correlation_exposure: float = 0.10,
        stress_correlation_floor: float = 0.90,
    ) -> None: ...

    def greeks(
        self, basket: DispersionBasket, surfaces: Mapping[str, VolSurface]
    ) -> pd.DataFrame:
        """Per-leg and aggregate: delta, vega, gamma, vanna, volga, theta."""
        raise NotImplementedError

    def correlation_exposure(self, basket: DispersionBasket) -> float:
        """dPnL / drho, in currency per correlation point."""
        raise NotImplementedError

    def weight_drift(
        self, basket: DispersionBasket, current_spots: Mapping[str, float]
    ) -> Tuple[float, bool]:
        """Returns (max relative weight drift, rehedge required)."""
        raise NotImplementedError

    def stress_pnl(
        self,
        basket: DispersionBasket,
        surfaces: Mapping[str, VolSurface],
        scenarios: Sequence[Mapping[str, float]],
    ) -> pd.DataFrame:
        """
        Mandatory scenario: index -15%, all pairwise correlations -> 0.90,
        constituent vols +50%, index skew steepened. Both P&L drivers move
        against the position at once and the correlation budget EXPANDS --
        this is the scenario that defines the strategy's tail.
        """
        raise NotImplementedError

    def kelly_moments(self, pnl_history: pd.Series) -> Tuple[float, float, float, float]:
        """
        Raw moments (m1..m4) for the Phase 5 higher-moment Kelly cubic. With
        m3 < 0 and large m4 the cubic root is materially below m1/m2, and the
        sizer MUST use the smaller value. Gate G12.
        """
        raise NotImplementedError


# ===========================================================================
# 8. ORCHESTRATION
# ===========================================================================

@dataclass
class VolArbConfig:
    index_symbol: str = "NIFTY"
    participant_type: Literal["client", "prop", "fpi_cat_i"] = "prop"
    target_tenor_days: Tuple[int, int] = (20, 45)
    use_monthly_index_cycle: bool = True     # matches stock option expiry
    close_before_physical_days: int = 3
    min_basket_names: int = 8
    max_basket_names: int = 12
    entry_spread_threshold: float = 0.05     # correlation points, net of cost
    max_quote_age_s: float = 5.0
    correlation_measure: CorrelationMeasure = CorrelationMeasure.CLEAN


class VolSurfaceArbEngine:
    """
    Top-level orchestration.

    ORDER OF OPERATIONS IS LOAD-BEARING:
        1. quote quality  ->  2. surface fit + arbitrage repair
     -> 3. local vol      ->  4. local correlation solve
     -> 5. admissibility diagnosis (WITH the quote-quality conjunct)
     -> 6. basket eligibility under MWPL
     -> 7. signal, sized from the higher-moment Kelly root
     -> 8. emit intents to the Phase 5 execution state machine

    Steps 1 and 5 cannot be reordered or skipped. Stale quotes and a genuine
    dislocation are observationally identical in the lambda solve, and stale
    quotes are overwhelmingly the more common cause in a market where
    single-name chains are thin and OTM liquidity evaporates near expiry.
    """

    def __init__(
        self,
        config: VolArbConfig,
        fitter: SurfaceFitter,
        arb_checker: ArbitrageChecker,
        local_vol: DupireLocalVol,
        local_corr: LocalCorrelationModel,
        implied_corr: ImpliedCorrelationCalculator,
        replicator: VarianceSwapReplicator,
        basket_constructor: BasketConstructor,
        mwpl: MWPLMonitor,
        risk: DispersionRiskEngine,
    ) -> None: ...

    # -- pipeline --------------------------------------------------------- #

    def build_surfaces(
        self, chains: Mapping[str, Sequence[OptionQuote]], now: pd.Timestamp
    ) -> Tuple[Dict[str, VolSurface], Dict[str, QuoteQualityReport]]:
        """Step 1-2. Underlyings failing quality are absent from the result."""
        raise NotImplementedError

    def generate_signal(
        self,
        surfaces: Mapping[str, VolSurface],
        quality: Mapping[str, QuoteQualityReport],
        weights: Mapping[str, float],
        realised_returns: pd.DataFrame,
        expiry: pd.Timestamp,
    ) -> Optional[DispersionSignal]:
        """Steps 3-5. Returns None when no admissible, quote-clean signal."""
        raise NotImplementedError

    def construct_trade(
        self, signal: DispersionSignal, equity: float, kelly_fraction: float
    ) -> Optional[DispersionBasket]:
        """
        Steps 6-7. Sizing routes through the Phase 5 KellySizer with the
        higher-moment correction ENABLED and the stressed correlation matrix
        as the second sizing pass -- take the smaller allocation.
        """
        raise NotImplementedError

    def to_order_intents(self, basket: DispersionBasket) -> List[Any]:
        """
        Step 8. Emits OrderIntent objects for the Phase 5 execution engine.
        Every leg carries reduce_only correctly, and the unwind schedule
        respects `close_before_physical_days` so no leg reaches physical
        settlement.
        """
        raise NotImplementedError

    # -- lifecycle -------------------------------------------------------- #

    def monitor(self, basket: DispersionBasket) -> Dict[str, Any]:
        """
        Per-bar: weight drift versus rehedge threshold, MWPL utilisation per
        leg, quote quality decay, days to the physical settlement window,
        realised-versus-implied correlation path.
        """
        raise NotImplementedError

    def unwind_triggers(self, basket: DispersionBasket) -> List[str]:
        """
        Any leg entering a ban period; quote quality failing on any leg;
        physical settlement window approaching; correlation stop; time stop.
        Returns the triggered reasons, empty if none.
        """
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        """
        Surfaces fitted / disqualified, admissibility event rate, fraction of
        admissibility events explained by quote quality (Gate G8), current
        correlation exposure, truncation cost as a share of expected carry.
        """
        raise NotImplementedError

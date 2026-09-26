"""
sortino_reward_env.py
=====================

Risk-adjusted reward shaping for `gym-anytrading` environments, intended for
PPO training under Stable-Baselines3.

Provides
--------
RollingSortino          : windowed Sortino estimator (correct n-denominator).
DifferentialSortino     : Moody & Saffell style incremental Sortino estimator.
SortinoStocksEnv        : StocksEnv subclass overriding `_calculate_reward`.

Design notes
------------
1.  `gym_anytrading.TradingEnv.step()` calls `_calculate_reward(action)` AFTER
    incrementing `_current_tick` but BEFORE flipping `_position`. Therefore
    inside `_calculate_reward`, `self._position` is the position that was held
    across the interval [t-1, t] -> correct for mark-to-market attribution.

2.  Returns are logarithmic so they are additive across time and symmetric in
    sign, which keeps the lower partial moment well behaved.

3.  Transaction cost is charged on the step where the flip occurs, as a log
    haircut: log(1 - fee) applied `fee_multiplier` times (2.0 for a reversal
    from full short to full long).

4.  Reward is clipped, not normalised by VecNormalize, because the ratio is
    already scale-aware. See `PPO integration` at the bottom of this file.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Literal, Optional, Tuple

import numpy as np

try:  # layout varies across gym-anytrading releases
    from gym_anytrading.envs import StocksEnv, Actions, Positions
except ImportError:  # pragma: no cover
    from gym_anytrading.envs.stocks_env import StocksEnv
    from gym_anytrading.envs.trading_env import Actions, Positions


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_POSITION_SIGN: Dict[int, float] = {
    Positions.Short.value: -1.0,
    Positions.Long.value: +1.0,
}

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #

@dataclass
class RollingSortino:
    window: int = 10
    mar: float = 0.0
    dd_floor: float = 1e-5
    max_abs_ratio: float = 10.0
    ddof: int = 0

    _buf: Deque[float] = field(init=False, repr=False)
    _prev_ratio: Optional[float] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        self._buf = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buf.clear()
        self._prev_ratio = None

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.window

    def downside_deviation(self) -> float:
        r = np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))
        shortfall = np.minimum(r - self.mar, 0.0)
        denom = max(r.size - self.ddof, 1)
        return float(np.sqrt(np.dot(shortfall, shortfall) / denom))

    def update(self, ret: float) -> Tuple[float, float, float]:
        self._buf.append(float(ret))

        if not self.ready:
            self._prev_ratio = None
            return 0.0, 0.0, self.dd_floor

        r = np.fromiter(self._buf, dtype=np.float64, count=self.window)
        dd = max(self.downside_deviation(), self.dd_floor)
        ratio = float((r.mean() - self.mar) / dd)
        ratio = float(np.clip(ratio, -self.max_abs_ratio, self.max_abs_ratio))

        delta = 0.0 if self._prev_ratio is None else ratio - self._prev_ratio
        self._prev_ratio = ratio
        return ratio, delta, dd

    def state(self) -> np.ndarray:
        dd = self.downside_deviation() if self._buf else 0.0
        mean = float(np.mean(self._buf)) if self._buf else 0.0
        fill = len(self._buf) / self.window
        return np.array([mean, dd, fill], dtype=np.float32)


@dataclass
class DifferentialSortino:
    eta: float = 0.02
    mar: float = 0.0
    l_floor: float = 2.5e-5
    max_abs: float = 25.0
    warmup: int = 32

    _A: float = field(init=False, default=0.0, repr=False)
    _L: float = field(init=False, default=0.0, repr=False)
    _n: int = field(init=False, default=0, repr=False)

    def reset(self) -> None:
        self._A = 0.0
        self._L = self.l_floor
        self._n = 0

    def __post_init__(self) -> None:
        self.reset()

    @property
    def ready(self) -> bool:
        return self._n >= self.warmup

    def ratio(self) -> float:
        return (self._A - self.mar) / np.sqrt(max(self._L, self.l_floor))

    def update(self, ret: float) -> Tuple[float, float]:
        r = float(ret)
        shortfall = min(r - self.mar, 0.0)

        A_prev, L_prev = self._A, max(self._L, self.l_floor)
        dA = r - A_prev
        dL = shortfall * shortfall - L_prev

        numerator = L_prev * dA - 0.5 * (A_prev - self.mar) * dL
        denominator = L_prev ** 1.5 + _EPS
        D = numerator / denominator

        self._A = A_prev + self.eta * dA
        self._L = max(L_prev + self.eta * dL, self.l_floor)
        self._n += 1

        if not self.ready:
            D = 0.0

        D = float(np.clip(D, -self.max_abs, self.max_abs))
        return D, float(np.sqrt(self._L))

    def state(self) -> np.ndarray:
        return np.array(
            [self._A, np.sqrt(max(self._L, self.l_floor)), self.ratio()],
            dtype=np.float32,
        )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

RewardMode = Literal["differential", "rolling", "rolling_delta"]


class SortinoStocksEnv(StocksEnv):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        df,
        window_size: int,
        frame_bound: Tuple[int, int],
        *,
        reward_mode: RewardMode = "differential",
        window: int = 10,
        eta: float = 0.02,
        mar: float = 0.0,
        trade_fee: float = 0.002,
        reverse_multiplier: float = 2.0,
        reward_scale: float = 0.05,
        lambda_dd: float = 1.0,
        z_clip: float = 4.0,
        reward_clip: float = 5.0,
        dd_floor: float = 1e-5,
        l_floor: float = 2.5e-5,
        warmup: int = 32,
        augment_observation: bool = True,
        **kwargs,
    ) -> None:
        self.reward_mode = reward_mode
        self.mar = float(mar)
        self.trade_fee = float(trade_fee)
        self.reverse_multiplier = float(reverse_multiplier)
        self.reward_scale = float(reward_scale)
        self.lambda_dd = float(lambda_dd)
        self.z_clip = float(z_clip)
        self.reward_clip = float(reward_clip)
        self.augment_observation = bool(augment_observation)

        self._rolling = RollingSortino(window=window, mar=mar, dd_floor=dd_floor)
        self._differential = DifferentialSortino(
            eta=eta, mar=mar, l_floor=l_floor, warmup=warmup
        )

        self._equity_log: float = 0.0
        self._last_components: Dict[str, float] = {}

        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound, **kwargs)

        self.trade_fee_bid_percent = self.trade_fee
        self.trade_fee_ask_percent = self.trade_fee

        if self.augment_observation:
            self._extend_observation_space()

    def _extend_observation_space(self) -> None:
        from gymnasium import spaces

        base = self.observation_space
        extra = self._reward_state().size
        low = np.concatenate(
            [base.low.reshape(-1), np.full(extra, -np.inf, dtype=np.float32)]
        )
        high = np.concatenate(
            [base.high.reshape(-1), np.full(extra, np.inf, dtype=np.float32)]
        )
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            shape=low.shape,
            dtype=np.float32,
        )

    def _reward_state(self) -> np.ndarray:
        if self.reward_mode == "differential":
            core = self._differential.state()
        else:
            core = self._rolling.state()
        pos = _POSITION_SIGN.get(
            getattr(self._position, "value", Positions.Short.value), -1.0
        )
        return np.concatenate([core, np.array([pos, self._equity_log], dtype=np.float32)])

    def _get_observation(self):
        obs = super()._get_observation()
        if not self.augment_observation:
            return obs
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32).reshape(-1), self._reward_state()]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self._rolling.reset()
        self._differential.reset()
        self._equity_log = 0.0
        self._last_components = {}
        return super().reset(seed=seed, options=options)

    def _is_reversal(self, action: int) -> bool:
        return (
            action == Actions.Buy.value and self._position == Positions.Short
        ) or (
            action == Actions.Sell.value and self._position == Positions.Long
        )

    def _step_log_return(self, action: int) -> float:
        t = self._current_tick
        p_now = float(self.prices[t])
        p_prev = float(self.prices[t - 1])
        if p_prev <= 0.0 or p_now <= 0.0:
            market_ret = 0.0
        else:
            market_ret = float(np.log(p_now / p_prev))

        sign = _POSITION_SIGN[self._position.value]
        gross = sign * market_ret

        cost = 0.0
        if self._is_reversal(action) and self.trade_fee > 0.0:
            cost = -self.reverse_multiplier * float(np.log1p(-self.trade_fee))

        return gross - cost

    def _calculate_reward(self, action: int) -> float:
        r = self._step_log_return(action)
        self._equity_log += r

        if self.reward_mode == "differential":
            core, dd = self._differential.update(r)
        else:
            level, delta, dd = self._rolling.update(r)
            core = level if self.reward_mode == "rolling" else delta

        shortfall = min(r - self.mar, 0.0)
        z = shortfall / max(dd, _EPS)
        z = float(np.clip(z, -self.z_clip, 0.0))
        penalty = self.lambda_dd * (z * z)

        reward = self.reward_scale * core - penalty
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self._last_components = {
            "step_log_return": r,
            "sortino_core": float(core),
            "downside_dev": float(dd),
            "downside_penalty": float(penalty),
            "equity_log": self._equity_log,
        }
        return reward

    def _get_info(self, *args, **kwargs):
        try:
            info = super()._get_info(*args, **kwargs)
        except TypeError:
            info = super()._get_info()
        if isinstance(info, dict):
            info.update(self._last_components)
            info["terminal_sortino"] = (
                self._differential.ratio()
                if self.reward_mode == "differential"
                else (self._rolling._prev_ratio or 0.0)
            )
        return info

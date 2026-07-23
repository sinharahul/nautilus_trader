# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Gymnasium-compatible ETHUSDT bar environment for RL research.

Observations and rewards are built from Nautilus-generated bars/features collected
via a short BacktestEngine feature dump, then stepped as a lightweight simulator
for training speed. Evaluate learned policies with the real Strategy runners.

*** RESEARCH ONLY — NOT FOR LIVE TRADING WITH REAL MONEY. ***
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.examples.strategies.rl.features import N_ACTIONS
from nautilus_trader.examples.strategies.rl.features import OBS_DIM
from nautilus_trader.examples.strategies.rl.features import RLAction
from nautilus_trader.examples.strategies.rl.features import build_obs
from nautilus_trader.examples.strategies.rl.features import z_score_from_bands
from nautilus_trader.indicators import AverageTrueRange
from nautilus_trader.indicators import BollingerBands
from nautilus_trader.indicators import EfficiencyRatio
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy


class _FeatureDumpConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType


class _FeatureDumpStrategy(Strategy):
    def __init__(self, config: _FeatureDumpConfig) -> None:
        super().__init__(config)
        self.rows: list[dict[str, float]] = []
        self.fast_ema = ExponentialMovingAverage(10)
        self.slow_ema = ExponentialMovingAverage(20)
        self.efficiency = EfficiencyRatio(20)
        self.bb = BollingerBands(20, 1.5)
        self.atr = AverageTrueRange(14)

    def on_start(self) -> None:
        for ind in (self.fast_ema, self.slow_ema, self.efficiency, self.bb, self.atr):
            self.register_indicator_for_bars(self.config.bar_type, ind)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        self.rows.append(
            {
                "close": close,
                "efficiency": self.efficiency.value,
                "z": z_score_from_bands(close, self.bb.middle, self.bb.upper, self.bb.lower),
                "atr": self.atr.value,
                "fast": self.fast_ema.value,
                "slow": self.slow_ema.value,
            },
        )


def collect_feature_rows(
    *,
    tick_bars: int = 500,
    log_level: str = "ERROR",
) -> list[dict[str, float]]:
    instrument = TestInstrumentProvider.ethusdt_binance()
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("FEATURE-001"),
            logging=LoggingConfig(log_level=log_level, log_colors=False, use_pyo3=False),
        ),
    )
    engine.add_venue(
        venue=BINANCE_VENUE,
        oms_type=OmsType.NETTING,
        book_type=BookType.L1_MBP,
        account_type=AccountType.CASH,
        base_currency=None,
        starting_balances=[Money(1_000_000.0, USDT), Money(10.0, ETH)],
        trade_execution=True,
    )
    engine.add_instrument(instrument)
    ticks = TradeTickDataWrangler(instrument=instrument).process(
        TestDataProvider().read_csv_ticks("binance/ethusdt-trades.csv"),
    )
    engine.add_data(ticks)
    dump = _FeatureDumpStrategy(
        config=_FeatureDumpConfig(
            instrument_id=instrument.id,
            bar_type=BarType.from_str(f"ETHUSDT.BINANCE-{tick_bars}-TICK-LAST-INTERNAL"),
        ),
    )
    engine.add_strategy(dump)
    engine.run()
    rows = list(dump.rows)
    engine.dispose()
    return rows


def save_feature_cache(path: str | Path, rows: list[dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        close=np.asarray([r["close"] for r in rows], dtype=np.float64),
        efficiency=np.asarray([r["efficiency"] for r in rows], dtype=np.float64),
        z=np.asarray([r["z"] for r in rows], dtype=np.float64),
        atr=np.asarray([r["atr"] for r in rows], dtype=np.float64),
        fast=np.asarray([r["fast"] for r in rows], dtype=np.float64),
        slow=np.asarray([r["slow"] for r in rows], dtype=np.float64),
    )


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


class NautilusEthEnv:
    """
    Minimal Gymnasium-style env (duck-typed). Subclasses gymnasium.Env when available.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_cache: str | Path = "examples/backtest/rl_artifacts/features.npz",
        *,
        trade_size: float = 0.10,
        fee_bps: float = 10.0,
        atr_stop_mult: float = 2.5,
        max_steps: int | None = None,
    ) -> None:
        cache_path = Path(feature_cache)
        if not cache_path.exists():
            rows = collect_feature_rows()
            save_feature_cache(cache_path, rows)
        feats = load_feature_cache(cache_path)
        self.close = feats["close"]
        self.efficiency = feats["efficiency"]
        self.z = feats["z"]
        self.atr = feats["atr"]
        self.fast = feats["fast"]
        self.slow = feats["slow"]
        self.n = len(self.close)
        if self.n < 5:
            raise RuntimeError("Feature cache too short; check ETHUSDT data path")

        self.trade_size = trade_size
        self.fee = fee_bps * 1e-4
        self.atr_stop_mult = atr_stop_mult
        self.max_steps = max_steps or (self.n - 2)
        self.churn_penalty = 0.02 * trade_size  # discourage flip/open spam
        self.hold_loser_penalty = 0.01 * trade_size

        self._t = 0
        self._pos = 0
        self._entry: float | None = None
        self._steps = 0
        self._hold_bars = 0
        self._equity_peak = 0.0
        self._realized = 0.0

        try:
            import gymnasium as gym
            from gymnasium import spaces

            self.observation_space = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(OBS_DIM,),
                dtype=np.float64,
            )
            self.action_space = spaces.Discrete(N_ACTIONS)
            self._gym = gym
        except ImportError:  # pragma: no cover
            self.observation_space = None
            self.action_space = None
            self._gym = None

    def _obs(self) -> np.ndarray:
        i = self._t
        return build_obs(
            efficiency=float(self.efficiency[i]),
            z_score=float(self.z[i]),
            atr=float(self.atr[i]),
            price=float(self.close[i]),
            fast_ema=float(self.fast[i]),
            slow_ema=float(self.slow[i]),
            position=self._pos,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._t = 0
        self._pos = 0
        self._entry = None
        self._steps = 0
        self._hold_bars = 0
        self._equity_peak = 0.0
        self._realized = 0.0
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        price = float(self.close[self._t])
        atr = float(self.atr[self._t])
        prev_mark = 0.0 if self._entry is None else self._pos * (price - self._entry) * self.trade_size
        opened_or_flipped = False

        reward = 0.0

        def _close_position(px: float) -> float:
            nonlocal opened_or_flipped
            if self._entry is None or self._pos == 0:
                return 0.0
            pnl = self._pos * (px - self._entry) * self.trade_size
            fee = abs(px * self.trade_size) * self.fee
            self._realized += pnl - fee
            self._pos = 0
            self._entry = None
            self._hold_bars = 0
            return pnl - fee

        # ATR stop
        if self._entry is not None and self._pos != 0 and atr > 0:
            stop = atr * self.atr_stop_mult
            if self._pos > 0 and price <= self._entry - stop:
                reward += _close_position(price)
            elif self._pos < 0 and price >= self._entry + stop:
                reward += _close_position(price)

        action = int(action)
        if action == RLAction.FLAT and self._pos != 0:
            reward += _close_position(price)
        elif action == RLAction.BUY:
            if self._pos < 0:
                reward += _close_position(price)
                opened_or_flipped = True
            if self._pos == 0:
                self._pos = 1
                self._entry = price
                self._hold_bars = 0
                reward -= abs(price * self.trade_size) * self.fee
                opened_or_flipped = True
        elif action == RLAction.SELL:
            if self._pos > 0:
                reward += _close_position(price)
                opened_or_flipped = True
            if self._pos == 0:
                self._pos = -1
                self._entry = price
                self._hold_bars = 0
                reward -= abs(price * self.trade_size) * self.fee
                opened_or_flipped = True

        if opened_or_flipped:
            reward -= self.churn_penalty

        if self._pos != 0:
            self._hold_bars += 1

        self._t = min(self._t + 1, self.n - 1)
        self._steps += 1
        new_price = float(self.close[self._t])
        new_mark = 0.0 if self._entry is None else self._pos * (new_price - self._entry) * self.trade_size
        delta_mark = new_mark - prev_mark
        reward += delta_mark

        # Penalize sitting in an open loser
        if self._pos != 0 and self._entry is not None:
            open_pnl = self._pos * (new_price - self._entry) * self.trade_size
            if open_pnl < 0:
                reward -= self.hold_loser_penalty

        # Risk-adjusted shaping: penalize drawdown from peak equity
        equity = self._realized + new_mark
        self._equity_peak = max(self._equity_peak, equity)
        dd = self._equity_peak - equity
        reward -= 0.1 * dd

        terminated = self._t >= self.n - 1
        truncated = self._steps >= self.max_steps
        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            {"position": self._pos, "equity": equity, "realized": self._realized},
        )


# Optional proper gymnasium subclass for SB3 registration convenience
try:
    import gymnasium as gym

    class NautilusEthGymEnv(NautilusEthEnv, gym.Env):
        """Gymnasium Env subclass."""

        def __init__(self, **kwargs: Any) -> None:
            gym.Env.__init__(self)
            NautilusEthEnv.__init__(self, **kwargs)

except ImportError:  # pragma: no cover

    class NautilusEthGymEnv(NautilusEthEnv):  # type: ignore[no-redef]
        pass

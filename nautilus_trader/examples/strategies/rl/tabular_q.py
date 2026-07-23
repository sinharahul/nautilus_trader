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
Tabular Q-learning strategy on ETHUSDT internal bars.

*** RESEARCH ONLY — NOT FOR LIVE TRADING WITH REAL MONEY. ***
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.examples.strategies.rl.features import N_ACTIONS
from nautilus_trader.examples.strategies.rl.features import N_TABULAR_STATES
from nautilus_trader.examples.strategies.rl.features import RLAction
from nautilus_trader.examples.strategies.rl.features import build_obs
from nautilus_trader.examples.strategies.rl.features import discretize_state
from nautilus_trader.examples.strategies.rl.features import z_score_from_bands
from nautilus_trader.indicators import AverageTrueRange
from nautilus_trader.indicators import BollingerBands
from nautilus_trader.indicators import EfficiencyRatio
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


class TabularQConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``TabularQStrategy``.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 10
    slow_ema_period: PositiveInt = 20
    efficiency_period: PositiveInt = 20
    efficiency_threshold: PositiveFloat = 0.35
    bb_period: PositiveInt = 20
    bb_k: PositiveFloat = 1.5
    atr_period: PositiveInt = 14
    atr_stop_mult: PositiveFloat = 2.5
    alpha: PositiveFloat = 0.20
    gamma: PositiveFloat = 0.95
    epsilon_start: PositiveFloat = 0.30
    epsilon_end: PositiveFloat = 0.05
    epsilon_decay_bars: PositiveInt = 200
    learn: bool = True
    collect_transitions: bool = True
    transitions_path: str = "examples/backtest/rl_artifacts/transitions.npz"
    q_table_path: str = "examples/backtest/rl_artifacts/q_table.npy"
    close_positions_on_stop: bool = True


class TabularQStrategy(Strategy):
    """
    ε-greedy tabular Q-learning over a compact regime/z/position state.
    """

    def __init__(self, config: TabularQConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument | None = None
        self._entry_price: float | None = None
        self._bar_i = 0
        self._prev_state: int | None = None
        self._prev_action: int | None = None
        self._prev_equity: float | None = None
        self._prev_obs: np.ndarray | None = None

        self.q = np.zeros((N_TABULAR_STATES, N_ACTIONS), dtype=np.float64)
        self._transitions: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
        self.efficiency = EfficiencyRatio(config.efficiency_period)
        self.bb = BollingerBands(config.bb_period, config.bb_k)
        self.atr = AverageTrueRange(config.atr_period)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        q_path = Path(self.config.q_table_path)
        if q_path.exists():
            loaded = np.load(q_path)
            if loaded.shape == self.q.shape:
                self.q = loaded
                self.log.info(f"Loaded Q-table from {q_path}", LogColor.BLUE)

        for ind in (self.fast_ema, self.slow_ema, self.efficiency, self.bb, self.atr):
            self.register_indicator_for_bars(self.config.bar_type, ind)
        self.subscribe_bars(self.config.bar_type)
        self.log.info("TabularQStrategy started", LogColor.BLUE)

    def _position_sign(self) -> int:
        iid = self.config.instrument_id
        if self.portfolio.is_net_long(iid):
            return 1
        if self.portfolio.is_net_short(iid):
            return -1
        return 0

    def _epsilon(self) -> float:
        t = min(self._bar_i, self.config.epsilon_decay_bars)
        frac = t / max(self.config.epsilon_decay_bars, 1)
        return float(
            self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * frac,
        )

    def _select_action(self, state: int) -> int:
        if self.config.learn and np.random.random() < self._epsilon():
            return int(np.random.randint(0, N_ACTIONS))
        return int(np.argmax(self.q[state]))

    def _mark_equity(self, price: float) -> float:
        # Lightweight proxy: cash PnL approx via open unrealized vs entry
        if self._entry_price is None or self._position_sign() == 0:
            return 0.0
        return self._position_sign() * (price - self._entry_price) * float(self.config.trade_size)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized():
            return
        if bar.is_single_price():
            return

        close = bar.close.as_double()
        if self._check_stop(close):
            # Treat stop as forced FLAT transition with negative reward signal
            return

        z = z_score_from_bands(close, self.bb.middle, self.bb.upper, self.bb.lower)
        pos = self._position_sign()
        state = discretize_state(
            efficiency=self.efficiency.value,
            z_score=z,
            position=pos,
            efficiency_threshold=self.config.efficiency_threshold,
        )
        obs = build_obs(
            efficiency=self.efficiency.value,
            z_score=z,
            atr=self.atr.value,
            price=close,
            fast_ema=self.fast_ema.value,
            slow_ema=self.slow_ema.value,
            position=pos,
        )
        equity = self._mark_equity(close)

        if (
            self.config.learn
            and self._prev_state is not None
            and self._prev_action is not None
            and self._prev_equity is not None
        ):
            reward = equity - self._prev_equity
            best_next = float(np.max(self.q[state]))
            td = (
                reward
                + self.config.gamma * best_next
                - self.q[self._prev_state, self._prev_action]
            )
            self.q[self._prev_state, self._prev_action] += self.config.alpha * td
            if self.config.collect_transitions and self._prev_obs is not None:
                self._transitions.append((self._prev_obs, self._prev_action, reward, obs, False))

        action = self._select_action(state)
        self._apply_action(action, close)

        self._prev_state = state
        self._prev_action = action
        self._prev_equity = self._mark_equity(close)
        self._prev_obs = obs
        self._bar_i += 1

    def _apply_action(self, action: int, price: float) -> None:
        iid = self.config.instrument_id
        pos = self._position_sign()

        if action == RLAction.HOLD:
            return
        if action == RLAction.FLAT:
            if pos != 0:
                self.close_all_positions(iid)
                self._entry_price = None
            return
        if action == RLAction.BUY:
            if pos < 0:
                self.close_all_positions(iid)
                self._entry_price = None
            if self.portfolio.is_flat(iid):
                self._submit(OrderSide.BUY)
            return
        if action == RLAction.SELL:
            if pos > 0:
                self.close_all_positions(iid)
                self._entry_price = None
            if self.portfolio.is_flat(iid):
                self._submit(OrderSide.SELL)

    def _submit(self, side: OrderSide) -> None:
        assert self.instrument is not None
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def _check_stop(self, close: float) -> bool:
        iid = self.config.instrument_id
        if self._entry_price is None or self.portfolio.is_flat(iid):
            return False
        stop_dist = self.atr.value * self.config.atr_stop_mult
        if stop_dist <= 0:
            return False
        if self.portfolio.is_net_long(iid) and close <= self._entry_price - stop_dist:
            self.close_all_positions(iid)
            self._entry_price = None
            return True
        if self.portfolio.is_net_short(iid) and close >= self._entry_price + stop_dist:
            self.close_all_positions(iid)
            self._entry_price = None
            return True
        return False

    def on_order_filled(self, event: OrderFilled) -> None:
        if self.portfolio.is_flat(self.config.instrument_id):
            self._entry_price = None
            return
        self._entry_price = event.last_px.as_double()

    def on_data(self, data: Data) -> None:
        pass

    def on_event(self, event: Event) -> None:
        pass

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)

        Path(self.config.q_table_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.config.q_table_path, self.q)
        self.log.info(f"Saved Q-table -> {self.config.q_table_path}", LogColor.BLUE)

        if self.config.collect_transitions and self._transitions:
            obs = np.stack([t[0] for t in self._transitions])
            actions = np.asarray([t[1] for t in self._transitions], dtype=np.int64)
            rewards = np.asarray([t[2] for t in self._transitions], dtype=np.float64)
            next_obs = np.stack([t[3] for t in self._transitions])
            dones = np.asarray([t[4] for t in self._transitions], dtype=np.bool_)
            path = Path(self.config.transitions_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
            )
            self.log.info(
                f"Saved {len(self._transitions)} transitions -> {path}",
                LogColor.BLUE,
            )

        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.efficiency.reset()
        self.bb.reset()
        self.atr.reset()
        self._entry_price = None
        self._bar_i = 0
        self._prev_state = None
        self._prev_action = None
        self._prev_equity = None
        self._prev_obs = None
        self._transitions.clear()

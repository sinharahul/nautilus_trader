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
Offline NumPy MLP policy for ETHUSDT RL research.

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
from nautilus_trader.examples.strategies.rl.features import OBS_DIM
from nautilus_trader.examples.strategies.rl.features import RLAction
from nautilus_trader.examples.strategies.rl.features import build_obs
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


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class NumpyMLP:
    """
    Tiny 2-layer MLP classifier (obs -> action logits), NumPy only.
    """

    def __init__(self, in_dim: int = OBS_DIM, hidden: int = 32, out_dim: int = N_ACTIONS) -> None:
        rng = np.random.default_rng(7)
        self.w1 = rng.normal(0, 0.2, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 0.2, size=(hidden, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def act(self, obs: np.ndarray) -> int:
        logits = self.forward(obs.reshape(1, -1))[0]
        return int(np.argmax(logits))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]


def train_behavior_cloning(
    transitions_path: str | Path,
    model_path: str | Path,
    *,
    epochs: int = 40,
    lr: float = 0.05,
    batch_size: int = 64,
    hidden: int = 32,
) -> NumpyMLP:
    """
    Train MLP to imitate logged actions (behavior cloning from tabular explorer).
    """
    data = np.load(transitions_path)
    obs = data["obs"].astype(np.float64)
    actions = data["actions"].astype(np.int64)
    n = len(actions)
    if n == 0:
        raise ValueError(f"No transitions in {transitions_path}")

    model = NumpyMLP(in_dim=obs.shape[1], hidden=hidden, out_dim=N_ACTIONS)
    rng = np.random.default_rng(42)

    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            x = obs[batch]
            y = actions[batch]

            h = np.tanh(x @ model.w1 + model.b1)
            logits = h @ model.w2 + model.b2
            probs = _softmax(logits)

            # one-hot cross-entropy grads
            target = np.zeros_like(probs)
            target[np.arange(len(y)), y] = 1.0
            dlogits = (probs - target) / len(y)
            dw2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ model.w2.T * (1.0 - h**2)
            dw1 = x.T @ dh
            db1 = dh.sum(axis=0)

            model.w2 -= lr * dw2
            model.b2 -= lr * db2
            model.w1 -= lr * dw1
            model.b1 -= lr * db1

    model.save(model_path)
    return model


class OfflinePolicyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    model_path: str = "examples/backtest/rl_artifacts/offline_mlp.npz"
    fast_ema_period: PositiveInt = 10
    slow_ema_period: PositiveInt = 20
    efficiency_period: PositiveInt = 20
    bb_period: PositiveInt = 20
    bb_k: PositiveFloat = 1.5
    atr_period: PositiveInt = 14
    atr_stop_mult: PositiveFloat = 2.5
    close_positions_on_stop: bool = True


class OfflinePolicyStrategy(Strategy):
    """
    Frozen MLP policy acting greedily on continuous observations.
    """

    def __init__(self, config: OfflinePolicyConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument | None = None
        self._entry_price: float | None = None
        self.model = NumpyMLP()
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

        path = Path(self.config.model_path)
        if not path.exists():
            self.log.error(f"Missing model weights at {path}; train offline first")
            self.stop()
            return
        self.model.load(path)
        self.log.info(f"Loaded offline policy from {path}", LogColor.BLUE)

        for ind in (self.fast_ema, self.slow_ema, self.efficiency, self.bb, self.atr):
            self.register_indicator_for_bars(self.config.bar_type, ind)
        self.subscribe_bars(self.config.bar_type)

    def _position_sign(self) -> int:
        iid = self.config.instrument_id
        if self.portfolio.is_net_long(iid):
            return 1
        if self.portfolio.is_net_short(iid):
            return -1
        return 0

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        if self._check_stop(close):
            return

        z = z_score_from_bands(close, self.bb.middle, self.bb.upper, self.bb.lower)
        obs = build_obs(
            efficiency=self.efficiency.value,
            z_score=z,
            atr=self.atr.value,
            price=close,
            fast_ema=self.fast_ema.value,
            slow_ema=self.slow_ema.value,
            position=self._position_sign(),
        )
        self._apply_action(self.model.act(obs))

    def _apply_action(self, action: int) -> None:
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
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.efficiency.reset()
        self.bb.reset()
        self.atr.reset()
        self._entry_price = None

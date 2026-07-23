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
Nautilus strategy that replays FinRL DRLAgent action signals (CSV).

Generate signals with ``examples/backtest/finrl_ethusdt_bridge.py``.

*** RESEARCH ONLY — NOT FOR LIVE TRADING. ***
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


class FinRLSignalConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    signals_path: str = "examples/backtest/rl_artifacts/finrl/finrl_signals.csv"
    action_threshold: PositiveFloat = 0.15
    close_positions_on_stop: bool = True


class FinRLSignalStrategy(Strategy):
    """
    Executes BUY/SELL/FLAT from a FinRL actions CSV (date, actions).
    """

    def __init__(self, config: FinRLSignalConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument | None = None
        self._signals: dict[pd.Timestamp, float] = {}

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Missing instrument {self.config.instrument_id}")
            self.stop()
            return

        path = Path(self.config.signals_path)
        if not path.exists():
            self.log.error(f"Missing FinRL signals at {path}")
            self.stop()
            return

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        for _, row in df.iterrows():
            action = row["actions"]
            if isinstance(action, str):
                action = float(action.strip("[]"))
            else:
                action = float(action)
            # Floor to minute for INTERNAL/minute alignment
            ts = row["date"].floor("min")
            self._signals[ts] = action

        self.subscribe_bars(self.config.bar_type)
        self.log.info(f"Loaded {len(self._signals)} FinRL signals from {path}", LogColor.BLUE)

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").floor("min")
        action = self._signals.get(ts)
        if action is None:
            return

        iid = self.config.instrument_id
        thr = self.config.action_threshold
        flat = self.portfolio.is_flat(iid)

        if action > thr:
            if self.portfolio.is_flat(iid):
                self._submit(OrderSide.BUY)
        elif action < -thr:
            # FinRL StockTradingEnv is cash/long-only: negative = sell/flat, not short.
            if self.portfolio.is_net_long(iid):
                self.close_all_positions(iid)
        elif abs(action) < thr * 0.5 and not flat:
            self.close_all_positions(iid)

    def _submit(self, side: OrderSide) -> None:
        assert self.instrument is not None
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def on_data(self, data: Data) -> None:
        pass

    def on_event(self, event: Event) -> None:
        pass

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

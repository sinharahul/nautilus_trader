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

from decimal import Decimal
from enum import Enum
from enum import unique

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import NonNegativeInt
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
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


# *** THIS IS A TEST STRATEGY WITH NO ALPHA ADVANTAGE WHATSOEVER. ***
# *** IT IS NOT INTENDED TO BE USED TO TRADE LIVE WITH REAL MONEY. ***


@unique
class MarketRegime(Enum):
    TREND = "TREND"
    MEAN_REVERT = "MEAN_REVERT"


class RegimeHybridConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``RegimeHybrid`` (quality-filtered research template).

    Fewer/better entries: higher z/efficiency, ``confirm_bars``, ``cooldown_bars``.
    Asymmetric exits: tight ``atr_loss_mult``, trail ``atr_trail_mult``, ``max_hold_bars``.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 10
    slow_ema_period: PositiveInt = 20
    efficiency_period: PositiveInt = 20
    efficiency_threshold: PositiveFloat = 0.45
    bb_period: PositiveInt = 20
    entry_z: PositiveFloat = 1.5
    atr_period: PositiveInt = 14
    atr_loss_mult: PositiveFloat = 3.5
    atr_trail_mult: PositiveFloat = 3.5
    confirm_bars: PositiveInt = 1
    cooldown_bars: NonNegativeInt = 0
    max_hold_bars: PositiveInt = 40
    close_positions_on_stop: bool = True


class RegimeHybrid(Strategy):
    """
    Regime-switching hybrid with quality filters and asymmetric exits.
    """

    def __init__(self, config: RegimeHybridConfig) -> None:
        if config.fast_ema_period >= config.slow_ema_period:
            raise ValueError(
                f"{config.fast_ema_period=} must be less than {config.slow_ema_period=}",
            )
        super().__init__(config)

        self.instrument: Instrument | None = None
        self.regime: MarketRegime = MarketRegime.MEAN_REVERT
        self._entry_price: float | None = None
        self._extreme_price: float | None = None
        self._bars_in_trade = 0
        self._cooldown_left = 0
        self._pending_side: int | None = None
        self._pending_count = 0

        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
        self.efficiency = EfficiencyRatio(config.efficiency_period)
        self.bb = BollingerBands(config.bb_period, config.entry_z)
        self.atr = AverageTrueRange(config.atr_period)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        for indicator in (
            self.fast_ema,
            self.slow_ema,
            self.efficiency,
            self.bb,
            self.atr,
        ):
            self.register_indicator_for_bars(self.config.bar_type, indicator)

        self.subscribe_bars(self.config.bar_type)
        self.log.info(
            "RegimeHybrid v2 started "
            f"(eff>={self.config.efficiency_threshold}, z={self.config.entry_z}, "
            f"confirm={self.config.confirm_bars}, cooldown={self.config.cooldown_bars}, "
            f"lossATR={self.config.atr_loss_mult}, trailATR={self.config.atr_trail_mult})",
            LogColor.BLUE,
        )

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return

        close = bar.close.as_double()
        self._update_regime()

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if not self.portfolio.is_flat(self.config.instrument_id):
            self._bars_in_trade += 1
            if self._manage_open_trade(close):
                return
        else:
            self._bars_in_trade = 0

        if self._cooldown_left > 0:
            return

        if self.regime == MarketRegime.TREND:
            self._trade_trend()
        else:
            self._trade_mean_revert(close)

    def _update_regime(self) -> None:
        new_regime = (
            MarketRegime.TREND
            if self.efficiency.value >= self.config.efficiency_threshold
            else MarketRegime.MEAN_REVERT
        )
        if new_regime != self.regime:
            self.log.info(
                f"Regime {self.regime.value} -> {new_regime.value} "
                f"(eff={self.efficiency.value:.4f})",
                LogColor.YELLOW,
            )
            self.regime = new_regime
            self._pending_side = None
            self._pending_count = 0

    def _manage_open_trade(self, close: float) -> bool:
        iid = self.config.instrument_id
        if self._entry_price is None or self.atr.value <= 0:
            return False

        long = self.portfolio.is_net_long(iid)
        short = self.portfolio.is_net_short(iid)
        if not (long or short):
            return False

        if long:
            self._extreme_price = (
                close if self._extreme_price is None else max(self._extreme_price, close)
            )
        else:
            self._extreme_price = (
                close if self._extreme_price is None else min(self._extreme_price, close)
            )

        loss_dist = self.atr.value * self.config.atr_loss_mult
        trail_dist = self.atr.value * self.config.atr_trail_mult

        if long and close <= self._entry_price - loss_dist:
            return self._exit("loss_stop", close)
        if short and close >= self._entry_price + loss_dist:
            return self._exit("loss_stop", close)

        if self._extreme_price is not None:
            if long and self._extreme_price > self._entry_price:
                if close <= self._extreme_price - trail_dist:
                    return self._exit("trail_stop", close)
            if short and self._extreme_price < self._entry_price:
                if close >= self._extreme_price + trail_dist:
                    return self._exit("trail_stop", close)

        if self._bars_in_trade >= self.config.max_hold_bars:
            return self._exit("time_stop", close)

        if self.regime == MarketRegime.MEAN_REVERT:
            if long and close >= self.bb.middle:
                return self._exit("mr_mid", close)
            if short and close <= self.bb.middle:
                return self._exit("mr_mid", close)

        return False

    def _exit(self, reason: str, close: float) -> bool:
        iid = self.config.instrument_id
        color = LogColor.RED if reason in {"loss_stop", "time_stop"} else LogColor.GREEN
        self.log.info(
            f"Exit {reason} close={close:.4f} entry={self._entry_price} "
            f"bars={self._bars_in_trade}",
            color,
        )
        was_protective = reason in {"loss_stop", "trail_stop", "time_stop"}
        self.close_all_positions(iid)
        self._entry_price = None
        self._extreme_price = None
        self._bars_in_trade = 0
        self._pending_side = None
        self._pending_count = 0
        if was_protective:
            self._cooldown_left = self.config.cooldown_bars
        return True

    def _request_entry(self, side: int) -> None:
        if self._pending_side == side:
            self._pending_count += 1
        else:
            self._pending_side = side
            self._pending_count = 1

        if self._pending_count < self.config.confirm_bars:
            return

        self._pending_side = None
        self._pending_count = 0
        if side > 0:
            self.buy()
        else:
            self.sell()

    def _trade_trend(self) -> None:
        iid = self.config.instrument_id
        if self.efficiency.value < self.config.efficiency_threshold:
            return
        if not self.portfolio.is_flat(iid):
            return

        if self.fast_ema.value > self.slow_ema.value:
            self._request_entry(+1)
        else:
            self._request_entry(-1)

    def _trade_mean_revert(self, close: float) -> None:
        iid = self.config.instrument_id
        if not self.portfolio.is_flat(iid):
            return

        if close <= self.bb.lower:
            self._request_entry(+1)
        elif close >= self.bb.upper:
            self._request_entry(-1)
        else:
            self._pending_side = None
            self._pending_count = 0

    def buy(self) -> None:
        assert self.instrument is not None
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def sell(self) -> None:
        assert self.instrument is not None
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def on_order_filled(self, event: OrderFilled) -> None:
        iid = self.config.instrument_id
        if self.portfolio.is_flat(iid):
            self._entry_price = None
            self._extreme_price = None
            return
        px = event.last_px.as_double()
        self._entry_price = px
        self._extreme_price = px
        self._bars_in_trade = 0

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
        self._extreme_price = None
        self._bars_in_trade = 0
        self._cooldown_left = 0
        self._pending_side = None
        self._pending_count = 0
        self.regime = MarketRegime.MEAN_REVERT

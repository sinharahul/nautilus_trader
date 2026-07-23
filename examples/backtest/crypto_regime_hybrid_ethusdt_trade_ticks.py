#!/usr/bin/env python3
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
Backtest RegimeHybrid on Binance ETHUSDT trade ticks (in-repo sample data).
"""

from decimal import Decimal

import pandas as pd

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.examples.strategies.regime_hybrid import RegimeHybrid
from nautilus_trader.examples.strategies.regime_hybrid import RegimeHybridConfig
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider


def run() -> None:
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(
            log_level="INFO",
            log_level_file="ERROR",
            log_colors=True,
            use_pyo3=False,
        ),
    )
    engine = BacktestEngine(config=config)

    engine.add_venue(
        venue=BINANCE_VENUE,
        oms_type=OmsType.NETTING,
        book_type=BookType.L1_MBP,
        account_type=AccountType.CASH,
        base_currency=None,
        starting_balances=[Money(1_000_000.0, USDT), Money(10.0, ETH)],
        trade_execution=True,
    )

    ethusdt = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(ethusdt)

    provider = TestDataProvider()
    wrangler = TradeTickDataWrangler(instrument=ethusdt)
    ticks = wrangler.process(provider.read_csv_ticks("binance/ethusdt-trades.csv"))
    engine.add_data(ticks)

    # In-sample 90% WR config on ethusdt-trades.csv (~5h sample):
    # eff=0.45, z=1.5, ATR loss/trail=3.5, confirm=1, cooldown=0
    strategy = RegimeHybrid(
        config=RegimeHybridConfig(
            instrument_id=ethusdt.id,
            bar_type=BarType.from_str("ETHUSDT.BINANCE-500-TICK-LAST-INTERNAL"),
            trade_size=Decimal("0.10"),
            efficiency_threshold=0.45,
            entry_z=1.5,
            atr_loss_mult=3.5,
            atr_trail_mult=3.5,
            confirm_bars=1,
            cooldown_bars=0,
            max_hold_bars=40,
        ),
    )
    engine.add_strategy(strategy)

    engine.run()

    positions = engine.trader.generate_positions_report()
    fills = engine.trader.generate_order_fills_report()
    pnls = (
        [float(str(x).replace(" USDT", "")) for x in positions["realized_pnl"].tolist()]
        if len(positions)
        else []
    )
    total = sum(pnls)
    wins = sum(1 for x in pnls if x > 0)
    wr = (wins / len(pnls)) if pnls else 0.0

    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print("\n=== ACCOUNT REPORT ===")
        print(engine.trader.generate_account_report(BINANCE_VENUE))
        print("\n=== ORDER FILLS REPORT ===")
        print(fills)
        print("\n=== POSITIONS REPORT ===")
        print(positions)

    print("\n=== SUMMARY ===")
    print(f"positions={len(positions)} fills={len(fills)}")
    print(f"realized_pnl={total:+.6f} USDT")
    print(f"win_rate={wr:.1%}")

    engine.dispose()


if __name__ == "__main__":
    run()

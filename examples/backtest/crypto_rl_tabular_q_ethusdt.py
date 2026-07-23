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
Train/run tabular Q-learning strategy on Binance ETHUSDT trade ticks.
"""

from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.examples.strategies.rl.tabular_q import TabularQConfig
from nautilus_trader.examples.strategies.rl.tabular_q import TabularQStrategy
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


ARTIFACT_DIR = Path("examples/backtest/rl_artifacts")


def _summarize(engine: BacktestEngine) -> None:
    positions = engine.trader.generate_positions_report()
    fills = engine.trader.generate_order_fills_report()
    pnls = (
        [float(str(x).replace(" USDT", "")) for x in positions["realized_pnl"].tolist()]
        if len(positions)
        else []
    )
    total = sum(pnls)
    wins = sum(1 for x in pnls if x > 0)
    print(f"positions={len(positions)} fills={len(fills)}")
    print(f"realized_pnl={total:+.6f} USDT")
    if pnls:
        print(f"win_rate={wins / len(pnls):.1%}")


def run(*, learn: bool = True) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("RL-TABULAR-001"),
            logging=LoggingConfig(log_level="ERROR", log_colors=False, use_pyo3=False),
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
    eth = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(eth)
    ticks = TradeTickDataWrangler(instrument=eth).process(
        TestDataProvider().read_csv_ticks("binance/ethusdt-trades.csv"),
    )
    engine.add_data(ticks)

    strategy = TabularQStrategy(
        config=TabularQConfig(
            instrument_id=eth.id,
            bar_type=BarType.from_str("ETHUSDT.BINANCE-500-TICK-LAST-INTERNAL"),
            trade_size=Decimal("0.10"),
            learn=learn,
            collect_transitions=True,
            transitions_path=str(ARTIFACT_DIR / "transitions.npz"),
            q_table_path=str(ARTIFACT_DIR / "q_table.npy"),
        ),
    )
    engine.add_strategy(strategy)
    engine.run()

    with pd.option_context("display.max_rows", 40, "display.width", 200):
        print("\n=== POSITIONS (tail) ===")
        print(engine.trader.generate_positions_report().tail(10))
    _summarize(engine)
    engine.dispose()


if __name__ == "__main__":
    print("=== Tabular Q pass 1 (learn + collect) ===")
    run(learn=True)
    print("\n=== Tabular Q pass 2 (reuse Q-table, lower exploration via loaded values) ===")
    run(learn=True)

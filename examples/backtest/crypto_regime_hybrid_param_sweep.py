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
Parameter sweep for RegimeHybrid on Binance ETHUSDT trade ticks.
"""

from __future__ import annotations

from decimal import Decimal
from itertools import product

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


def _pnl_usdt(value: object) -> float:
    return float(str(value).replace(" USDT", ""))


def run_once(
    ticks,
    instrument,
    *,
    tick_bars: int,
    efficiency_threshold: float,
    entry_z: float,
    atr_stop_mult: float,
) -> dict[str, float | int]:
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SWEEP-001"),
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
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    engine.add_strategy(
        RegimeHybrid(
            config=RegimeHybridConfig(
                instrument_id=instrument.id,
                bar_type=BarType.from_str(
                    f"ETHUSDT.BINANCE-{tick_bars}-TICK-LAST-INTERNAL",
                ),
                trade_size=Decimal("0.10"),
                efficiency_threshold=efficiency_threshold,
                entry_z=entry_z,
                atr_loss_mult=atr_stop_mult,
                atr_trail_mult=max(atr_stop_mult, 2.0),
                confirm_bars=1,
                cooldown_bars=0,
            ),
        ),
    )
    engine.run()

    positions = engine.trader.generate_positions_report()
    fills = engine.trader.generate_order_fills_report()
    pnls = [_pnl_usdt(x) for x in positions["realized_pnl"].tolist()] if len(positions) else []
    total = sum(pnls)
    wins = sum(1 for x in pnls if x > 0)
    engine.dispose()

    return {
        "tick_bars": tick_bars,
        "efficiency_threshold": efficiency_threshold,
        "entry_z": entry_z,
        "atr_stop_mult": atr_stop_mult,
        "positions": len(positions),
        "fills": len(fills),
        "realized_pnl": round(total, 6),
        "win_rate": round(wins / len(pnls), 4) if pnls else 0.0,
    }


def main() -> None:
    instrument = TestInstrumentProvider.ethusdt_binance()
    ticks = TradeTickDataWrangler(instrument=instrument).process(
        TestDataProvider().read_csv_ticks("binance/ethusdt-trades.csv"),
    )

    grid = list(
        product(
            (100, 250, 500),  # tick bar size
            (0.25, 0.35, 0.45),  # efficiency threshold
            (1.5, 2.0, 2.5),  # BB / z entry
            (1.5, 2.5, 3.5),  # ATR stop multiple
        ),
    )

    rows: list[dict[str, float | int]] = []
    for i, (tick_bars, eff, entry_z, atr_mult) in enumerate(grid, start=1):
        row = run_once(
            ticks,
            instrument,
            tick_bars=tick_bars,
            efficiency_threshold=eff,
            entry_z=entry_z,
            atr_stop_mult=atr_mult,
        )
        rows.append(row)
        print(
            f"[{i:02d}/{len(grid)}] bars={tick_bars} eff={eff} z={entry_z} "
            f"atr={atr_mult} -> pnl={row['realized_pnl']:+.4f} "
            f"pos={row['positions']} win={row['win_rate']:.0%}",
        )

    df = pd.DataFrame(rows).sort_values("realized_pnl", ascending=False)
    print("\n=== TOP 10 ===")
    print(df.head(10).to_string(index=False))
    print("\n=== BOTTOM 5 ===")
    print(df.tail(5).to_string(index=False))

    best = df.iloc[0].to_dict()
    print("\n=== BEST ===")
    for key, value in best.items():
        print(f"  {key}: {value}")

    out = "examples/backtest/regime_hybrid_sweep_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

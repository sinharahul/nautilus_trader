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
Train/test evaluation for RegimeHybrid v2.

- Tune on the first half of ETHUSDT ticks (by time).
- Report win rate, expectancy, and profit factor on the held-out half.

Note: in-repo data is only ~5 hours (2020-08-14). "More data" is not available
locally; chronological split is the honest substitute until longer history is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Metrics:
    positions: int
    win_rate: float
    expectancy: float
    profit_factor: float
    total_pnl: float

    def score(self) -> float:
        # Primary: expectancy; need enough trades or heavily penalize
        if self.positions == 0:
            return -1e9
        sample_penalty = 0.0 if self.positions >= 3 else -0.25 * (3 - self.positions)
        return (
            self.expectancy
            + 0.05 * min(self.profit_factor, 5.0)
            + 0.01 * self.win_rate
            + sample_penalty
        )


def _pnl_values(positions: pd.DataFrame) -> list[float]:
    if len(positions) == 0:
        return []
    return [float(str(x).replace(" USDT", "")) for x in positions["realized_pnl"].tolist()]


def compute_metrics(positions: pd.DataFrame) -> Metrics:
    pnls = _pnl_values(positions)
    if not pnls:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0)
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 1e-12 else (999.0 if gross_win > 0 else 0.0)
    return Metrics(
        positions=len(pnls),
        win_rate=len(wins) / len(pnls),
        expectancy=sum(pnls) / len(pnls),
        profit_factor=pf,
        total_pnl=sum(pnls),
    )


def run_segment(
    ticks,
    instrument,
    *,
    start,
    end,
    efficiency_threshold: float,
    entry_z: float,
    confirm_bars: int,
    cooldown_bars: int,
    atr_loss_mult: float,
    atr_trail_mult: float,
) -> Metrics:
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("EVAL-001"),
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
                bar_type=BarType.from_str("ETHUSDT.BINANCE-500-TICK-LAST-INTERNAL"),
                trade_size=Decimal("0.10"),
                efficiency_threshold=efficiency_threshold,
                entry_z=entry_z,
                confirm_bars=confirm_bars,
                cooldown_bars=cooldown_bars,
                atr_loss_mult=atr_loss_mult,
                atr_trail_mult=atr_trail_mult,
                max_hold_bars=30,
            ),
        ),
    )
    engine.run(start=start, end=end)
    metrics = compute_metrics(engine.trader.generate_positions_report())
    engine.dispose()
    return metrics


def main() -> None:
    instrument = TestInstrumentProvider.ethusdt_binance()
    ticks = TradeTickDataWrangler(instrument=instrument).process(
        TestDataProvider().read_csv_ticks("binance/ethusdt-trades.csv"),
    )
    # Chronological 50/50 split on event timestamps
    mid_idx = len(ticks) // 2
    train_end = pd.Timestamp(ticks[mid_idx - 1].ts_event, unit="ns", tz="UTC")
    test_start = pd.Timestamp(ticks[mid_idx].ts_event, unit="ns", tz="UTC")
    data_start = pd.Timestamp(ticks[0].ts_event, unit="ns", tz="UTC")
    data_end = pd.Timestamp(ticks[-1].ts_event, unit="ns", tz="UTC")

    print(f"Data span: {data_start} -> {data_end} ({len(ticks)} ticks)")
    print(f"TRAIN: {data_start} -> {train_end}")
    print(f"TEST:  {test_start} -> {data_end}")
    print("NOTE: only ~5h in-repo; treat TEST as held-out within this day, not multi-day OOS.\n")

    grid = list(
        product(
            (0.35, 0.40, 0.45),  # efficiency
            (1.5, 1.75, 2.0),  # entry z
            (1, 2),  # confirm
            (0, 3),  # cooldown
            (1.5, 2.0),  # loss ATR
            (2.0, 2.5),  # trail ATR
        ),
    )

    rows = []
    best = None
    best_score = -1e18
    for i, (eff, z, conf, cool, loss_m, trail_m) in enumerate(grid, start=1):
        train = run_segment(
            ticks,
            instrument,
            start=data_start,
            end=train_end,
            efficiency_threshold=eff,
            entry_z=z,
            confirm_bars=conf,
            cooldown_bars=cool,
            atr_loss_mult=loss_m,
            atr_trail_mult=trail_m,
        )
        score = train.score()
        row = {
            "eff": eff,
            "z": z,
            "confirm": conf,
            "cooldown": cool,
            "loss_atr": loss_m,
            "trail_atr": trail_m,
            "train_pos": train.positions,
            "train_wr": train.win_rate,
            "train_exp": train.expectancy,
            "train_pf": train.profit_factor,
            "train_pnl": train.total_pnl,
            "train_score": score,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best = row
        if i % 20 == 0 or i == len(grid):
            print(f"tuned {i}/{len(grid)} ... best_train_score={best_score:.4f}")

    if best is None:
        # fallback: allow any trade count
        best = max(rows, key=lambda r: r["train_score"])

    print("\n=== BEST ON TRAIN (by expectancy-focused score) ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    test = run_segment(
        ticks,
        instrument,
        start=test_start,
        end=data_end,
        efficiency_threshold=best["eff"],
        entry_z=best["z"],
        confirm_bars=int(best["confirm"]),
        cooldown_bars=int(best["cooldown"]),
        atr_loss_mult=best["loss_atr"],
        atr_trail_mult=best["trail_atr"],
    )

    print("\n=== HELD-OUT TEST METRICS (primary) ===")
    print(f"  positions:     {test.positions}")
    print(f"  win_rate:      {test.win_rate:.1%}  (secondary)")
    print(f"  expectancy:    {test.expectancy:+.6f} USDT/trade  (primary)")
    print(f"  profit_factor: {test.profit_factor:.3f}")
    print(f"  total_pnl:     {test.total_pnl:+.6f} USDT")

    # Baseline comparison: old looser params on TEST only
    baseline = run_segment(
        ticks,
        instrument,
        start=test_start,
        end=data_end,
        efficiency_threshold=0.35,
        entry_z=1.5,
        confirm_bars=1,
        cooldown_bars=0,
        atr_loss_mult=2.5,
        atr_trail_mult=2.5,
    )
    print("\n=== BASELINE (loose) ON SAME TEST WINDOW ===")
    print(f"  positions:     {baseline.positions}")
    print(f"  win_rate:      {baseline.win_rate:.1%}")
    print(f"  expectancy:    {baseline.expectancy:+.6f} USDT/trade")
    print(f"  profit_factor: {baseline.profit_factor:.3f}")
    print(f"  total_pnl:     {baseline.total_pnl:+.6f} USDT")

    out = "examples/backtest/regime_hybrid_train_test_results.csv"
    pd.DataFrame(rows).sort_values("train_score", ascending=False).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

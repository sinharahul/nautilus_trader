# Python strategy development — code map

Navigate the repo from strategy API → runners → examples.

## Core Python types

| Role | Symbol | Path |
| --- | --- | --- |
| Strategy base | `Strategy` | `nautilus_trader/trading/strategy.pyx` |
| Actor base | `Actor` | `nautilus_trader/common/actor.pyx` |
| Strategy config / factory | `StrategyConfig`, `StrategyFactory` | `nautilus_trader/trading/config.py` |
| Order factory | `OrderFactory` | `nautilus_trader/common/factories.pyx` |
| Trader (registers strategies) | `Trader` | `nautilus_trader/trading/trader.py` |
| Kernel | `NautilusKernel` | `nautilus_trader/system/kernel.py` |
| Kernel config | `NautilusKernelConfig` | `nautilus_trader/system/config.py` |
| Backtest engine | `BacktestEngine` | `nautilus_trader/backtest/engine.pyx` |
| Backtest node | `BacktestNode` | `nautilus_trader/backtest/node.py` |
| Backtest configs | `BacktestEngineConfig`, `BacktestRunConfig` | `nautilus_trader/backtest/config.py` |
| Live node | `TradingNode` | `nautilus_trader/live/node.py` |
| Live config | `TradingNodeConfig` | `nautilus_trader/live/config.py` |
| Portfolio / cache | `Portfolio`, `Cache` | `nautilus_trader/portfolio/`, `nautilus_trader/cache/` |

## Related Rust crates

Strategy authors rarely edit these for Python strategies, but they back the runtime:

| Crate | Path | Role |
| --- | --- | --- |
| `nautilus-model` | `crates/model/` | Orders, events, instruments |
| `nautilus-common` | `crates/common/` | Actors, msgbus, cache, clocks |
| `nautilus-trading` | `crates/trading/` | Strategy machinery |
| `nautilus-data` | `crates/data/` | Data engine |
| `nautilus-execution` | `crates/execution/` | Execution engine / emulator |
| `nautilus-risk` | `crates/risk/` | Risk engine |
| `nautilus-backtest` | `crates/backtest/` | Simulated exchange / matching |
| `nautilus-live` | `crates/live/` | Live runtime |
| `nautilus-system` | `crates/system/` | Kernel-style orchestration |
| `nautilus-indicators` | `crates/indicators/` | Indicators |
| Adapters | `crates/adapters/*` | Venue clients |

## Call chains

### Handler dispatch (orders)

```text
ExecutionEngine applies fill
  → publishes OrderFilled (and related)
  → Strategy:
       on_order_filled(event)
       on_order_event(event)
       on_event(event)
  → may also emit PositionOpened/Changed/Closed → on_position_* → on_event
```

### Submit order (happy path, non-emulated)

```text
Strategy.submit_order(order)
  → OrderInitialized (+ SubmitOrder command on msgbus)
  → RiskEngine (pre-trade checks)  OR deny → OrderDenied
  → ExecutionEngine
  → ExecutionClient
       backtest: SimulatedExchange matching
       live: venue REST/WS
  → Order events return via ExecutionEngine → Strategy handlers
```

Branches before risk: `emulation_trigger` → `OrderEmulator`; `exec_algorithm_id` → `ExecAlgorithm`.
See `docs/concepts/execution.md`.

### on_start data path (EMACross pattern)

```text
on_start
  → cache.instrument(id)
  → register_indicator_for_bars(bar_type, ema)
  → request_bars(bar_type, callback=subscribe_bars)
  → (optional) subscribe_quote_ticks / subscribe_trade_ticks
```

Implementation reference: `nautilus_trader/examples/strategies/ema_cross.py`.

## Example index

### Strategy implementations

| File | Notes |
| --- | --- |
| `nautilus_trader/examples/strategies/blank.py` | Minimal shell |
| `nautilus_trader/examples/strategies/ema_cross.py` | Canonical bar + EMA + market |
| `nautilus_trader/examples/strategies/ema_cross_bracket.py` | Bracket exits |
| `nautilus_trader/examples/strategies/ema_cross_bracket_algo.py` | Bracket + exec algorithm |
| `nautilus_trader/examples/strategies/ema_cross_twap.py` | TWAP routing |
| `nautilus_trader/examples/strategies/orderbook_imbalance.py` | Book-driven |
| `nautilus_trader/examples/strategies/grid_market_maker.py` | Grid MM |
| `nautilus_trader/examples/strategies/signal_strategy.py` | Actor signals |

### Backtest tutorials (progressive)

Under `examples/backtest/`:

| Dir | Focus |
| --- | --- |
| `example_01_load_bars_from_custom_csv` | Load bars, run engine |
| `example_02_use_clock_timer` | Timers |
| `example_03_bar_aggregation` | Aggregation |
| `example_04_using_data_catalog` | Catalog |
| `example_05_using_portfolio` | Portfolio |
| `example_06_using_cache` | Cache |
| `example_07_using_indicators` | Indicators |
| `example_08_cascaded_indicator` | Indicator chains |
| `example_09`–`example_11` | Messaging / actor signals |

### Full scripts

| File | Runner |
| --- | --- |
| `examples/backtest/crypto_ema_cross_ethusdt_trade_ticks.py` | `BacktestEngine` + TWAP |
| `examples/backtest/fx_ema_cross_bracket_gbpusd_bars_internal.py` | Brackets / FX bars |
| `examples/backtest/tardis_option_chain.py` | `BacktestNode` |
| `examples/live/binance/binance_spot_ema_cross_bracket_algo.py` | `TradingNode` |
| `examples/live/bybit/bybit_ema_cross.py` | Simpler live EMA |
| `examples/sandbox/*` | Real data + simulated execution |
| `examples/other/minimal_reproducible_example/` | Minimal repro |

## Docs cross-links

| Topic | Official doc |
| --- | --- |
| Strategies | `docs/concepts/strategies.md` |
| Actors | `docs/concepts/actors.md` |
| Execution | `docs/concepts/execution.md` |
| Backtest APIs | `docs/concepts/backtesting/apis-and-runs.md` |
| Backtest exec flow | `docs/concepts/backtesting/execution-flow.md` |
| Live | `docs/concepts/live.md` |
| Data / bars | `docs/concepts/data/index.md` |
| Rust strategy how-to | `docs/how_to/write_rust_strategy.md` |

## Where to edit what

| Goal | Start here |
| --- | --- |
| Change strategy logic | Your `Strategy` subclass / `examples/strategies/*` |
| Change default order helpers | `nautilus_trader/trading/strategy.pyx`, `common/factories.pyx` |
| Change handler wiring | `nautilus_trader/common/actor.pyx`, `trading/strategy.pyx` |
| Change backtest loop timing vs strategy callbacks | `nautilus_trader/backtest/engine.pyx` + `crates/backtest/` |
| Change live bootstrap | `nautilus_trader/live/node.py`, `system/kernel.py` |
| Change risk checks | `nautilus_trader/risk/engine.pyx`, `crates/risk/` |

## Related RD_DOCS

- Orientation: [00_orientation.md](00_orientation.md)
- Deep dive: [01_deep_dive.md](01_deep_dive.md)
- Engine code map: [../02_trading_engine/02_code_map.md](../02_trading_engine/02_code_map.md)

# Trading engine — code map

File and crate map for kernel, engines, backtest loop, live node, and order flow.

## Python bootstrap and orchestration

| Role | Path |
| --- | --- |
| Kernel | `nautilus_trader/system/kernel.py` |
| Kernel config | `nautilus_trader/system/config.py` |
| Trader | `nautilus_trader/trading/trader.py` |
| Controller | `nautilus_trader/trading/controller.py` |
| Backtest engine (loop entry) | `nautilus_trader/backtest/engine.pyx` |
| Backtest node | `nautilus_trader/backtest/node.py` |
| Backtest configs | `nautilus_trader/backtest/config.py` |
| Simulated exchange (Python face) | `nautilus_trader/backtest/` (exchange / matching modules) |
| Trading node | `nautilus_trader/live/node.py` |
| Live configs | `nautilus_trader/live/config.py` |
| Live data engine | `nautilus_trader/live/data_engine.py` (and related) |
| Live execution engine | `nautilus_trader/live/execution_engine.py` |
| Live risk engine | `nautilus_trader/live/risk_engine.py` |

## Python engines and messaging

| Role | Path |
| --- | --- |
| MessageBus / clocks | `nautilus_trader/common/component.pyx` |
| DataEngine | `nautilus_trader/data/engine.pyx` |
| ExecutionEngine | `nautilus_trader/execution/engine.pyx` |
| OrderEmulator | `nautilus_trader/execution/emulator.pyx` (and package) |
| ExecAlgorithm base | `nautilus_trader/execution/algorithm.pyx` |
| RiskEngine | `nautilus_trader/risk/engine.pyx` |
| Cache | `nautilus_trader/cache/cache.py` (+ Cython pieces) |
| Portfolio | `nautilus_trader/portfolio/portfolio.py` |
| Strategy submit APIs | `nautilus_trader/trading/strategy.pyx` |

## Rust crates (performance core)

| Crate | Path | Engine concern |
| --- | --- | --- |
| `nautilus-core` | `crates/core/` | Time, UUIDs, primitives |
| `nautilus-model` | `crates/model/` | Domain types, order/events |
| `nautilus-common` | `crates/common/` | Msgbus, cache, clocks, actors |
| `nautilus-data` | `crates/data/` | Data engine |
| `nautilus-execution` | `crates/execution/` | Execution engine, emulator paths |
| `nautilus-risk` | `crates/risk/` | Risk engine |
| `nautilus-portfolio` | `crates/portfolio/` | Portfolio |
| `nautilus-trading` | `crates/trading/` | Strategy/trading machinery |
| `nautilus-system` | `crates/system/` | System orchestration |
| `nautilus-backtest` | `crates/backtest/` | Matching, simulated exchange, backtest engine |
| `nautilus-live` | `crates/live/` | Live runtime |
| `nautilus-network` | `crates/network/` | Async networking for adapters |
| `nautilus-persistence` | `crates/persistence/` | Catalog / streaming I/O |
| Adapters | `crates/adapters/<venue>/` | Venue data + exec clients |

## Kernel construction (code trail)

```text
TradingNode(config) / BacktestEngine(config)
  → NautilusKernel.__init__(name, config, loop=...)
       MessageBus(...)
       Cache(...)
       Portfolio(...)
       DataEngine | LiveDataEngine
       RiskEngine | LiveRiskEngine
       ExecutionEngine | LiveExecutionEngine
       OrderEmulator(...)
       Trader(...)
  → (backtest) add_venue / add_data / add_strategy
  → (live) add_*_client_factory → build()
  → run()
```

Read environment checks around live vs non-live config types in
`nautilus_trader/system/kernel.py` (raises `InvalidConfiguration` on mismatch).

## Backtest data-point pipeline (code trail)

Official sequence: `docs/concepts/backtesting/execution-flow.md`.

```text
BacktestEngine.run()
  → for each data point at ts=T:
       SimulatedExchange process_*  # match resting
       DataEngine.process(data)     # strategy on_* 
       _process_and_settle_venues(T)
           drain commands → matching iterate
           repeat while pending (cascades)
           simulation modules / expirations
  → end/stop → strategy on_stop → settle → stop engines
```

Implementation gravity: `nautilus_trader/backtest/engine.pyx` + `crates/backtest/`.

## Live pipeline (code trail)

```text
TradingNode(config)
  → register strategies on node.trader
  → add_data_client_factory / add_exec_client_factory
  → build()   # adapter clients connect into Live* engines
  → run()     # asyncio loop, LiveClock
  → dispose()
```

Example: `examples/live/binance/binance_spot_ema_cross_bracket_algo.py`  
Factories: e.g. `BinanceLiveDataClientFactory`, `BinanceLiveExecClientFactory` under
`nautilus_trader/adapters/binance/`.

## Submit → fill (code trail)

```text
strategy.pyx: submit_order
  → msgbus command SubmitOrder (+ OrderInitialized event)
  → optional: execution/emulator or execution/algorithm
  → risk/engine: checks → OrderDenied OR forward
  → execution/engine: route to ExecutionClient
  → backtest: crates/backtest matching OR live: adapters/<venue>
  → execution/engine: apply events, update cache/portfolio
  → strategy.pyx handlers: on_order_* / on_position_* / on_event
```

Diagram source of truth: `docs/concepts/execution.md`.

## Adapter attachment points

| Side | Python package | Rust crate |
| --- | --- | --- |
| Data client | `nautilus_trader/adapters/<venue>/` | `crates/adapters/<venue>/` |
| Exec client | same | same |
| Listing / tiers | `ADAPTERS.md` | — |

New venues need project approval — see `ROADMAP.md` and `ADAPTERS.md`.

## Example scripts by engine concern

| Concern | Example |
| --- | --- |
| BacktestEngine + strategy + exec algo | `examples/backtest/crypto_ema_cross_ethusdt_trade_ticks.py` |
| BacktestNode / catalog-style runs | `examples/backtest/tardis_option_chain.py` |
| Tutorial engine usage | `examples/backtest/example_01_…` through `example_11_…` |
| Live TradingNode | `examples/live/binance/binance_spot_ema_cross_bracket_algo.py` |
| Sandbox | `examples/sandbox/` |
| Minimal system | `examples/other/minimal_reproducible_example/` |

## Docs cross-links

| Topic | Path |
| --- | --- |
| Architecture | `docs/concepts/architecture.md` |
| Overview / use cases | `docs/concepts/overview.md` |
| Message bus | `docs/concepts/message_bus.md` |
| Execution | `docs/concepts/execution.md` |
| Cache | `docs/concepts/cache.md` |
| Portfolio | `docs/concepts/portfolio.md` |
| Backtesting index | `docs/concepts/backtesting/index.md` |
| Execution flow | `docs/concepts/backtesting/execution-flow.md` |
| APIs and runs | `docs/concepts/backtesting/apis-and-runs.md` |
| Live | `docs/concepts/live.md` |
| Events | `docs/concepts/events/index.md` |

## Where to edit what

| Goal | Start here |
| --- | --- |
| Change kernel wiring / engine selection | `nautilus_trader/system/kernel.py` |
| Change backtest timestamp phases | `nautilus_trader/backtest/engine.pyx`, `crates/backtest/` |
| Change risk denial rules | `nautilus_trader/risk/`, `crates/risk/` |
| Change order state / fill application | `nautilus_trader/execution/engine.pyx`, `crates/execution/` |
| Change matching / sim fills | `crates/backtest/` |
| Change live reconciliation | `nautilus_trader/live/`, `crates/live/` |
| Change msgbus semantics | `nautilus_trader/common/component.pyx`, `crates/common/` |
| Add venue | `crates/adapters/<venue>/` + `nautilus_trader/adapters/<venue>/` (after approval) |

## Related RD_DOCS

- Orientation: [00_orientation.md](00_orientation.md)
- Deep dive: [01_deep_dive.md](01_deep_dive.md)
- Strategy code map: [../01_python_strategy_dev/02_code_map.md](../01_python_strategy_dev/02_code_map.md)

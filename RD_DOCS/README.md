# RD_DOCS

Research / developer narrative docs for NautilusTrader.

These pages complement the official Sphinx docs under `docs/`. Prefer `docs/` as the
canonical API and concept reference; use `RD_DOCS/` for end-to-end flows, diagrams, and
code maps when navigating the repo.

## How to read

Each topic has three layers:

| Layer | File | Purpose |
| --- | --- | --- |
| Orientation | `00_orientation.md` | Day-one picture and minimal path |
| Deep dive | `01_deep_dive.md` | Lifecycle, semantics, gotchas |
| Code map | `02_code_map.md` | Concrete files, crates, call chains, examples |

Start at `00`, go deeper as needed, use `02` when editing or debugging.

## Topics

1. [Python strategy development flow](01_python_strategy_dev/00_orientation.md)
2. [How the trading engine runs](02_trading_engine/00_orientation.md)
3. [Regime-hybrid ETHUSDT design](03_regime_hybrid_design.md) — example strategy + backtest
4. [RL strategies (tabular / offline / gym)](04_rl_strategies.md)
5. [FinRL bridge](05_finrl_bridge.md) — FinRL FeatureEngineer + DRLAgent on ETHUSDT

## Scope notes

- Examples and paths target the **v1** package (`nautilus_trader/`), which is what most
  strategies and tutorials use today (Cython + Rust core).
- The **v2** package under `python/` is PyO3-first and separate (own venv / `target-v2`).
  Do not mix v1 and v2 imports in one process.
- Official concepts: [Strategies](../docs/concepts/strategies.md),
  [Architecture](../docs/concepts/architecture.md),
  [Execution](../docs/concepts/execution.md),
  [Backtesting](../docs/concepts/backtesting/index.md),
  [Live](../docs/concepts/live.md).

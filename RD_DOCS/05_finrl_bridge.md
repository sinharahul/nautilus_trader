# FinRL bridge

Uses code from local FinRL checkout:

`/Users/rahulsinha/Documents/GitHub/FinRL`

## What is reused

| FinRL piece | Role |
| --- | --- |
| `finrl.meta.preprocessor.FeatureEngineer` | MACD/Boll/RSI/CCI/DX/SMA features (`INDICATORS`) |
| `finrl.meta.preprocessor.data_split` | Train/test date split |
| `finrl.meta.env_stock_trading.StockTradingEnv` | Gym trading env (cash / long-only) |
| `finrl.agents.stablebaselines3.DRLAgent` | A2C train + `DRL_prediction` |
| MACD+RSI rule on FinRL features | Fallback Nautilus signals when RL collapses on short samples |

## Run

```bash
# From nautilus_trader root (venv with gymnasium, stable-baselines3, stockstats, matplotlib, yfinance)
python examples/backtest/finrl_ethusdt_bridge.py
python examples/backtest/crypto_finrl_signal_ethusdt.py
```

The bridge stubs `finrl/__init__.py` (skips Alpaca paper-trading imports) and reuses:

- `FeatureEngineer` + `INDICATORS`
- `StockTradingEnv`
- `DRLAgent` (A2C)

Artifacts land in `examples/backtest/rl_artifacts/finrl/`.

Nautilus replay of exported signals:

- Strategy: `nautilus_trader/examples/strategies/finrl_signal.py`
- Runner: `examples/backtest/crypto_finrl_signal_ethusdt.py`
- Signals CSV: `examples/backtest/rl_artifacts/finrl/finrl_signals.csv`

## Caveat

In-repo ETHUSDT ticks are ~5 hours. FinRL was designed for multi-year daily equity data.
Treat results as an integration demo, not a 90% win-rate claim.

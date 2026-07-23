# RL strategies (ETHUSDT research stack)

*** Research templates only — not for live trading with real money. ***

## Layout

| Piece | Path |
| --- | --- |
| Shared features / actions | `nautilus_trader/examples/strategies/rl/features.py` |
| (1) Tabular Q strategy | `.../rl/tabular_q.py` |
| (2) Offline NumPy MLP | `.../rl/offline_policy.py` |
| (3) Gymnasium env | `.../rl/env.py` |
| Tabular runner | `examples/backtest/crypto_rl_tabular_q_ethusdt.py` |
| Offline runner | `examples/backtest/crypto_rl_offline_policy_ethusdt.py` |
| Gym / SB3 runner | `examples/backtest/crypto_rl_gym_sb3_ethusdt.py` |
| Artifacts | `examples/backtest/rl_artifacts/` |

## How to run

```bash
# 1) Tabular Q (learns online, writes q_table + transitions)
python examples/backtest/crypto_rl_tabular_q_ethusdt.py

# 2) Behavior-clone MLP from transitions, then backtest frozen policy
python examples/backtest/crypto_rl_offline_policy_ethusdt.py

# 3) Feature cache + optional PPO (needs extras)
pip install gymnasium stable-baselines3
python examples/backtest/crypto_rl_gym_sb3_ethusdt.py
```

## Design notes

1. **Tabular Q** — 18 discrete states (regime × z-bucket × position), 4 actions
   (HOLD/BUY/SELL/FLAT), ε-greedy, ATR stop overlay, same 500-tick ETHUSDT sample.
2. **Offline policy** — NumPy MLP trained by behavior cloning on logged transitions;
   no PyTorch required.
3. **Gym env** — Nautilus `BacktestEngine` dumps bar features once; env steps a fast
   simulator for training. Evaluate policies with the Strategy backtests above.

In-sample on a short CSV; treat PnL as a plumbing check, not alpha.

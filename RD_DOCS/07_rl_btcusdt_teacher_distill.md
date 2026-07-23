# RL data/backtest design — BTCUSDT teacher distillation

## Problem

On ~5h in-repo ETH ticks, RL collapsed to flat/exit (safe local optimum) and never
learned RegimeHybrid’s high win-rate behavior.

## Goal

Give RL enough history and a teacher so the student recovers (then lightly improves)
a high-WR policy on **BTCUSDT spot**.

## Data

| Item | Choice |
| --- | --- |
| Symbol | `BTCUSDT` spot |
| Source | [Binance Vision](https://data.binance.vision) monthly/daily 1m klines |
| Default window | Last **60 calendar days** (configurable) |
| Bar type | `BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL` |
| Cache | `examples/backtest/rl_artifacts/btcusdt_data/` |

## Split

Chronological **70% train / 15% val / 15% test** by bar time (no shuffle).

## Teacher

`RegimeHybrid` with the locked high-WR-style params (eff=0.45, z=1.5, ATR 3.5/3.5,
confirm=1, cooldown=0), retuned only if 1m BTC needs different bar dynamics.

Each bar: log `obs` (`build_obs`) + discrete `RLAction` inferred from teacher intent
(BUY/SELL/FLAT/HOLD).

## Student

1. **Behavior clone** NumPy MLP on train transitions  
2. **Optional fine-tune** with win/loss round-trip reward on train (val early-stop)  
3. **Eval** frozen policy on test via Nautilus backtest (or bar simulator)

## Success criteria

- Teacher test WR baseline recorded  
- Student test WR within **10 percentage points** of teacher  
- ≥ **20** closed test trades  
- Beat flat / random action baselines on composite (WR + PF + PnL)

## Runner

```bash
python examples/backtest/crypto_rl_teacher_distill_btcusdt.py
```

## Non-goals (v1)

Tick-level 500-bar teacher, FinRL continuous agents, live trading.

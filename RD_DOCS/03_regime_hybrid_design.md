# Regime-hybrid ETHUSDT strategy — design (Approach A)

Approved design for a research-light hybrid strategy on Binance ETHUSDT trade ticks.

## Goal

Demonstrate a regime-switching strategy (trend vs mean-reversion) with a simple ATR stop,
backed by in-repo trade-tick data.

## Signal design

- Bars: INTERNAL tick bars from trade ticks (e.g. 250-TICK-LAST-INTERNAL).
- Regime: Kaufman `EfficiencyRatio` — high → TREND, low → MEAN_REVERT.
- TREND entries: fast/slow EMA cross while flat.
- MEAN_REVERT entries: close z-score vs rolling mid beyond ±entry_z (fade) while flat.
- Exit: ATR multiple stop from entry price; opposing regime signal may flatten/flip via market orders.
- Non-goals: daily circuit breaker, TP ladder, ML, live node.

## Deliverables

- `nautilus_trader/examples/strategies/regime_hybrid.py`
- `examples/backtest/crypto_regime_hybrid_ethusdt_trade_ticks.py`

## Backtest

- Venue: Binance spot CASH, NETTING, L1_MBP, `trade_execution=True`
- Data: `tests/test_data/binance/ethusdt-trades.csv`
- Print account / fills / positions reports
- Sweep: `examples/backtest/crypto_regime_hybrid_param_sweep.py`
## Improvements tried (v2)

| Idea | Implementation | Result on ~5h sample |
| --- | --- | --- |
| Fewer/better entries | Higher z/eff, `confirm_bars`, `cooldown_bars` | Cuts trades; too strict → 0 trades |
| Asymmetric exits | `atr_loss_mult`, `atr_trail_mult`, `max_hold_bars` | Loss stops fire; mid exits still dominate MR |
| Train/test split | `crypto_regime_hybrid_train_test_eval.py` | Tune on AM, report PM held-out |
| Optimize expectancy/PF | Score uses expectancy + PF, win rate secondary | Best train params ≈ loose settings here |
| More data | Not in repo (only 5h CSV) | Split is within-day only |

Eval: `python examples/backtest/crypto_regime_hybrid_train_test_eval.py`

## 90% win-rate config (in-sample on `ethusdt-trades.csv`)

| Param | Value |
| --- | --- |
| Bars | `500-TICK-LAST-INTERNAL` |
| `efficiency_threshold` | 0.45 |
| `entry_z` | 1.5 |
| `atr_loss_mult` / `atr_trail_mult` | 3.5 / 3.5 |
| `confirm_bars` / `cooldown_bars` | 1 / 0 |

Observed: **10** positions, **90%** win rate, **~+0.77 USDT** realized.

Caveat: in-sample on ~5 hours of 2020-08-14 ticks — not a live edge claim.


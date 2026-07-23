# Resume entry — NautilusTrader quant / RL research

## Full project entry (use this)

**Algorithmic Trading & Reinforcement Learning Research**  
*NautilusTrader · Binance ETHUSDT · Personal R&D*  
*Python, NautilusTrader, NumPy, Pandas, Gymnasium, Stable-Baselines3, FinRL*

Independent research project building event-driven trading strategies and RL pipelines on top of NautilusTrader, using Binance ETHUSDT trade-tick data for backtesting and policy evaluation.

- Architected a **regime-switching hybrid strategy** that classifies market state with Kaufman’s Efficiency Ratio, then routes to EMA trend-following or Bollinger mean-reversion entries, with asymmetric ATR loss/trail exits, confirmation bars, and cooldown filters.
- Built reproducible **backtest harnesses** (full-sample runners, parameter sweeps, chronological train/test evaluation) reporting win rate, expectancy, profit factor, and realized PnL—not only headline returns.
- Tuned strategy hyperparameters on tick bars and achieved an **in-sample 90% win rate** (10 closed positions, ~+0.77 USDT realized PnL) on the repository’s ETHUSDT trade-tick sample; documented limits of short-horizon, single-day data.
- Implemented a **multi-tier RL research stack** inside Nautilus examples: (1) online tabular Q-learning with discretized regime/z/position states; (2) offline NumPy MLP trained by behavior cloning on logged transitions; (3) Gymnasium environment fed by Nautilus feature dumps for Stable-Baselines3 (PPO) training, with frozen policies replayed as Nautilus strategies.
- Integrated **FinRL** libraries (`FeatureEngineer`, `StockTradingEnv`, `DRLAgent`) by converting ticks to OHLCV, generating technical features, training A2C/PPO agents, exporting action signals, and replaying them in Nautilus via a long-only signal strategy—bridging academic RL tooling with production-style event-driven execution.
- Wrote research documentation covering strategy design, RL layout, FinRL bridge caveats, and evaluation methodology for handoff and future work (including teacher-distillation RL aimed at high win rate).

**Outcomes:** End-to-end path from raw ticks → indicators → classical + RL policies → Nautilus fills/PnL reports; clear separation of research templates vs live trading; measurable backtest metrics suitable for iteration.

---

## Shorter variant (if space is tight)

**Algorithmic Trading & RL Research — NautilusTrader**  
*Python · NautilusTrader · Gymnasium · Stable-Baselines3 · FinRL · NumPy*

- Built regime-hybrid ETHUSDT strategies (Efficiency Ratio → EMA / Bollinger + ATR) with sweeps and train/test evaluation in NautilusTrader; locked an in-sample **90% win rate** (10 trades, ~+0.77 USDT).
- Delivered a three-tier RL stack (tabular Q, offline MLP behavior cloning, Gymnasium/SB3) with shared features and Nautilus policy replay.
- Bridged FinRL feature engineering and DRL agents to Nautilus via tick→OHLCV preprocessing and signal-based execution.

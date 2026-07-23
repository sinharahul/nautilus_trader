# SPY synthetic options strategy bake-off

## Goal

Backtest **10 classic options strategies** on **SPY** (last **2 years**, daily) using a
**Black–Scholes synthetic chain**, then rank by a composite of win rate, profit factor,
and total PnL.

## Approach

Self-contained research simulator (Approach A). Not venue fills / real IV surface.

## Strategies

1. Long Call  
2. Long Put  
3. Covered Call  
4. Cash-Secured Put  
5. Bull Call Debit Spread  
6. Bear Put Debit Spread  
7. Long Straddle  
8. Long Strangle  
9. Iron Condor  
10. Long Call Butterfly  

## Shared rules

- Entries: each Monday (or next session) when flat  
- Expiry: ~35 calendar days  
- Vol: 20-day realized σ (annualized); r = 0.04  
- Strike grid: $1 steps near spot  
- Exit: 50% of max theoretical credit/debit target, or 2× debit loss, or ≤7 DTE, or expiry  
- Position size: 1 contract (100 shares multiplier) unless noted (covered call holds 100 sh)

## Ranking

```
score = 0.40 * win_rate + 0.30 * tanh(PF/5) + 0.30 * tanh(total_pnl / scale)
```

Require ≥5 closed trades; else disqualified.

## Runner

`python examples/backtest/options_spy_strategy_bakeoff.py`

## Result (run 2026-07-23)

**Winner: Covered Call** — composite score ≈ 0.72 (WR 77%, PF ≈ 1.97, PnL ≈ $13.2k over 31 trades).

Runner-up: Cash-Secured Put (higher WR, lower total PnL).

Caveat: SPY was in a broad uptrend over the 2y window; covered-call PnL includes the long stock leg. Synthetic BS (realized vol), not live IV/fills.


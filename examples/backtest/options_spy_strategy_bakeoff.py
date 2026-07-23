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
SPY synthetic options bake-off: 10 classic strategies, Black-Scholes MTM, composite ranking.

Research simulator only — not venue fills or live trading.

  python examples/backtest/options_spy_strategy_bakeoff.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


ARTIFACT_DIR = Path("examples/backtest/rl_artifacts/options_bakeoff")
R = 0.04
MULT = 100.0  # shares per contract
DTE_ENTRY = 35
DTE_EXIT = 7
VOL_WINDOW = 20
STRIKE_STEP = 1.0


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------


def _bs_d1_d2(s: float, k: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    if t <= 1e-12 or sigma <= 1e-12 or s <= 0 or k <= 0:
        return 0.0, 0.0
    vol_sqrt = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / vol_sqrt
    d2 = d1 - vol_sqrt
    return float(d1), float(d2)


def bs_price(s: float, k: float, t: float, r: float, sigma: float, kind: str) -> float:
    """European option mid price; at expiry returns intrinsic."""
    if t <= 1e-12:
        if kind == "C":
            return max(s - k, 0.0)
        return max(k - s, 0.0)
    d1, d2 = _bs_d1_d2(s, k, t, r, sigma)
    if kind == "C":
        return float(s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))


def round_strike(x: float, step: float = STRIKE_STEP) -> float:
    return float(np.round(x / step) * step)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


def load_spy(years: float = 2.0) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(days=int(years * 365.25))
    raw = yf.download("SPY", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["rv"] = log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    df["rv"] = df["rv"].clip(lower=0.08, upper=0.80).bfill()
    return df


def year_frac(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return max((d1 - d0).days / 365.25, 0.0)


# ---------------------------------------------------------------------------
# Legs & positions
# ---------------------------------------------------------------------------


@dataclass
class Leg:
    kind: str  # "C" or "P"
    strike: float
    qty: int  # +1 long, -1 short
    entry_premium: float = 0.0


@dataclass
class Position:
    strategy: str
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    legs: list[Leg]
    stock_qty: int = 0  # shares (+ long)
    stock_entry: float = 0.0
    entry_cash: float = 0.0  # cash paid (+) or received (-) at open (options*mult + stock)
    max_profit_ref: float = 0.0  # positive target for 50% rule (debit strategies use debit)
    is_credit: bool = False
    closed: bool = False
    exit_date: pd.Timestamp | None = None
    exit_pnl: float = 0.0


@dataclass
class TradeResult:
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    pnl: float
    won: bool


def mark_stock(pos: Position, spot: float) -> float:
    if pos.stock_qty == 0:
        return 0.0
    return pos.stock_qty * (spot - pos.stock_entry)


def mtm_pnl(pos: Position, spot: float, asof: pd.Timestamp, sigma: float) -> float:
    """Mark-to-market PnL vs entry (options + optional stock hedge)."""
    opt_now = sum(
        leg.qty * bs_price(spot, leg.strike, year_frac(asof, pos.expiry), R, sigma, leg.kind) * MULT
        for leg in pos.legs
    )
    opt_entry = sum(leg.qty * leg.entry_premium * MULT for leg in pos.legs)
    return (opt_now - opt_entry) + mark_stock(pos, spot)


def fill_entry_premiums(legs: list[Leg], spot: float, asof: pd.Timestamp, expiry: pd.Timestamp, sigma: float) -> None:
    t = year_frac(asof, expiry)
    for leg in legs:
        leg.entry_premium = bs_price(spot, leg.strike, t, R, sigma, leg.kind)


# ---------------------------------------------------------------------------
# Strategy builders (return Position or None)
# ---------------------------------------------------------------------------

StrategyFn = Callable[[pd.Timestamp, float, float, pd.Timestamp], Position | None]


def _expiry(asof: pd.Timestamp) -> pd.Timestamp:
    return asof + pd.Timedelta(days=DTE_ENTRY)


def long_call(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k = round_strike(spot)
    legs = [Leg("C", k, +1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    debit = sum(l.entry_premium * MULT for l in legs)
    return Position("Long Call", asof, expiry, legs, entry_cash=debit, max_profit_ref=debit, is_credit=False)


def long_put(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k = round_strike(spot)
    legs = [Leg("P", k, +1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    debit = sum(l.entry_premium * MULT for l in legs)
    return Position("Long Put", asof, expiry, legs, entry_cash=debit, max_profit_ref=debit, is_credit=False)


def covered_call(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k = round_strike(spot * 1.02)  # ~2% OTM
    legs = [Leg("C", k, -1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    credit = legs[0].entry_premium * MULT
    return Position(
        "Covered Call",
        asof,
        expiry,
        legs,
        stock_qty=100,
        stock_entry=spot,
        entry_cash=100 * spot - credit,
        max_profit_ref=credit + (k - spot) * 100,
        is_credit=True,
    )


def cash_secured_put(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k = round_strike(spot * 0.98)
    legs = [Leg("P", k, -1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    credit = legs[0].entry_premium * MULT
    return Position(
        "Cash-Secured Put",
        asof,
        expiry,
        legs,
        entry_cash=-credit,
        max_profit_ref=credit,
        is_credit=True,
    )


def bull_call_spread(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k1 = round_strike(spot)
    k2 = round_strike(spot + 5)
    legs = [Leg("C", k1, +1), Leg("C", k2, -1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    debit = sum(l.qty * l.entry_premium * MULT for l in legs)
    width = (k2 - k1) * MULT
    return Position(
        "Bull Call Spread",
        asof,
        expiry,
        legs,
        entry_cash=debit,
        max_profit_ref=max(width - debit, debit),
        is_credit=False,
    )


def bear_put_spread(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k1 = round_strike(spot)
    k2 = round_strike(spot - 5)
    legs = [Leg("P", k1, +1), Leg("P", k2, -1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    debit = sum(l.qty * l.entry_premium * MULT for l in legs)
    width = (k1 - k2) * MULT
    return Position(
        "Bear Put Spread",
        asof,
        expiry,
        legs,
        entry_cash=debit,
        max_profit_ref=max(width - debit, debit),
        is_credit=False,
    )


def long_straddle(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k = round_strike(spot)
    legs = [Leg("C", k, +1), Leg("P", k, +1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    debit = sum(l.entry_premium * MULT for l in legs)
    return Position("Long Straddle", asof, expiry, legs, entry_cash=debit, max_profit_ref=debit, is_credit=False)


def long_strangle(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    kc = round_strike(spot * 1.02)
    kp = round_strike(spot * 0.98)
    legs = [Leg("C", kc, +1), Leg("P", kp, +1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    debit = sum(l.entry_premium * MULT for l in legs)
    return Position("Long Strangle", asof, expiry, legs, entry_cash=debit, max_profit_ref=debit, is_credit=False)


def iron_condor(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    # Short 5-wide wings ~5% OTM
    put_short = round_strike(spot * 0.95)
    put_long = round_strike(put_short - 5)
    call_short = round_strike(spot * 1.05)
    call_long = round_strike(call_short + 5)
    legs = [
        Leg("P", put_long, +1),
        Leg("P", put_short, -1),
        Leg("C", call_short, -1),
        Leg("C", call_long, +1),
    ]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    credit = -sum(l.qty * l.entry_premium * MULT for l in legs)  # net credit > 0
    if credit <= 0:
        return None
    return Position(
        "Iron Condor",
        asof,
        expiry,
        legs,
        entry_cash=-credit,
        max_profit_ref=credit,
        is_credit=True,
    )


def long_call_butterfly(asof: pd.Timestamp, spot: float, sigma: float, expiry: pd.Timestamp) -> Position | None:
    k = round_strike(spot)
    legs = [Leg("C", k - 5, +1), Leg("C", k, -2), Leg("C", k + 5, +1)]
    fill_entry_premiums(legs, spot, asof, expiry, sigma)
    debit = sum(l.qty * l.entry_premium * MULT for l in legs)
    if debit <= 0:
        return None
    return Position(
        "Long Call Butterfly",
        asof,
        expiry,
        legs,
        entry_cash=debit,
        max_profit_ref=5 * MULT - debit,
        is_credit=False,
    )


STRATEGIES: list[tuple[str, StrategyFn]] = [
    ("Long Call", long_call),
    ("Long Put", long_put),
    ("Covered Call", covered_call),
    ("Cash-Secured Put", cash_secured_put),
    ("Bull Call Spread", bull_call_spread),
    ("Bear Put Spread", bear_put_spread),
    ("Long Straddle", long_straddle),
    ("Long Strangle", long_strangle),
    ("Iron Condor", iron_condor),
    ("Long Call Butterfly", long_call_butterfly),
]


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def should_exit(pos: Position, pnl: float, asof: pd.Timestamp) -> bool:
    dte = (pos.expiry - asof).days
    if dte <= DTE_EXIT or asof >= pos.expiry:
        return True
    if pos.is_credit:
        # Take 50% of credit; stop at 2x credit loss
        if pnl >= 0.5 * pos.max_profit_ref:
            return True
        if pnl <= -2.0 * pos.max_profit_ref:
            return True
    else:
        if pos.max_profit_ref > 0 and pnl >= 0.5 * pos.max_profit_ref:
            return True
        if pos.entry_cash > 0 and pnl <= -2.0 * abs(pos.entry_cash):
            return True
    return False


def run_strategy(df: pd.DataFrame, name: str, builder: StrategyFn) -> list[TradeResult]:
    trades: list[TradeResult] = []
    pos: Position | None = None
    dates = df.index.to_list()

    for i, asof in enumerate(dates):
        if i < VOL_WINDOW:
            continue
        spot = float(df.loc[asof, "Close"])
        sigma = float(df.loc[asof, "rv"])

        if pos is not None and not pos.closed:
            pnl = mtm_pnl(pos, spot, asof, sigma)
            if should_exit(pos, pnl, asof):
                # At expiry use intrinsic
                if asof >= pos.expiry:
                    pnl = mtm_pnl(pos, spot, pos.expiry, sigma)
                pos.closed = True
                pos.exit_date = asof
                pos.exit_pnl = pnl
                trades.append(
                    TradeResult(name, pos.entry_date, asof, pnl, pnl > 0),
                )
                pos = None

        # Enter on Mondays when flat
        if pos is None and asof.weekday() == 0:
            expiry = _expiry(asof)
            # Need enough calendar left in sample
            if expiry > dates[-1]:
                continue
            new_pos = builder(asof, spot, sigma, expiry)
            if new_pos is not None:
                pos = new_pos

    # Force close leftover
    if pos is not None and not pos.closed:
        asof = dates[-1]
        spot = float(df.loc[asof, "Close"])
        sigma = float(df.loc[asof, "rv"])
        pnl = mtm_pnl(pos, spot, asof, sigma)
        trades.append(TradeResult(name, pos.entry_date, asof, pnl, pnl > 0))

    return trades


@dataclass
class Metrics:
    strategy: str
    trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    expectancy: float
    score: float
    qualified: bool


def compute_metrics(name: str, trades: list[TradeResult], pnl_scale: float) -> Metrics:
    if not trades:
        return Metrics(name, 0, 0.0, 0.0, 0.0, 0.0, -1.0, False)
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    gw = sum(wins)
    gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else (99.0 if gw > 0 else 0.0)
    total = sum(pnls)
    exp = total / len(pnls)
    score = 0.40 * wr + 0.30 * float(np.tanh(pf / 5.0)) + 0.30 * float(np.tanh(total / pnl_scale))
    return Metrics(name, len(pnls), wr, pf, total, exp, score, len(pnls) >= 5)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading SPY (2y daily) ...")
    df = load_spy(2.0)
    print(f"Bars: {len(df)}  {df.index[0].date()} → {df.index[-1].date()}")

    rows: list[Metrics] = []
    all_trades: list[dict] = []
    for name, fn in STRATEGIES:
        trades = run_strategy(df, name, fn)
        for t in trades:
            all_trades.append(
                {
                    "strategy": t.strategy,
                    "entry": str(t.entry_date.date()),
                    "exit": str(t.exit_date.date()),
                    "pnl": t.pnl,
                    "won": t.won,
                },
            )
        print(f"  {name}: {len(trades)} trades")
        rows.append(compute_metrics(name, trades, pnl_scale=1.0))  # scale filled later

    # Recompute scores with scale = median |total_pnl| of qualified
    abs_totals = [abs(m.total_pnl) for m in rows if m.qualified and abs(m.total_pnl) > 1]
    scale = float(np.median(abs_totals)) if abs_totals else 1000.0
    scale = max(scale, 100.0)
    ranked: list[Metrics] = []
    for name, fn in STRATEGIES:
        trades = [TradeResult(r["strategy"], pd.Timestamp(r["entry"]), pd.Timestamp(r["exit"]), r["pnl"], r["won"]) for r in all_trades if r["strategy"] == name]
        ranked.append(compute_metrics(name, trades, pnl_scale=scale))

    ranked.sort(key=lambda m: (m.qualified, m.score), reverse=True)

    table = pd.DataFrame(
        [
            {
                "strategy": m.strategy,
                "trades": m.trades,
                "win_rate": round(m.win_rate, 4),
                "profit_factor": round(m.profit_factor, 3),
                "total_pnl": round(m.total_pnl, 2),
                "expectancy": round(m.expectancy, 2),
                "score": round(m.score, 4),
                "qualified": m.qualified,
            }
            for m in ranked
        ],
    )
    out_csv = ARTIFACT_DIR / "bakeoff_results.csv"
    table.to_csv(out_csv, index=False)
    pd.DataFrame(all_trades).to_csv(ARTIFACT_DIR / "bakeoff_trades.csv", index=False)

    winner = next((m for m in ranked if m.qualified), ranked[0])
    summary = {
        "winner": winner.strategy,
        "score": winner.score,
        "win_rate": winner.win_rate,
        "profit_factor": winner.profit_factor,
        "total_pnl": winner.total_pnl,
        "trades": winner.trades,
        "pnl_scale": scale,
        "underlier": "SPY",
        "lookback_years": 2,
        "note": "Synthetic BS options — research only",
    }
    (ARTIFACT_DIR / "bakeoff_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SPY options bake-off (composite rank) ===")
    print(table.to_string(index=False))
    print(f"\nWINNER: {winner.strategy}")
    print(
        f"  score={winner.score:.4f}  WR={winner.win_rate:.1%}  "
        f"PF={winner.profit_factor:.2f}  PnL=${winner.total_pnl:,.2f}  trades={winner.trades}",
    )
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()

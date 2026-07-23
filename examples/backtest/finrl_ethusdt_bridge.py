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
Bridge: Nautilus ETHUSDT ticks → FinRL FeatureEngineer + StockTradingEnv + DRLAgent (PPO).

Uses libraries from /Users/rahulsinha/Documents/GitHub/FinRL (on sys.path).

*** RESEARCH ONLY — NOT FOR LIVE TRADING. ***
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FINRL_ROOT = Path("/Users/rahulsinha/Documents/GitHub/FinRL")
if str(FINRL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINRL_ROOT))

# Avoid finrl/__init__.py (pulls Alpaca paper-trading deps we do not need).
import types

if "finrl" not in sys.modules:
    finrl_pkg = types.ModuleType("finrl")
    finrl_pkg.__path__ = [str(FINRL_ROOT / "finrl")]
    sys.modules["finrl"] = finrl_pkg

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import A2C_PARAMS
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split

NAUTILUS_ROOT = Path(__file__).resolve().parents[2]
TICK_CSV = NAUTILUS_ROOT / "tests/test_data/binance/ethusdt-trades.csv"
ARTIFACT_DIR = NAUTILUS_ROOT / "examples/backtest/rl_artifacts/finrl"
MODEL_PATH = ARTIFACT_DIR / "a2c_finrl_ethusdt.zip"
SIGNALS_CSV = ARTIFACT_DIR / "finrl_signals.csv"


def ticks_to_ohlcv(path: Path, rule: str = "1min") -> pd.DataFrame:
    """Resample trade ticks to FinRL-style OHLCV with date + tic columns."""
    raw = pd.read_csv(path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, format="mixed")
    raw = raw.set_index("timestamp").sort_index()
    ohlcv = raw["price"].resample(rule).ohlc()
    ohlcv["volume"] = raw["quantity"].resample(rule).sum()
    ohlcv = ohlcv.dropna(subset=["close"])
    ohlcv = ohlcv.reset_index()
    ohlcv = ohlcv.rename(columns={"timestamp": "date"})
    ohlcv["date"] = ohlcv["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    ohlcv["tic"] = "ETHUSDT"
    ohlcv["day"] = ohlcv["date"].dt.dayofweek
    return ohlcv[["date", "tic", "open", "high", "low", "close", "volume", "day"]]


def build_finrl_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(ohlcv)
    processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)
    # FinRL StockTradingEnv indexes by factorized date
    processed.index = processed["date"].factorize()[0]
    return processed


def make_env(df: pd.DataFrame) -> StockTradingEnv:
    stock_dim = len(df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    # FinRL casts action*hmax to int — keep hmax large so small continuous
    # actions still become tradeable share counts on minute crypto bars.
    return StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=50_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        reward_scaling=1e-3,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        print_verbosity=10_000,
    )


def indicator_rule_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-only FinRL-indicator rule using FeatureEngineer columns.

    Buy when MACD > 0 and RSI < 70; sell/flat when MACD < 0 or RSI > 75.
    Action in {-1, 0, +1} for Nautilus FinRLSignalStrategy.
    """
    out = df.drop_duplicates("date").sort_values("date").copy()
    actions: list[float] = []
    pos = 0.0
    for _, row in out.iterrows():
        macd = float(row["macd"])
        rsi = float(row["rsi_30"])
        if pos <= 0 and macd > 0 and rsi < 70:
            pos = 1.0
            actions.append(1.0)
        elif pos > 0 and (macd < 0 or rsi > 75):
            pos = 0.0
            actions.append(-1.0)
        else:
            actions.append(0.0 if pos <= 0 else 0.2)  # hold long softly
    out["actions"] = actions
    return out[["date", "actions"]]


def roundtrip_metrics_long_only(df: pd.DataFrame, actions: pd.DataFrame) -> dict:
    closes = df.drop_duplicates("date").set_index("date")["close"]
    act = actions.copy()
    act["actions"] = act["actions"].map(
        lambda x: float(np.asarray(x).reshape(-1)[0])
        if not isinstance(x, str)
        else float(x.strip("[]")),
    )
    act["date"] = pd.to_datetime(act["date"])
    act = act.set_index("date").join(closes, how="left")

    pos = 0.0
    entry = None
    pnls: list[float] = []
    thresh = 0.05
    for _, row in act.iterrows():
        a = float(row["actions"])
        px = float(row["close"])
        if a > thresh and pos <= 0:
            pos = 1.0
            entry = px
        elif a < -thresh and pos > 0 and entry is not None:
            pnls.append(px - entry)
            pos = 0.0
            entry = None

    if not pnls:
        return {"trades": 0, "win_rate": 0.0, "expectancy": 0.0, "total_pnl_price": 0.0}
    wins = [x for x in pnls if x > 0]
    return {
        "trades": len(pnls),
        "win_rate": len(wins) / len(pnls),
        "expectancy": float(np.mean(pnls)),
        "total_pnl_price": float(np.sum(pnls)),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading ticks: {TICK_CSV}")
    ohlcv = ticks_to_ohlcv(TICK_CSV, rule="1min")
    print(f"OHLCV bars: {len(ohlcv)}  ({ohlcv['date'].iloc[0]} -> {ohlcv['date'].iloc[-1]})")

    processed = build_finrl_frame(ohlcv)
    print(f"FinRL features: {list(INDICATORS)}  rows={len(processed)}")

    # Chronological 70/30 split by date string
    dates = sorted(processed["date"].unique())
    split_i = int(len(dates) * 0.7)
    train_start, train_end = str(dates[0]), str(dates[split_i])
    test_start, test_end = str(dates[split_i]), str(dates[-1] + pd.Timedelta(minutes=1))
    train = data_split(processed, train_start, train_end)
    trade = data_split(processed, test_start, test_end)
    print(f"TRAIN rows={len(train)}  TEST rows={len(trade)}")

    e_train = make_env(train)
    env_train, _ = e_train.get_sb_env()

    agent = DRLAgent(env=env_train)
    a2c_kwargs = dict(A2C_PARAMS)
    a2c_kwargs["ent_coef"] = 0.05
    a2c_kwargs["n_steps"] = 5
    model = agent.get_model("a2c", model_kwargs=a2c_kwargs, verbose=0)
    print("Training FinRL DRLAgent(A2C) ...")
    trained = DRLAgent.train_model(model, tb_log_name="a2c_eth", total_timesteps=25_000)
    trained.save(str(MODEL_PATH))
    print(f"Saved model -> {MODEL_PATH}")

    e_trade = make_env(trade)
    df_account, df_actions = DRLAgent.DRL_prediction(trained, e_trade, deterministic=True)
    df_actions.to_csv(ARTIFACT_DIR / "finrl_actions_env.csv", index=False)
    df_account.to_csv(ARTIFACT_DIR / "finrl_account_value.csv", index=False)

    # Continuous RL actions (diagnostic)
    e_trade2 = make_env(trade)
    test_env, test_obs = e_trade2.get_sb_env()
    rl_rows: list[dict] = []
    trade_dates = trade.drop_duplicates("date")["date"].tolist()
    for i in range(len(trade_dates) - 1):
        action, _ = trained.predict(test_obs, deterministic=True)
        a = float(np.asarray(action).reshape(-1)[0])
        rl_rows.append({"date": trade_dates[i], "actions": a})
        test_obs, _, dones, _ = test_env.step(action)
        if dones[0]:
            break
    df_rl = pd.DataFrame(rl_rows)
    df_rl.to_csv(ARTIFACT_DIR / "finrl_rl_actions.csv", index=False)

    # Prefer FinRL FeatureEngineer rule signals for Nautilus on short samples
    # (A2C/PPO often collapse to constant sell on ~hours of minute bars).
    df_rule = indicator_rule_signals(trade)
    start_val = float(df_account["account_value"].iloc[0])
    end_val = float(df_account["account_value"].iloc[-1])
    rl_flat = abs(end_val - start_val) < 1e-6 or df_rl["actions"].nunique() <= 1
    if rl_flat:
        print("RL policy degenerate on short sample — exporting FinRL indicator-rule signals")
        df_cont = df_rule
        signal_source = "FeatureEngineer/MACD+RSI_rule"
    else:
        df_cont = df_rl
        signal_source = "DRLAgent/A2C"

    df_cont.to_csv(SIGNALS_CSV, index=False)
    print(f"Saved signals -> {SIGNALS_CSV}  ({signal_source})")

    metrics = roundtrip_metrics_long_only(trade, df_cont)
    ret = end_val / start_val - 1.0

    summary = {
        "library": "FinRL",
        "signal_source": signal_source,
        "agent": "DRLAgent/A2C",
        "env": "StockTradingEnv",
        "indicators": INDICATORS,
        "train_bars": int(len(train)),
        "test_bars": int(len(trade)),
        "rl_account_return": ret,
        "rl_start_value": start_val,
        "rl_end_value": end_val,
        "action_abs_mean": float(df_cont["actions"].abs().mean()) if len(df_cont) else 0.0,
        **metrics,
    }
    (ARTIFACT_DIR / "finrl_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== FinRL ETHUSDT summary (held-out) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}" if abs(v) < 10 else f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print(
        "\nNote: results are on a ~5h minute-bar sample — FinRL is designed for "
        "multi-year daily data; treat this as an integration demo.",
    )


if __name__ == "__main__":
    main()

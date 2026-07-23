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
Gymnasium env smoke test + optional Stable-Baselines3 PPO training on ETHUSDT features.
"""

from __future__ import annotations

from pathlib import Path

from nautilus_trader.examples.strategies.rl.env import NautilusEthGymEnv
from nautilus_trader.examples.strategies.rl.env import collect_feature_rows
from nautilus_trader.examples.strategies.rl.env import save_feature_cache
from nautilus_trader.examples.strategies.rl.features import RLAction


ARTIFACT_DIR = Path("examples/backtest/rl_artifacts")
FEATURES = ARTIFACT_DIR / "features.npz"
SB3_MODEL = ARTIFACT_DIR / "ppo_ethusdt.zip"


def ensure_features() -> None:
    if FEATURES.exists():
        return
    print("Building feature cache via Nautilus BacktestEngine ...")
    rows = collect_feature_rows()
    save_feature_cache(FEATURES, rows)
    print(f"Saved {len(rows)} bars -> {FEATURES}")


def smoke_random_policy(env: NautilusEthGymEnv, steps: int = 200) -> float:
    import numpy as np

    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(steps):
        action = int(np.random.randint(0, env.action_space.n if env.action_space else 4))
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


def train_sb3(timesteps: int = 5_000) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        print("stable-baselines3 not installed.")
        print("Install with: pip install gymnasium stable-baselines3")
        raise SystemExit(1) from e

    env = NautilusEthGymEnv(feature_cache=FEATURES)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64)
    print(f"Training PPO for {timesteps} timesteps ...")
    model.learn(total_timesteps=timesteps)
    model.save(str(SB3_MODEL))
    print(f"Saved SB3 model -> {SB3_MODEL}")

    obs, _ = env.reset(seed=1)
    total = 0.0
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total += reward
        if terminated or truncated:
            break
    print(f"PPO eval episodic reward proxy: {total:+.6f}")
    print(f"Action enum example: FLAT={int(RLAction.FLAT)}")


if __name__ == "__main__":
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_features()

    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("gymnasium not installed — installing is required for SB3 path.")
        print("pip install gymnasium stable-baselines3")
        # Still allow duck-typed env smoke without spaces.n
        env = NautilusEthGymEnv(feature_cache=FEATURES)
        obs, _ = env.reset()
        print(f"Env reset OK obs_dim={len(obs)} (gymnasium missing; skip SB3)")
        raise SystemExit(0)

    env = NautilusEthGymEnv(feature_cache=FEATURES)
    rand_ret = smoke_random_policy(env)
    print(f"Random policy reward proxy ({200} steps): {rand_ret:+.6f}")
    train_sb3(timesteps=5_000)

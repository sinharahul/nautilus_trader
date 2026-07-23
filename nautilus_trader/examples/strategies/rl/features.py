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
Shared feature helpers for ETHUSDT RL research strategies.

*** RESEARCH ONLY — NOT FOR LIVE TRADING WITH REAL MONEY. ***
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class RLAction(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2
    FLAT = 3


N_ACTIONS = len(RLAction)

# Continuous observation layout used by offline MLP / gym:
# [efficiency, z_score, atr_norm, ema_diff_norm, position]
OBS_DIM = 5


def clip_z(z: float, limit: float = 5.0) -> float:
    return float(np.clip(z, -limit, limit))


def z_score_from_bands(close: float, middle: float, upper: float, lower: float) -> float:
    """
    Approximate z-score from Bollinger mid/upper (k-sigma bands).
    """
    width = upper - middle
    if width <= 1e-12:
        return 0.0
    return clip_z((close - middle) / width * 2.0)  # bands are typically k=2 → scale


def ema_diff_norm(fast: float, slow: float, price: float) -> float:
    if price <= 1e-12:
        return 0.0
    return float(np.clip((fast - slow) / price, -0.05, 0.05) / 0.05)


def atr_norm(atr: float, price: float) -> float:
    if price <= 1e-12:
        return 0.0
    return float(np.clip(atr / price, 0.0, 0.05) / 0.05)


def build_obs(
    *,
    efficiency: float,
    z_score: float,
    atr: float,
    price: float,
    fast_ema: float,
    slow_ema: float,
    position: int,
) -> np.ndarray:
    return np.asarray(
        [
            float(np.clip(efficiency, 0.0, 1.0)),
            clip_z(z_score),
            atr_norm(atr, price),
            ema_diff_norm(fast_ema, slow_ema, price),
            float(position),
        ],
        dtype=np.float64,
    )


def discretize_state(
    *,
    efficiency: float,
    z_score: float,
    position: int,
    efficiency_threshold: float = 0.35,
) -> int:
    """
    Compact tabular state id.

    Bits:
    - regime: 0 = mean-revert, 1 = trend  (x1)
    - z bucket: 0=low, 1=mid, 2=high     (x3)
    - position: 0=flat, 1=long, 2=short  (x3)

    Total states = 2 * 3 * 3 = 18.
    """
    regime = 1 if efficiency >= efficiency_threshold else 0
    if z_score <= -1.0:
        z_bucket = 0
    elif z_score >= 1.0:
        z_bucket = 2
    else:
        z_bucket = 1
    if position > 0:
        pos_bucket = 1
    elif position < 0:
        pos_bucket = 2
    else:
        pos_bucket = 0
    return regime * 9 + z_bucket * 3 + pos_bucket


N_TABULAR_STATES = 18

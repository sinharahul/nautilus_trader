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
RL research strategies for ETHUSDT demos.
"""

from nautilus_trader.examples.strategies.rl.env import NautilusEthEnv
from nautilus_trader.examples.strategies.rl.env import NautilusEthGymEnv
from nautilus_trader.examples.strategies.rl.env import collect_feature_rows
from nautilus_trader.examples.strategies.rl.offline_policy import NumpyMLP
from nautilus_trader.examples.strategies.rl.offline_policy import OfflinePolicyConfig
from nautilus_trader.examples.strategies.rl.offline_policy import OfflinePolicyStrategy
from nautilus_trader.examples.strategies.rl.offline_policy import train_behavior_cloning
from nautilus_trader.examples.strategies.rl.tabular_q import TabularQConfig
from nautilus_trader.examples.strategies.rl.tabular_q import TabularQStrategy

__all__ = [
    "NautilusEthEnv",
    "NautilusEthGymEnv",
    "NumpyMLP",
    "OfflinePolicyConfig",
    "OfflinePolicyStrategy",
    "TabularQConfig",
    "TabularQStrategy",
    "collect_feature_rows",
    "train_behavior_cloning",
]

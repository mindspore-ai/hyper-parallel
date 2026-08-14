# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Task reward functions consumed by agentic environments."""
import re
from typing import Callable, Optional, Sequence
from rl.registry import Registry
RewardInput = str | Sequence[str]
RewardOutput = float | list[float]
RewardFunction = Callable[[RewardInput, str], RewardOutput]
REWARDS = Registry[RewardFunction]("reward")
_ANSWER_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
_REWARD_WINDOW = 300
def register_reward(name: str) -> Callable[[RewardFunction], RewardFunction]:
    """Register a reward function under a stable task name."""
    return REWARDS.register(name)
def get_reward(name: str) -> RewardFunction:
    """Return a registered reward function by task name."""
    return REWARDS.get(name)
def extract_answer(solution: str) -> Optional[str]:
    """Extract the final strict numeric answer from the response tail."""
    matches = _ANSWER_PATTERN.findall(solution[-_REWARD_WINDOW:])
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "")
@register_reward("gsm8k")
def compute_rule_reward(solution: RewardInput, ground_truth: str) -> RewardOutput:
    """Return strict numeric exact-match reward for one or more responses."""
    if not isinstance(solution, str):
        return [
            float(compute_rule_reward(candidate, ground_truth))
            for candidate in solution
        ]
    predicted = extract_answer(solution)
    normalized_ground_truth = ground_truth.replace(",", "").replace("$", "")
    return float(predicted is not None and predicted == normalized_ground_truth)

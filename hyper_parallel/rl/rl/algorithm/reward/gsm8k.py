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
"""Strict GSM8K answer extraction and rule rewards."""

import re
from typing import Optional, Sequence

_ANSWER_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
_REWARD_WINDOW = 300


def extract_answer(solution: str) -> Optional[str]:
    """Extract the final strict numeric answer from the response tail."""
    matches = _ANSWER_PATTERN.findall(solution[-_REWARD_WINDOW:])
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "")


def compute_rule_reward(solution: str, ground_truth: str) -> float:
    """Return one for a strict numeric exact match, otherwise zero."""
    predicted = extract_answer(solution)
    normalized_ground_truth = ground_truth.replace(",", "").replace("$", "")
    return float(predicted is not None and predicted == normalized_ground_truth)


def compute_rule_rewards(solutions: Sequence[str], ground_truth: str) -> list[float]:
    """Score responses belonging to the same prompt in input order."""
    return [compute_rule_reward(solution, ground_truth) for solution in solutions]

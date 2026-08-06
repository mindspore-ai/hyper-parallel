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
"""Unit tests for the strict numeric rule reward."""

import pytest

from rl.algorithm.reward import compute_rule_reward, extract_answer


@pytest.mark.parametrize(
    ("solution", "ground_truth", "expected"),
    [
        ("Reasoning\n#### 42", "42", 1.0),
        ("Reasoning\n#### 41", "42", 0.0),
        ("No strict final answer", "42", 0.0),
        ("Reasoning\n#### -3.5", "-3.5", 1.0),
        ("Reasoning\n#### 1,234", "$1,234", 1.0),
        ("First #### 1\nCorrection #### 2", "2", 1.0),
        ("Wrong spacing\n####42", "42", 0.0),
    ],
)
def test_compute_rule_reward_strict_cases(solution: str, ground_truth: str, expected: float) -> None:
    """Verify exact-match, normalization, strict formatting, and final-match behavior."""
    actual = compute_rule_reward(solution, ground_truth)
    assert actual == expected, f"Unexpected reward: expected={expected}, got={actual}"


def test_extract_answer_only_inspects_last_300_characters() -> None:
    """Verify an otherwise valid answer outside the reward window is ignored."""
    solution = "#### 42" + ("x" * 301)
    actual = extract_answer(solution)
    assert actual is None, f"Answer outside final 300 characters should be ignored, got={actual}"


def test_extract_answer_accepts_answer_at_window_boundary() -> None:
    """Verify strict matching still works inside the final reward window."""
    solution = ("x" * 400) + "#### 12.50"
    actual = extract_answer(solution)
    expected = "12.50"
    assert actual == expected, f"Unexpected extracted answer: expected={expected}, got={actual}"

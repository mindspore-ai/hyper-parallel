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
"""Validate memory optimization for adjacent PyTorch checkpoint SAVE regions."""

from tests.common.mark_utils import arg_mark
from tests.torch.activation_checkpoint.checkpoint_exclude_matmul import (
    _HIDDEN_SIZE,
    _TOKEN_NUM,
    _run_mode_in_subprocess,
)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_adjacent_save_regions_elide_intermediate_output() -> None:
    """A four-SAVE chain should elide two non-activation intermediate outputs."""
    optimized = _run_mode_in_subprocess("save_chain")
    legacy = _run_mode_in_subprocess("legacy_save_chain")

    assert optimized["loss"] == legacy["loss"]
    assert optimized["input_grad"] == legacy["input_grad"]
    assert optimized["first_calls"] == 1
    assert optimized["second_calls"] == 1
    assert optimized["third_calls"] == 1
    assert optimized["fourth_calls"] == 1
    assert legacy["first_calls"] == 1
    assert legacy["second_calls"] == 1
    assert legacy["third_calls"] == 1
    assert legacy["fourth_calls"] == 1

    # The first output is already a ReLU backward activation. The second and
    # third outputs account for two extra allocations in the legacy mode.
    expected_gap = 2 * _TOKEN_NUM * _HIDDEN_SIZE * 2
    tolerance = expected_gap // 4
    forward_gap = legacy["forward_bytes"] - optimized["forward_bytes"]
    assert abs(forward_gap - expected_gap) <= tolerance

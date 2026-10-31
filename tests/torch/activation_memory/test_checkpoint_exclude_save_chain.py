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
from tests.torch.activation_memory.checkpoint_exclude_matmul import (
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
def test_adjacent_save_regions_execute_once_and_replay_cached_outputs() -> None:
    """A four-SAVE chain should execute each excluded region only once."""
    result = _run_mode_in_subprocess("save_chain")

    assert result["first_calls"] == 1
    assert result["second_calls"] == 1
    assert result["third_calls"] == 1
    assert result["fourth_calls"] == 1
    assert result["forward_bytes"] > _TOKEN_NUM * _HIDDEN_SIZE * 2


def _assert_swap_save_chain_matches_baseline(baseline: dict, actual: dict) -> None:
    """Validate numerical parity, exclusion behavior, and swap storage cleanup."""
    loss_tolerance = max(1e-4, abs(baseline["loss"]) * 1e-6)
    grad_tolerance = max(1e-5, abs(baseline["input_grad"]) * 1e-5)
    assert abs(actual["loss"] - baseline["loss"]) <= loss_tolerance
    assert abs(actual["input_grad"] - baseline["input_grad"]) <= grad_tolerance
    assert baseline["calls"] == [1, 1, 1, 1]
    assert actual["calls"] == [1, 1, 1, 1]
    assert actual["registered_storage_count"] > 0
    assert actual["offloaded_tensor_count"] > 0
    assert actual["remaining_storage_count"] == 0


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_checkpoint_exclude_save_chain_swap() -> None:
    """checkpoint_exclude_wrapper(swap=True) should offload and restore SAVE activations."""
    baseline = _run_mode_in_subprocess("swap_save_chain_baseline")
    swapped = _run_mode_in_subprocess("swap_save_chain")

    _assert_swap_save_chain_matches_baseline(baseline, swapped)
    assert swapped["group_managed_tensor_count"] == 0
    assert swapped["pool_allocated_bytes"] == 0


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_checkpoint_exclude_save_chain_group_swap_cpu_pool() -> None:
    """Group swap should fuse SAVE tensors and allocate their host buffers from the explicit pool."""
    baseline = _run_mode_in_subprocess("swap_save_chain_baseline")
    pooled = _run_mode_in_subprocess("group_swap_pool_save_chain")

    _assert_swap_save_chain_matches_baseline(baseline, pooled)
    assert pooled["group_managed_tensor_count"] > 0
    assert 0 < pooled["pool_allocated_bytes"] <= pooled["pool_capacity_bytes"]

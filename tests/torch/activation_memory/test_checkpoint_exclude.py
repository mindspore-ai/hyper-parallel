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
"""Validate replay behavior for adjacent PyTorch checkpoint SAVE regions."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

CHECKPOINT_EXCLUDE = str(Path(__file__).with_name("checkpoint_exclude.py"))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_checkpoint_exclude_wrapper():
    """
    Feature: checkpoint exclude
    Description:
        1. test_adjacent_save_regions_execute_once_and_replay_cached_outputs
        2. test_checkpoint_exclude_save_chain_swap
        3. test_checkpoint_exclude_save_chain_group_swap_cpu_pool
        4. test_rmsnorm_matmul_checkpoint_exclude_memory
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(CHECKPOINT_EXCLUDE, "test_adjacent_save_regions_execute_once_and_replay_cached_outputs", 12414, 1),
        TorchCase(CHECKPOINT_EXCLUDE, "test_checkpoint_exclude_save_chain_swap", 12415, 1),
        TorchCase(CHECKPOINT_EXCLUDE, "test_checkpoint_exclude_save_chain_group_swap_cpu_pool", 12416, 1),
        TorchCase(CHECKPOINT_EXCLUDE, "test_rmsnorm_matmul_checkpoint_exclude_memory", 12417, 1),
    ])

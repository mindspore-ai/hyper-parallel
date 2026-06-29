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
"""swap optimizer test"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

SWAP_OPTIMIZER = "swap_optimizer.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_swap_optimizer_1():
    """
    Feature: parallel run case in swap_optimizer
    Description:
        1. test_native_adam_fully_shard_swap_optimizer_state_align_worker
        2. test_mindformers_adamw_fully_shard_swap_optimizer_state_align_worker
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(
            SWAP_OPTIMIZER,
            "test_native_adam_fully_shard_swap_optimizer_state_align_worker",
            master_port=11737,
            worker_num=4,
            local_worker_num=4,
        ),
        MindSporeCase(
            SWAP_OPTIMIZER,
            "test_mindformers_adamw_fully_shard_swap_optimizer_state_align_worker",
            master_port=11746,
            worker_num=4,
            local_worker_num=4,
        ),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_optimizer_2():
    """
    Feature: parallel run case in swap_optimizer
    Description:
        1. test_native_adam_swap_optimizer_checkpoint_cpu_mirror_roundtrip
        2. test_native_adam_swap_optimizer_checkpoint_fresh_load_builds_slots
        3. test_mindformers_adamw_non_fused_swap_optimizer_state_align
        4. test_mindformers_adamw_fused_swap_optimizer_state_align
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SWAP_OPTIMIZER, "test_native_adam_swap_optimizer_checkpoint_cpu_mirror_roundtrip", 11742, 1),
        MindSporeCase(SWAP_OPTIMIZER, "test_native_adam_swap_optimizer_checkpoint_fresh_load_builds_slots", 11743, 1),
        MindSporeCase(SWAP_OPTIMIZER, "test_mindformers_adamw_non_fused_swap_optimizer_state_align", 11744, 1),
        MindSporeCase(SWAP_OPTIMIZER, "test_mindformers_adamw_fused_swap_optimizer_state_align", 11745, 1),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_optimizer_3():
    """
    Feature: parallel run case in swap_optimizer
    Description:
        1. test_mindformers_adamw_packed_swap_optimizer_checkpoint_roundtrip
        2. test_native_adam_swap_optimizer_state_align
        3. test_native_adam_nesterov_swap_optimizer_state_align
        4. test_native_adam_amsgrad_swap_optimizer_state_align
        5. test_native_adam_weight_decay_swap_optimizer_state_align
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(
            SWAP_OPTIMIZER,
            "test_mindformers_adamw_packed_swap_optimizer_checkpoint_roundtrip",
            11747,
            1,
        ),
        MindSporeCase(SWAP_OPTIMIZER, "test_native_adam_swap_optimizer_state_align", 11738, 1),
        MindSporeCase(SWAP_OPTIMIZER, "test_native_adam_nesterov_swap_optimizer_state_align", 11739, 1),
        MindSporeCase(SWAP_OPTIMIZER, "test_native_adam_amsgrad_swap_optimizer_state_align", 11740, 1,),
        MindSporeCase(SWAP_OPTIMIZER, "test_native_adam_weight_decay_swap_optimizer_state_align", 11741, 1),
    ])

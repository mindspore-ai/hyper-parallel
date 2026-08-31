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
"""test swap optimizer"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

SWAP_OPTIMIZER = "swap_optimizer.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_swap_optimizer_1():
    """
    Feature: parallel run case in swap_optimizer
    Description:
        1. test_fully_shard_adamw_mixed_precision_swap_optimizer_parameter_align
        2. test_fully_shard_optimizer_swap_adamw_4card_parameter_align
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SWAP_OPTIMIZER, "test_fully_shard_adamw_mixed_precision_swap_optimizer_parameter_align", 12504, 4),
        TorchCase(SWAP_OPTIMIZER, "test_fully_shard_optimizer_swap_adamw_4card_parameter_align", 12511, 4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_optimizer_2():
    """
    Feature: parallel run case in swap_optimizer
    Description:
        1. test_new_adamw_amsgrad_swap_optimizer_parameter_align
        2. test_torch_adam_swap_optimizer_multi_param_group_align
        3. test_torch_adam_swap_optimizer_checkpoint_host_state
        4. test_torch_adamw_eager_state_swap_optimizer_parameter_align
        5. test_torch_fused_adamw_swap_optimizer_parameter_align
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SWAP_OPTIMIZER, "test_new_adamw_amsgrad_swap_optimizer_parameter_align", 12504, 1),
        TorchCase(SWAP_OPTIMIZER, "test_torch_adam_swap_optimizer_multi_param_group_align", 12505, 1),
        TorchCase(SWAP_OPTIMIZER, "test_torch_adam_swap_optimizer_checkpoint_host_state", 12506, 1),
        TorchCase(SWAP_OPTIMIZER, "test_torch_adamw_eager_state_swap_optimizer_parameter_align", 12509, 1),
        TorchCase(SWAP_OPTIMIZER, "test_torch_fused_adamw_swap_optimizer_parameter_align", 12510, 1),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_optimizer_3():
    """
    Feature: Eight-card fully_shard packed/per swap optimizer.
    Description: 
        1. test_torch_adam_swap_optimizer_parameter_align
        2. test_torch_adamw_swap_optimizer_parameter_align
        3. test_torch_adam_amsgrad_swap_optimizer_parameter_align
        4. test_new_adamw_swap_optimizer_parameter_align
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SWAP_OPTIMIZER, "test_torch_adam_swap_optimizer_parameter_align", 12505, 1),
        TorchCase(SWAP_OPTIMIZER, "test_torch_adamw_swap_optimizer_parameter_align", 12506, 1),
        TorchCase(SWAP_OPTIMIZER, "test_torch_adam_amsgrad_swap_optimizer_parameter_align", 12507, 1),
        TorchCase(SWAP_OPTIMIZER, "test_new_adamw_swap_optimizer_parameter_align", 12508, 1),
    ])

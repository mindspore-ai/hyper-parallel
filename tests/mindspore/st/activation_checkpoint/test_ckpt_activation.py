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
"""selective activation checkpoint test"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

BASE_SHARD = "ckpt_activation.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sac_group():
    """
    Feature: parallel run case in ckpt_activation
    Description:
        1. test_ac_memory_comparison
        2. test_group_swap_correctness_and_memory
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_SHARD, "test_basic_ac_memory_comparison", 11637, 1),
        MindSporeCase(BASE_SHARD, "test_func_ac_memory_comparison", 11638, 1),
        MindSporeCase(BASE_SHARD, "test_group_swap_correctness_and_memory", 11639, 1),
        MindSporeCase(BASE_SHARD, "test_swap_manager_manual_group_api", 11640, 1),
        MindSporeCase(BASE_SHARD, "test_inplace_modification", 11641, 1),
        MindSporeCase(BASE_SHARD, "test_wrapper_overlap_detection_cases", 11642, 1),
        MindSporeCase(BASE_SHARD, "test_wrapper_non_overlapping_allowed_cases", 11643, 1),
        MindSporeCase(
            "checkpoint_exclude_matmul.py", "test_rmsnorm_matmul_checkpoint_exclude_memory", 11644, 1
        ),
    ])

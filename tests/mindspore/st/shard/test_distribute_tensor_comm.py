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
"""Pytest launcher for distribute_tensor src_data_rank ST coverage."""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

DISTRIBUTE_TENSOR_COMM = "distribute_tensor_comm.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_distribute_tensor_comm_group1():
    """
    Feature: distribute_tensor scatter/broadcast with src_data_rank.
    Description:
        1. test_rank0_only_shard0
        2. test_src_only_nonzero_src
        3. test_rank0_only_replicate
        4. test_rank0_only_2d_shard_replicate
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(DISTRIBUTE_TENSOR_COMM, "test_rank0_only_shard0", 18410, 2, 2),
        MindSporeCase(DISTRIBUTE_TENSOR_COMM, "test_src_only_nonzero_src", 18411, 2, 2),
        MindSporeCase(DISTRIBUTE_TENSOR_COMM, "test_rank0_only_replicate", 18412, 2, 2),
        MindSporeCase(DISTRIBUTE_TENSOR_COMM, "test_rank0_only_2d_shard_replicate", 18413, 2, 2),
    ])

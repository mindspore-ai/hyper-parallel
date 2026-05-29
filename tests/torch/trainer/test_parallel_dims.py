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
"""Launch trainer ``ParallelDims`` and rank-aware logging ST cases."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_PARALLEL_DIMS = "_test_parallel_dims.py"


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_parallel_dims_group1():
    """
    Feature: trainer ParallelDims pure-FSDP mesh and rank-aware logging.
    Description:
        1. test_parallel_dims_pure_fsdp_mesh_and_rank_logging
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_PARALLEL_DIMS, "test_parallel_dims_pure_fsdp_mesh_and_rank_logging", 11731, 4),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_parallel_dims_group2():
    """
    Feature: trainer ParallelDims mixed parallel compositions.
    Description:
        1. test_parallel_dims_tp_plus_fsdp_mesh
        2. test_parallel_dims_hsdp_dp_combines_replicate_and_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_PARALLEL_DIMS, "test_parallel_dims_tp_plus_fsdp_mesh", 11733, 4),
        TorchCase(_TEST_PARALLEL_DIMS, "test_parallel_dims_hsdp_dp_combines_replicate_and_shard", 11734, 4),
    ])

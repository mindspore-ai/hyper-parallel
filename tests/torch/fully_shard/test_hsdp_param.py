# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test TorchHSDPParamV2"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_HSDP_PARAM = "_test_hsdp_param.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_hsdp_param_group1():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_hsdp_param_v2_fsdp_1d_mesh
        2.test_hsdp_param_v2_sharded_state_transitions
        3.test_hsdp_param_v2_custom_shard_placement
        4.test_hsdp_param_v2_mixed_precision
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_fsdp_1d_mesh", 12343, 2),
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_sharded_state_transitions", 12345, 2),
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_custom_shard_placement", 12346, 2),
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_mixed_precision", 12347, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_hsdp_param_group2():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_hsdp_param_v2_all_gather_comm
        2.test_hsdp_param_v2_prefetch_unshard
        3.test_hsdp_param_v2_unshard_shard_cycle
        4.test_hsdp_param_v2_reduce_scatter_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_all_gather_comm", 12348, 2),
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_prefetch_unshard", 12349, 2),
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_unshard_shard_cycle", 12350, 2),
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_reduce_scatter_grad", 12351, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_param_group3():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_hsdp_param_v2_hsdp_2d_mesh
        2.test_hsdp_param_v2_all_reduce_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_hsdp_2d_mesh", 12344, 4),
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_all_reduce_grad", 12352, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_hsdp_param_group4():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_hsdp_param_v2_accumulate_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_HSDP_PARAM, "test_hsdp_param_v2_accumulate_grad", 12355, 2),
    ])

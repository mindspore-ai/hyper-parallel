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
"""Launch _test_fully_shard_precision.py and _test_fully_shard_precision_list.py cases."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_FULLY_SHARD_PRECISION = "_test_fully_shard_precision.py"
_TEST_FULLY_SHARD_PRECISION_LIST = "_test_fully_shard_precision_list.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fully_shard_precision_group1():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_zero3_fully_shard
        2.test_zero3_fully_shard_with_mp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_PRECISION, "test_zero3_fully_shard", 12343, 4),
        TorchCase(_TEST_FULLY_SHARD_PRECISION, "test_zero3_fully_shard_with_mp", 12401, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fully_shard_precision_group2():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_zero3_fully_shard_with_offload
        2.test_zero3_partial_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_PRECISION, "test_zero3_fully_shard_with_offload", 12345, 4),
        TorchCase(_TEST_FULLY_SHARD_PRECISION, "test_zero3_partial_shard", 12344, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fully_shard_precision_list_unit():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_list_unit_precision_zero3
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_PRECISION_LIST, "test_list_unit_precision_zero3", 12350, 4),
    ])


def test_zero3_fully_shard_comm_fusion():
    """
    Feature: Test_zero3_fully_shard_comm_fusion.
    Description: Test_zero3_fully_shard with comm fusion.
    Expectation: case run successfully.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_PRECISION, "test_zero3_fully_shard_comm_fusion", 12315, 4),
        TorchCase(_TEST_FULLY_SHARD_PRECISION, "test_zero3_partial_shard_comm_fusion", 12314, 4)
    ])

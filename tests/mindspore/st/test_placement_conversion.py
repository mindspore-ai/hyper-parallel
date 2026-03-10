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
"""test placement and tensor_map conversion"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

PLACEMENT_CONVERSION = "placement_conversion.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_placement_conversion_group1():
    """
    Feature: parallel run case in placement_conversion
    Description:
        1. test_basic_conversion
        2. test_shard_combinations
        3. test_full_replication
        4. test_partial_shard_mixed
        5. test_partial_reduce_ops
        6. test_conversion_errors
        7. test_from_device_mesh
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(PLACEMENT_CONVERSION, "test_basic_conversion", 11664),
        MindSporeCase(PLACEMENT_CONVERSION, "test_shard_combinations", 11665),
        MindSporeCase(PLACEMENT_CONVERSION, "test_full_replication", 11666),
        MindSporeCase(PLACEMENT_CONVERSION, "test_partial_shard_mixed", 11667),
        MindSporeCase(PLACEMENT_CONVERSION, "test_partial_reduce_ops", 11668),
        MindSporeCase(PLACEMENT_CONVERSION, "test_conversion_errors", 11669),
        MindSporeCase(PLACEMENT_CONVERSION, "test_from_device_mesh", 11670)
    ])

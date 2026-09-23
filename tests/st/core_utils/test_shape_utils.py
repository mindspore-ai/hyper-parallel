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
"""Launch ``core/utils/shape_utils`` real-mesh ST cases."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_SHAPE_UTILS = "_test_shape_utils.py"


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_shape_utils_group1():
    """
    Feature: real-mesh shape utility contracts.
    Description:
        1. test_shape_utils_alias_shard_uneven_split_matches_chunk
        2. test_shape_utils_placement_objects_uneven_match_alias_string
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_SHAPE_UTILS, "test_shape_utils_alias_shard_uneven_split_matches_chunk", 12822, 4),
        TorchCase(_TEST_SHAPE_UTILS, "test_shape_utils_placement_objects_uneven_match_alias_string", 12823, 4),
    ])

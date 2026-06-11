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
"""Launch _test_fully_shard_with_defer_init.py cases."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_FULLY_SHARD_WITH_DEFER_INIT = "_test_fully_shard_with_defer_init.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_fully_shard_meta_init():
    """
    Feature: Test fully_shard with meta device initialization
    Description:
        1.test_fully_shard_meta_init
        2.test_fully_shard_init_empty_weights_with_prefetch
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_WITH_DEFER_INIT, "test_fully_shard_meta_init", 12360, 4),
        TorchCase(_TEST_FULLY_SHARD_WITH_DEFER_INIT, "test_fully_shard_init_empty_weights_with_prefetch", 12361, 4),
    ])

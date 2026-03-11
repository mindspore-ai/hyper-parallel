# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Launch _test_dcp_with_fully_shard_optimizer.py (same pattern as parallel_run + TorchCase under tests/torch/fully_shard)."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_DCP_WITH_FULLY_SHARD_OPTIMIZER = "_test_dcp_with_fully_shard_optimizer.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_with_fully_shard_optimizer_single():
    """
    Feature: fully_shard + distributed checkpoint + flatten_state_dict + bytes (single process, num_proc=1).
    Description:
        1. test_dcp_with_fully_shard_optimizer
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_DCP_WITH_FULLY_SHARD_OPTIMIZER, "test_dcp_with_fully_shard_optimizer", 12501, 1),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_with_fully_shard_optimizer_multi():
    """
    Feature: fully_shard + DCP + flatten_state_dict + bytes (multi-GPU).

    Same style as test_fully_shard_auto_grad.
    Description:
        1. test_dcp_with_fully_shard_optimizer
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_DCP_WITH_FULLY_SHARD_OPTIMIZER, "test_dcp_with_fully_shard_optimizer", 12502, 4),
    ])

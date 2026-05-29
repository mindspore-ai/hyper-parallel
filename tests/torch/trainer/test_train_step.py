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
"""Launch trainer ``train_step`` ST cases."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_TRAIN_STEP = "_test_train_step.py"


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level0",
    card_mark="allcards", essential_mark="essential",
)
def test_train_step_group1():
    """
    Feature: trainer ``train_step`` token-weighted path.
    Description:
        1. test_train_step_token_weighted_end_to_end_4card
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_TRAIN_STEP, "test_train_step_token_weighted_end_to_end_4card", 11811, 4),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_train_step_group2():
    """
    Feature: trainer ``train_step`` rank-average path.
    Description:
        1. test_train_step_rank_average_end_to_end_4card
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_TRAIN_STEP, "test_train_step_rank_average_end_to_end_4card", 11812, 4),
    ])

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
"""ST entry for ``.claude/agents/init.md`` meta -> to_empty -> reset_parameters (MindSpore)."""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_INIT_MD = "_test_empty_weights_and_reset_parameter.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_empty_weights_and_reset_parameter():
    """
    Feature: parallel run case for init.md to_empty + reset_parameters flow (MindSpore)
    Description:
        1. test_empty_weights_and_reset_parameter
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(_TEST_INIT_MD, "test_empty_weights_and_reset_parameter", 12353, 2, 2),
    ])

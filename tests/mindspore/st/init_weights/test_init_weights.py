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
"""Launcher for init_weights distributed ST (MindSpore).

Keep this module free of ``mindspore`` / ``hyper_parallel`` imports so pytest
collection of the ST tree does not pay framework startup in the parent process.
"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

# Worker module next to this launcher (cwd typically tests/mindspore/st).
_TEST_INIT_WEIGHTS = "_test_init_weights.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_init_weights():
    """
    Feature: parallel run case in init_weights (MindSpore)
    Description:
        1. test_init_weights_consistency
        2. test_init_weights_with_randn_like
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(_TEST_INIT_WEIGHTS, "test_init_weights_with_randn_like", 12351, 2, 2),
    ])

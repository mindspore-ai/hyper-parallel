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
"""parallel_reshape test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase
from tests.mindspore.st.utils import skip_if_ms_version_ge

PARALLEL_RESHAPE = "parallel_reshape.py"


@skip_if_ms_version_ge("2.9.0")
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_reshape_group1():
    """
    Feature: parallel run case in parallel_reshape
    Description:
        1. test_parallel_reshape_0
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(PARALLEL_RESHAPE, "test_parallel_reshape_0", 10676, 8, 8)
    ])

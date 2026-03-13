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
"""parallel_base_custom_shard test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase
from tests.mindspore.st.utils import skip_if_ms_version_ge

BASE_CUSTOM_SHARD = "base_custom_shard.py"


@skip_if_ms_version_ge("2.9.0")
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_base_custom_shard_group1():
    """
    Feature: parallel run case in base_custom_shard
    Description:
        1. test_base_custom_shard
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_CUSTOM_SHARD, "test_base_custom_shard", 18301, 8, 8)
    ])

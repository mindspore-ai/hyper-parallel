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
"""test base dtensor init"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

BASE_DTENSOR_INIT = "base_dtensor_init.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_base_dtensor_init_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_ones
        2.test_empty
        3.test_full
        4.test_zeros
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(BASE_DTENSOR_INIT, "test_ones", 11335, 2),
        TorchCase(BASE_DTENSOR_INIT, "test_empty", 11336, 2),
        TorchCase(BASE_DTENSOR_INIT, "test_full", 11337, 2),
        TorchCase(BASE_DTENSOR_INIT, "test_zeros", 11338, 2),
    ])

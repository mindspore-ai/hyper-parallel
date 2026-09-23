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
"""test fully_shard module api"""
import os
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_FULLY_SHARD_MODULE = os.path.join(os.path.dirname(__file__), "_test_fully_shard_module.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_fully_shard_module_group1():
    """
    Feature: parallel run cases in fully_shard module
    Description:
        1.test_fully_shard_module_01 — HSDPModule interface methods
        2.test_fully_shard_module_02 — set_reshard_after_backward/forward recurse on nested model
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_MODULE, "test_fully_shard_module_01", 12343, 4),
        TorchCase(_TEST_FULLY_SHARD_MODULE, "test_fully_shard_module_02", 12344, 4),
    ])

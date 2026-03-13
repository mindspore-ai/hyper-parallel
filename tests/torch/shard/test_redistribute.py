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
"""test redistribute"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

REDISTRIBUTE = "redistribute.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_redistribute_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_shard_to_replicate
        2.test_replicate_to_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(REDISTRIBUTE, "test_shard_to_replicate", 11333, 2),
        TorchCase(REDISTRIBUTE, "test_replicate_to_shard", 11334, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_redistribute_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_different_mesh
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(REDISTRIBUTE, "test_different_mesh", 11335, 2),
    ])

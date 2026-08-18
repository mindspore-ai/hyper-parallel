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
"""test checkpoint DCP resharding API"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run

DCP_RESHARDING_API = "dcp_resharding_api.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_resharding_api_group1():
    """
    Feature: parallel run smoke case for checkpoint DCP safe_open reads under resharding.
    Description:
        1.test_dcp_safe_open_basic_resharding_load
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_RESHARDING_API, "test_dcp_safe_open_basic_resharding_load", 12256, 4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_resharding_api_group1_gloo():
    """
    Feature: parallel run smoke case for checkpoint DCP safe_open reads under resharding.
    Description:
        1.test_dcp_safe_open_basic_resharding_load
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_RESHARDING_API, "test_dcp_safe_open_basic_resharding_load", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="essential")
def test_dcp_resharding_api_group2():
    """
    Feature: parallel run case for checkpoint DCP safe_open reads across multiple resharding layouts.
    Description:
        1.test_dcp_safe_open_with_real_resharding_load
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_RESHARDING_API, "test_dcp_safe_open_with_real_resharding_load", 12257, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="essential")
def test_dcp_resharding_api_group3():
    """
    Feature: parallel run case for checkpoint DCP safe_open reads from fully_shard shards under resharding.
    Description:
        1.test_dcp_safe_open_with_fully_shard_tp_dp_resharding_load
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_RESHARDING_API, "test_dcp_safe_open_with_fully_shard_tp_dp_resharding_load", 12258, 4),
    ])

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
"""test checkpoint minimal plan-cache API"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_MIN_PLAN_CACHE_API = "dcp_plan_cache_minimal_api.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_dcp_plan_cache_minimal_api_group1():
    """
    Feature: parallel run case in checkpoint minimal plan-cache API.
    Description:
        1.test_dcp_minimal_plan_cache_hit
        2.test_dcp_minimal_plan_cache_model_optimizer_isolation
        3.test_dcp_minimal_plan_cache_hit_async
        4.test_dcp_minimal_plan_cache_model_optimizer_isolation_async
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_MIN_PLAN_CACHE_API, "test_dcp_minimal_plan_cache_hit", 12257, 2),
        TorchCase(DCP_MIN_PLAN_CACHE_API, "test_dcp_minimal_plan_cache_model_optimizer_isolation", 12258, 2),
        #TorchCase(DCP_MIN_PLAN_CACHE_API, "test_dcp_minimal_plan_cache_hit_async", 12259, 2),
        #TorchCase(DCP_MIN_PLAN_CACHE_API, "test_dcp_minimal_plan_cache_model_optimizer_isolation_async", 12260, 2),
    ])

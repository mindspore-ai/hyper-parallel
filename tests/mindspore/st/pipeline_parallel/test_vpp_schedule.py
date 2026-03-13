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
"""vpp test"""

from tests.common.parallel_case import parallel_run, MindSporeCase


def test_simple_mlp():
    """
    Feature: schedule 1f1b + hsdp + shard + shared_param.
    Description: Test pp with shared parameter.
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase("vpp_schedule.py", "test", 12346, 8, 8, 3)
    ])

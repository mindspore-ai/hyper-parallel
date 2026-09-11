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
"""Launch MindSpore reentrant-checkpoint validation on one Ascend device."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase


_CASE_FILE = "reentrant_checkpoint.py"
_CASES = (
    "test_reentrant_exclude_result_correctness",
    "test_reentrant_checkpoint_device_peak_memory",
    "test_reentrant_checkpoint_host_dispatch_performance",
    "test_reentrant_checkpoint_device_memory_no_leak",
    "test_reentrant_checkpoint_host_memory_no_leak",
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_reentrant_checkpoint_validation() -> None:
    """Run every validation in a clean process to isolate memory statistics.

    Feature: Reentrant checkpoint exclusion.
    Description: Run correctness, memory, dispatch, and leak cases independently.
    Expectation: Every worker exits successfully and satisfies its assertions.
    """
    for index, case_name in enumerate(_CASES):
        parallel_run([MindSporeCase(_CASE_FILE, case_name, 11660 + index, 1)])

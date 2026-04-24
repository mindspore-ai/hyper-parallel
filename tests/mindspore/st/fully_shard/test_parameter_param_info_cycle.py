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
"""ST entry for the MindSpore Parameter.param_info cycle patch case."""

from tests.common.mark_utils import arg_mark

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor
import hyper_parallel  # pylint: disable=unused-import


NUM_ELEMENTS = 256 * 1024 * 1024
FLOAT32_BYTES = 4
MEMORY_ACCOUNTING_OVERHEAD_BYTES = 4 * 1024


def _collect_runtime_memory():
    """Return current runtime memory after flushing MindSpore's idle cache."""
    ms.runtime.empty_cache()
    return ms.runtime.memory_allocated()


def _run_parameter_lifecycle():
    """Allocate a large Parameter, trigger ParamInfo init, then measure release timing."""
    baseline = _collect_runtime_memory()
    data = np.random.randn(NUM_ELEMENTS).astype(np.float32)
    param = Parameter(Tensor(data), name="big_param_patched")

    # Trigger MindSpore Parameter initialization paths that bind ParamInfo.
    param += 1
    after_create = _collect_runtime_memory()

    del param
    after_del = _collect_runtime_memory()

    return {
        "baseline": baseline,
        "after_create": after_create,
        "after_del": after_del,
    }


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_parameter_param_info_cycle_patch_releases_memory_after_del():
    """
    Feature: MindSpore Parameter param_info cycle patch.
    Description: Import hyper_parallel so the package-level patch is installed, then verify
                 deleting a large Parameter releases its device memory without explicit gc.collect().
    Expectation: The extra memory allocated by the Parameter should be released right after del.
    """
    assert getattr(Parameter, "_hyper_parallel_param_info_cycle_patched", False), (
        "Importing hyper_parallel should install the Parameter.param_info cycle patch"
    )

    stats = _run_parameter_lifecycle()
    allocated_bytes = NUM_ELEMENTS * FLOAT32_BYTES
    retained_bytes = stats["after_del"] - stats["baseline"]

    print(f"allocated_bytes={allocated_bytes}")
    print(f"stats={stats}")

    assert allocated_bytes <= stats["after_create"] - stats["baseline"] <= (
        allocated_bytes + MEMORY_ACCOUNTING_OVERHEAD_BYTES
    ), (
        "The large Parameter allocation should match MindSpore runtime memory stats"
    )
    assert retained_bytes == 0, (
        "Patched Parameter should release device memory immediately after del"
    )

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
"""Launch FSDP CPU offload and mixed-precision lifecycle guards."""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run


_TEST_FSDP_OFFLOAD = "_test_fsdp_offload.py"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_fsdp_offload_guard_group():
    """Run the CPU offload and parameter conversion lifecycle guards."""
    parallel_run([
        TorchCase(_TEST_FSDP_OFFLOAD, "test_cpu_offload_mixed_precision_training", 12362, 4),
        TorchCase(_TEST_FSDP_OFFLOAD, "test_apply_and_reset_mixed_precision_storage", 12363, 4),
    ])

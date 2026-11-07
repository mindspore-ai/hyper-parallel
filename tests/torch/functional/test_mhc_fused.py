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
"""Launch the fused-only mHC contract tests on one Ascend NPU."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="unessential")
def test_mhc_requires_fused_operators() -> None:
    """An unavailable custom-op package is an error, not a fallback."""
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_mhc_fused.py")),
        case_name="test_missing_fused_operators",
        num_proc=1,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="unessential")
def test_mhc_post_fused_forward_backward() -> None:
    """Compare the real post kernel and its gradients to the mathematical reference."""
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_mhc_fused.py")),
        case_name="test_post_forward_backward",
        num_proc=1,
    )

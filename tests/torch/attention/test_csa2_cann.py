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
"""Framework-free launcher for CSA2 CANN numerical parity."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_csa2_cann_parity() -> None:
    """Compare CANN attention and KL with the existing numerical reference."""
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_csa2_cann.py")),
        case_name="test_csa2_cann_forward_backward_parity",
        num_proc=1,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_csa2_cann_cp_parity() -> None:
    """Compare the target and existing CSA2 implementation on two CP ranks."""
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_csa2_cann.py")),
        case_name="test_csa2_cann_two_rank_cp_parity",
        num_proc=2,
    )

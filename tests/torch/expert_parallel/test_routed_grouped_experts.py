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
"""Launch staged grouped-expert acceptance without importing Torch in the parent."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).resolve().parent / "_test_routed_grouped_experts.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_local_routed_grouped_experts() -> None:
    """Feature: Local staged grouped experts.
    Description: Compare fused permutation and grouped GEMM with direct expert math.
    Expectation: Both routing-weight modes match in forward and backward with one permutation.
    """
    torchrun_case(_WORKER, "test_local_grouped_experts_forward_backward", num_proc=1)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_ep_routed_grouped_experts() -> None:
    """Feature: Expert-parallel staged grouped experts.
    Description: Compare four-rank slot transport with global direct expert math.
    Expectation: Both modes match in forward and backward with source and receive permutations.
    """
    torchrun_case(_WORKER, "test_ep_grouped_experts_forward_backward", num_proc=4)

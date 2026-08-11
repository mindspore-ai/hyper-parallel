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
"""Launch eight-card dual-mode Trainer FSDP accuracy workers."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case


_WORKER = str(Path(__file__).resolve().parent / "_test_fsdp_accuracy.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_tp_accuracy() -> None:
    """Compare 20-layer HSDP(2x2)+TP(2) training with standalone."""
    torchrun_case(_WORKER, "test_hsdp_tp_accuracy", num_proc=8)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_dp_cp_tp_accuracy() -> None:
    """Compare FSDP-shard(4)+DP(2)+CP(2)+TP(2) with standalone."""
    torchrun_case(_WORKER, "test_dp_cp_tp_accuracy", num_proc=8)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_tp_ep_moe_accuracy() -> None:
    """Compare HF MoE HSDP+TP+EP training with its standalone reference."""
    torchrun_case(_WORKER, "test_hsdp_tp_ep_moe_accuracy", num_proc=8)

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
"""Launch the eight-card dual-mode Trainer FSDP accuracy worker."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).resolve().parent / "_test_fsdp_accuracy.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_tp_cp_ep_fsdp_global_accuracy() -> None:
    """Compare Qwen2-MoE fp32 main_param training on DP(2)+CP(2)+TP(2)+EP(8)."""
    torchrun_case(
        _WORKER,
        "test_qwen2_tp_cp_ep_fsdp_global_accuracy",
        num_proc=8,
    )

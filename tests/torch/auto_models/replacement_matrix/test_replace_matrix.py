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
"""Multi-model distributed training launchers for the replacement interface."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).resolve().parent / "_test_replace_matrix.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_llama_rms_module_replacement() -> None:
    """Compare ten native and module-replaced dense Llama training updates."""
    torchrun_case(_WORKER, "test_llama_rms_module_replacement", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen3_moe_rms_module_replacement() -> None:
    """Compare ten native and module-replaced Qwen3-MoE training updates."""
    torchrun_case(_WORKER, "test_qwen3_moe_rms_module_replacement", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_cp_only_forward_replacement() -> None:
    """Train two ranks with the CP attention action active and EP inactive."""
    torchrun_case(_WORKER, "test_qwen2_moe_cp_only_forward_replacement", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_module_only_replacement() -> None:
    """Train two FSDP ranks with nine RMSNorm replacements and inactive CP/EP."""
    torchrun_case(_WORKER, "test_qwen2_moe_module_only_replacement", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_module_and_cp_forward_replacements() -> None:
    """Train two ranks with RMSNorm and CP attention replacements."""
    torchrun_case(_WORKER, "test_qwen2_moe_module_and_cp_forward_replacements", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_module_and_ep_forward_replacements() -> None:
    """Train two ranks with RMSNorm and EP MoE replacements."""
    torchrun_case(_WORKER, "test_qwen2_moe_module_and_ep_forward_replacements", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_ep_only_forward_replacement() -> None:
    """Train two ranks with the EP MoE action active and CP inactive."""
    torchrun_case(_WORKER, "test_qwen2_moe_ep_only_forward_replacement", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_cp_ep_four_ranks() -> None:
    """Train four ranks with both forward actions active."""
    torchrun_case(_WORKER, "test_qwen2_moe_cp_ep_four_ranks", num_proc=4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_cp_ep_forward_replacements() -> None:
    """Train Qwen2-MoE for ten updates with concurrent CP and EP replacements."""
    torchrun_case(_WORKER, "test_qwen2_moe_cp_ep_forward_replacements", num_proc=8)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen2_moe_module_and_forward_replacements() -> None:
    """Train eight ranks with RMSNorm module, CP attention, and EP MoE actions."""
    torchrun_case(_WORKER, "test_qwen2_moe_module_and_forward_replacements", num_proc=8)

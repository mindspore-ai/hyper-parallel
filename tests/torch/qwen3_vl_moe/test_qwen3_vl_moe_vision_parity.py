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
"""Thin parity entry for the Qwen3-VL-MoE vision tower."""
from __future__ import annotations

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.distributed_launcher import torchrun_case

_WORKER = str(Path(__file__).resolve().parent / "_qwen3_vl_moe_vision_parity_impl.py")


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen3_vl_moe_vision_forward_matches_legacy_eager_path():
    """See implementation case with the same name."""
    torchrun_case(
        _WORKER,
        "test_qwen3_vl_moe_vision_forward_matches_legacy_eager_path",
        master_port=13916,
        num_proc=1,
    )

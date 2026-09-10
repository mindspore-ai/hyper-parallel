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
"""Thin launcher for the four-process CPU/Gloo distributed-data test."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).resolve().parent / "_test_dynamic_packing_gloo.py")


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_dynamic_packing_dp2_mp2_gloo() -> None:
    """Run sample routing, MP delivery, exactly-once, and collective STOP coverage."""
    torchrun_case(
        file_name=_WORKER,
        case_name="test_dynamic_packing_dp2_mp2_gloo",
        num_proc=4,
    )

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
"""Framework-free launcher for DSV4.1 framework-balanced router EP parity."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_deepseek_v41_fix_router_ep_gloo() -> None:
    """
    Feature: DSV4.1 framework-balanced expert parallelism.
    Description: Run two CPU/Gloo ranks against an explicit dense expert oracle.
    Expectation: Learned/balanced outputs and all gradients match the oracle.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_deepseek_v41_fix_router.py")),
        case_name="test_framework_balanced_router_ep_parity",
        num_proc=2,
    )

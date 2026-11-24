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
"""Lightweight launcher for collective MoonEP family parity."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_collective_moonep_family() -> None:
    """Feature: Collective MoonEP family.
    Description: Compare ordinary EP and capacity-bounded replicas across modes and checkpoint replay.
    Expectation: Forward, hidden/router gradients and home parameter gradients agree on four NPUs.
    """
    worker = str(Path(__file__).with_name("_test_moonep_family.py"))
    torchrun_case(worker, "test_collective_moonep_family", num_proc=4)

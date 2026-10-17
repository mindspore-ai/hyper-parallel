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
"""Lightweight launchers for cached KDA recursive doubling."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_rd_boundary_cpu():
    """Feature: Cached RD on ordered CP groups.

    Description: Run a non-power-of-two Gloo scan and interleaved subgroups.
    Expectation: Forward and backward match independent FP64 ordered references."""
    torchrun_case(str(Path(__file__).with_name("_test_kda_rd_boundary.py")), "test_kda_rd_boundary_cpu", num_proc=6)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_rd_boundary_cp8():
    """Feature: NPU cached RD transport and epilogues.

    Description: Run CP8 and DP2/CP4 with public and coalesced transports.
    Expectation: FP64 boundaries, compact caches and transport equivalence hold."""
    torchrun_case(str(Path(__file__).with_name("_test_kda_rd_boundary.py")), "test_kda_rd_boundary_npu", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_rd_layer_cp8():
    """Feature: KDA RD full-layer compatibility and cache lifetime.

    Description: Use identical K3-shaped weights, inputs and output gradients.
    Expectation: Reverse live calls and checkpoint pass; P2P differences stay within 0.003."""
    torchrun_case(str(Path(__file__).with_name("_test_kda_rd_layer.py")), "test_kda_rd_layer", num_proc=8)

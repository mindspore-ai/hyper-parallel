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
"""Thin launchers for KDA AG, subgroup and hybrid qualifications."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_ag_boundary_cpu() -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Run eight Gloo workers covering ordered boundaries and root-preserving subgroups.
    Expectation: Boundaries match independent FP64 recurrences and retain only local transfers.
    """
    torchrun_case(str(Path(__file__).with_name("_test_kda_ag_boundary.py")),
                  "test_kda_ag_boundary_gloo", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_ag_boundary_cp4() -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Compare fused AG and group-owner boundaries with independent FP64 recurrences.
    Expectation: Independent boundary, layout or same-arithmetic compatibility checks pass.
    """
    torchrun_case(str(Path(__file__).with_name("_test_kda_ag_boundary.py")),
                  "test_kda_ag_boundary_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_ag_layer_cp4() -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Compare AG and two-way combinations on local 4K, H96/D128 inputs.
    Expectation: Independent boundary, layout or same-arithmetic compatibility checks pass.
    """
    torchrun_case(str(Path(__file__).with_name("_test_kda_ag_layer.py")), "test_kda_ag_layer", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_ag_layer_cp8() -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Exercise all three nontrivial axes, with U2/g2/two owner groups.
    Expectation: Independent boundary, layout or same-arithmetic compatibility checks pass.
    """
    torchrun_case(str(Path(__file__).with_name("_test_kda_ag_layer.py")), "test_kda_ag_layer", num_proc=8)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_kda_hybrid_layout_cpu() -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Check head order, full-CP convolution halos and parameter gradients against FP32.
    Expectation: Independent boundary, layout or same-arithmetic compatibility checks pass.
    """
    torchrun_case(str(Path(__file__).with_name("_test_kda_hybrid_layout.py")),
                  "test_kda_hybrid_layout", num_proc=4)

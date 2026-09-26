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
"""Thin launchers for KDA boundary, layout and layer qualifications."""
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("case,workers", [("test_kda_ag_boundary_gloo", 8), ("test_kda_hybrid_layout", 4)])
def test_kda_ag_hybrid_cpu(case: str, workers: int) -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Check ordered subgroups and hybrid head/halo layout with Gloo.
    Expectation: Boundaries and layer gradients match independent serial references.
    """
    torchrun_case(str(Path(__file__).with_name("_test_kda_ag_boundary.py")), case, num_proc=workers)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("worker,case,workers", [
    ("boundary", "test_kda_ag_boundary_npu", 4),
    ("layer", "test_kda_ag_layer", 4),
    ("layer", "test_kda_ag_layer", 8),
])
def test_kda_ag_hybrid_npu(worker: str, case: str, workers: int) -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Check FP64 boundaries, local-4K H96/D128 gradients, caches and three-way CP8.
    Expectation: Numerical tolerances and local-M ownership hold across backward and checkpoint replay.
    """
    torchrun_case(str(Path(__file__).with_name(f"_test_kda_ag_{worker}.py")), case, num_proc=workers)

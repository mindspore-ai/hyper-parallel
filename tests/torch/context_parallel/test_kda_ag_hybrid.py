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
"""Thin launchers for KDA state exchange, layout and layer qualifications."""
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("case,workers", [("test_kda_ag_state_gloo", 8), ("test_kda_hybrid_layout", 4)])
def test_kda_ag_hybrid_cpu(case: str, workers: int) -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Check ordered subgroups and hybrid head/halo layout with Gloo.
    Expectation: Recurrent states and layer gradients match independent serial references.
    """
    # Gloo workers do not need accelerator extensions or an activated CANN runtime.
    with patch.dict("os.environ", {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}):
        torchrun_case(str(Path(__file__).with_name("_test_kda_ag_state.py")), case, num_proc=workers)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("worker,case,workers", [
    ("state", "test_kda_ag_state_npu", 4),
    ("layer", "test_kda_ag_layer", 4),
    ("layer", "test_kda_ag_layer", 8),
])
def test_kda_ag_hybrid_npu(worker: str, case: str, workers: int) -> None:
    """Feature: KDA AG and hybrid context parallelism.

    Description: Check states against FP64, local-4K H96/D128 gradients, caches and three-way CP8.
    Expectation: Numerical tolerances and local-M ownership hold across backward and checkpoint replay.
    """
    # Check package presence before worker collection imports the Triton summary.
    # Triton-Ascend provides the Python module named "triton".
    # Installed but incompatible backends must still fail in the worker.
    packages = ("torch_npu", "triton", "fla") if worker == "layer" else ("torch_npu", "triton")
    for package in packages:
        if find_spec(package) is None:
            pytest.skip(f"KDA {worker} NPU test requires the optional package {package!r}")
    torchrun_case(str(Path(__file__).with_name(f"_test_kda_ag_{worker}.py")), case, num_proc=workers)

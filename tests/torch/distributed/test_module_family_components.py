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
"""Framework-free launchers for local high-performance family acceptance."""
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.torch.multicore._integration import activate_multicore_environment

_WORKER_FILE = str(Path(__file__).with_name("_test_module_family_components.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_local_families_npu() -> None:
    """Run RMSNorm and clamped MLP precision acceptance on one NPU."""
    torchrun_case(_WORKER_FILE, "test_local_families_npu", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_mega_mhc_family_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run native mHC output and all-gradient parity on one NPU with a multicore payload."""
    activate_multicore_environment(monkeypatch)
    torchrun_case(_WORKER_FILE, "test_mega_mhc_family_npu", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_ordinary_mega_mhc_family_npu() -> None:
    """Run the ordinary shifted mHC math and gradient comparison on one NPU."""
    torchrun_case(_WORKER_FILE, "test_ordinary_mega_mhc_family_npu", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_source_contracts_npu() -> None:
    """Compare real source dtype, activation and gradient contracts on a single NPU."""
    torchrun_case(_WORKER_FILE, "test_source_contracts_npu", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_router_contracts_npu() -> None:
    """Validate projection, bias selection, normalization and routing gradients on one NPU."""
    torchrun_case(_WORKER_FILE, "test_router_contracts_npu", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_mega_gate_family_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare the public Gate mega member with original MegaGate native routing and Torch math."""
    activate_multicore_environment(monkeypatch)
    torchrun_case(_WORKER_FILE, "test_mega_gate_family_npu", num_proc=1)

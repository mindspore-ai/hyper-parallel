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
"""Lightweight launcher for the graph-mode CPU offload loss regression."""

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


def _torch_has_ao_ops() -> bool:
    """Whether the installed torch ships the private ``ao::`` offload ops.

    ``torch._functorch._activation_offloading.offload_ops`` is private and is
    absent from some stock builds, so the package version alone is not a
    reliable proxy. Probe the module file through package metadata only:
    launchers must not import torch. The worker re-checks real op registration
    and skips when the build does not provide them.
    """
    try:
        module_file = distribution("torch").locate_file(
            "torch/_functorch/_activation_offloading/offload_ops.py"
        )
    except (PackageNotFoundError, OSError):
        return False
    return Path(module_file).is_file()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.skipif(
    not _torch_has_ao_ops(),
    reason="torch build does not ship the private ao:: offload ops",
)
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_attention_cpu_offload_loss(dtype):
    """
    Feature: Graph-mode attention activation CPU offload.
    Description: Train one toy attention LM eager, as a joint graph, and with
        activation_swap="attention" from identical weights on a single NPU.
    Expectation: Offload loss, gradients and updated parameters match the
        no-offload graph and eager baselines over several optimizer steps.
    """
    worker = str(Path(__file__).with_name("_test_cpu_offload.py"))
    torchrun_case(
        file_name=worker,
        case_name=f"test_attention_cpu_offload_loss[{dtype}]",
        num_proc=1,
    )

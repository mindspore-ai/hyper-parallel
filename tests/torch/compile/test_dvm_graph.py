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
"""Launcher for the graph-mode DVM precision regression."""

import importlib.util
import re
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Optional

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_DVM_MIN_TORCH_VERSION = (2, 10)


def _torch_version_tuple() -> Optional[tuple[int, int]]:
    """Read the torch version from package metadata without importing torch."""
    try:
        version = distribution("torch").version
    except PackageNotFoundError:
        return None
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _torch_npu_package_paths() -> list[Path]:
    """Return torch_npu package roots without importing it."""
    try:
        spec = importlib.util.find_spec("torch_npu")
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    locations = spec.submodule_search_locations
    if locations:
        return [Path(location) for location in locations]
    if spec.origin is None:
        return []
    return [Path(spec.origin).parent]


def _supports_dvm_graph() -> bool:
    """Whether this install has torch >= 2.10 and torch_npu's DVM backend."""
    version = _torch_version_tuple()
    if version is None or version < _DVM_MIN_TORCH_VERSION:
        return False
    return any(
        (root / "_inductor" / "dvm" / "__init__.py").is_file()
        for root in _torch_npu_package_paths()
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.skipif(
    not _supports_dvm_graph(),
    reason="requires torch >= 2.10 and torch_npu with the DVM Inductor backend",
)
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_dvm_graph_precision(dtype):
    """
    Feature: Graph-mode DVM Inductor backend.
    Description: Train one toy model eager, as a plain joint graph, and with
        ``inductor_backend="dvm"`` from identical weights on a single NPU.
    Expectation: DVM loss, gradients and updated parameters match the plain
        graph and eager baselines, and the compiled artifact contains DVM
        kernel code.
    """
    worker = str(Path(__file__).with_name("_test_dvm_graph.py"))
    torchrun_case(
        file_name=worker,
        case_name=f"test_dvm_graph_precision[{dtype}]",
        num_proc=1,
    )

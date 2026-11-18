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
"""Single-card launchers for the Triton symmetric matrix product."""

import os
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).with_name("_test_symmetric_matmul.py"))


def _has_triton_ascend() -> bool:
    """Check for the optional Ascend backend without importing Triton in the launcher."""
    spec = find_spec("triton")
    return spec is not None and any(
        (Path(path) / "backends" / "ascend").is_dir()
        for path in spec.submodule_search_locations or ()
    )


pytestmark = pytest.mark.skipif(not _has_triton_ascend(), reason="requires the optional Triton-Ascend backend")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_symmetric_matmul_accuracy() -> None:
    """Check dense products, tail blocks and both output precision contracts."""
    with patch.dict(os.environ, {"CLOSE_MATMUL_K_SHIFT": "1"}):
        torchrun_case(_WORKER, "test_symmetric_matmul_accuracy", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_symmetric_matmul_layouts() -> None:
    """Check both operand layouts, independent symmetric operands and guarded outputs."""
    with patch.dict(os.environ, {"CLOSE_MATMUL_K_SHIFT": "1"}):
        torchrun_case(_WORKER, "test_symmetric_matmul_layouts", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_symmetric_matmul_streams_and_graph() -> None:
    """Check current-stream execution, temporary lifetimes and graph replay."""
    with patch.dict(os.environ, {"CLOSE_MATMUL_K_SHIFT": "1"}):
        torchrun_case(_WORKER, "test_symmetric_matmul_streams_and_graph", num_proc=1)

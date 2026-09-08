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
"""Pytest launcher for PrecompiledBoundary / local_region CPU Gloo integration.

The worker initializes a real single-process Gloo group and builds real
DTensors; per the test-layering rules these cases live under ``tests/torch``
(Gate-2) instead of ``tests/ut``. This launcher only forks the torchrun
runner and must not import torch / hyper_parallel itself.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_precompiled_boundary_gloo.py")


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_precompiled_boundary_gloo():
    """
    Feature: PrecompiledBoundary redistribute IO and local_region wrap/unwrap on CPU Gloo
    Description:
        1. test_redistribute_io
        2. test_local_region_error_paths
        3. test_local_region_wrap_unwrap
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_WORKER, "test_redistribute_io", num_proc=1),
        TorchCase(_WORKER, "test_local_region_error_paths", num_proc=1),
        TorchCase(_WORKER, "test_local_region_wrap_unwrap", num_proc=1),
    ])

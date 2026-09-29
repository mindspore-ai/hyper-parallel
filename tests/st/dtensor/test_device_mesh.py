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
"""Pytest launchers for DeviceMesh NPU distributed tests.

Each launcher uses ``parallel_run`` to execute several worker cases from
``_test_device_mesh.py`` concurrently on disjoint NPU subsets.

Typical packing:
  - up to four 2-card worker cases per launcher

Port allocation:
  10520–10529  2-card tests (single-node ``num_proc=2``)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

# Absolute path: torchrun inherits cwd (repo root or this dir); basename alone can fail.
_WORKER = str(Path(__file__).resolve().parent / "_test_device_mesh.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_dtensor_device_mesh_group1():
    """
    Feature: parallel_run launcher for 2-card DTensor DeviceMesh coverage
    Description:
        1. test_device_mesh_init_1d_eight_ranks_npu
        2. test_device_mesh_get_current_mesh_raises_without_context_npu
        3. test_device_mesh_with_mesh_current_mesh_identity_npu
        4. test_device_mesh_nested_with_get_current_mesh_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_device_mesh_init_1d_eight_ranks_npu", 10520, 2),
        ("test_device_mesh_get_current_mesh_raises_without_context_npu", 10521, 2),
        ("test_device_mesh_with_mesh_current_mesh_identity_npu", 10522, 2),
        ("test_device_mesh_nested_with_get_current_mesh_npu", 10523, 2),
    )

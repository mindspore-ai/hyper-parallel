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
"""Pytest entry-points for DeviceMesh NPU distributed tests.

Each test spawns worker processes via torchrun and delegates to the
corresponding test function in device_mesh.py (NPU/hccl).

Run from ``tests/torch/dtensor/`` so the worker module path resolves (same pattern
as ``tests/torch/context_parallel/test_cp_npu.py``).

Port allocation:
  10520–10529  8-card tests (single-node ``num_proc=8``)
"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

_FILE = "device_mesh.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_device_mesh_init_1d_eight_ranks_npu():
    """DeviceMesh 1-D init with eight ranks.

    Feature: DeviceMesh construction on NPU
    Description: ``init_device_mesh`` with 1-D mesh (8,) under hccl.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_device_mesh_init_1d_eight_ranks_npu", master_port=10520, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_device_mesh_get_current_mesh_raises_without_context_npu():
    """get_current_mesh raises outside of mesh context (8-rank).

    Feature: DeviceMesh context API
    Description: No active mesh stack entry before entering context.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_device_mesh_get_current_mesh_raises_without_context_npu", master_port=10521, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_device_mesh_with_mesh_current_mesh_identity_npu():
    """with device_mesh sets current mesh (8-rank).

    Feature: DeviceMesh context API
    Description: ``get_current_mesh`` matches entered mesh.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_device_mesh_with_mesh_current_mesh_identity_npu", master_port=10522, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_device_mesh_nested_with_get_current_mesh_npu():
    """Nested mesh contexts and get_current_mesh (8-rank).

    Feature: DeviceMesh nested context
    Description: Inner ``with`` restores outer current mesh after exit.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_device_mesh_nested_with_get_current_mesh_npu", master_port=10523, num_proc=8)

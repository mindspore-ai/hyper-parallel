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
"""Distributed NPU tests for ``parallelize_module`` (torchrun, hccl).

Launched from ``test_parallelize_module_distributed.py`` via ``parallel_run``
(2-card functional cases). Linear precision ST lives in tp_styles.
"""
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import init_device_mesh, parallelize_module
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


class _VerifyMeshParallelStyle(ParallelStyle):
    """Asserts process group world size matches *device_mesh* for this rank."""

    def apply(self, module: nn.Module, device_mesh: object) -> nn.Module:
        """Return *module* after asserting mesh matches the process group.

        Args:
            module: Module being parallelized.
            device_mesh: Device mesh for the current plan.

        Returns:
            The same *module* instance.
        """
        assert dist.is_initialized(), "process group must be initialized"
        ws = dist.get_world_size()
        rk = dist.get_rank()
        assert ws == len(device_mesh.rank_list), (
            f"world_size {ws} != len(mesh.rank_list) {device_mesh.rank_list}"
        )
        assert rk in device_mesh.rank_list, f"rank {rk} not in mesh {device_mesh.rank_list}"
        return module


class _CountingParallelStyle(ParallelStyle):
    """Increments a module attribute each time ``apply`` runs (per-rank)."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def apply(self, module: nn.Module, device_mesh: object) -> nn.Module:
        """Increment counters and return *module* unchanged.

        Args:
            module: Module being parallelized.
            device_mesh: Device mesh for the current plan.

        Returns:
            The same *module* instance.
        """
        self.count += 1
        if not hasattr(module, "_hp_parallelize_apply_count"):
            module._hp_parallelize_apply_count = 0
        module._hp_parallelize_apply_count += 1
        return module


def _make_tp_mesh_1d():
    """1-D mesh covering all ranks in the default process group."""
    return init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )


def test_parallelize_module_mesh_aligned_with_process_group_npu():
    """
    Feature: parallelize_module mesh aligns with HCCL process group on NPU
    Description: init dist, 1-D mesh over all ranks, parallelize_module with _VerifyMeshParallelStyle
    Expectation: style apply passes; mesh rank_list matches world size and contains current rank
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    m = to_device(nn.Identity(), _DEVICE_TYPE)
    parallelize_module(m, mesh, _VerifyMeshParallelStyle())


def test_parallelize_module_dict_fnmatch_npu():
    """
    Feature: parallelize_module dict plan with fnmatch wildcard on NPU
    Description: MLP with net1, net2, other on NPU; plan {"net*": one CountingParallelStyle}
    Expectation: style.count==2; net1/net2 apply count 1; other has no apply count attr
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net1 = nn.Identity()
            self.net2 = nn.Identity()
            self.other = nn.Identity()

    model = to_device(_MLP(), _DEVICE_TYPE)
    style = _CountingParallelStyle()
    parallelize_module(model, mesh, {"net*": style})
    assert style.count == 2
    assert model.net1._hp_parallelize_apply_count == 1
    assert model.net2._hp_parallelize_apply_count == 1
    assert not hasattr(model.other, "_hp_parallelize_apply_count")


def test_parallelize_module_src_data_rank_npu():
    """
    Feature: parallelize_module propagates src_data_rank on NPU
    Description: two calls with src_data_rank=1 then src_data_rank=None on same Linear module
    Expectation: first style src_data_rank==1; second style src_data_rank is None
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    m = to_device(nn.Linear(2, 2), _DEVICE_TYPE)
    style = _VerifyMeshParallelStyle()
    parallelize_module(m, mesh, style, src_data_rank=1)
    assert style.src_data_rank == 1
    style_none = _VerifyMeshParallelStyle()
    parallelize_module(m, mesh, style_none, src_data_rank=None)
    assert style_none.src_data_rank is None


def test_parallelize_module_single_style_root_npu():
    """
    Feature: parallelize_module single ParallelStyle on root module on NPU
    Description: Linear on NPU with one CountingParallelStyle as parallelize_plan
    Expectation: style.count==1 and root module parallelize apply count is 1
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    m = to_device(nn.Linear(3, 3), _DEVICE_TYPE)
    style = _CountingParallelStyle()
    parallelize_module(m, mesh, style)
    assert style.count == 1
    assert m._hp_parallelize_apply_count == 1

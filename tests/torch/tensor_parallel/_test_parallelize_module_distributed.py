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

Launched from ``test_parallelize_module_distributed.py`` via ``parallel_run``:
functional cases use ``num_proc=2``; precision cases use ``num_proc=4`` for
``world_size==4`` linear sharding checks.

Precision cases shard ``nn.Linear`` in the same way as PyTorch
``ColwiseParallel`` / ``RowwiseParallel`` (output dim vs input dim); the reference is
``F.linear`` on CPU with the full weight (PyTorch numerical semantics).
"""
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


# --- PyTorch-aligned TP shard patterns (golden = CPU F.linear full weight) ---


class _ColwiseParallelLinear(nn.Module):
    """Column-parallel Linear: shard ``weight`` along out_features; ``all_gather`` on output (like ColwiseParallel)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        weight_shard: torch.Tensor,
        bias_shard: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.weight = nn.Parameter(weight_shard)
        self.bias = nn.Parameter(bias_shard) if bias_shard is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Column-parallel linear then all-gather along the last dim.

        Args:
            x: Input activations.

        Returns:
            Full output after all-gather.
        """
        y_local = F.linear(x, self.weight, self.bias)
        chunks = [torch.empty_like(y_local) for _ in range(self.world_size)]
        dist.all_gather(chunks, y_local)
        return torch.cat(chunks, dim=-1)


class _ColwiseLinearPrecisionStyle(ParallelStyle):
    """Apply column-parallel sharding to ``nn.Linear`` (PyTorch ``ColwiseParallel`` layout)."""

    def apply(self, module: nn.Module, device_mesh: object) -> nn.Module:
        """Replace ``nn.Linear`` with a column-sharded parallel module.

        Args:
            module: ``nn.Linear`` to shard.
            device_mesh: Device mesh (unused; 1-D mesh matches process group).

        Returns:
            A column-parallel linear module for this rank.
        """
        del device_mesh  # 1-D mesh already matches process group
        if not isinstance(module, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(module)}")
        world = dist.get_world_size()
        rank = dist.get_rank()
        out_f = module.out_features
        in_f = module.in_features
        if out_f % world != 0:
            raise ValueError(f"out_features {out_f} not divisible by world_size {world}")
        per = out_f // world
        start = rank * per
        end = start + per
        w_full = module.weight.data
        b_full = module.bias.data if module.bias is not None else None
        w_shard = w_full[start:end].contiguous().clone()
        b_shard = b_full[start:end].contiguous().clone() if b_full is not None else None
        return _ColwiseParallelLinear(in_f, out_f, world, w_shard, b_shard)


class _RowwiseParallelLinear(nn.Module):
    """Row-parallel Linear: shard ``weight`` along in_features; partial matmul + sum allreduce (like RowwiseParallel)."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        world_size: int,
        rank: int,
        weight_shard: torch.Tensor,
        bias_full: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.world_size = world_size
        self.rank = rank
        self.weight = nn.Parameter(weight_shard)
        self.bias = nn.Parameter(bias_full) if bias_full is not None else None
        per = in_features // world_size
        self._in_start = rank * per
        self._in_len = per

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Row-parallel linear: partial matmul, all-reduce, then optional bias.

        Args:
            x: Input activations (full width; each rank uses its shard of features).

        Returns:
            Full output after all-reduce (and bias).
        """
        x_r = x[:, self._in_start : self._in_start + self._in_len]
        y = F.linear(x_r, self.weight, None)
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            y = y + self.bias
        return y


class _RowwiseLinearPrecisionStyle(ParallelStyle):
    """Apply row-parallel sharding to ``nn.Linear`` (PyTorch ``RowwiseParallel`` layout)."""

    def apply(self, module: nn.Module, device_mesh: object) -> nn.Module:
        """Replace ``nn.Linear`` with a row-sharded parallel module.

        Args:
            module: ``nn.Linear`` to shard.
            device_mesh: Device mesh (unused).

        Returns:
            A row-parallel linear module for this rank.
        """
        del device_mesh
        if not isinstance(module, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(module)}")
        world = dist.get_world_size()
        rank = dist.get_rank()
        out_f = module.out_features
        in_f = module.in_features
        if in_f % world != 0:
            raise ValueError(f"in_features {in_f} not divisible by world_size {world}")
        per = in_f // world
        start = rank * per
        end = start + per
        w_full = module.weight.data
        w_shard = w_full[:, start:end].contiguous().clone()
        b_full = module.bias.data.clone() if module.bias is not None else None
        return _RowwiseParallelLinear(out_f, in_f, world, rank, w_shard, b_full)


def _npu_precision_close(a: torch.Tensor, b: torch.Tensor) -> None:
    """Assert NPU vs CPU reference within typical float32 tolerance (HCCL matmul)."""
    torch.testing.assert_close(
        a.cpu().float(),
        b.cpu().float(),
        rtol=1.5e-4,
        atol=1e-5,
    )


def test_parallelize_module_colwise_linear_precision_vs_pytorch_ref_npu():
    """
    Feature: column-parallel Linear via parallelize_module matches CPU F.linear
    Description: shard out_features; all_gather outputs — same layout as PyTorch ColwiseParallel
    Expectation: gathered NPU output close to PyTorch reference on CPU
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    torch.manual_seed(42)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(42)
    in_f, out_f, batch = 32, 64, 8
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x, w, b)

    linear = to_device(nn.Linear(in_f, out_f, bias=True), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    x_npu = to_device(x, _DEVICE_TYPE)

    sharded = parallelize_module(linear, mesh, _ColwiseLinearPrecisionStyle())
    y_hp = sharded(x_npu)
    _npu_precision_close(y_hp, y_ref)


def test_parallelize_module_rowwise_linear_precision_vs_pytorch_ref_npu():
    """
    Feature: row-parallel Linear via parallelize_module matches CPU F.linear
    Description: shard in_features; allreduce partials — same layout as PyTorch RowwiseParallel
    Expectation: NPU output close to PyTorch reference on CPU
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    torch.manual_seed(43)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(43)
    in_f, out_f, batch = 32, 24, 8
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x, w, b)

    linear = to_device(nn.Linear(in_f, out_f, bias=True), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    x_npu = to_device(x, _DEVICE_TYPE)

    sharded = parallelize_module(linear, mesh, _RowwiseLinearPrecisionStyle())
    y_hp = sharded(x_npu)
    _npu_precision_close(y_hp, y_ref)


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

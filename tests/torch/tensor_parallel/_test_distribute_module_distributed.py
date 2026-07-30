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
"""Distributed NPU integration tests for :func:`distribute_module` (torchrun, HCCL).

Aligned with PyTorch ``test/distributed/tensor/test_api.py`` (``DTensorAPITest``):
``test_distribute_module``, ``test_distribute_module_input_fn_output_fn``.

Launched from ``test_distribute_module_distributed.py`` via ``parallel_run``
(2-card functional cases). Linear precision ST lives in tp_styles.
"""
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import DTensor, distribute_module, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


def _make_1d_mesh_tp():
    """1-D mesh over all ranks in the default process group (launcher uses 2 or 4 ranks)."""
    return init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )


def _is_all_replicate(param) -> bool:
    return all(p.is_replicate() for p in param.placements)


def _is_shard_dim0(param) -> bool:
    pl = param.placements
    return len(pl) == 1 and pl[0].is_shard() and pl[0].dim == 0


class _SeqMLP(nn.Module):
    """Two-layer MLP for partial-shard tests (names ``layers.0`` / ``layers.1``)."""

    def __init__(self, d_in: int, d_h: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.Linear(d_h, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def test_distribute_module_replicate_all_params_npu():
    """
    Feature: ``distribute_module`` without ``partition_fn`` fully replicates parameters
    Description: Same as PyTorch ``test_distribute_module`` replica branch — all
        ``nn.Parameter`` become ``DTensor`` with ``Replicate`` on 1-D mesh.
    Expectation: Every parameter is ``DTensor`` and placements are all-replicate.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_1d_mesh_tp()
    ws = dist.get_world_size()
    m = to_device(nn.Linear(12, 8 * ws, bias=True), _DEVICE_TYPE)
    torch.manual_seed(0)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(0)
    with torch.no_grad():
        m.weight.normal_(0, 0.02)
        m.bias.normal_(0, 0.02)

    distributed = distribute_module(m, mesh, partition_fn=None)
    assert distributed is m
    for p in m.parameters():
        assert isinstance(p, DTensor), f"expected DTensor param, got {type(p)}"
        assert _is_all_replicate(p), f"expected full replicate, got {p.placements}"


def test_distribute_module_shard_all_linears_npu():
    """
    Feature: ``partition_fn`` shards every ``nn.Linear`` like PyTorch ``shard_fn``
    Description: ``distribute_tensor(..., [Shard(0)])`` per parameter on 1-D mesh;
        ``out_features`` divisible by ``world_size``.
    Expectation: All parameters are ``DTensor`` with ``Shard(0)`` placement.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_1d_mesh_tp()
    ws = dist.get_world_size()
    in_f, out_f = 16, 10
    assert out_f % ws == 0
    spec = [Shard(0)]

    def shard_fn(mod_name: str, module: nn.Module, device_mesh) -> None:
        del mod_name
        if isinstance(module, nn.Linear):
            for pname, param in module.named_parameters(recurse=False):
                dist_tensor = distribute_tensor(param.data, device_mesh, spec)
                module.register_parameter(pname, nn.Parameter(dist_tensor))

    m = to_device(nn.Sequential(
        nn.Linear(in_f, out_f, bias=True),
        nn.ReLU(),
        nn.Linear(out_f, in_f, bias=True),
    ), _DEVICE_TYPE)
    torch.manual_seed(1)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.02)

    distribute_module(m, mesh, partition_fn=shard_fn)
    for p in m.parameters():
        assert isinstance(p, DTensor)
        assert _is_shard_dim0(p), f"expected Shard(0), got {p.placements}"


def test_distribute_module_partial_shard_replicate_rest_npu():
    """
    Feature: partial shard + implicit replicate (PyTorch ``test_distribute_module`` tail)
    Description: Only ``layers.0`` Linear is sharded in ``partition_fn``; ``layers.1``
        is left dense then converted to replicate ``DTensor`` by default path.
    Expectation: ``layers.0`` weights/bias shard dim0; ``layers.1`` replicate.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_1d_mesh_tp()
    ws = dist.get_world_size()
    d_in, d_h, d_out = 8, 6 * ws, 10
    assert d_h % ws == 0
    shard_spec = [Shard(0)]

    def shard_fn(mod_name: str, module: nn.Module, device_mesh) -> None:
        if isinstance(module, nn.Linear) and mod_name == "layers.0":
            for pname, param in module.named_parameters(recurse=False):
                dist_tensor = distribute_tensor(param.data, device_mesh, shard_spec)
                module.register_parameter(pname, nn.Parameter(dist_tensor))

    model = to_device(_SeqMLP(d_in, d_h, d_out), _DEVICE_TYPE)
    torch.manual_seed(2)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(2)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)

    distribute_module(model, mesh, partition_fn=shard_fn)
    for fqname, p in model.named_parameters():
        assert isinstance(p, DTensor), fqname
        if fqname.startswith("layers.0."):
            assert _is_shard_dim0(p), (fqname, p.placements)
        elif fqname.startswith("layers.1."):
            assert _is_all_replicate(p), (fqname, p.placements)
        else:
            raise AssertionError(f"unexpected param name {fqname}")


def test_distribute_module_input_output_hooks_npu():
    """
    Feature: ``input_fn`` / ``output_fn`` on root (PyTorch ``test_distribute_module_input_fn_output_fn``)
    Description: Replicate all params; pre-hook wraps batch dim with ``Shard(0)``;
        post-hook returns ``to_local()`` tensor.
    Expectation: Forward output is plain ``torch.Tensor`` on NPU, not ``DTensor``.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_1d_mesh_tp()
    ws = dist.get_world_size()
    batch, in_f, out_f = 4 * ws, 20, 12
    assert batch % ws == 0

    def input_fn(mod, inputs, device_mesh):
        del mod
        x = inputs[0]
        dt = DTensor.from_local(x, device_mesh, [Shard(0)])
        return (dt,) + tuple(inputs[1:])

    def output_fn(mod, outputs, device_mesh):
        del mod, device_mesh
        assert isinstance(outputs, DTensor)
        return outputs.to_local()

    m = to_device(nn.Linear(in_f, out_f, bias=True), _DEVICE_TYPE)
    torch.manual_seed(3)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(3)
    with torch.no_grad():
        m.weight.normal_(0, 0.02)
        m.bias.zero_()

    distribute_module(m, mesh, partition_fn=None, input_fn=input_fn, output_fn=output_fn)
    x = torch.randn(batch, in_f, device=_DEVICE_TYPE, dtype=torch.float32)
    y = m(x)
    assert isinstance(y, torch.Tensor)
    assert not isinstance(y, DTensor)

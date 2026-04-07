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
"""Precision test for fully_shard(module=list): chained layers as one FSDP unit.

Compares distributed training (HyperParallel fully_shard with list roots + grouped hooks)
against standalone eager training. Semantics match PyTorch FSDP2 ``fully_shard([...])``
(one unshard / one post-forward boundary per group per step).
"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np
import torch
import torch_npu
from torch import nn

from hyper_parallel import DTensor, DeviceMesh, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from tests.torch.utils import init_dist

torch.manual_seed(0)
HIDDEN = 8
standalone_x = torch.rand(8, HIDDEN)


class ListUnitModel(nn.Module):
    """pre -> lin1 -> lin2 -> scalar sum (list unit covers lin1, lin2)."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.pre = nn.Linear(hidden, hidden, bias=False)
        self.block = nn.Module()
        self.block.lin1 = nn.Linear(hidden, hidden, bias=False)
        self.block.lin2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through pre, list-wrapped linears, scalar sum.

        Args:
            x: Input batch tensor.
        """
        x = self.pre(x)
        x = self.block.lin1(x)
        x = self.block.lin2(x)
        return x.sum()


def _init_deterministic_weights(module: nn.Module) -> None:
    torch.manual_seed(12345)
    for p in module.parameters():
        nn.init.normal_(p, mean=0.0, std=0.05)


def _list_fsdp_kwargs(mesh: DeviceMesh):
    """Shared kwargs; pass ``reshard_after_forward`` per ``fully_shard`` call."""
    return {
        "mesh": mesh,
        "shard_placement_fn": None,
        "mp_policy": MixedPrecisionPolicy(),
        "offload_policy": OffloadPolicy(),
        "ignored_params": None,
    }


def _one_step_standalone_list_unit():
    """Single forward-backward (no optimizer): reference loss and full lin1 grad."""
    model = ListUnitModel(HIDDEN).npu()
    _init_deterministic_weights(model)
    x = standalone_x.npu()
    model.zero_grad()
    loss = model(x)
    loss.backward()
    return loss.detach(), model.block.lin1.weight.grad.detach().clone()


def _one_step_hp_list_unit(mesh: DeviceMesh):
    """Same as nested list FSDP setup; backward scaled like ``_test_fully_shard_precision``."""
    model = ListUnitModel(HIDDEN).npu()
    _init_deterministic_weights(model)
    fsdp_kw = _list_fsdp_kwargs(mesh)
    fully_shard(model.pre, **fsdp_kw, reshard_after_forward=True)
    list_modules = [model.block.lin1, model.block.lin2]
    fully_shard(list_modules, **fsdp_kw, reshard_after_forward=False)
    fully_shard(model, **fsdp_kw, reshard_after_forward=True)
    model.set_reduce_op_type("sum")

    x = standalone_x.npu()
    model.zero_grad()
    loss = model(x)
    # Match ``_test_fully_shard_precision`` grad shard check: default backward on loss.
    loss.backward()
    g = model.block.lin1.weight.grad
    assert isinstance(g, DTensor), type(g)
    return loss.detach(), g.data.clone()


def _compare_against_standalone(mesh: DeviceMesh) -> None:
    """Assert distributed list-unit step matches standalone loss and ``lin1`` grad local shard."""
    init_dist()
    rank, _ = init_dist()
    shard_size = mesh.mesh_shape[-1]
    standalone_loss, standalone_grad = _one_step_standalone_list_unit()
    dist_loss, dist_grad = _one_step_hp_list_unit(mesh)

    assert np.allclose(
        standalone_loss.cpu().numpy(),
        dist_loss.cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
    ), (standalone_loss.item(), dist_loss.item())

    dp_stride = HIDDEN // shard_size
    dp_offset = (rank % shard_size) * dp_stride
    sg = standalone_grad.cpu().numpy()[dp_offset : dp_offset + dp_stride, :]
    dg = dist_grad.cpu().numpy()
    assert np.allclose(sg, dg, rtol=1e-3, atol=1e-3), (sg.shape, dg.shape)


def test_list_unit_precision_zero3():
    """
    Feature: fully_shard list roots precision vs standalone.
    Description: Nested fully_shard with ``fully_shard([lin1, lin2], reshard_after_forward=False)``;
        one forward-backward: scalar loss and ``lin1`` grad local shard match standalone eager.
    Expectation: Run success.
    """
    _compare_against_standalone(init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",)))

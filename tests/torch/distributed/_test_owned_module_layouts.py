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
"""Real planner/application parity for replicated coefficients and sparse EP rows."""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.modules import Engram, PipelinedMhc
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.mesh import MeshContext
from tests.common.engram_family import small_config


class _ComponentModel(nn.Module):
    """Expose a real parameter tree to the standard planner without a large model dependency."""

    def __init__(self, component: nn.Module) -> None:
        super().__init__()
        self.component = component
        self.config = SimpleNamespace(model_type="owned_family_test", num_attention_heads=4,
                                      num_key_value_heads=4, num_experts=2)

    def forward(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Call the registered component through its ordinary module boundary."""
        return self.component(*args)


def _apply(model: nn.Module, mesh: MeshContext) -> nn.Module:
    """Use normal parameter localization and communication binding without YAML layout injections."""
    plan = ShardingPlanner().plan(model, mesh.device_mesh, tp_size=mesh.tp_size,
                                  cp_size=mesh.cp_size, ep_size=mesh.ep_size,
                                  sequence_parallel=mesh.sequence_parallel)
    model, _ = apply_sharding_plan(model, plan, mesh)
    return model


def _check_mhc(device: torch.device, *, sequence_parallel: bool) -> None:
    """Check CP2 and TP2/SP local outputs, input gradients and globally summed coefficient gradients."""
    context = {"tp_size": 2 if sequence_parallel else 1, "cp_size": 1 if sequence_parallel else 2,
               "sequence_parallel": sequence_parallel}
    mesh = MeshContext(tp_size=context["tp_size"], cp_size=context["cp_size"],
                       dp_shard_size=context["cp_size"], sequence_parallel=sequence_parallel)
    mesh.build_meshs(device.type, dist.get_world_size())
    torch.manual_seed(42)
    settings = SimpleNamespace(num_stream=4, hidden_size=16, sinkhorn_iters=20,
                               hc_eps=1e-6, norm_eps=1e-6, use_gamma=False, init_scale=1.0)
    model = _ComponentModel(PipelinedMhc(config=settings, context=context)).to(device)
    reference = _ComponentModel(PipelinedMhc(config=settings, context=context)).to(device)
    reference.load_state_dict(model.state_dict())
    model = _apply(model, mesh)
    values = torch.linspace(-1, 1, 512).reshape(1, 8, 4, 16).to(device)
    full = values.detach().clone().requires_grad_()
    local = values.chunk(2, dim=1)[dist.get_rank()].detach().clone().requires_grad_()
    expected, actual = reference(full), model(local)
    for output, target in zip(actual, expected):
        torch.testing.assert_close(output, target.chunk(2, dim=1)[dist.get_rank()], rtol=3e-4, atol=3e-5)
    sum(output.square().sum() for output in actual).backward()
    sum(output.square().sum() for output in expected).backward()
    torch.testing.assert_close(local.grad, full.grad.chunk(2, dim=1)[dist.get_rank()], rtol=3e-4, atol=3e-5)
    for name, parameter in model.named_parameters():
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, rtol=3e-4, atol=3e-5)


def _check_engram(device: torch.device, *, context_parallel: bool) -> None:
    """Compare sparse owner-row and replicated fusion gradients with one full-table reference."""
    cp_size = 2 if context_parallel else 1
    mesh = MeshContext(dp_size=1 if context_parallel else 2, cp_size=cp_size, dp_shard_size=2, ep_size=2)
    mesh.build_meshs(device.type, dist.get_world_size())
    torch.manual_seed(43)
    model = _ComponentModel(Engram(config=small_config(), context={"ep_size": 2, "cp_size": cp_size})).to(device)
    reference = _ComponentModel(Engram(config=small_config(), context={"ep_size": 2, "cp_size": cp_size})).to(device)
    reference.load_state_dict(model.state_dict())
    model = _apply(model, mesh)
    values = torch.linspace(-0.7, 1.3, 64).reshape(1, 8, 2, 4).to(device)
    ids = torch.tensor([[1, 2, 3, 4, 2, 1, 4, 3]], device=device)
    full = values.detach().clone().requires_grad_()
    local = values.chunk(2, dim=1)[dist.get_rank()].detach().clone().requires_grad_()
    local_ids = ids.chunk(2, dim=1)[dist.get_rank()]
    actual = model(local, local_ids)
    # CP retains the global n-gram halo; independent data replicas start separate sequences.
    expected = (reference(full, ids) if context_parallel else
                torch.cat([reference.component(chunk, token_ids)
                           for chunk, token_ids in zip(full.chunk(2, dim=1), ids.chunk(2, dim=1))], dim=1))
    torch.testing.assert_close(actual, expected.chunk(2, dim=1)[dist.get_rank()], rtol=3e-4, atol=3e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(local.grad, full.grad.chunk(2, dim=1)[dist.get_rank()], rtol=3e-4, atol=3e-5)
    for name, parameter in model.named_parameters():
        target = reference.get_parameter(name).grad
        if name.endswith("embed.weight"):
            target = target.chunk(2, dim=0)[dist.get_rank()]
        else:
            dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad, target, rtol=3e-4, atol=3e-5)


def _run(device_type: str) -> None:
    """Initialize only this worker group's resources and always release them."""
    if device_type == "npu":
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(device_type, int(os.environ["LOCAL_RANK"])) if device_type == "npu" else torch.device("cpu")
    dist.init_process_group("hccl" if device_type == "npu" else "gloo", timeout=timedelta(seconds=180))
    try:
        _check_mhc(device, sequence_parallel=False)
        _check_mhc(device, sequence_parallel=True)
        _check_engram(device, context_parallel=False)
        _check_engram(device, context_parallel=True)
    finally:
        dist.destroy_process_group()


def test_owned_module_layouts_gloo() -> None:
    """Run the two-rank CPU planner/application regression."""
    _run("cpu")


def test_owned_module_layouts_hccl() -> None:
    """Run the same two-rank regression on Ascend HCCL."""
    _run("npu")

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
"""Real Gloo checks through existing CP helpers and EP recipe execution."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.components.modules.family import get_family_parallel_adapter
from hyper_parallel.distributed.expert_parallel import experts as ep_execution
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.models._transformers.model_builder import _plan_and_apply_sharding
from tests.common.module_parallel_utils import (
    PackedExperts, cpu_grouped_matmul, make_attention, make_grouped_experts,
)


class _RoutedMoe(nn.Module):
    """Native model semantics stay in the owner after Experts replacement."""

    def __init__(self) -> None:
        """Build native routing and packed expert parameters."""
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False, dtype=torch.float64)
        self.experts = PackedExperts()
        self.top_k = 2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the local routing oracle before automatic EP binding."""
        flat = hidden_states.reshape(-1, 4)
        weights, indices = self.gate(flat).softmax(-1, dtype=torch.float32).topk(self.top_k, -1)
        weights = weights / weights.sum(-1, keepdim=True)
        return self.experts(hidden_states, indices, weights)


def _model_with_boundary(name, module):
    model = nn.Module()
    model.config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"], num_experts=4)
    model.layer = nn.Module()
    setattr(model.layer, name, module)
    return model


def _check_automatic_cp_build(rank):
    """The builder plans and installs CP without a manually supplied wrapper."""
    torch.manual_seed(31)
    model = _model_with_boundary("self_attn", make_attention())
    reference = deepcopy(model.layer.self_attn)
    mesh = MeshContext(cp_size=2, dp_shard_size=2, edp_shard_size=2)
    mesh.build_meshs("cpu", 2)
    model, _ = _plan_and_apply_sharding(model, mesh, ShardingPlanner(), True, False)
    full = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    local = full.detach()[:, rank * 3:(rank + 1) * 3].clone().requires_grad_()
    actual, _ = model.layer.self_attn(local)
    expected, _ = reference(full)
    torch.testing.assert_close(actual, expected[:, rank * 3:(rank + 1) * 3])
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(local.grad, full.grad[:, rank * 3:(rank + 1) * 3])
    for name, parameter in model.layer.self_attn.named_parameters():
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)


def _check_automatic_ep_build(rank):
    """The builder owns expert sharding and Family factory installation."""
    torch.manual_seed(37)
    reference = _RoutedMoe()
    owner = deepcopy(reference)
    owner.experts = make_grouped_experts(owner.experts)
    model = _model_with_boundary("mlp", owner)
    mesh = MeshContext(dp_size=2, dp_shard_size=2, ep_size=2)
    mesh.build_meshs("cpu", 2)
    model, source_info = _plan_and_apply_sharding(model, mesh, ShardingPlanner(), True, False)
    if owner.experts.gate_up_proj.shape[0] != 2 or not source_info:
        raise AssertionError("The builder must shard Experts and return source layout metadata")
    torch.manual_seed(41 + rank)
    hidden = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    with patch("hyper_parallel.components.modules.grouped_experts.grouped_matmul",
               side_effect=cpu_grouped_matmul):
        actual = model.layer.mlp(hidden)
        actual.square().sum().backward()
    expected = reference(ref_hidden)
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(owner.gate.weight.grad, reference.gate.weight.grad)
    for name, parameter in owner.experts.named_parameters():
        reference_grad = dict(reference.experts.named_parameters())[name].grad
        dist.all_reduce(reference_grad)
        torch.testing.assert_close(parameter.grad.transpose(-1, -2), reference_grad[rank * 2:(rank + 1) * 2])


def _check_cp(rank):
    torch.manual_seed(13)
    module = make_attention()
    reference = deepcopy(module)
    full = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    local = full.detach()[:, rank * 3:(rank + 1) * 3].clone().requires_grad_()
    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: rank, get_group=lambda: dist.group.WORLD)
    adapter = get_family_parallel_adapter(type(module), "cp")
    _commit_forward_rewrite(adapter(module, None, None, mesh, None))
    actual, _ = module(local)
    expected, _ = reference(full)
    torch.testing.assert_close(actual, expected[:, rank * 3:(rank + 1) * 3])
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(local.grad, full.grad[:, rank * 3:(rank + 1) * 3])
    for name, parameter in module.named_parameters():
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)


def _check_ep(rank, mesh, *, replacement, route, split_dispatch, chunks):
    torch.manual_seed(17)
    source = PackedExperts()
    reference = deepcopy(source)
    module = make_grouped_experts(source) if replacement else source
    owner = nn.Module()
    owner.experts = module
    count, split = (3, 0) if route == "empty_source" else (5, 3)
    full_states = torch.randn(1, count, 4, dtype=torch.float64, requires_grad=True)
    full_weights = torch.randn(count, 2, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[3, 0], [2, 0], [1, 1], [3, 2], [0, 3]])[:count]
    if route == "empty_destination":
        ids = torch.zeros_like(ids)
    region = slice(0, split) if rank == 0 else slice(split, count)
    states = full_states.detach()[:, region].clone().requires_grad_()
    weights = full_weights.detach()[region].clone().requires_grad_()
    events = []

    def _router(block, hidden):
        del block, hidden
        return ids[region], weights

    def _overlap(block, hidden):
        del block
        events.append("overlap")
        return hidden * 0

    def _combine(block, hidden, routed, shared=None):
        del block, hidden
        return routed if shared is None else routed + shared

    def _expert_hook(experts, args):
        del experts, args
        events.append("experts")

    if replacement:
        adapter = get_family_parallel_adapter(type(module), "ep")
        compute = adapter(module=owner, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh,
                          router_fn=_router, combine=_combine,
                          overlap_fn=_overlap if split_dispatch else None)
    else:
        compute = build_ep_compute(
            owner, mesh, router_fn=_router, archetype_key="native_test", expected_attrs=["experts"],
            combine=_combine, use_grouped_gemm=False, overlap_fn=_overlap if split_dispatch else None)
    # Only parameter sharding is emulated; routing and execution use production recipes.
    for name, parameter in list(module.named_parameters()):
        setattr(module, name, nn.Parameter(parameter.detach()[rank * 2:(rank + 1) * 2].clone()))
    module.register_forward_pre_hook(_expert_hook)
    with patch.object(ep_execution, "_DISPATCH_CHUNKS_RAW", str(chunks)), patch(
            "hyper_parallel.components.modules.grouped_experts.grouped_matmul", side_effect=cpu_grouped_matmul):
        actual = compute(owner, states)
        actual.square().sum().backward()
    expected = reference(full_states, ids, full_weights)
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected[:, region])
    torch.testing.assert_close(states.grad, full_states.grad[:, region])
    torch.testing.assert_close(weights.grad, full_weights.grad[region])
    for name, parameter in module.named_parameters():
        expected_grad = dict(reference.named_parameters())[name].grad[rank * 2:(rank + 1) * 2]
        actual_grad = parameter.grad.transpose(-1, -2) if replacement else parameter.grad
        torch.testing.assert_close(actual_grad, expected_grad)
    assert events.count("experts") == chunks, f"Expert hook events: {events}, expected {chunks} calls"
    if split_dispatch:
        assert events[0] == "overlap", f"Shared branch ordering: {events}"


def test_existing_cp_ep_adapters_gloo():
    """Native and replacement modules retain existing execution and gradients."""
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        _check_automatic_cp_build(rank)
        _check_automatic_ep_build(rank)
        _check_cp(rank)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        for replacement in (False, True):
            for route in ("uneven", "empty_destination", "empty_source"):
                for split_dispatch in (False, True):
                    for chunks in (1, 2):
                        _check_ep(rank, mesh, replacement=replacement, route=route,
                                  split_dispatch=split_dispatch, chunks=chunks)
    finally:
        dist.destroy_process_group()

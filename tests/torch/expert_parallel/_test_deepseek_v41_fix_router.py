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
"""CPU/Gloo worker for DSV4.1 consuming the generic balanced router."""

import copy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn import functional
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_ep_compute_fn,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter


def _make_moe() -> torch.nn.Module:
    """Build real native gate and expert parameters without external model assets."""
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        hidden_size=4, moe_intermediate_size=6, n_routed_experts=8, num_experts_per_tok=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.25, swiglu_limit=1.5,
        v41_vision_enabled=True,
    )
    module = torch.nn.Module()
    module.gate = DeepseekV41TopKRouter(config)
    module.experts = DeepseekV4Experts(config)
    module.shared_experts = torch.nn.Linear(4, 4, bias=False)
    module.is_hash = False
    module.forward = lambda hidden_states, input_ids=None, image_mask=None: hidden_states
    for parameter in module.parameters():
        torch.nn.init.uniform_(parameter, -0.5, 0.5)
    return module


def _dense_oracle(
        module: torch.nn.Module,
        states: torch.Tensor,
        image_mask: torch.Tensor,
        fixed: bool,
) -> torch.Tensor:
    """Evaluate every expert explicitly, keeping the native learned Top-K weights."""
    _, weights, indices = module.gate(states, image_mask=image_mask)
    if fixed:
        # Each sender restarts its destination cycle, including a partial final cycle.
        local_experts = module.experts.num_experts // dist.get_world_size()
        expert_cycle = [
            rank * local_experts + offset
            for offset in range(local_experts) for rank in range(dist.get_world_size())
        ]
        sender_indices = torch.tensor([
            expert_cycle[slot % len(expert_cycle)] for slot in range(states.shape[1] * module.gate.top_k)
        ]).view(states.shape[1], module.gate.top_k)
        indices = sender_indices.repeat(states.shape[0], 1)
    flattened = states.reshape(-1, states.shape[-1])
    output = torch.zeros_like(flattened)
    for expert_id in range(module.experts.num_experts):
        gate, up = functional.linear(  # pylint: disable=not-callable
            flattened, module.experts.gate_up_proj[expert_id],
        ).chunk(2, dim=-1)
        activated = functional.silu(gate.clamp(max=1.5)) * up.clamp(min=-1.5, max=1.5)
        projected = functional.linear(  # pylint: disable=not-callable
            activated, module.experts.down_proj[expert_id],
        )
        output = output + projected * ((indices == expert_id) * weights).sum(-1, keepdim=True)
    return output.view_as(states) + module.shared_experts(states)


def _check_parity(ep_mesh: object, fixed: bool, tokens: int) -> None:
    """Compare outputs, token gradients, and sharded/replicated parameter gradients."""
    torch.manual_seed(2026)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    module = _make_moe()
    reference = copy.deepcopy(module)
    checkpoint_keys = set(module.state_dict())
    local_experts = module.experts.num_experts // world_size
    expert_start, expert_end = rank * local_experts, (rank + 1) * local_experts
    module.experts.gate_up_proj = torch.nn.Parameter(
        module.experts.get_parameter("gate_up_proj").detach()[expert_start:expert_end].clone(),
    )
    module.experts.down_proj = torch.nn.Parameter(
        module.experts.get_parameter("down_proj").detach()[expert_start:expert_end].clone(),
    )
    compute = deepseek_v41_ep_compute_fn(
        module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=ep_mesh, fix_router=fixed,
    )
    if set(module.state_dict()) != checkpoint_keys:
        raise AssertionError("The framework routing switch must preserve checkpoint keys")
    states = torch.randn(world_size, tokens, 4)
    upstream = torch.randn_like(states)
    images = torch.arange(world_size * tokens).view(world_size, tokens) % 3 == 0
    local_states = states[rank:rank + 1].clone().requires_grad_()
    reference_states = states.clone().requires_grad_()
    actual = compute(module, local_states, image_mask=images[rank:rank + 1])
    expected = _dense_oracle(reference, reference_states, images, fixed)
    torch.testing.assert_close(actual, expected[rank:rank + 1])
    (actual * upstream[rank:rank + 1]).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(local_states.grad, reference_states.grad[rank:rank + 1])
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in module.named_parameters():
        expected_gradient = reference_parameters[name].grad
        if expected_gradient is None:
            if parameter.grad is not None:
                raise AssertionError(f"Unexpected gradient for {name}: {parameter.grad}")
            continue
        if name.startswith("experts."):
            actual_gradient = parameter.grad
            expected_gradient = expected_gradient[expert_start:expert_end]
        else:
            actual_gradient = parameter.grad.detach().clone()
            dist.all_reduce(actual_gradient)
        torch.testing.assert_close(
            actual_gradient, expected_gradient, msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )


def test_framework_balanced_router_ep_parity() -> None:
    """
    Feature: DSV4.1 learned/balanced EP dispatch.
    Description: Exercise complete and partial expert cycles through real all-to-all.
    Expectation: Outputs, input gradients and all parameter gradients match a dense oracle.
    """
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This parity worker requires exactly two CPU ranks")
        ep_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        for fixed in (False, True):
            for tokens in (3, 4, 5):
                _check_parity(ep_mesh, fixed, tokens)
    finally:
        dist.destroy_process_group()

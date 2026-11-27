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
"""Ascend forward/backward parity for staged local and EP grouped experts."""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from hyper_parallel import init_device_mesh
from hyper_parallel.components.ops import moe_token_permute
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
)
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext


_EXPERTS = 4
_HIDDEN = 64
_INTERMEDIATE = 96
_TOKENS = 12
_TOP_K = 2


def _reference_forward(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    probabilities: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Evaluate the authoritative packed-expert formula route by route."""
    outputs = []
    for token_index, hidden in enumerate(hidden_states):
        routes = []
        for route_index, expert_id in enumerate(expert_ids[token_index]):
            expert = int(expert_id)
            gate, up = F.linear(  # pylint: disable=not-callable
                hidden,
                gate_up_proj[expert],
            ).chunk(2, dim=-1)
            activated = F.silu(gate) * up
            probability = probabilities[token_index, route_index]
            if mode == "before_fc2":
                activated = activated * probability
            output = F.linear(activated, down_proj[expert])  # pylint: disable=not-callable
            routes.append(output if mode == "before_fc2" else output * probability)
        outputs.append(torch.stack(routes).sum(dim=0))
    return torch.stack(outputs)


def _inputs(device: torch.device, rank: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic rank-specific routes with every expert represented."""
    generator = torch.Generator(device="cpu").manual_seed(1009 + rank)
    hidden = torch.randn(_TOKENS, _HIDDEN, generator=generator).to(device=device, dtype=torch.bfloat16)
    hidden.requires_grad_()
    token = torch.arange(_TOKENS, device=device)
    expert_ids = torch.stack(((token + rank) % _EXPERTS, (token + rank + 2) % _EXPERTS), dim=1)
    logits = torch.randn(_TOKENS, _TOP_K, generator=generator).to(device=device, dtype=torch.bfloat16)
    probabilities = logits.softmax(-1).requires_grad_()
    upstream = torch.randn(_TOKENS, _HIDDEN, generator=generator).to(device=device, dtype=torch.bfloat16)
    return hidden, expert_ids, probabilities, upstream


def _full_weights(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one deterministic global expert table on every rank."""
    generator = torch.Generator(device="cpu").manual_seed(2027)
    gate_up = torch.randn(_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN, generator=generator) * 0.02
    down = torch.randn(_EXPERTS, _HIDDEN, _INTERMEDIATE, generator=generator) * 0.02
    return gate_up.to(device=device, dtype=torch.bfloat16), down.to(device=device, dtype=torch.bfloat16)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Compare BF16 grouped-GEMM results and gradients on the host."""
    actual_float = actual.detach().float().cpu()
    expected_float = expected.detach().float().cpu()
    difference = actual_float - expected_float
    max_abs = difference.abs().max().item()
    relative_l2 = difference.norm().item() / max(expected_float.norm().item(), 1.0e-12)
    torch.testing.assert_close(
        actual_float,
        expected_float,
        rtol=1.0e-2,
        atol=2.0e-3,
        msg=f"{label}: max_abs={max_abs:.6e}, relative_l2={relative_l2:.6e}",
    )


def _local_case(device: torch.device, mode: str) -> None:
    """Compare one fused local permutation/GMM path with the direct formula."""
    config = GroupedExpertsConfig(_EXPERTS, _HIDDEN, _INTERMEDIATE, routing_weight_mode=mode, num_layers=1)
    candidate = GroupedExpertsModule(config=config).to(device=device, dtype=torch.bfloat16)
    gate_up, down = _full_weights(device)
    candidate.gate_up_proj.data.copy_(gate_up)
    candidate.down_proj.data.copy_(down)
    hidden, expert_ids, probabilities, upstream = _inputs(device)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_probabilities = probabilities.detach().clone().requires_grad_()
    reference_gate_up = gate_up.detach().clone().requires_grad_()
    reference_down = down.detach().clone().requires_grad_()

    with patch(
        "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.moe_token_permute",
        wraps=moe_token_permute,
    ) as permute:
        actual = candidate(hidden, expert_ids, probabilities)
    expected = _reference_forward(
        reference_hidden,
        expert_ids,
        reference_probabilities,
        reference_gate_up,
        reference_down,
        mode,
    )
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()

    _assert_close(actual, expected, f"local {mode} output")
    _assert_close(hidden.grad, reference_hidden.grad, f"local {mode} input gradient")
    _assert_close(probabilities.grad, reference_probabilities.grad, f"local {mode} probability gradient")
    _assert_close(candidate.gate_up_proj.grad, reference_gate_up.grad, f"local {mode} FC1 gradient")
    _assert_close(candidate.down_proj.grad, reference_down.grad, f"local {mode} FC2 gradient")
    assert permute.call_count == 1, f"EP=1 must permute once, got calls={permute.call_count}"


def test_local_grouped_experts_forward_backward() -> None:
    """EP=1 fused grouped experts match the direct formula in both weighting modes."""
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    for mode in ("output", "before_fc2"):
        _local_case(device, mode)


def _initialize_distributed() -> tuple[torch.device, int, int]:
    """Initialize one HCCL EP group and select this worker's device."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    return device, local_rank, world_size


def _ep_case(device: torch.device, rank: int, world_size: int, mode: str) -> None:
    """Compare four-rank transport with a global reference on every source rank."""
    if world_size != _EXPERTS:
        raise ValueError(f"test requires world_size={_EXPERTS}, got {world_size}")
    config = GroupedExpertsConfig(_EXPERTS, _HIDDEN, _INTERMEDIATE, routing_weight_mode=mode, num_layers=1)
    candidate = EPGroupedExperts(config=config).to(device=device, dtype=torch.bfloat16)
    gate_up, down = _full_weights(device)
    candidate.gate_up_proj = nn.Parameter(gate_up[rank:rank + 1].clone())
    candidate.down_proj = nn.Parameter(down[rank:rank + 1].clone())
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("ep",))
    candidate.ep_context = EPContext(mesh, num_experts=_EXPERTS)
    hidden, expert_ids, probabilities, upstream = _inputs(device, rank)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_probabilities = probabilities.detach().clone().requires_grad_()
    reference_gate_up = gate_up.detach().clone().requires_grad_()
    reference_down = down.detach().clone().requires_grad_()

    with patch(
        "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.moe_token_permute",
        wraps=moe_token_permute,
    ) as source_permute, patch(
        "hyper_parallel.distributed.expert_parallel.ep_context.moe_token_permute",
        wraps=moe_token_permute,
    ) as receive_permute:
        actual = candidate(hidden, expert_ids, probabilities)
    expected = _reference_forward(
        reference_hidden,
        expert_ids,
        reference_probabilities,
        reference_gate_up,
        reference_down,
        mode,
    )
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    dist.all_reduce(reference_gate_up.grad)
    dist.all_reduce(reference_down.grad)

    _assert_close(actual, expected, f"EP {mode} output rank {rank}")
    _assert_close(hidden.grad, reference_hidden.grad, f"EP {mode} input gradient rank {rank}")
    _assert_close(probabilities.grad, reference_probabilities.grad, f"EP {mode} probability gradient rank {rank}")
    _assert_close(candidate.gate_up_proj.grad, reference_gate_up.grad[rank:rank + 1], f"EP {mode} FC1 rank {rank}")
    _assert_close(candidate.down_proj.grad, reference_down.grad[rank:rank + 1], f"EP {mode} FC2 rank {rank}")
    assert source_permute.call_count == 1, (
        f"EP source must permute once: rank={rank}, calls={source_permute.call_count}"
    )
    assert receive_permute.call_count == 1, (
        f"EP receive must permute once: rank={rank}, calls={receive_permute.call_count}"
    )


def test_ep_grouped_experts_forward_backward() -> None:
    """EP>1 adds one receive permutation and matches both direct formulas."""
    device, rank, world_size = _initialize_distributed()
    try:
        for mode in ("output", "before_fc2"):
            _ep_case(device, rank, world_size, mode)
            dist.barrier()
    finally:
        clear_device_mesh_cache()
        dist.destroy_process_group()

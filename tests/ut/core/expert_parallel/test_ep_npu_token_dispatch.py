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

"""Unit tests for the NPU EP token dispatcher."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from hyper_parallel.distributed.expert_parallel import experts as ep_experts


class _SingleRankGroup:
    """Minimal process-group stand-in for a single-rank EP unit test."""

    @staticmethod
    def size() -> int:
        """Return the single-rank group size."""
        return 1


class _MockNpuDevice(str):
    """Report NPU metadata while keeping mocked-kernel tensor allocations on CPU."""

    type = "npu"
    index = None


class _ExpertMajorExperts:
    """Deterministic local expert computation used by the dispatcher test."""

    def __init__(self, num_experts: int) -> None:
        """Store the number of local experts."""
        self.local_expert_count = num_experts

    def __call__(
        self,
        states: torch.Tensor,
        local_expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Scale each expert-major token by its one-based expert index."""
        scales = local_expert_indices.to(states.dtype) + 1
        return states * scales.unsqueeze(-1)


class _BindableExpertMajorExperts(nn.Module):
    """Expert module exposing the grouped-compute contract used by adapters."""

    def __init__(self, num_experts: int) -> None:
        """Store the global expert count used by the binder."""
        super().__init__()
        self.num_experts = num_experts

    @staticmethod
    def forward_expert_major(
        states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a deterministic expert-major transformation."""
        del tokens_per_expert
        return states + 1


def _reference_token_permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_indices = torch.arange(tokens.shape[0]).repeat_interleave(indices.shape[1])
    order = indices.reshape(-1).argsort(stable=True)
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.numel())
    return tokens[source_indices[order]], inverse_order.to(torch.int32)


def _reference_token_unpermute(
    permuted_tokens: torch.Tensor,
    order: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    restored_routes = permuted_tokens[order]
    weighted = restored_routes * probabilities.reshape(-1, 1)
    return weighted.view(probabilities.shape[0], probabilities.shape[1], -1).sum(dim=1)


def test_token_dispatch_preserves_local_expert_forward():
    """Fused token routing must keep the established expert binding contract."""
    experts = _BindableExpertMajorExperts(num_experts=4)
    module = SimpleNamespace(
        experts=experts,
        config=SimpleNamespace(hidden_act="silu"),
    )

    ep_experts.bind_local_expert_forward(
        module,
        ep_size=2,
        use_grouped_gemm=True,
    )

    states = torch.arange(8, dtype=torch.float32).view(2, 4)
    local_expert_indices = torch.tensor([1, 0])
    torch.testing.assert_close(experts(states, local_expert_indices), states + 1)
    assert experts.local_expert_count == 2, (
        f"Expected two local experts, got {experts.local_expert_count}"
    )


@pytest.mark.parametrize("token_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("enabled", [None, False])
def test_npu_ep_dispatcher_matches_reference(monkeypatch, token_dtype, enabled):
    """Preserve FP32 routing weights and weighted Top-K forward/backward semantics."""
    permute = Mock(side_effect=_reference_token_permute)
    monkeypatch.setitem(
        sys.modules,
        "hyper_parallel.components.functional.moe_token_permute",
        SimpleNamespace(moe_token_permute=permute),
    )
    monkeypatch.setattr(torch.Tensor, "device", property(lambda tensor: _MockNpuDevice("cpu")))

    def checked_unpermute(
        states: torch.Tensor,
        order: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Match the NPU output dtype without rounding the router probabilities."""
        assert probabilities is topk_weights, (
            f"Expected original routing weights ({topk_weights.dtype}), got {probabilities.dtype}"
        )
        return _reference_token_unpermute(states, order, probabilities).to(states.dtype)

    unpermute = Mock(side_effect=checked_unpermute)
    monkeypatch.setitem(
        sys.modules,
        "hyper_parallel.components.functional.moe_token_unpermute",
        SimpleNamespace(moe_token_unpermute=unpermute),
    )
    monkeypatch.setattr(ep_experts.dist, "get_rank", lambda group: 0)

    monkeypatch.setattr(ep_experts, "ep_exchange_counts", lambda counts, group: counts)
    monkeypatch.setattr(
        ep_experts,
        "ep_all_to_all",
        lambda tensor, send_counts, receive_counts, group: tensor,
    )

    hidden_states = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        dtype=token_dtype,
        requires_grad=True,
    )
    topk_indices = torch.tensor([[2, 0], [1, 2], [0, 1]])
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float32, requires_grad=True
    )
    module = SimpleNamespace(experts=_ExpertMajorExperts(num_experts=3))

    options = {} if enabled is None else {"use_fused_moe_token_permute_and_unpermute": enabled}
    output = ep_experts.ep_routed_forward(
        module,
        hidden_states,
        router_fn=lambda owner, states: (topk_indices, topk_weights),
        ep_group=_SingleRankGroup(),
        **options,
    )
    if enabled is False:
        permute.assert_not_called()
        unpermute.assert_not_called()
    else:
        permute.assert_called_once()
        unpermute.assert_called_once()

    scales = topk_indices.to(hidden_states.dtype) + 1
    reference_states = hidden_states.detach().clone().requires_grad_()
    reference_weights = topk_weights.detach().clone().requires_grad_()
    expert_routes = reference_states.flatten(0, 1).unsqueeze(1) * scales.unsqueeze(-1)
    weights = reference_weights if enabled is None else reference_weights.to(token_dtype)
    expected = (expert_routes * weights.unsqueeze(-1)).sum(dim=1)
    expected = expected.to(token_dtype).view_as(hidden_states)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        hidden_states.grad, reference_states.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        topk_weights.grad, reference_weights.grad, rtol=0, atol=0
    )

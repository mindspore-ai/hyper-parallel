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

from types import SimpleNamespace

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


class _ExpertMajorExperts:
    """Deterministic expert-major computation used by the dispatcher test."""

    def __init__(self, num_experts: int) -> None:
        """Store the number of local experts."""
        self.local_expert_count = num_experts

    def __call__(
        self,
        states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Scale each expert-major token by its one-based expert index."""
        expert_scales = torch.arange(
            1,
            self.local_expert_count + 1,
            dtype=states.dtype,
            device=states.device,
        )
        scales = torch.repeat_interleave(expert_scales, tokens_per_expert)
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
    return tokens[source_indices[order]], inverse_order


def _reference_token_unpermute(
    permuted_tokens: torch.Tensor,
    order: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    restored_routes = permuted_tokens[order]
    weighted = restored_routes * probabilities.reshape(-1, 1)
    return weighted.view(probabilities.shape[0], probabilities.shape[1], -1).sum(dim=1)


def test_reorder_variable_chunks_round_trip():
    """Source-major and expert-major chunk permutations must be inverse."""
    chunk_sizes = [2, 0, 1, 3, 2, 1]
    source_to_expert = [0, 2, 4, 1, 3, 5]
    expert_to_source = [0, 3, 1, 4, 2, 5]
    tensor = torch.arange(sum(chunk_sizes)).unsqueeze(-1)

    expert_major = ep_experts._reorder_variable_chunks(
        tensor,
        chunk_sizes,
        source_to_expert,
    )
    restored = ep_experts._reorder_variable_chunks(
        expert_major,
        [chunk_sizes[index] for index in source_to_expert],
        expert_to_source,
    )

    torch.testing.assert_close(restored, tensor)


def test_bind_npu_dispatch_installs_expert_major_forward():
    """The public binder must preserve the model-adapter keyword contract."""
    experts = _BindableExpertMajorExperts(num_experts=4)
    module = SimpleNamespace(
        experts=experts,
        config=SimpleNamespace(hidden_act="silu"),
    )

    ep_experts.bind_local_expert_forward(
        module,
        ep_size=2,
        use_grouped_gemm=True,
        use_npu_moe_token_dispatch=True,
    )

    states = torch.arange(8, dtype=torch.float32).view(2, 4)
    counts = torch.tensor([1, 1])
    torch.testing.assert_close(experts(states, counts), states + 1)
    assert experts.local_expert_count == 2, (
        f"Expected two local experts, got {experts.local_expert_count}"
    )


@pytest.mark.parametrize("token_dtype", [torch.float32, torch.bfloat16])
def test_npu_ep_dispatcher_matches_reference(monkeypatch, token_dtype):
    """Preserve FP32 routing weights and weighted Top-K forward/backward semantics."""
    monkeypatch.setattr(ep_experts, "_npu_moe_token_permute", _reference_token_permute)

    def checked_unpermute(states, order, probabilities):
        """Match the NPU output dtype without rounding the router probabilities."""
        assert probabilities is topk_weights, (
            f"Expected original routing weights ({topk_weights.dtype}), got {probabilities.dtype}"
        )
        return _reference_token_unpermute(states, order, probabilities).to(states.dtype)

    monkeypatch.setattr(ep_experts, "_npu_moe_token_unpermute", checked_unpermute)
    monkeypatch.setattr(ep_experts.dist, "get_rank", lambda group: 0)

    def fake_all_gather(
        output: torch.Tensor,
        local_counts: torch.Tensor,
        group: object,
    ) -> None:
        """Copy the single rank's counts into the gathered output."""
        del group
        output[0].copy_(local_counts)

    monkeypatch.setattr(ep_experts.dist, "all_gather_into_tensor", fake_all_gather)
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

    output = ep_experts.ep_routed_forward(
        module,
        hidden_states,
        router_fn=lambda owner, states: (topk_indices, topk_weights),
        ep_group=_SingleRankGroup(),
        use_npu_moe_token_dispatch=True,
    )

    scales = topk_indices.to(hidden_states.dtype) + 1
    reference_states = hidden_states.detach().clone().requires_grad_()
    reference_weights = topk_weights.detach().clone().requires_grad_()
    expert_routes = reference_states.flatten(0, 1).unsqueeze(1) * scales.unsqueeze(-1)
    expected = (expert_routes * reference_weights.unsqueeze(-1)).sum(dim=1)
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

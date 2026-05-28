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
"""Distributed worker tests for sequence_partition_group in _compute_load_balance_loss.

Launched from ``test_moe_load_balance_seq_partition.py`` via ``parallel_run``.

Test strategy:
  - 4-card gloo process group simulates TP+SP or CP ranks on CPU.
  - Each rank produces different routing data (rank-specific seed) so
    expert_fraction differs before sync.
  - ``expert_fraction`` is all-reduced across the group (synchronized),
    while ``expert_prob`` is kept local for gradient flow.
  - Loss is verified against manual computation using the synchronized
    expert_fraction and local expert_prob.

Test IDs:
  LB-S03: expert_fraction all-reduced — matches manual global computation
  LB-S04: Loss with group uses global normalization (smaller than without)
"""
import torch
import torch.distributed as dist

from hyper_parallel.platform.torch.common.moe import _compute_load_balance_loss
from tests.torch.utils import init_dist_gloo


def _make_routing_data(
    num_experts: int = 4,
    top_k: int = 2,
    num_tokens: int = 16,
    seed: int = 42,
):
    """Create per-rank routing data (top_scores + selected_experts).

    Uses rank-specific seed so each rank produces different expert assignments.

    Args:
        num_experts: Total number of experts.
        top_k: Top-k routing.
        num_tokens: Number of tokens.
        seed: Base seed (rank is added to make per-rank data different).

    Returns:
        Tuple of (top_scores, selected_experts) tensors.
    """
    rank = dist.get_rank()
    torch.manual_seed(seed + rank)
    logits = torch.randn(num_tokens, num_experts)
    top_scores, selected_experts = logits.topk(top_k, dim=-1)
    top_scores = torch.sigmoid(top_scores)
    return top_scores, selected_experts


def _manual_load_balance_loss(top_scores, selected_experts, num_experts, group):
    """Manually compute expected loss with global expert_fraction.

    Replicates the logic of _compute_load_balance_loss step-by-step,
    doing the all-reduce explicitly so we can verify intermediate states.

    Returns:
        Tuple of (expected_loss, local_expert_fraction_before_sync,
                  global_expert_fraction).
    """
    num_tokens, top_k = top_scores.shape
    world_size = dist.get_world_size(group)

    flat_experts = selected_experts.flatten()
    flat_scores = top_scores.flatten()

    # Local expert_fraction (before sync)
    local_expert_fraction = torch.zeros(
        num_experts, dtype=top_scores.dtype, device=top_scores.device
    )
    local_expert_fraction.scatter_add_(0, flat_experts, torch.ones_like(flat_scores))

    local_before_sync = local_expert_fraction.clone()

    # All-reduce to get global token counts
    dist.all_reduce(local_expert_fraction, group=group)

    # Global expert_fraction
    global_expert_fraction = local_expert_fraction / (num_tokens * world_size * top_k)

    # Local expert_prob (same as _compute_load_balance_loss: local scores, global denominator)
    expert_prob = torch.zeros(
        num_experts, dtype=top_scores.dtype, device=top_scores.device
    )
    expert_prob.scatter_add_(0, flat_experts, flat_scores)
    expert_prob = expert_prob / (num_tokens * world_size)

    expected_loss = num_experts * (global_expert_fraction * expert_prob).sum()
    return expected_loss, local_before_sync, global_expert_fraction


# ---------------------------------------------------------------------------
# LB-S03: expert_fraction all-reduced — matches manual global computation
# ---------------------------------------------------------------------------


def test_lbs03_expert_fraction_sync_matches_manual():
    """LB-S03: Loss with sequence_partition_group matches manual global computation.

    Each rank has different routing data. _compute_load_balance_loss with
    sequence_partition_group all-reduces expert_fraction internally. The result
    should match a manual computation where we all-reduce expert_fraction
    ourselves and apply the same formula.
    """
    init_dist_gloo()
    group = dist.group.WORLD
    num_experts = 4

    top_scores, selected_experts = _make_routing_data(
        num_experts=num_experts, top_k=2, num_tokens=16, seed=42
    )

    # Compute loss via the function under test
    loss = _compute_load_balance_loss(
        top_scores,
        selected_experts,
        num_experts,
        sequence_partition_group=group,
    )

    # Compute expected loss manually
    expected_loss, _, _ = _manual_load_balance_loss(
        top_scores, selected_experts, num_experts, group
    )

    assert torch.allclose(loss, expected_loss, atol=1e-6), (
        f"LB-S03: Loss with sequence_partition_group does not match manual "
        f"computation: function={loss.item():.8f}, manual={expected_loss.item():.8f}"
    )


# ---------------------------------------------------------------------------
# LB-S04: Loss with group uses global normalization (different from no-group)
# ---------------------------------------------------------------------------


def test_lbs04_global_normalization_changes_loss():
    """LB-S04: Loss with sequence_partition_group differs from no-group on same input.

    With the group, both expert_fraction and expert_prob are divided by
    (num_tokens * num_sub_sequence), using global token count for normalization.
    Without the group, only local num_tokens is used. Since expert_fraction is
    also all-reduced (different values), the loss values should differ.

    Additionally, verify that expert_fraction before sync differs across ranks
    (precondition: each rank has different input), and that the global
    normalization produces a smaller loss than the un-normalized version.
    """
    init_dist_gloo()
    group = dist.group.WORLD
    num_experts = 4

    top_scores, selected_experts = _make_routing_data(
        num_experts=num_experts, top_k=2, num_tokens=16, seed=100
    )

    # Precondition: expert_fraction differs across ranks before sync
    local_frac = torch.zeros(num_experts, dtype=top_scores.dtype)
    flat_experts = selected_experts.flatten()
    local_frac.scatter_add_(0, flat_experts, torch.ones_like(top_scores.flatten()))
    all_frac = [torch.zeros_like(local_frac) for _ in range(dist.get_world_size())]
    dist.all_gather(all_frac, local_frac)
    fracs_differ = not all(
        torch.allclose(all_frac[0], all_frac[i], atol=1e-6)
        for i in range(1, len(all_frac))
    )
    assert fracs_differ, (
        "LB-S04 precondition: expert_fraction should differ across ranks "
        "before sync (each rank has different input)"
    )

    # Loss WITHOUT sequence_partition_group
    loss_nosync = _compute_load_balance_loss(
        top_scores, selected_experts, num_experts, sequence_partition_group=None
    )

    # Loss WITH sequence_partition_group
    loss_sync = _compute_load_balance_loss(
        top_scores, selected_experts, num_experts, sequence_partition_group=group
    )

    # The two losses should differ (different normalization + different expert_fraction)
    assert not torch.allclose(loss_sync, loss_nosync, atol=1e-6), (
        f"LB-S04: Loss with and without sequence_partition_group should differ, "
        f"but both are {loss_sync.item():.8f}"
    )

    # With global normalization, both expert_fraction and expert_prob have
    # larger denominators, so loss should be smaller
    assert loss_sync < loss_nosync, (
        f"LB-S04: Loss with global normalization should be smaller: "
        f"sync={loss_sync.item():.8f}, nosync={loss_nosync.item():.8f}"
    )

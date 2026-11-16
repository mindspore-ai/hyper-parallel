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
"""Distributed worker tests for router load synchronization (torchrun, Gloo CPU).

Launched from ``test_moe_load_balance_distributed.py`` via ``parallel_run``.

Test strategy:
  - 4-card gloo process group simulates DP ranks on CPU.
  - Each rank supplies deterministic expert counts to the shared sync function.
  - After all_reduce via dp_group, all ranks should see identical tokens_per_expert
    and produce identical expert_bias updates.
  - With identical local counts, each rank updates its bias independently.

Test IDs:
  LB-D01: 4-card DP — expert_bias identical after sync
  LB-D02: Compare with/without all_reduce — unsynced diverge, synced match
  LB-D03: Pure EP (no dp_group) — independent update, no error
"""
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.modules.moe_utils import sync_and_update_expert_bias
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from tests.torch.utils import init_dist_gloo


def _make_dp_group() -> GroupInfo:
    """Create a DP GroupInfo spanning all ranks."""
    group = dist.group.WORLD
    return GroupInfo("dp_group", group, dist.get_world_size())


class _RouterWithLoad(nn.Module):
    """Supply the expert-bias update contract used by the shared sync function."""

    def __init__(self, expert_offset: int) -> None:
        super().__init__()
        counts = torch.roll(torch.tensor([8, 8, 0, 0]), shifts=expert_offset)
        # Keep the global load uneven so the synchronized bias update is observable.
        counts[0] += 4
        self.register_buffer("tokens_per_expert", counts)
        self.register_buffer("bias", torch.zeros(counts.shape))

    def update_expert_bias(self, lr: float, num_recomputations: int = 1) -> None:
        """Apply the requested bias step to experts above or below the mean load."""
        counts = self.tokens_per_expert.float() / num_recomputations
        self.bias.add_((counts.mean() - counts).sign(), alpha=lr)
        self.tokens_per_expert.zero_()


# ---------------------------------------------------------------------------
# LB-D01: 4-card DP sync — expert_bias identical across ranks
# ---------------------------------------------------------------------------


def test_lbd01_dp_sync_expert_bias_identical():
    """LB-D01: After sync_and_update_expert_bias with dp_group, all ranks have identical expert_bias.

    Each rank starts with different token counts. After all_reduce, all ranks
    see the global sum and produce the same bias update.
    """
    init_dist_gloo()
    dp_group = _make_dp_group()
    router = _RouterWithLoad(dist.get_rank())

    # Verify tokens_per_expert differs across ranks BEFORE sync
    local_tokens = router.tokens_per_expert.clone()
    all_tokens = [torch.zeros_like(local_tokens) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tokens, local_tokens)
    ranks_differ = not all(
        torch.allclose(all_tokens[0], all_tokens[i]) for i in range(1, len(all_tokens))
    )
    assert ranks_differ, (
        "LB-D01 precondition: tokens_per_expert should differ across ranks before sync"
    )

    # Sync and update
    sync_and_update_expert_bias(router, lr=1e-3, dp_group=dp_group)

    # After sync, all ranks should have identical expert_bias
    local_bias = router.bias.clone()
    torch.testing.assert_close(local_bias, torch.tensor([-1e-3, 1e-3, 1e-3, 1e-3]))
    all_bias = [torch.zeros_like(local_bias) for _ in range(dist.get_world_size())]
    dist.all_gather(all_bias, local_bias)
    for i in range(1, len(all_bias)):
        assert torch.allclose(all_bias[0], all_bias[i], atol=1e-6), (
            f"LB-D01: expert_bias differs between rank 0 and rank {i}: "
            f"rank0={all_bias[0]}, rank{i}={all_bias[i]}"
        )

    # tokens_per_expert should be zeroed after update
    assert (router.tokens_per_expert == 0).all(), (
        f"LB-D01: tokens_per_expert should be zeroed after update, "
        f"got {router.tokens_per_expert}"
    )


# ---------------------------------------------------------------------------
# LB-D02: Compare with/without all_reduce
# ---------------------------------------------------------------------------


def test_lbd02_sync_vs_nosync():
    """LB-D02: Without sync, expert_bias diverges across ranks; with sync, they match.

    Two router instances per rank:
      - router_nosync: update without dp_group → rank-local bias
      - router_sync: update with dp_group → global bias (identical)
    """
    init_dist_gloo()
    dp_group = _make_dp_group()

    rank = dist.get_rank()
    router_nosync = _RouterWithLoad(rank)
    router_sync = _RouterWithLoad(rank)

    # Update without sync
    sync_and_update_expert_bias(router_nosync, lr=1e-3)

    # Update with sync
    sync_and_update_expert_bias(router_sync, lr=1e-3, dp_group=dp_group)

    # Gather both biases
    nosync_bias = router_nosync.bias.clone()
    sync_bias = router_sync.bias.clone()
    all_nosync = [torch.zeros_like(nosync_bias) for _ in range(dist.get_world_size())]
    all_sync = [torch.zeros_like(sync_bias) for _ in range(dist.get_world_size())]
    dist.all_gather(all_nosync, nosync_bias)
    dist.all_gather(all_sync, sync_bias)

    # nosync biases should differ across ranks (different token distributions)
    nosync_differ = not all(
        torch.allclose(all_nosync[0], all_nosync[i], atol=1e-6)
        for i in range(1, len(all_nosync))
    )
    assert nosync_differ, (
        "LB-D02: Without sync, expert_bias should differ across ranks "
        "(each rank has different token distribution)"
    )

    # sync biases should be identical
    for i in range(1, len(all_sync)):
        assert torch.allclose(all_sync[0], all_sync[i], atol=1e-6), (
            f"LB-D02: With sync, expert_bias should be identical across ranks, "
            f"but rank0={all_sync[0]}, rank{i}={all_sync[i]}"
        )


# ---------------------------------------------------------------------------
# LB-D03: Pure EP scenario — no dp_group, independent update
# ---------------------------------------------------------------------------


def test_lbd03_pure_ep_no_dp_group():
    """LB-D03: Pure EP scenario without dp_group — each rank updates independently, no error.

    All ranks receive the same routing data in this case, so updating the
    bias from local counts gives the same result without a synchronization group.
    """
    init_dist_gloo()

    router = _RouterWithLoad(0)

    # Update without any group
    sync_and_update_expert_bias(router, lr=1e-3)

    # In EP, tokens_per_expert is identical across ranks (same model, same input)
    # so bias should also be identical even without sync
    local_bias = router.bias.clone()
    all_bias = [torch.zeros_like(local_bias) for _ in range(dist.get_world_size())]
    dist.all_gather(all_bias, local_bias)

    # All ranks should have identical bias (same input → same tokens → same update)
    for i in range(1, len(all_bias)):
        assert torch.allclose(all_bias[0], all_bias[i], atol=1e-6), (
            f"LB-D03: In pure EP (same input), expert_bias should be identical "
            f"even without sync: rank0={all_bias[0]}, rank{i}={all_bias[i]}"
        )

    # tokens_per_expert reset
    assert (router.tokens_per_expert == 0).all(), (
        f"LB-D03: tokens_per_expert should be zeroed after update, "
        f"got {router.tokens_per_expert}"
    )

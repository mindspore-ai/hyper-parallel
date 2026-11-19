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
"""Send one hidden vector per (token, destination), preserving every router score."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.distributed.expert_parallel.moonep_planner import MoonEPPlan


@dataclass(frozen=True)
class MoonEPTokenDedup:
    """Peer splits and selection fanout shared by dispatch and its adjoint."""

    source_tokens: torch.Tensor
    receive_rows: torch.Tensor
    send: list[int]
    recv: list[int]

    def dispatch(self, hidden: torch.Tensor, group: Any) -> torch.Tensor:
        """Expand unique received vectors into source-major expert selections.

        Args:
            hidden: Source token vectors before expert selection fanout.
            group: Expert-parallel process group.
        """
        received = ep_all_to_all(hidden.index_select(0, self.source_tokens), self.send, self.recv, group)
        return received.index_select(0, self.receive_rows)

    def combine(self, weighted: torch.Tensor, tokens: int, group: Any) -> torch.Tensor:
        """Sum weighted selections before returning unique vectors to their sources.

        Args:
            weighted: Weighted expert outputs in received selection order.
            tokens: Number of token rows on the source rank.
            group: Expert-parallel process group.
        """
        compact = weighted.new_zeros((sum(self.recv), weighted.shape[-1]))
        compact = compact.index_add(0, self.receive_rows, weighted)
        returned = ep_all_to_all(compact, self.recv, self.send, group)
        return weighted.new_zeros((tokens, weighted.shape[-1])).index_add(0, self.source_tokens, returned)


@torch.no_grad()
def compile_token_dedup(plan: MoonEPPlan, top_k: int, group: Any) -> MoonEPTokenDedup:
    """Exchange one count header per peer and one local unique-row index per selection.

    Large selection maps remain on the device. Only peer counts are copied to
    the host for collective split sizes. The result is also reused in recompute.

    Args:
        plan: Original selection-level dispatch plan.
        top_k: Number of expert selections per source token.
        group: Expert-parallel process group exchanging selection metadata.
    """
    order = plan.source_order
    ranks = len(plan.token_send)
    tokens = max(1, order.numel() // top_k)
    send_counts = order.new_tensor(plan.token_send)
    peers = torch.repeat_interleave(torch.arange(ranks, device=order.device), send_counts,
                                    output_size=order.numel())
    keys, inverse = torch.unique(peers * tokens + order // top_k, sorted=True, return_inverse=True)
    unique_counts = torch.bincount(keys // tokens, minlength=ranks)
    offsets = unique_counts.cumsum(0) - unique_counts
    relative = inverse - offsets.index_select(0, peers)

    # Each peer packet starts with its unique-vector count, including empty peers.
    headers = send_counts.cumsum(0) - send_counts + torch.arange(ranks, device=order.device)
    packet = order.new_empty(order.numel() + ranks)
    packet[headers] = unique_counts
    packet[torch.arange(order.numel(), device=order.device) + peers + 1] = relative
    recv_counts = order.new_tensor(plan.token_recv)
    received = order.new_empty(sum(plan.token_recv) + ranks)
    dist.all_to_all_single(received, packet, [n + 1 for n in plan.token_recv],
                           [n + 1 for n in plan.token_send], group=group)
    recv_headers = recv_counts.cumsum(0) - recv_counts + torch.arange(ranks, device=order.device)
    unique_recv = received.index_select(0, recv_headers)
    counts = torch.stack((unique_counts, unique_recv)).cpu().tolist()
    recv_peers = torch.repeat_interleave(torch.arange(ranks, device=order.device), recv_counts,
                                         output_size=sum(plan.token_recv))
    positions = torch.arange(sum(plan.token_recv), device=order.device) + recv_peers + 1
    lookup = received.index_select(0, positions) + (unique_recv.cumsum(0) - unique_recv)[recv_peers]
    return MoonEPTokenDedup(keys % tokens, lookup, counts[0], counts[1])

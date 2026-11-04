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
"""Explicit experimental KDA head-packet producer and one-sided state transport."""

from __future__ import annotations

# Context parallel is Torch-only; this optional component specifically targets NPU.
# pylint: disable=forbidden-backend-import

from typing import Any

import torch

from hyper_parallel.core.context_parallel.onesided import StatePullArena
from hyper_parallel.components.functional._triton.kimi_delta_attention.state_summary import (
    kda_split_state_summary_kernel,
    kda_state_transition_kernel,
    kda_state_grad_ext_kernel,
    kda_bounded_state_summary_kernel,
    kda_bounded_state_gradient_kernel,
)


class KDAOneSidedP2P:
    """Own KDA's optional SHMEM state transport; construction and close are collective.

    This experimental adapter uses group-local peers without a power-of-two
    restriction. It targets A2, B1/K=V128 and BT64 with configurable heads.
    It preserves full-head prepare and local h/dhu.
    With H96 and four packets, two groups each publish two H24 packets.
    Other head counts use balanced nonempty ranges. Setting producer_groups=1
    supplies the full-summary control. No global interception
    or implicit transport fallback is used.

    Args:
        group: Chronological HCCL group with independently reachable SDK peers.
        bootstrap_uri: SHMEM bootstrap endpoint, distinct from HCCL.
        num_heads: Number of equal query/key/value heads.
        packets: Maximum number of head packets, capped at num_heads.
        producer_groups: Requested producer groups, capped at the packet count.
        reserved_workers: Zero, or 16 programs for the second group's S/G only.
        heap_bytes: SDK heap shared by compatible instances in this process.
    """

    def __init__(self, group: Any, *, bootstrap_uri: str, num_heads: int = 96, packets: int = 4,
                 producer_groups: int = 2,
                 reserved_workers: int = 0, heap_bytes: int = 192 * 2**20) -> None:
        """Create a qualified producer configuration and its explicitly owned arena."""
        if num_heads < 1 or packets < 1 or producer_groups < 1:
            raise ValueError("Heads, packets and producer groups must be positive.")
        packets = min(packets, num_heads)
        producer_groups = min(producer_groups, packets)
        if producer_groups not in (1, 2) or reserved_workers not in (0, 16):
            raise ValueError("Qualified producer groups are 1/2 and worker limits are 0/16.")
        if reserved_workers and producer_groups != 2:
            raise ValueError("The worker limit applies only to a second producer group.")
        self.num_heads = num_heads
        self.producer_groups = producer_groups
        self.group_packets = tuple(index * packets // producer_groups for index in range(producer_groups + 1))
        self.reserved_workers = reserved_workers
        self.arena = StatePullArena(group, (1, num_heads, 128, 128), packets=packets,
                                    bootstrap_uri=bootstrap_uri, heap_bytes=heap_bytes)

    def validate(self, group: Any, query: torch.Tensor, value: torch.Tensor, chunk_size: int) -> None:
        """Reject unsupported layouts before reserving an invocation or communicating."""
        self.arena.check_stream()
        if group is not self.arena.group:
            raise ValueError("KDA transport must use the same precreated state process group.")
        if (query.device != self.arena.device or value.device != self.arena.device
                or query.dtype != torch.bfloat16 or value.dtype != torch.bfloat16
                or query.shape[0] != 1 or query.shape[2:] != (self.num_heads, 128)
                or value.shape[0] != 1 or value.shape[2:] != (self.num_heads, 128) or chunk_size != 64):
            raise ValueError("KDA state pull requires NPU BF16 B1/K=V128 with the configured head count and BT64.")

    def prepare_forward(self, key: torch.Tensor, weight: torch.Tensor, value: torch.Tensor,
                        gate: torch.Tensor, sequence_info: Any, chunk_size: int,
                        exchange: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Publish each group's S/M before producing the next group's summaries."""
        if not exchange.sends:
            return None, None
        tail = slice(None) if sequence_info is None else sequence_info.tail
        key, weight, value, gate = (tensor[:, tail].contiguous() for tensor in (key, weight, value, gate))
        length = key.shape[1]
        state = torch.empty(self.arena.state_shape, device=key.device, dtype=torch.float32)
        needs_matrix = exchange.receives and (sequence_info is None or sequence_info.transfer_active)
        matrix = torch.empty_like(state) if needs_matrix else None
        for group in range(self.producer_groups):
            first = self.group_packets[group]
            count = self.group_packets[group + 1] - first
            start = self.arena.packet_offsets[first]
            width = self.arena.packet_offsets[first + count] - start
            part = state[:, start:start + width]
            transfer = None if matrix is None else matrix[:, start:start + width]
            workers = min(self.reserved_workers, width) if group else 0
            kernel = kda_bounded_state_summary_kernel if workers else kda_split_state_summary_kernel
            options = {"NUM_WORKERS": workers} if workers else {}
            kernel[(workers,) if workers else (1, width)](
                key, weight, value, gate, part, transfer if transfer is not None else part, length,
                H=width, H_TOTAL=self.num_heads, HEAD_OFFSET=start, K=128, V=128, BT=chunk_size, BV=128,
                OUTPUT_MODE=1, num_warps=2, **options)
            if transfer is not None:
                kda_state_transition_kernel[(2, width)](
                    key, weight, gate, transfer, length, H=width, H_TOTAL=self.num_heads, HEAD_OFFSET=start,
                    K=128, BT=chunk_size, BV=64, num_warps=4, num_stages=2)
            exchange.publish_group(part, transfer, first, count)
        return state, matrix

    def prepare_backward(self, query: torch.Tensor, key: torch.Tensor, weight: torch.Tensor,
                         gate: torch.Tensor, grad_output: torch.Tensor, grad_value: torch.Tensor,
                         scale: float, chunk_size: int, sequence_info: Any,
                         matrix: torch.Tensor | None, exchange: Any) -> torch.Tensor | None:
        """Publish reverse G packets using the saved local M, transposed once."""
        if not exchange.sends:
            return None
        head = slice(None) if sequence_info is None else sequence_info.head
        query, key, weight, gate, grad_output, grad_value = (
            tensor[:, head].contiguous() for tensor in (query, key, weight, gate, grad_output, grad_value))
        length = query.shape[1]
        state = torch.empty(self.arena.state_shape, device=query.device, dtype=torch.float32)
        transposed = matrix.transpose(-2, -1).contiguous() if matrix is not None else None
        for group in range(self.producer_groups):
            first = self.group_packets[group]
            count = self.group_packets[group + 1] - first
            start = self.arena.packet_offsets[first]
            width = self.arena.packet_offsets[first + count] - start
            part = state[:, start:start + width]
            transfer = None if transposed is None else transposed[:, start:start + width]
            workers = min(self.reserved_workers, width) if group else 0
            kernel = kda_bounded_state_gradient_kernel if workers else kda_state_grad_ext_kernel
            options = {"NUM_WORKERS": workers} if workers else {}
            kernel[(workers,) if workers else (1, width)](
                query, key, weight, gate, grad_output, grad_value, part, float(scale), length,
                H=width, H_TOTAL=self.num_heads, HEAD_OFFSET=start, K=128, V=128, BT=chunk_size, BV=128,
                num_warps=2, **options)
            exchange.publish_group(part, transfer, first, count)
        return state

    def close(self) -> None:
        """Close collectively after all first backward calls or discarded graphs."""
        self.arena.close()

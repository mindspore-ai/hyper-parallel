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
"""Routed-slot transport, independent of routing and expert computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from hyper_parallel.distributed.expert_parallel.collectives import (
    ep_all_to_all, ep_exchange_counts, ep_split_sizes,
)


@dataclass
class DispatchReceipt:
    """Per-call inverse permutations and split sizes; never cached on a module."""

    owner: object
    send_counts: list[int]
    recv_counts: list[int]
    send_order: Tensor
    expert_order: Tensor
    tokens: int
    top_k: int
    consumed: bool = False


@dataclass(frozen=True)
class DispatchedExperts:
    """Expert-major slots; probability application belongs to the expert algorithm."""

    hidden_states: Tensor
    routing_weights: Tensor
    expert_indices: Tensor
    tokens_per_expert: Tensor
    receipt: DispatchReceipt


def _validate_slots(hidden: Tensor, indices: Tensor, weights: Tensor, experts: int) -> None:
    """Validate routed token slots, expert indices and optional routing probabilities."""
    if hidden.ndim != 2 or indices.ndim != 2 or weights.shape != indices.shape:
        raise ValueError("expected hidden[T,H], expert_indices[T,K], routing_weights[T,K]")
    if hidden.shape[0] != indices.shape[0] or indices.shape[1] < 1:
        raise ValueError("token dimensions must agree and top-k must be positive")
    if indices.dtype not in (torch.int32, torch.int64) or not weights.is_floating_point():
        raise ValueError("expert IDs must be integers and routing weights floating point")
    if hidden.device != indices.device or hidden.device != weights.device:
        raise ValueError("all routing inputs must be on the same device")
    if bool(((indices < 0) | (indices >= experts)).any()):
        raise ValueError("every supplied slot must have an expert ID in [0, num_experts); drop is unsupported")


class EPContext:
    """Balanced contiguous expert ownership from HP mesh and Shard(0) layout."""

    def __init__(self, mesh: Any, *, num_experts: int) -> None:
        """Borrow the EP mesh; never create a group or reduce parameter gradients."""
        if mesh is None or mesh.ndim != 1:
            raise ValueError("EPContext requires an existing one-dimensional HP mesh")
        if num_experts < 1 or num_experts % mesh.size():
            raise ValueError("num_experts must be positive and divisible by EP size")
        self.mesh = mesh
        self.num_experts = num_experts
        self.local_experts = num_experts // mesh.size()
        self.rank = mesh.get_local_rank()
        self._identity = object()

    def _exchange(self, tensor, send, receive):
        if self.mesh.size() == 1:
            return tensor
        return ep_all_to_all(tensor, send, receive, self.mesh.get_group())

    def dispatch(self, hidden_states: Tensor, expert_indices: Tensor, routing_weights: Tensor) -> DispatchedExperts:
        """Transport every already-selected slot, including zero-probability slots.

        Args:
            hidden_states: Source tokens [T,H].
            expert_indices: Global selected expert IDs [T,K]; no capacity/drop mode.
            routing_weights: Already normalized/scaled probabilities [T,K], with gradients.
        """
        _validate_slots(hidden_states, expert_indices, routing_weights, self.num_experts)
        tokens, top_k = expert_indices.shape
        indices = expert_indices.reshape(-1).to(torch.int64)
        destinations = indices // self.local_experts
        send_order = torch.argsort(destinations, stable=True)
        counts = torch.bincount(destinations, minlength=self.mesh.size())
        received_counts = (ep_exchange_counts(counts, self.mesh.get_group())
                           if self.mesh.size() > 1 else counts)
        send, receive = ep_split_sizes(counts), ep_split_sizes(received_counts)
        expanded = hidden_states[:, None, :].expand(-1, top_k, -1).reshape(-1, hidden_states.shape[-1])
        values = self._exchange(expanded[send_order], send, receive)
        weights = self._exchange(routing_weights.reshape(-1, 1)[send_order], send, receive).reshape(-1)
        ids = self._exchange(indices[send_order, None], send, receive).reshape(-1) - self.rank * self.local_experts
        expert_order = torch.argsort(ids, stable=True)
        receipt = DispatchReceipt(self._identity, send, receive, send_order, expert_order, tokens, top_k)
        return DispatchedExperts(values[expert_order], weights[expert_order], ids[expert_order],
                                 torch.bincount(ids, minlength=self.local_experts), receipt)

    def combine(self, values: Tensor, receipt: DispatchReceipt) -> Tensor:
        """Return computed slots to their source tokens and sum top-k contributions.

        Args:
            values: Expert-major weighted slot outputs [received_slots,H].
            receipt: Receipt from exactly one dispatch on this context.
        """
        if receipt.owner is not self._identity or receipt.consumed:
            raise ValueError("dispatch receipt belongs to another context or was already consumed")
        if values.ndim != 2 or values.shape[0] != sum(receipt.recv_counts):
            raise ValueError("expert output must have one row per received slot")
        peer_major = values[torch.argsort(receipt.expert_order)]
        source_slots = self._exchange(peer_major, receipt.recv_counts, receipt.send_counts)
        source_slots = source_slots[torch.argsort(receipt.send_order)]
        output = source_slots.reshape(receipt.tokens, receipt.top_k, values.shape[-1]).sum(dim=1)
        receipt.consumed = True
        return output

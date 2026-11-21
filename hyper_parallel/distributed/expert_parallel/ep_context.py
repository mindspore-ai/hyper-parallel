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

from typing import Any

import torch
from torch import Tensor

from hyper_parallel.distributed.expert_parallel.collectives import (
    ep_all_to_all, ep_exchange_counts, ep_split_sizes,
)


def _validate_slot_inputs(hidden: Tensor, ids: Tensor, weights: Tensor | None) -> None:
    """Validate resource-free source slots before entering any collective."""
    if hidden.ndim != 2 or hidden.shape[1] < 1 or not hidden.is_floating_point():
        raise ValueError("slot hidden_states must be floating-point [slots,H] with H>0")
    if ids.shape != (hidden.shape[0],) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_ids must have integer dtype and shape [slots]")
    if ids.device != hidden.device:
        raise ValueError("slot activations and expert IDs must share the same device")
    if weights is not None:
        if weights.shape != ids.shape or not weights.is_floating_point() or weights.device != hidden.device:
            raise ValueError("slot routing_weights must be floating-point [slots] on the activation device")


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

    def _exchange(self, tensor, send, receive):
        if self.mesh.size() == 1:
            return tensor
        return ep_all_to_all(tensor, send, receive, self.mesh.get_group())

    def _slot_split_sizes(self, counts: Tensor, weighted: bool) -> tuple[list[int], list[int]]:
        """Exchange counts and verify optional probability transport agrees across peers."""
        metadata = torch.stack((counts, torch.full_like(counts, int(weighted))), dim=1)
        received = (ep_exchange_counts(metadata, self.mesh.get_group())
                    if self.mesh.size() > 1 else metadata)
        fields = ep_split_sizes(received.reshape(-1))
        if any(flag != int(weighted) for flag in fields[1::2]):
            raise ValueError("all EP ranks must agree on whether slot routing_weights are transported")
        return ep_split_sizes(counts), fields[::2]

    def dispatch(
        self, hidden_states: Tensor, expert_ids: Tensor, routing_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, list[int], list[int], Tensor]:
        """Transport prepared slots and group received rows by local expert.

        Args:
            hidden_states: Source expert-major slot activations [S,H].
            expert_ids: Matching global expert IDs [S], sorted in ascending order.
            routing_weights: Optional differentiable per-slot probabilities [S].

        Returns:
            Owner-local hidden [R,H], counts [E_local], optional weights [R],
            send/receive split lists and the inverse mapping from received rows to
            owner-local expert-major rows [R]. The caller retains them for combine.
        """
        _validate_slot_inputs(hidden_states, expert_ids, routing_weights)
        ids = expert_ids.to(torch.int64)
        # Fixed-size counting rejects invalid/drop IDs on device and retains empty experts.
        global_counts = ids.new_zeros(self.num_experts).scatter_add_(0, ids, torch.ones_like(ids))
        counts = global_counts.reshape(self.mesh.size(), self.local_experts).sum(dim=1)
        send, receive = self._slot_split_sizes(counts, routing_weights is not None)
        # Ascending global IDs already group rows by rank under contiguous expert ownership.
        hidden = self._exchange(hidden_states.contiguous(), send, receive)
        global_ids = self._exchange(ids[:, None].contiguous(), send, receive).reshape(-1)
        weights = None
        if routing_weights is not None:
            weights = self._exchange(routing_weights[:, None].contiguous(), send, receive).reshape(-1)
        local_ids = global_ids - self.rank * self.local_experts
        local_counts = local_ids.new_zeros(self.local_experts).scatter_add_(
            0, local_ids, torch.ones_like(local_ids),
        )
        if not hidden.shape[0]:
            return hidden, local_counts, weights, send, receive, local_ids.to(torch.int32)

        if hidden.device.type == "npu":
            # Keep the optional torch_npu dependency inside the NPU path.
            from hyper_parallel.components.functional.moe_token_permute import (  # pylint: disable=C0415
                moe_token_permute,
            )

            # One expert per received slot keeps this a one-to-one rearrangement.
            hidden, recv_inverse = moe_token_permute(hidden.contiguous(), local_ids.to(torch.int32).view(-1, 1))
        else:
            order = local_ids.argsort(stable=True)
            hidden = hidden.index_select(0, order)
            rows = torch.arange(order.numel(), dtype=torch.int32, device=order.device)
            recv_inverse = torch.empty_like(rows).scatter_(0, order, rows)
        if weights is not None:
            weights = torch.empty_like(weights).scatter_(0, recv_inverse.to(torch.int64), weights)
        return hidden, local_counts, weights, send, receive, recv_inverse

    def combine(
        self, values: Tensor, send_splits: list[int], recv_splits: list[int], recv_inverse: Tensor,
    ) -> Tensor:
        """Return outputs to source prepared-slot order, without weighting or reduction.

        Args:
            values: Local expert-major outputs [received_slots,H].
            send_splits: Original per-peer send sizes from this call's dispatch.
            recv_splits: Original per-peer receive sizes, reversed for combine.
            recv_inverse: Mapping from peer-major rows to local expert-major rows [R].

        Returns:
            Source-side slot outputs; the Experts algorithm restores original tokens.
        """
        if len(send_splits) != self.mesh.size() or len(recv_splits) != self.mesh.size():
            raise ValueError("slot split lists must follow the current EP group's rank order")
        if values.ndim != 2 or values.shape[0] != sum(recv_splits):
            raise ValueError("slot output must have one row per received slot")
        if recv_inverse.shape != (values.shape[0],) or recv_inverse.dtype != torch.int32:
            raise ValueError("recv_inverse must have dtype int32 and one index per received slot")
        if recv_inverse.device != values.device:
            raise ValueError("slot outputs and recv_inverse must be on the same device")
        if values.shape[0] and values.device.type == "npu":
            # Keep the optional torch_npu dependency inside the NPU path.
            from hyper_parallel.components.functional.moe_token_unpermute import (  # pylint: disable=C0415
                moe_token_unpermute,
            )

            values = moe_token_unpermute(values.contiguous(), recv_inverse)
        elif values.shape[0]:
            values = values.index_select(0, recv_inverse)
        return self._exchange(values, recv_splits, send_splits)

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
"""Collective MoonEP transport with invocation-local plans and fixed home ownership."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from hyper_parallel.components.ops import moe_token_permute

from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from hyper_parallel.distributed.expert_parallel.moonep_planner import compile_plan, select_allocation


class MoonEPContext(EPContext):
    """Reuse EP collectives while dynamically balancing execution, without prefetch caches."""

    def __init__(self, mesh: Any, *, num_experts: int, replica_capacity: int | None = None,
                 overlap_shared_expert: bool = True) -> None:
        """Borrow the existing mesh and bound the number of remote experts per rank.

        Args:
            mesh: Existing one-dimensional TP-extend-EP mesh.
            num_experts: Global number of home experts, divisible by the mesh size.
            replica_capacity: Maximum distinct remote experts per rank; None means E/EP.
            overlap_shared_expert: Run bound shared FC1/FC2 during token exchanges.
        """
        super().__init__(mesh, num_experts=num_experts, overlap_shared_expert=overlap_shared_expert)
        capacity = self.local_experts if replica_capacity is None else replica_capacity
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 0:
            raise ValueError("moonep_replica_capacity must be a nonnegative integer or None")
        self.replica_capacity = int(capacity)

    def plan(self, expert_ids: Tensor) -> dict:
        """Compile a deterministic plan from all sources' prepared slot counts.

        Args:
            expert_ids: Global integer expert IDs [S], in source expert-major order.

        Returns:
            Per-call transport splits, permutations and compute-table counts.
        """
        ids = expert_ids.to(torch.int64)
        counts = ids.new_zeros(self.num_experts).scatter_add_(0, ids, torch.ones_like(ids))
        gathered = [torch.empty_like(counts) for _ in range(self.mesh.size())]
        if self.mesh.size() > 1:
            dist.all_gather(gathered, counts, group=self.mesh.get_group())
        else:
            gathered[0] = counts
        # The collective baseline plans on small integer histograms, like PR #1485.
        # This eager synchronization is not the official fused device planner.
        global_counts = torch.stack(gathered).cpu().numpy()
        allocation = select_allocation(global_counts, self.replica_capacity)
        return compile_plan(global_counts, allocation, ids, self.rank)

    # MoonEP deliberately keeps the plan-based entry point while inheriting EP runtime state.
    def dispatch(  # pylint: disable=arguments-differ
        self, hidden_states: Tensor, routing_weights: Tensor | None = None,
        *, plan: dict,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        """Send source slots to planned execution ranks, then perform the second permute.

        Args:
            hidden_states: Prepared expert-major slots [S,H].
            routing_weights: Optional differentiable before-FC2 probabilities [S].
            plan: Invocation-local result of plan().

        Returns:
            Compute-major activations [R,H], counts [C], optional probabilities [R],
            and the receiver inverse permutation [R]. C includes home and remote rows.
        """
        send, receive, order = plan["token_send"], plan["token_recv"], plan["source_order"]
        asynchronous = self._overlap_ready("fc1")
        hidden = self._exchange(hidden_states.index_select(0, order), send, receive, async_op=asynchronous)
        weights = None
        if routing_weights is not None:
            weights = self._exchange(routing_weights.index_select(0, order).unsqueeze(-1), send, receive,
                                     async_op=asynchronous).reshape(-1)
        if asynchronous:
            self._run_overlap("fc1", hidden, weights)
        counts, receive_order = plan["counts"], plan["receive_order"]
        if hidden.shape[0] and hidden.device.type == "npu":
            rows = torch.arange(counts.numel(), device=counts.device).repeat_interleave(
                counts, output_size=hidden.shape[0],
            )
            ids = torch.empty_like(rows).scatter_(0, receive_order, rows)
            hidden, _, inverse = moe_token_permute(hidden.contiguous(), ids.unsqueeze(-1))
            inverse = inverse.to(torch.int32)
        else:
            hidden = hidden.index_select(0, receive_order)
            inverse = torch.empty_like(receive_order, dtype=torch.int32).scatter_(
                0, receive_order, torch.arange(receive_order.numel(), device=receive_order.device, dtype=torch.int32),
            )
        if weights is not None:
            weights = weights.index_select(0, receive_order)
        return hidden, counts, weights, inverse

    def expert_weights(self, gate_up: Tensor, down: Tensor, plan: dict) -> tuple[Tensor, Tensor]:
        """Exchange only needed replicas using differentiable all-to-all.

        Args:
            gate_up: Materialized home FC1 weights [E/EP,2I,H].
            down: Materialized home FC2 weights [E/EP,H,I].
            plan: This invocation's remote expert requests and split sizes.

        Returns:
            Home rows followed by temporary remote rows. The collective adjoint
            returns replica gradients to home before the original Parameters' hooks.
            No Parameter, FSDP storage state or process group is created here.
        """
        if not plan["weight_exchange"]:
            return gate_up, down
        indices = plan["send_experts"]
        packet = torch.cat((gate_up.index_select(0, indices).flatten(1),
                            down.index_select(0, indices).flatten(1)), dim=1)
        remote = self._exchange(packet, plan["weight_send"], plan["weight_recv"])
        first, second = remote.split((gate_up[0].numel(), down[0].numel()), dim=1)
        return (torch.cat((gate_up, first.reshape(-1, *gate_up.shape[1:]))),
                torch.cat((down, second.reshape(-1, *down.shape[1:]))))

    # The forward plan owns the split sizes and source order required by MoonEP combine.
    def combine(  # pylint: disable=arguments-differ
        self, values: Tensor, plan: dict, recv_inverse: Tensor,
    ) -> Tensor:
        """Unpermute at the compute owner and return source expert-major slots.

        Args:
            values: Compute-major expert outputs [R,H].
            plan: Saved forward transport and source inverse mapping.
            recv_inverse: Receiver permutation returned by dispatch().

        Returns:
            Source expert-major outputs [S,H], before the algorithm's final unpermute.
        """
        values = super().combine(values, plan["token_send"], plan["token_recv"], recv_inverse)
        return values.index_select(0, plan["source_inverse"].to(torch.int64))

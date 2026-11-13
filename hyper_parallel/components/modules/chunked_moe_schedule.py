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
"""Bounded, call-local communication schedules for explicit ChunkMoE training."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from hyper_parallel.core.expert_parallel.expert_parallel import DispatchContext, _generate_permute_indices
from hyper_parallel.core.expert_parallel.row_permutation import invert_permutation, permute_rows
from hyper_parallel.core.utils.communication import (
    differentiable_all_to_all_single_async,
    exchange_splits_via_all_to_all,
    wait_async_tensor,
)


@dataclass
class _Plan:
    indices: torch.Tensor
    order: torch.Tensor
    source_rows: torch.Tensor
    counts: torch.Tensor
    context: Any = None


@dataclass
class _Prepared:
    start: int
    end: int
    scores: torch.Tensor
    source_counts: torch.Tensor
    plan: Any


@dataclass
class _Pending:
    # Own the send tensor through the consumer wait; never overwrite an in-flight buffer.
    send: torch.Tensor | None
    receive: torch.Tensor | None
    distributed: bool

    def drain(self) -> None:
        """Keep in-flight storage alive until the existing work has been waited."""
        if self.receive is not None:
            self.take()

    def take(self) -> torch.Tensor:
        """Wait once on the launch stream, then transfer ownership to the consumer."""
        if self.receive is None:
            raise RuntimeError("A pending exchange can only be consumed once")
        result = wait_async_tensor(self.receive) if self.distributed else self.receive
        self.send = self.receive = None
        return result


def _prepare(module, hidden, start, end, mesh):
    scores, indices, source_counts = module.router(hidden[start:end], module.expert_bias)
    order = indices.reshape(-1).argsort(stable=True)
    plan = _Plan(indices, order, order // module.top_k, source_counts)
    if mesh is not None:
        degree = mesh.size()
        local_experts = module.num_experts // degree
        received_counts = exchange_splits_via_all_to_all(source_counts, mesh.get_group())
        input_splits = source_counts.view(degree, local_experts).sum(1).tolist()
        output_splits = received_counts.view(degree, local_experts).sum(1).tolist()
        order_receive, plan.counts = _generate_permute_indices(
            received_counts, local_experts, degree, total_received=sum(output_splits),
        )
        plan.context = DispatchContext(
            input_splits=input_splits, output_splits=output_splits,
            input_shape=torch.Size((sum(output_splits), hidden.shape[-1])),
            permuted_indices=order_receive, inverse_indices=invert_permutation(order_receive),
            probs_input_shape=None,
        )
    return _Prepared(start, end, scores, source_counts, plan)


def _exchange(values, plan, mesh, reverse=False):
    if plan.context is None:
        return _Pending(values, values, False)
    context = plan.context
    incoming, outgoing = context.input_splits, context.output_splits
    if reverse:
        incoming, outgoing = outgoing, incoming
    values = values.contiguous()
    result = differentiable_all_to_all_single_async(values, incoming, outgoing, mesh.get_group())
    return _Pending(values, result, True)


def _dispatch(hidden, prepared, mesh):
    values = hidden[prepared.start:prepared.end].index_select(0, prepared.plan.source_rows)
    return _exchange(values, prepared.plan, mesh)


def _receive(pending, prepared):
    values = pending.take()
    context = prepared.plan.context
    return values if context is None else values.index_select(0, context.permuted_indices)


def _compute(values, prepared, experts, mesh):
    result = experts.forward(values, prepared.plan.counts)
    context = prepared.plan.context
    if context is not None:
        result = permute_rows(result, context.inverse_indices, context.permuted_indices)
    return _exchange(result, prepared.plan, mesh, reverse=True)


def _finish(pending, prepared, hidden, module, output):
    values = pending.take()
    values = values * prepared.scores.to(values.dtype).reshape(-1)[prepared.plan.order, None]
    part = hidden[prepared.start:prepared.end]
    result = torch.zeros_like(part).index_add_(0, prepared.plan.source_rows, values)
    if module.shared_expert is not None:
        result.add_(module.shared_expert(part))
    output[prepared.start:prepared.end].copy_(result)


def _window(hidden, prepared, experts, mesh, module, output):
    pending = []
    try:
        first = prepared[0]
        first_input = _dispatch(hidden, first, mesh)
        pending.append(first_input)
        values = _receive(first_input, first)
        if len(prepared) == 1:
            combined = _compute(values, first, experts, mesh)
            pending.append(combined)
            del values
            _finish(combined, first, hidden, module, output)
            return
        second = prepared[1]
        # Both plans are ready; metadata waits must not drain this payload window.
        second_input = _dispatch(hidden, second, mesh)
        pending.append(second_input)
        first_output = _compute(values, first, experts, mesh)
        pending.append(first_output)
        del values
        values = _receive(second_input, second)
        second_output = _compute(values, second, experts, mesh)
        pending.append(second_output)
        del values
        _finish(first_output, first, hidden, module, output)
        _finish(second_output, second, hidden, module, output)
    finally:
        for exchange in pending:
            exchange.drain()


def forward_overlapped(module: Any, flat: torch.Tensor, output: torch.Tensor,
                       experts: Any, mesh: Any) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Evaluate pairs with at most two blocks of pending feature exchanges.

    Args:
        module: Common MoE with a validated chunk size and routing contract.
        flat: Local flattened source input, shared by all windows.
        output: Output destination, written before each window releases its buffers.
        experts: Per-pass expert compute views.
        mesh: Expert mesh or None for local execution.

    Returns:
        Source counts, selected score sums and ordered call-local routing plans.
    """
    counts = torch.zeros_like(module.tokens_per_expert, dtype=torch.float32)
    score_sums = torch.zeros_like(counts)
    plans = []
    for window_start in range(0, flat.shape[0], 2 * module.chunk_size):
        prepared = []
        for start in range(window_start, min(window_start + 2 * module.chunk_size, flat.shape[0]), module.chunk_size):
            end = min(start + module.chunk_size, flat.shape[0])
            item = _prepare(module, flat, start, end, mesh)
            prepared.append(item)
            counts.add_(item.source_counts.float())
            score_sums.scatter_add_(0, item.plan.indices.reshape(-1), item.scores.reshape(-1))
            plans.append((start, end, item.plan))
        _window(flat, prepared, experts, mesh, module, output)
        del item, prepared
    return counts, score_sums, plans


class BackwardPrefetch:
    """Own one future X exchange within a single active layer backward."""

    def __init__(self, flat: torch.Tensor, plans: list, mesh: Any) -> None:
        """Reuse the outer backward's flat input and remaining reverse plans."""
        self.flat, self.plans, self.mesh = flat, plans, mesh
        self.pending = None
        self.plan = None

    def consume(self, chunk: torch.Tensor, plan: Any) -> torch.Tensor:
        """Wait for this plan's X and convert it to expert order.

        Args:
            chunk: Current source block; used for the cold first block.
            plan: Original route plan to consume.

        Returns:
            Ready expert-major input rows.
        """
        if self.pending is None:
            self.pending = _exchange(chunk.detach().index_select(0, plan.source_rows), plan, self.mesh)
            self.plan = plan
        if self.plan is not plan:
            raise RuntimeError("Chunk MoE prefetch does not match the next routing plan.")
        raw = self.pending.take()
        self.pending, self.plan = None, None
        context = plan.context
        return raw if context is None else raw.index_select(0, context.permuted_indices)

    def launch_next(self) -> None:
        """Launch the next reverse block only after current p/G are ready."""
        if self.pending is not None:
            raise RuntimeError("Chunk MoE supports only one future X exchange.")
        if self.plans:
            start, end, self.plan = self.plans[-1]
            values = self.flat[start:end].detach().index_select(0, self.plan.source_rows)
            self.pending = _exchange(values, self.plan, self.mesh)

    def close(self) -> None:
        """Drain exceptional pending work before releasing its buffers."""
        if self.pending is not None:
            self.pending.drain()
            self.pending, self.plan = None, None


@contextmanager
def backward_prefetch(flat: torch.Tensor, plans: list, mesh: Any,
                      enabled: bool) -> Iterator[BackwardPrefetch | None]:
    """Keep a future exchange alive through its consumer or exceptional exit.

    Args:
        flat: Flattened source input already unpacked by the outer backward.
        plans: Mutable reverse-consumption list owned by this invocation.
        mesh: Expert mesh or None for local execution.
        enabled: Whether this forward selected backward overlap.

    Yields:
        A single-slot prefetch owner, or None for synchronous execution.
    """
    prefetch = BackwardPrefetch(flat, plans, mesh) if enabled else None
    try:
        yield prefetch
    finally:
        if prefetch is not None:
            prefetch.close()

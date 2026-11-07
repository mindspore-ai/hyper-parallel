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
"""Explicit routed backward with complete owner gradients returned exactly once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.multicore.modules.routed_experts.compute import (
    ExpertState, combine, combine_gradient,
)
from hyper_parallel.core.multicore.modules.routed_experts.transport import prepare_exchange, submit_prepared
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import is_replaying
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import prepare_gradient_storage


@dataclass
class _Phase:
    """Keep one phase's admitted storage alive through its last submitted reader."""

    incoming: Any
    compute: Any
    returned: Any
    result: torch.Tensor
    arena_token: Any
    score_gradient: torch.Tensor | None = None
    derivative: torch.Tensor | None = None
    group_metadata: torch.Tensor | None = None


def _prepare_forward(hidden: torch.Tensor, gate: list, down: list, intermediate: int,
                     executor: Any, plan: Any, runtime: Any) -> _Phase:
    """Converge allocation failures once before either token direction starts."""
    phase, error = None, None
    token = runtime.arena.begin()
    try:
        group_metadata = (torch.empty((plan.route.received_rows, 8), dtype=torch.int64, device=hidden.device)
                         if executor.token_dedup else None)
        pipeline = plan.compute.forward_pipeline
        publish_each_block = (pipeline.enabled
                              and pipeline.supports(hidden.shape[-1], intermediate, gate, down,
                                                    len(plan.route.boundaries)))
        if publish_each_block:
            pipeline.prepare_route_metadata(plan.route, runtime.workspace, hidden,
                                            bool(gate and gate[0].stride(-2) == 1))
        incoming = prepare_exchange(hidden.contiguous(), plan.route, runtime.workspace,
                                    mapping=plan.order // executor.top_k,
                                    dedup=executor.token_dedup, group_metadata=group_metadata,
                                    publish_each_block=publish_each_block,
                                    forward_compact_max_unique_ratio=executor.forward_ratio)
        compute = plan.compute.prepare_forward(incoming.value, gate, down, intermediate,
                                               allocator=runtime.arena.allocate)
        returned = prepare_exchange(compute.output, plan.route, runtime.workspace, reverse=True,
                                    dependency=compute.ready)
        phase = _Phase(incoming, compute, returned, hidden.new_empty(hidden.shape), token,
                       group_metadata=group_metadata)
    except Exception as exc:
        error = exc
    try:
        runtime.workspace.check_preflight(error)
    except BaseException:
        runtime.arena.end(token)
        raise
    return phase


def _prepare_backward(dy: torch.Tensor, state: ExpertState, values: torch.Tensor, scores: torch.Tensor,
                      gate: list, down: list, executor: Any, plan: Any, runtime: Any, key: Any, shapes: tuple,
                      group_metadata: torch.Tensor | None) -> _Phase:
    """Allocate one reverse phase, reusing source dV after dispatch has consumed it."""
    phase, error = None, None
    token = runtime.arena.begin()
    try:
        dy = dy.contiguous()
        runtime.gradient_outputs[key] = {
            role: torch.empty(shape, dtype=torch.bfloat16, device=dy.device)
            for role, shape in zip(("gate", "down"), shapes)
        }
        dv = runtime.arena.allocate("selections", tuple(values.shape), values.dtype)
        ds = torch.empty_like(scores)
        received, accumulator, staging = prepare_gradient_storage(
            runtime.arena.allocate, (plan.route.received_rows, dy.shape[1]), shapes,
            staging_elements=runtime.gradient_domain.quota_elements)
        runtime.gradient_accumulators[key] = accumulator
        runtime.gradient_staging[key] = staging
        incoming = prepare_exchange(dv, plan.route, runtime.workspace, output=received)
        compute = plan.compute.prepare_backward(incoming.value, state, gate, down,
                                                allocator=runtime.arena.allocate)
        # T serializes dispatch and return, so its final write can reuse dV.
        returned = prepare_exchange(compute.output, plan.route, runtime.workspace, reverse=True,
                                    output=dv, dependency=compute.ready, reduce=group_metadata is not None,
                                    group_metadata=group_metadata,
                                    reduce_dense_ratio=executor.backward_ratio)
        phase = _Phase(incoming, compute, returned, dy.new_empty(dy.shape), token, ds, dy)
    except Exception as exc:
        error = exc
    try:
        runtime.workspace.check_preflight(error)
    except BaseException:
        runtime.arena.end(token)
        raise
    return phase


class SparseExpertFunction(torch.autograd.Function):
    """Own activation saves; borrow fresh weights independently in each phase."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, scores: torch.Tensor, home_gate: torch.Tensor,
                home_down: torch.Tensor, key: Any, runtime: Any) -> torch.Tensor:
        """Dispatch original tokens, evaluate actual experts, and combine once.

        Args:
            ctx: Autograd context retaining explicit backward state.
            hidden: Original BF16 token rows with their upstream gradient edge.
            scores: Differentiable BF16 or FP32 Top-K router weights.
            home_gate: FSDP-owned gate/up input for expert gradient attribution.
            home_down: FSDP-owned down input for expert gradient attribution.
            key: Planned execution identifying this layer and micro-batch.
            runtime: Shared weight, transport, and scratch lifecycle manager.

        Returns:
            Combined token rows whose history is owned by this autograd function.
        """
        # Queued native handlers retain their tensor arguments. Storage-sharing
        # aliases keep those references from retaining the checkpoint replay graph;
        # Function.apply still owns the original input edges for explicit backward.
        hidden, scores = hidden.detach(), scores.detach()
        executor, plan = runtime.executions[key]
        try:
            gate, down, weight_ready = runtime.consume(key)
            phase = _prepare_forward(hidden, gate, down, home_down.shape[1], executor, plan, runtime)
            incoming = submit_prepared(phase.incoming, runtime.workspace)
            torch.npu.current_stream(hidden.device).wait_event(weight_ready)
            _, state, _ = plan.compute.run_forward(phase.compute, incoming.ready)
            returned = submit_prepared(phase.returned, runtime.workspace).wait()
            result = combine(returned, plan.inverse_order, scores, hidden.shape[0], executor.top_k,
                             output=phase.result)
            runtime.arena.end(phase.arena_token)
            runtime.statistics["scratch_reserved_bytes"] = runtime.arena.reserved_bytes
            ctx.save_for_backward(state.x, state.preact, returned, scores, plan.order)
            ctx.group_metadata = phase.group_metadata
            ctx.key, ctx.runtime = key, runtime
            ctx.shapes = home_gate.shape, home_down.shape
            ctx.tokens = hidden.shape[0]
            if not is_replaying():
                runtime.release(key)
            # Autograd must attach output history to a different TensorImpl from
            # the buffer retained by the queued combine handler.
            return result.detach()
        except BaseException:
            runtime.poisoned = True
            raise

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dy: torch.Tensor) -> tuple:
        """Finish both replica gradients before returning to FSDP AccumulateGrad."""
        runtime, key = ctx.runtime, ctx.key
        runtime.validate_backward(key)
        runtime.backward_layers.add(key.layer)
        try:
            # Unpacking may replay the checkpoint and restore parameter storage.
            # Bind home aliases only after that replay has completed.
            saved = ctx.saved_tensors
            executor, plan = runtime.executions[key]
            gate, down, weight_ready = runtime.consume(key)
            x, z, values, scores, order = saved
            phase = _prepare_backward(dy, ExpertState(x, z), values, scores, gate, down, executor,
                                      plan, runtime, key, ctx.shapes, ctx.group_metadata)
            # The reverse output aliases source dV only after T finishes reading it.
            _, ds = combine_gradient(phase.derivative, values, order, scores,
                                     outputs=(phase.returned.value, phase.score_gradient))
            incoming = submit_prepared(phase.incoming, runtime.workspace)
            torch.npu.current_stream(phase.derivative.device).wait_event(weight_ready)
            packets = {}
            scheduler = getattr(executor.owner, "hsdp_scheduler", None)

            def publish(role: str, gradient: torch.Tensor) -> None:
                """Expose down and gate readiness without waiting for later autograd nodes."""
                if scheduler is None:
                    packets[role] = runtime.start_gradient(key, role, gradient)
                else:
                    packets[role] = scheduler.launch_gradient_extension((key, role, gradient))

            plan.compute.run_backward(phase.compute, publish, incoming.ready)
            selections_dx = submit_prepared(phase.returned, runtime.workspace).wait()
            dx = combine(selections_dx, plan.inverse_order, None, ctx.tokens, executor.top_k, output=phase.result)
            current = torch.npu.current_stream(dy.device)
            for packet in packets.values():
                current.wait_event(packet.ready)
                packet.value.record_stream(current)
            runtime.arena.end(phase.arena_token)
            runtime.statistics["scratch_reserved_bytes"] = runtime.arena.reserved_bytes
            home_gate, home_down = packets["gate"].value, packets["down"].value
            if (home_gate.shape, home_down.shape) != ctx.shapes:
                raise RuntimeError("complete sparse gradients must match the actual autograd parameter inputs")
            runtime.release(key)
            runtime.complete(key)
            ctx.group_metadata = None
            return dx, ds, home_gate, home_down, None, None
        except BaseException:
            runtime.poisoned = True
            raise

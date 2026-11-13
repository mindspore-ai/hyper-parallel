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
"""First-order head-chunk MLA with one native FA forward per stage."""

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.distributed.context_parallel.collectives import async_ulysses_seq_to_head_launch
from ._head_chunk_mla_offload import SavedStatsOffloadStore
from ._head_chunk_mla_math import (
    _SAVED_TENSOR_COUNT, _accumulator, _head_to_seq, _project_shared, _release_stage_outputs,
    _seq_to_head, _owner_kv_stage_vjp, _owner_stage, _project_owner_kv_stage, _project_query_stage,
    _publish_compressed_kv, _query_stage_vjp, _reduce_owner_kv_gradients,
)


def _launch_exchange(tensor, mesh, *, inverse=False):
    """Use the existing differentiable ticket with inverse scatter/gather axes."""
    if mesh is None or mesh.size() == 1:
        return tensor
    seq_dim, head_dim = (1, 2) if inverse else (2, 1)
    return async_ulysses_seq_to_head_launch(tensor, seq_dim, head_dim, mesh)


def _wait_exchange(ticket):
    """Keep each ticket alive until the dependent computation can read it."""
    return ticket if isinstance(ticket, torch.Tensor) else ticket.wait()


class _HeadChunkMLAFunction(torch.autograd.Function):
    """Manual inverse collectives return activation and parameter gradients separately."""

    @staticmethod
    def forward(ctx, hidden, weight, query_weight, kv_weight, output_weight, query_norm, kv_norm, cos, sin, execution):
        store = None
        if execution.pool is not None:
            store = SavedStatsOffloadStore(
                hidden.device, execution.pool, execution.prefetch_slots, statistics=execution.statistics,
                retire_d2h_sources=execution.retire_d2h_sources,
            )
        ctx.store, ctx.offload_used = store, False
        try:
            shared = _project_shared(execution, hidden, weight, query_norm, kv_norm, cos, sin, replay=False)
            latent, rope = _publish_compressed_kv(execution, shared, "forward")
            output = hidden.new_zeros((*hidden.shape[:2], output_weight.shape[0]))
            output_flat = output.flatten(0, 1)
            retained = []
            for index, stage in enumerate(execution.stages):
                state = _project_query_stage(execution, shared, query_weight, stage, cos, sin, replay=False)
                query = _seq_to_head(state.outputs[0], execution.cp_mesh)
                key, value = _project_owner_kv_stage(execution, latent, rope, kv_weight, stage)
                attention, saved = execution.backend.forward_saved(query, key, value, execution.scale)
                if execution.retain:
                    if store is None:
                        retained.extend(saved)
                    else:
                        store.offload_stage(index, saved)
                local = _head_to_seq(attention, execution.cp_mesh).transpose(1, 2).contiguous()
                start, count = stage
                columns = output_weight.narrow(1, start * execution.value_dim, count * execution.value_dim)
                output_flat.addmm_(local.flatten(0, 1).flatten(1), columns.transpose(0, 1))
                del state, query, key, value, attention, saved, local, columns
            if store is not None and execution.retire_d2h_sources:
                store.finish_forward()
            if execution.retain:
                ctx.save_for_backward(hidden, weight, query_weight, kv_weight, output_weight,
                                      query_norm, kv_norm, cos, sin, *retained)
                ctx.execution = execution
            execution.compressed_statistics["completed_forward"] += 1
            return output
        except Exception:
            if store is not None:
                store.close()
                ctx.store = None
            raise


    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.offload_used:
            raise RuntimeError("offloaded MLA saved states support only one backward per forward")
        store = ctx.store
        if store is not None:
            ctx.offload_used = True
        pending_exchange = None
        try:
            hidden, weight, query_weight, kv_weight, output_weight, query_norm, kv_norm, cos, sin, *retained = (
                ctx.saved_tensors
            )
            execution = ctx.execution
            loads = {}
            shared = _project_shared(execution, hidden, weight, query_norm, kv_norm, cos, sin, replay=True)
            latent, rope = _publish_compressed_kv(execution, shared, "backward")
            grad_query_latent = _accumulator(shared.query_latent)
            owner_accumulators = [_accumulator(latent), _accumulator(rope)]
            grad_query_weight = torch.empty_like(query_weight)
            # Head ownership is stage-dependent and is not an FSDP parameter shard.
            # The existing FSDP SUM/P consumes this full-shaped sparse contribution;
            # an extra parameter AllReduce here would double-count it.
            grad_kv_weight = torch.zeros_like(kv_weight)
            grad_output_weight = torch.empty_like(output_weight)
            grad_flat = grad_output.flatten(0, 1)
            if store is not None:
                for index in range(min(store.slots, len(execution.stages))):
                    if index not in loads:
                        loads[index] = store.prefetch_stage(index)
            for index, stage in enumerate(execution.stages):
                ticket = loads.pop(index) if store is not None else None
                state = _project_query_stage(execution, shared, query_weight, stage, cos, sin, replay=True)
                query = _seq_to_head(state.outputs[0].detach(), execution.cp_mesh)
                if execution.use_output_gradient_edges:
                    _release_stage_outputs(state)
                key, value = _project_owner_kv_stage(execution, latent, rope, kv_weight, stage)
                if store is None:
                    saved = tuple(retained[index * _SAVED_TENSOR_COUNT:(index + 1) * _SAVED_TENSOR_COUNT])
                else:
                    saved = store.consume(ticket)
                start, count = stage
                columns = output_weight.narrow(1, start * execution.value_dim, count * execution.value_dim)
                local = _head_to_seq(saved[0], execution.cp_mesh).transpose(1, 2).contiguous()
                if execution.overlap_backward:
                    grad_attention = grad_flat.mm(columns).view(*hidden.shape[:2], count, execution.value_dim)
                    pending_exchange = _launch_exchange(grad_attention.transpose(1, 2).contiguous(), execution.cp_mesh)
                    del grad_attention
                grad_output_weight.narrow(1, start * execution.value_dim, count * execution.value_dim).copy_(
                    grad_flat.transpose(0, 1).mm(local.reshape(grad_flat.shape[0], -1)),
                )
                del local
                if execution.overlap_backward:
                    grad_attention = _wait_exchange(pending_exchange)
                    pending_exchange = None
                else:
                    grad_attention = grad_flat.mm(columns).view(*hidden.shape[:2], count, execution.value_dim)
                    grad_attention = _seq_to_head(grad_attention.transpose(1, 2).contiguous(), execution.cp_mesh)
                gradients = execution.backend.backward_saved(query, key, value, grad_attention, execution.scale, saved)
                if execution.overlap_backward:
                    pending_exchange = _launch_exchange(gradients[0], execution.cp_mesh, inverse=True)
                    grad_kv_block = _owner_kv_stage_vjp(
                        execution, gradients[1], gradients[2], latent, kv_weight, stage, owner_accumulators,
                    )
                    grad_query = _wait_exchange(pending_exchange)
                    pending_exchange = None
                    grad_query_block = _query_stage_vjp(
                        execution, state, grad_query, shared, query_weight, stage, grad_query_latent,
                    )
                else:
                    grad_query = _head_to_seq(gradients[0], execution.cp_mesh)
                    grad_query_block = _query_stage_vjp(
                        execution, state, grad_query, shared, query_weight, stage, grad_query_latent,
                    )
                    grad_kv_block = _owner_kv_stage_vjp(
                        execution, gradients[1], gradients[2], latent, kv_weight, stage, owner_accumulators,
                    )
                grad_query_weight.narrow(0, start * execution.query_dim, count * execution.query_dim).copy_(
                    grad_query_block,
                )
                owner_start, owner_count = _owner_stage(execution, stage)
                kv_width = execution.nope_dim + execution.value_dim
                grad_kv_weight.narrow(0, owner_start * kv_width, owner_count * kv_width).copy_(grad_kv_block)
                del state, query, key, value, saved, columns, gradients, grad_query
                del grad_attention, grad_query_block, grad_kv_block
                if store is not None:
                    store.release_slot(ticket)
                    successor = index + store.slots
                    if successor < len(execution.stages):
                        loads[successor] = store.prefetch_stage(successor)
            if execution.release_global_buffers:
                del latent, rope
            grad_kv_latent, grad_key_rope = _reduce_owner_kv_gradients(execution, owner_accumulators)
            del owner_accumulators
            if not execution.release_global_buffers:
                del latent, rope
            shared_outputs = (shared.query_latent, shared.kv_latent, shared.key_rope)
            shared_gradients = tuple(
                grad.to(tensor.dtype) for grad, tensor in
                zip((grad_query_latent, grad_kv_latent, grad_key_rope), shared_outputs)
            )
            if execution.release_global_buffers:
                del grad_query_latent, grad_kv_latent, grad_key_rope
            grad_raw, grad_query_norm, grad_kv_norm = torch.autograd.grad(
                shared_outputs, (shared.raw, shared.query_norm_weight, shared.kv_norm_weight), shared_gradients,
            )
            raw_flat = grad_raw.flatten(0, 1)
            grad_hidden = raw_flat.mm(weight).view_as(hidden)
            grad_weight = raw_flat.transpose(0, 1).mm(hidden.flatten(0, 1))
            gradients = (grad_hidden, grad_weight, grad_query_weight, grad_kv_weight, grad_output_weight,
                         grad_query_norm, grad_kv_norm)
            execution.compressed_statistics["completed_backward"] += 1
            return tuple(gradient if ctx.needs_input_grad[index] else None
                         for index, gradient in enumerate(gradients)) + (None, None, None)
        finally:
            try:
                if (pending_exchange is not None and not isinstance(pending_exchange, torch.Tensor)
                        and not pending_exchange.waited):
                    _wait_exchange(pending_exchange)
            finally:
                if store is not None:
                    try:
                        store.close()
                    finally:
                        ctx.store = None

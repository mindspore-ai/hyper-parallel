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
"""MLA backward scheduling with paired input A2A nodes and one output projection node.

Paired launch/wait nodes require a complete first-order training backward.
Partial autograd.grad calls can prune the consumer and are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any

# This CP scheduler uses the native Torch autograd and communication APIs.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.collectives import (
    AsyncCPCollective, async_ulysses_seq_to_head_launch,
)


@dataclass
class MLAOutputProjectionState:
    """Output layout for ordinary MLA; its projection node owns O A2A."""

    mesh: Any
    head_dim: int


class _OutputProjection(torch.autograd.Function):
    """Own the ordinary output Linear derivative and its dO A2A."""

    @staticmethod
    def forward(ctx: Any, attention_output: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None, state: MLAOutputProjectionState) -> torch.Tensor:
        """Restore the FA output to local tokens, then apply Wo.

        Args:
            ctx: Autograd context for operands, layout and autocast settings.
            attention_output: FA output in [batch, sequence, local_heads, value_dim].
            weight: Managed output-projection weight.
            bias: Optional output-projection bias.
            state: Ulysses output layout owned by this projection node.

        Returns:
            Projected local hidden states.
        """
        output_a2a_handle = async_ulysses_seq_to_head_launch(attention_output, 2, 1, state.mesh)
        # Function.forward owns both the numeric A2A and its backward. The
        # collective's internal autograd wrapper is not recorded in this scope.
        local_output = output_a2a_handle.wait().flatten(2)
        ctx.save_for_backward(local_output, weight)
        ctx.state, ctx.has_bias = state, bias is not None
        ctx.device_type = local_output.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type) if ctx.autocast_enabled else None
        return F.linear(local_output, weight, bias)  # pylint: disable=not-callable

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Submit dO A2A before wgrad and wait before returning the FA gradient.

        Args:
            ctx: Saved operands, autocast settings and output layout.
            gradient: Gradient of the projected hidden states.

        Returns:
            FA output, weight, bias and no state gradients.
        """
        with torch.autocast(ctx.device_type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype):
            return _output_projection_backward(ctx, gradient)


def _output_projection_backward(ctx, gradient):
    """Keep the dO communication handle local to this backward invocation."""
    local_output, weight = ctx.saved_tensors
    state = ctx.state
    needs_attention_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
    grad_output_a2a = None
    if needs_attention_grad:
        grad_local_output = gradient.matmul(weight).to(local_output.dtype)
        grad_local_output = grad_local_output.reshape(*local_output.shape[:2], -1, state.head_dim)
        grad_output_a2a = async_ulysses_seq_to_head_launch(grad_local_output, 1, 2, state.mesh)
    flat_input = local_output.reshape(-1, local_output.shape[-1])
    flat_gradient = gradient.reshape(-1, gradient.shape[-1])
    grad_weight = None
    if needs_weight_grad:
        grad_weight = flat_input.transpose(0, 1).matmul(flat_gradient).transpose(0, 1)
    grad_bias = flat_gradient.sum(0) if ctx.has_bias and needs_bias_grad else None
    grad_attention_output = grad_output_a2a.wait() if grad_output_a2a is not None else None
    return grad_attention_output, grad_weight, grad_bias, None


def mla_output_projection_request(projection: nn.Linear) -> _ForwardRewriteRequest:
    """Keep parameter materialization hooks while allowing dgrad/wgrad scheduling.

    Args:
        projection: Plain, unwrapped output Linear, before FSDP installation.

    Returns:
        An atomic child forward rewrite; ordinary calls remain unchanged.
    """
    # Quantized and LoRA subclasses need their own derivative contracts.
    if type(projection) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("MLA backward overlap requires a plain output Linear")
    if getattr(projection.forward, "__func__", None) is not nn.Linear.forward:
        raise ValueError("MLA backward overlap requires an unwrapped plain output Linear")
    original = projection.forward

    @wraps(original)
    def output_projection(*args: torch.Tensor, mla_output_state: MLAOutputProjectionState | None = None,
                          **kwargs: Any) -> torch.Tensor:
        """Read managed parameters after their normal module pre-hooks.

        Args:
            *args: Positional arguments of the original Linear.
            mla_output_state: Optional layout selecting the scheduled ordinary output.
            **kwargs: Keyword arguments of the original Linear.

        Returns:
            Projected hidden states using ordinary or scheduled autograd.
        """
        if mla_output_state is None:
            return original(*args, **kwargs)
        # Scheduled calls consume the FA output; ordinary calls retain Linear input= support.
        if len(args) != 1 or kwargs:
            raise TypeError("Scheduled MLA output projection requires one positional activation")
        return _OutputProjection.apply(args[0], projection.weight, projection.bias, mla_output_state)

    return _ForwardRewriteRequest(projection, output_projection)


@dataclass
class MLAInputA2AState:
    """Own one input A2A and its inverse for a single forward invocation.

    Attributes:
        mesh: Ulysses device mesh.
        forward_a2a_handle: Q/K/V transfer from token-local to head-local layout.
        backward_a2a_handle: Gradient transfer from head-local to token-local layout.
        needs_send_grad: Whether the original input requires its inverse transfer.

    Each handle owns the communication work and its buffers. The paired consumer
    waits for it exactly once before reading the result, then clears the field.
    """

    mesh: Any
    forward_a2a_handle: AsyncCPCollective | None = None
    backward_a2a_handle: AsyncCPCollective | None = None
    needs_send_grad: bool = False


class _InputA2ALaunch(torch.autograd.Function):
    """Keep independent native compute between a send and its gradient wait."""

    @staticmethod
    def forward(ctx: Any, send_tensor: torch.Tensor, compute_input: torch.Tensor,
                state: MLAInputA2AState) -> torch.Tensor:
        """Launch token-to-head A2A and attach the intervening compute edge.

        Args:
            ctx: Autograd context holding this invocation's A2A state.
            send_tensor: Token-local tensor ready for communication.
            compute_input: Input of the independent projection branch.
            state: Forward and backward handles shared with the paired wait.

        Returns:
            A view of compute_input retaining both gradient dependencies.
        """
        ctx.state = state
        state.needs_send_grad = ctx.needs_input_grad[0]
        state.forward_a2a_handle = async_ulysses_seq_to_head_launch(send_tensor, 1, 2, state.mesh)
        # Pass through the value without a copy while the paired nodes bracket
        # the independent projection's native backward graph.
        return compute_input.view_as(compute_input)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, compute_gradient: torch.Tensor) -> tuple:
        """Consume the inverse A2A after the independent branch backward.

        Args:
            ctx: Autograd context containing the paired A2A state.
            compute_gradient: Gradient returned by the independent branch.

        Returns:
            Exchanged source gradient, compute gradient and no state gradient.
        """
        state = ctx.state
        send_gradient = None
        if state.needs_send_grad:
            if state.backward_a2a_handle is None:
                raise RuntimeError("MLA input consumer did not submit its inverse A2A")
            send_gradient = state.backward_a2a_handle.wait()
            state.backward_a2a_handle = None
        return send_gradient, compute_gradient, None


class _LatentInputA2AWait(torch.autograd.Function):
    """Launch inverse Q before releasing the local KV projection gradients."""

    @staticmethod
    def forward(ctx: Any, key: torch.Tensor, value: torch.Tensor, state: MLAInputA2AState) -> tuple:
        """Attach completed Q and both KV edges to the paired consumer.

        Args:
            ctx: Autograd context for the paired Q A2A.
            key: Expanded head-local keys.
            value: Expanded head-local values.
            state: Q A2A submitted before the KV projection.

        Returns:
            Attention inputs in query, key, value order.
        """
        ctx.state = state
        query = state.forward_a2a_handle.wait()
        state.forward_a2a_handle = None
        return query, key.view_as(key), value.view_as(value)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_query: torch.Tensor, grad_key: torch.Tensor,
                 grad_value: torch.Tensor) -> tuple:
        """Run KV backward while dQ is being returned to its token owner.

        Args:
            ctx: Autograd context containing the paired Q A2A.
            grad_query: Attention gradient for head-local queries.
            grad_key: Attention gradient for head-local keys.
            grad_value: Attention gradient for head-local values.

        Returns:
            Key and value gradients, followed by no state gradient.
        """
        state = ctx.state
        if state.needs_send_grad:
            state.backward_a2a_handle = async_ulysses_seq_to_head_launch(grad_query, 2, 1, state.mesh)
        return grad_key, grad_value, None


@dataclass
class MLAExpandedInputA2AState:
    """Own only the Q/K/V transfers for one expanded forward invocation."""

    query: MLAInputA2AState
    key: MLAInputA2AState
    value: MLAInputA2AState

    def wait(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Consume Q/K/V in attention order after all forward work is submitted."""
        results = []
        for a2a_state in (self.query, self.key, self.value):
            results.append(a2a_state.forward_a2a_handle.wait())
            a2a_state.forward_a2a_handle = None
        return results[0], results[1], results[2]

    def launch_backward(self, grad_query: torch.Tensor, grad_key: torch.Tensor,
                        grad_value: torch.Tensor) -> None:
        """Submit each required inverse before releasing the independent compute.

        Args:
            grad_query: Head-local query gradient from attention.
            grad_key: Head-local key gradient from attention.
            grad_value: Head-local value gradient from attention.
        """
        for a2a_state, gradient in ((self.query, grad_query), (self.key, grad_key), (self.value, grad_value)):
            if a2a_state.needs_send_grad:
                a2a_state.backward_a2a_handle = async_ulysses_seq_to_head_launch(gradient, 2, 1, a2a_state.mesh)


class _ExpandedInputA2AWait(torch.autograd.Function):
    """Finish ordinary MLA inputs and return only Q's gradient to its projection."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, state: MLAExpandedInputA2AState) -> tuple:
        """Submit Q after the shared V/K launches, then consume all three inputs.

        Args:
            ctx: Autograd context holding the three input transfers.
            query: Final token-local query, with no following gate computation.
            state: Per-forward Q/K/V communication state.

        Returns:
            Completed attention inputs in query, key, value order.
        """
        ctx.state = state
        state.query.needs_send_grad = ctx.needs_input_grad[0]
        state.query.forward_a2a_handle = async_ulysses_seq_to_head_launch(query, 1, 2, state.query.mesh)
        return state.wait()

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_query: torch.Tensor, grad_key: torch.Tensor,
                 grad_value: torch.Tensor) -> tuple:
        """Submit dQ/dK/dV and release Q backward after dQ has arrived.

        Args:
            ctx: Autograd context holding the three paired input transfers.
            grad_query: Attention gradient for head-local queries.
            grad_key: Attention gradient for head-local keys.
            grad_value: Attention gradient for head-local values.

        Returns:
            Token-local Q gradient and no communication-state gradient.
        """
        state = ctx.state
        state.launch_backward(grad_query, grad_key, grad_value)
        query_gradient = None
        if state.query.needs_send_grad:
            query_gradient = state.query.backward_a2a_handle.wait()
            state.query.backward_a2a_handle = None
        return query_gradient, None


def launch_input_a2a(send_tensor: torch.Tensor, compute_input: torch.Tensor,
                     a2a_state: MLAInputA2AState) -> torch.Tensor:
    """Submit an input A2A while retaining its native collective backward.

    Args:
        send_tensor: Final token-local Q, K or V ready to send.
        compute_input: Input of the independent branch to execute next.
        a2a_state: State receiving this forward communication handle.

    Returns:
        Unchanged compute_input, without adding a paired autograd dependency.
    """
    a2a_state.forward_a2a_handle = async_ulysses_seq_to_head_launch(send_tensor, 1, 2, a2a_state.mesh)
    return compute_input


def finish_expanded_inputs(query: torch.Tensor, state: MLAExpandedInputA2AState,
                           overlap_backward: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finish ungated expanded inputs using paired or native collective gradients.

    Args:
        query: Final token-local query, after V and K have been submitted.
        state: This invocation's Q/K/V communication state.
        overlap_backward: Pair input autograd nodes for the pure-Ulysses backward.

    Returns:
        Completed query, key and value tensors in head-local layout.
    """
    if overlap_backward:
        return _ExpandedInputA2AWait.apply(query, state)
    state.query.forward_a2a_handle = async_ulysses_seq_to_head_launch(query, 1, 2, state.query.mesh)
    return state.wait()


class _LatentKVLinear(torch.autograd.Function):
    """Schedule the gathered latent adjoint before selected-head weight gradients."""

    @staticmethod
    def forward(ctx: Any, payload: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                gathered: torch.Tensor, mesh: Any) -> tuple:
        """Reuse the completed gather inside the managed Linear module.

        Args:
            ctx: Autograd state for this projection.
            payload: Local payload supplying the single gather gradient edge.
            weight: Selected head rows of the managed projection weight.
            bias: Optional selected projection bias.
            gathered: Completed full-sequence latent and shared-key payload.
            mesh: Chronological CP mesh used by the forward gather.

        Returns:
            Expanded KV heads and the shared key, without duplicating its gradient.
        """
        del payload
        latent = gathered[..., :weight.shape[-1]].contiguous()
        shared_key = gathered[..., weight.shape[-1]:]
        ctx.save_for_backward(latent, weight)
        ctx.mesh, ctx.has_bias = mesh, bias is not None
        ctx.device_type = latent.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type) if ctx.autocast_enabled else None
        return F.linear(latent, weight, bias), shared_key.view_as(shared_key)  # pylint: disable=not-callable

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor, shared_gradient: torch.Tensor) -> tuple:
        """Overlap SUM ReduceScatter with the independent KV weight gradient.

        Args:
            ctx: Saved projection operands, mesh and autocast state.
            gradient: Gradient of the expanded head-local KV features.
            shared_gradient: Gradient of the gathered shared key features.

        Returns:
            Local payload, weight and optional bias gradients; no gather or mesh gradient.
        """
        with torch.autocast(ctx.device_type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype):
            return _latent_projection_backward(ctx, gradient, shared_gradient)


def _latent_projection_backward(ctx, gradient, shared_gradient):
    """Keep both reduction buffers alive until their local gradient is consumed."""
    activation, weight = ctx.saved_tensors
    needs_payload_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
    work = None
    if needs_payload_grad:
        grad_latent = gradient.matmul(weight).to(activation.dtype)
        grad_payload = torch.cat((grad_latent, shared_gradient), dim=-1)
        send = grad_payload.movedim(1, 0).contiguous()
        shape = list(send.shape)
        shape[0] //= ctx.mesh.size()
        receive = send.new_empty(shape)
        work = dist.reduce_scatter_tensor(
            receive, send, op=dist.ReduceOp.SUM, group=ctx.mesh.get_group(), async_op=True,
        )
    flat_gradient = gradient.reshape(-1, gradient.shape[-1])
    grad_weight = None
    if needs_weight_grad:
        flat_activation = activation.reshape(-1, activation.shape[-1])
        grad_weight = flat_gradient.transpose(0, 1).matmul(flat_activation)
    grad_bias = flat_gradient.sum(0) if ctx.has_bias and needs_bias_grad else None
    local = None
    if work is not None:
        work.wait()
        local = receive.movedim(0, 1).contiguous()
    return local, grad_weight, grad_bias, None, None

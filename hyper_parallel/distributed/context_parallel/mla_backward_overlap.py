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
"""MLA input and output backward scheduling using paired Ulysses exchanges.

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
from hyper_parallel.distributed.context_parallel.collectives import async_ulysses_seq_to_head_launch


@dataclass
class MLAOutputExchange:
    """Per-forward state owned by a paired output projection and gradient consumer."""

    mesh: Any
    head_dim: int
    pending: Any = None


class _RestoreOutput(torch.autograd.Function):
    """Connect a completed forward exchange to the projection's inverse exchange."""

    @staticmethod
    def forward(ctx: Any, attention: torch.Tensor, restored: torch.Tensor,
                state: MLAOutputExchange) -> torch.Tensor:
        """Attach the original attention edge without retaining its tensor storage.

        Args:
            ctx: Autograd context holding only the invocation state.
            attention: Head-sharded attention result owning the gradient edge.
            restored: Token-local output from the completed forward exchange.
            state: Shared only with this invocation's output projection.

        Returns:
            A view of the restored output with the paired gradient edge."""
        del attention
        ctx.state = state
        return restored.view_as(restored)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, _gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Consume the exchange submitted by the paired output projection.

        Args:
            ctx: Autograd context with the paired communication handle.
            _gradient: Local gradient; its exchanged value is already in flight.

        Returns:
            Head-sharded attention gradient and empty metadata gradients."""
        state = ctx.state
        if state.pending is None:
            raise RuntimeError("MLA output projection did not submit its gradient exchange")
        result = state.pending.wait()
        state.pending = None
        return result, None, None


class _OutputLinear(torch.autograd.Function):
    """Launch dO before wgrad while preserving the ordinary Linear and gate math."""

    @staticmethod
    def forward(ctx: Any, activation: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None, gate: torch.Tensor | None,
                state: MLAOutputExchange) -> torch.Tensor:
        """Save Linear operands and the optional token-local gate.

        Args:
            ctx: Autograd context for operands and autocast settings.
            activation: Restored token-local attention output.
            weight: Managed output-projection weight.
            bias: Optional output-projection bias.
            gate: Optional token-local gate after its sigmoid.
            state: Paired invocation state for the inverse output exchange.

        Returns:
            Gated and projected local hidden states."""
        projected = activation if gate is None else activation * gate
        ctx.save_for_backward(activation, projected, weight, gate)
        ctx.state, ctx.has_bias = state, bias is not None
        ctx.device_type = activation.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type) if ctx.autocast_enabled else None
        return F.linear(projected, weight, bias)  # pylint: disable=not-callable

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Submit the inverse output exchange before computing weight gradients.

        Args:
            ctx: Saved operands, autocast settings and exchange state.
            gradient: Gradient of the projected hidden states.

        Returns:
            Gradients for activation, weight, bias and gate; no state gradient."""
        with torch.autocast(ctx.device_type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype):
            return _output_linear_backward(ctx, gradient)


def _output_linear_backward(ctx, gradient):
    """Match forward autocast and skip communication when attention is frozen."""
    activation, projected, weight, gate = ctx.saved_tensors
    grad_projected = gradient.matmul(weight).to(projected.dtype)
    grad_activation = None
    if ctx.needs_input_grad[0]:
        grad_activation = grad_projected if gate is None else grad_projected * gate
        state = ctx.state
        exchange_input = grad_activation.reshape(*activation.shape[:2], -1, state.head_dim)
        state.pending = async_ulysses_seq_to_head_launch(exchange_input, 1, 2, state.mesh)
    grad_gate = grad_projected * activation if gate is not None and ctx.needs_input_grad[3] else None
    flat_input = projected.reshape(-1, projected.shape[-1])
    flat_gradient = gradient.reshape(-1, gradient.shape[-1])
    grad_weight = None
    if ctx.needs_input_grad[1]:
        grad_weight = flat_input.transpose(0, 1).matmul(flat_gradient).transpose(0, 1)
    grad_bias = flat_gradient.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
    return grad_activation, grad_weight, grad_bias, grad_gate, None


def restore_mla_output_gradient(attention: torch.Tensor, restored: torch.Tensor,
                                state: MLAOutputExchange) -> torch.Tensor:
    """Attach the paired backward consumer to a completed output exchange.

    Args:
        attention: Head-sharded attention result owning the gradient edge.
        restored: Completed token-local forward result, without a gradient edge.
        state: Fresh state for this invocation; never shared across microbatches.

    Returns:
        Token-local result whose backward waits for the projection's dO exchange.
    """
    return _RestoreOutput.apply(attention, restored, state)


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
    def output_projection(input: torch.Tensor, *, mla_output_exchange: MLAOutputExchange | None = None,
                          mla_output_gate: torch.Tensor | None = None) -> torch.Tensor:
        """Read managed parameters after their normal module pre-hooks.

        Args:
            input: Token-local attention output.
            mla_output_exchange: Optional paired output-gradient exchange state.
            mla_output_gate: Optional post-sigmoid output gate.

        Returns:
            Projected hidden states using ordinary or scheduled autograd."""
        if mla_output_exchange is None:
            return original(input)
        return _OutputLinear.apply(input, projection.weight, projection.bias, mla_output_gate, mla_output_exchange)

    return _ForwardRewriteRequest(projection, output_projection)


@dataclass
class MLAInputExchange:
    """Per-invocation ownership of one input exchange and its inverse."""

    mesh: Any
    pending_forward: Any = None
    pending_backward: Any = None
    needs_first_gradient: bool = False


class _InputLaunch(torch.autograd.Function):
    """Keep independent projection work between a launch and its gradient wait."""

    @staticmethod
    def forward(ctx: Any, first: torch.Tensor, second: torch.Tensor,
                state: MLAInputExchange) -> torch.Tensor:
        """Launch the first input and retain both source gradient edges."""
        ctx.state = state
        state.needs_first_gradient = ctx.needs_input_grad[0]
        state.pending_forward = async_ulysses_seq_to_head_launch(first, 1, 2, state.mesh)
        return second.view_as(second)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Consume the first inverse exchange after the second branch backward."""
        state = ctx.state
        first_gradient = None
        if state.needs_first_gradient:
            if state.pending_backward is None:
                raise RuntimeError("MLA input consumer did not submit its inverse exchange")
            first_gradient = state.pending_backward.wait()
            state.pending_backward = None
        return first_gradient, gradient, None


class _ExpandedInputWait(torch.autograd.Function):
    """Return expanded KV/Q after completing their paired forward exchanges."""

    @staticmethod
    def forward(ctx: Any, second: torch.Tensor, state: MLAInputExchange) -> tuple:
        """Complete both inputs before exposing them to attention."""
        ctx.state = state
        second = async_ulysses_seq_to_head_launch(second, 1, 2, state.mesh).wait()
        first = state.pending_forward.wait()
        state.pending_forward = None
        return first, second

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, first_gradient: torch.Tensor, second_gradient: torch.Tensor) -> tuple:
        """Release dQ while the inverse expanded KV exchange remains in flight."""
        state = ctx.state
        second = async_ulysses_seq_to_head_launch(second_gradient, 2, 1, state.mesh)
        if state.needs_first_gradient:
            state.pending_backward = async_ulysses_seq_to_head_launch(first_gradient, 2, 1, state.mesh)
        return second.wait(), None


class _LatentInputWait(torch.autograd.Function):
    """Launch inverse Q before releasing the local KV projection gradients."""

    @staticmethod
    def forward(ctx: Any, key: torch.Tensor, value: torch.Tensor, state: MLAInputExchange) -> tuple:
        """Attach completed Q and both KV edges to the paired consumer."""
        ctx.state = state
        query = state.pending_forward.wait()
        state.pending_forward = None
        return query, key.view_as(key), value.view_as(value)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_query: torch.Tensor, grad_key: torch.Tensor,
                 grad_value: torch.Tensor) -> tuple:
        """Run KV backward while dQ is being returned to its token owner."""
        state = ctx.state
        if state.needs_first_gradient:
            state.pending_backward = async_ulysses_seq_to_head_launch(grad_query, 2, 1, state.mesh)
        return grad_key, grad_value, None


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
        """Overlap SUM ReduceScatter with the independent KV weight gradient."""
        with torch.autocast(ctx.device_type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype):
            return _latent_projection_backward(ctx, gradient, shared_gradient)


def _latent_projection_backward(ctx, gradient, shared_gradient):
    """Keep both reduction buffers alive until their local gradient is consumed."""
    activation, weight = ctx.saved_tensors
    work = None
    if ctx.needs_input_grad[0]:
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
    if ctx.needs_input_grad[1]:
        flat_activation = activation.reshape(-1, activation.shape[-1])
        grad_weight = flat_gradient.transpose(0, 1).matmul(flat_activation)
    grad_bias = flat_gradient.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
    local = None
    if work is not None:
        work.wait()
        local = receive.movedim(0, 1).contiguous()
    return local, grad_weight, grad_bias, None, None

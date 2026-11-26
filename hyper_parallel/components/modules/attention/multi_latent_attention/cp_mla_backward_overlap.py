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
"""MLA-specific latent and output projection backward scheduling.

Paired launch/wait nodes require a complete first-order training backward.
Partial autograd.grad calls can prune the consumer and are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# This CP scheduler uses the native Torch autograd and communication APIs.
import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from hyper_parallel.distributed.context_parallel.collectives import (
    async_ulysses_seq_to_head_launch,
)
from hyper_parallel.distributed.context_parallel.ulysses_overlap import UlyssesInputA2AState


class CPMLAOutputLinear(nn.Linear):
    """A managed Linear boundary with explicit optional CP output scheduling."""

    def __init__(self, source: nn.Linear) -> None:
        """Retain source Parameters without registering an extra child prefix."""
        nn.Module.__init__(self)
        if type(source) is not nn.Linear and not isinstance(source, CPMLAOutputLinear):  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA scheduled output requires a plain Linear")
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight, self.bias = source.weight, source.bias
        self.train(source.training)

    def forward(self, attention_output: torch.Tensor, *, mla_output_state: Any = None) -> torch.Tensor:
        """Read materialized parameters after the normal Module pre-hooks.

        Args:
            attention_output: Local attention output or head-partitioned CP output.
            mla_output_state: Invocation-local CP layout; absent for an ordinary projection.
        """
        if mla_output_state is None:
            return F.linear(attention_output, self.weight, self.bias)
        return _OutputProjection.apply(attention_output, self.weight, self.bias, mla_output_state)


class CPMLAKVLinear(nn.Linear):
    """Select computation heads inside the managed KV up-projection boundary."""

    def __init__(self, source: nn.Linear, head_dim: int) -> None:
        """Reuse full TP-local parameter storage; CP changes only computation ownership."""
        nn.Module.__init__(self)
        if type(source) is not nn.Linear and not isinstance(source, CPMLAKVLinear):  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA KV projection requires a plain Linear")
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight, self.bias = source.weight, source.bias
        self.head_dim = head_dim
        self.train(source.training)

    def forward(self, latent: torch.Tensor, *, mla_head_range: tuple[int, int] | None = None,
                mla_gathered: torch.Tensor | None = None, mla_cp_mesh: Any = None) -> Any:
        """Project all heads or selected rows with the existing latent RS adjoint.

        Args:
            latent: KV latent tokens.
            mla_head_range: Optional half-open interval of computation heads.
            mla_gathered: Gathered latent storage used by the scheduled derivative.
            mla_cp_mesh: CP mesh required by the scheduled latent derivative.
        """
        if mla_head_range is None:
            return F.linear(latent, self.weight, self.bias)
        begin, end = mla_head_range
        if not 0 <= begin < end <= self.weight.shape[0] // self.head_dim:
            raise ValueError("MLA projection head range is outside the managed parameter")
        rows = slice(begin * self.head_dim, end * self.head_dim)
        bias = None if self.bias is None else self.bias[rows]
        if mla_gathered is not None:
            if mla_cp_mesh is None:
                raise ValueError("Scheduled latent projection requires its CP mesh")
            return _LatentKVLinear.apply(latent, self.weight[rows], bias, mla_gathered, mla_cp_mesh)
        return F.linear(latent, self.weight[rows], bias)


@dataclass
class MLAOutputProjectionState:
    """Value-only output layout; the projection node owns its O/dO A2A handles.

    FSDP may rebuild this container without affecting the derivative: no paired
    node relies on mutating it to deliver a gradient or communication handle.
    """

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


class _LatentInputA2AWait(torch.autograd.Function):
    """Launch inverse Q before releasing the local KV projection gradients."""

    @staticmethod
    def forward(ctx: Any, key: torch.Tensor, value: torch.Tensor, state: UlyssesInputA2AState) -> tuple:
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

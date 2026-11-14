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
"""Gate-specific input dependencies and output derivatives for MLA overlap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Gate math and its autograd scheduling use native Torch.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.collectives import (
    AsyncCPCollective, async_ulysses_seq_to_head_launch,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.cp_mla_backward_overlap import (
    MLAExpandedInputA2AState, _InputA2ALaunch, launch_input_a2a,
)


class CPGatedMLAOutputLinear(nn.Linear):
    """Managed output projection with the gated MLA derivative contract."""

    def __init__(self, source: nn.Linear) -> None:
        """Preserve the source parameter tree and train/freeze state."""
        nn.Module.__init__(self)
        if type(source) is not nn.Linear and not isinstance(source, CPGatedMLAOutputLinear):  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Gated MLA scheduled output requires a plain Linear")
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight, self.bias = source.weight, source.bias
        self.train(source.training)

    def forward(self, input: torch.Tensor, *, mla_output_state: Any = None,
                mla_output_gate: torch.Tensor | None = None) -> torch.Tensor:
        """Enter the existing output Function only for an explicit CP invocation.

        Args:
            input: Local or head-partitioned attention output.
            mla_output_state: Invocation-local layout and optional gate-gradient handoff.
            mla_output_gate: Sigmoid gate paired with a scheduled CP output.
        """
        if mla_output_state is None:
            if mla_output_gate is not None:
                raise ValueError("Scheduled gated output requires a state")
            return F.linear(input, self.weight, self.bias)
        if mla_output_gate is None:
            raise ValueError("Scheduled gated output requires its sigmoid gate")
        return _GatedOutputProjection.apply(input, self.weight, self.bias, mla_output_gate, mla_output_state)


class GatedMLAInputState:
    """Invocation-local gate values and the optional deferred native derivative.

    The adapter supplies the model-owned gate computation. It is released after
    forward; the remaining state belongs only to this invocation's paired nodes.
    With defer_gate_gradient, the output saves dG and the input consumer returns
    it only after submitting the input-gradient A2As.
    """

    def __init__(self, hidden: torch.Tensor | None,
                 projection: Callable[[torch.Tensor], torch.Tensor] | None,
                 defer_gate_gradient: bool) -> None:
        """Keep one identity across the input/output autograd nodes and FSDP hooks.

        FSDP maps tensors recursively inside dataclass arguments. This shared
        handoff is an opaque control object, not a tensor container to rebuild.
        Its forward-only hidden/projection references are released by project().
        """
        self.hidden = hidden
        self.projection = projection
        self.defer_gate_gradient = defer_gate_gradient
        self.value: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None

    def project(self, source: torch.Tensor) -> torch.Tensor:
        """Run the managed gate and release its temporary forward-only references.

        Args:
            source: Local hidden states, optionally carrying the input launch edge.

        Returns:
            The model's post-activation gate in local-token layout.
        """
        if self.projection is None:
            raise RuntimeError("MLA input gate must be projected exactly once")
        self.value = self.projection(source)
        self.hidden, self.projection = None, None
        return self.value

    def take_gradient(self) -> torch.Tensor:
        """Consume the output multiplication's deferred gate gradient once."""
        if self.gradient is None:
            raise RuntimeError("MLA output projection did not release its gate gradient")
        gradient, self.gradient = self.gradient, None
        return gradient


class _GatedExpandedInputA2AWait(torch.autograd.Function):
    """Release gate backward only after the attention input gradients are sent."""

    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, state: MLAExpandedInputA2AState,
                gate_state: GatedMLAInputState) -> tuple:
        """Consume Q/K/V after native gate compute, keeping its gradient edge.

        Args:
            ctx: Autograd context for communication and gate-gradient ownership.
            gate: Post-sigmoid local-token gate, whose backward is deferred here.
            state: Q/K/V transfers already submitted before gate compute.
            gate_state: Invocation-local handoff from the gated output derivative.

        Returns:
            Completed attention inputs in query, key, value order.
        """
        # Output projection consumes the gate value. This argument keeps the
        # edge on which this consumer will return the deferred dG.
        del gate
        ctx.state, ctx.gate_state = state, gate_state
        return state.wait()

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_query: torch.Tensor, grad_key: torch.Tensor,
                 grad_value: torch.Tensor) -> tuple:
        """Launch dQ/dK/dV before delivering dG to native sigmoid/g_proj backward.

        Args:
            ctx: This invocation's communication and gate-gradient states.
            grad_query: Head-local query gradient from attention.
            grad_key: Head-local key gradient from attention.
            grad_value: Head-local value gradient from attention.

        Returns:
            Gate gradient and no gradients for either state object.
        """
        ctx.state.launch_backward(grad_query, grad_key, grad_value)
        return ctx.gate_state.take_gradient(), None, None


def finish_gated_expanded_inputs(
    query: torch.Tensor, state: MLAExpandedInputA2AState, gate_state: GatedMLAInputState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cover the last Q A2A with the model-owned native gate projection.

    Args:
        query: Final token-local query, after V and K have been submitted.
        state: This invocation's Q/K/V communication state.
        gate_state: Model gate computation and its selected gradient handoff.

    Returns:
        Completed attention inputs in head-local layout; gate is saved in gate_state.
    """
    launch_input = _InputA2ALaunch.apply if gate_state.defer_gate_gradient else launch_input_a2a
    gate_input = launch_input(query, gate_state.hidden, state.query)
    gate = gate_state.project(gate_input)
    if gate_state.defer_gate_gradient:
        return _GatedExpandedInputA2AWait.apply(gate, state, gate_state)
    return state.wait()


@dataclass
class GatedMLAOutputProjectionState:
    """Own gated output communication and optional input-side gate gradients.

    A pre-launched O transfer covers the native gate projection. For input-side
    gating, input_gate instead delivers dG to the paired input consumer, which
    releases the native gate backward after submitting the input gradients.
    """

    mesh: Any
    head_dim: int
    input_gate: GatedMLAInputState | None = None
    forward_a2a_handle: AsyncCPCollective | None = None


class _GatedOutputProjection(torch.autograd.Function):
    """Own gated output multiplication, Linear and the inverse output A2A."""

    @staticmethod
    def forward(ctx: Any, attention_output: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None, gate: torch.Tensor,
                state: GatedMLAOutputProjectionState) -> torch.Tensor:
        """Restore the FA output, multiply its required gate and apply Wo.

        Args:
            ctx: Autograd context for operands, layout and autocast settings.
            attention_output: FA output in [batch, sequence, local_heads, value_dim].
            weight: Managed output-projection weight.
            bias: Optional output-projection bias.
            gate: Native post-sigmoid gate in token-local layout.
            state: Output layout, O transfer and deferred gate consumer.

        Returns:
            Gated and projected local hidden states.
        """
        output_a2a_handle, state.forward_a2a_handle = state.forward_a2a_handle, None
        if output_a2a_handle is None:
            output_a2a_handle = async_ulysses_seq_to_head_launch(attention_output, 2, 1, state.mesh)
        # This Function owns the collective's derivative. Native g_proj and
        # sigmoid stay outside it so their module hooks and autograd still run.
        local_output = output_a2a_handle.wait().flatten(2)
        projected = local_output * gate
        ctx.save_for_backward(local_output, projected, weight, gate)
        ctx.state, ctx.has_bias = state, bias is not None
        ctx.device_type = local_output.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type) if ctx.autocast_enabled else None
        return F.linear(projected, weight, bias)  # pylint: disable=not-callable

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Overlap dO A2A with gate multiplication gradients and Wo wgrad.

        Args:
            ctx: Saved operands, autocast settings and output layout.
            gradient: Gradient of the projected hidden states.

        Returns:
            FA output, weight, bias, gate and no state gradients. Input-side
            gating delivers dG through its paired consumer instead.
        """
        with torch.autocast(ctx.device_type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype):
            return _gated_output_projection_backward(ctx, gradient)


def _gated_output_projection_backward(ctx, gradient):
    """Keep the dO launch before wgrad and the wait before returning dF."""
    local_output, projected, weight, gate = ctx.saved_tensors
    state = ctx.state
    needs_attention_grad, needs_weight_grad, needs_bias_grad, needs_gate_grad = ctx.needs_input_grad[:4]
    needs_gate = needs_gate_grad or state.input_gate is not None
    grad_projected = gradient.matmul(weight).to(projected.dtype) if needs_attention_grad or needs_gate else None
    grad_output_a2a = None
    if needs_attention_grad:
        grad_local_output = (grad_projected * gate).reshape(*local_output.shape[:2], -1, state.head_dim)
        grad_output_a2a = async_ulysses_seq_to_head_launch(grad_local_output, 1, 2, state.mesh)
    grad_gate = grad_projected * local_output if needs_gate else None
    if state.input_gate is not None:
        state.input_gate.gradient = grad_gate
        grad_gate = None
    flat_input = projected.reshape(-1, projected.shape[-1])
    flat_gradient = gradient.reshape(-1, gradient.shape[-1])
    grad_weight = None
    if needs_weight_grad:
        grad_weight = flat_input.transpose(0, 1).matmul(flat_gradient).transpose(0, 1)
    grad_bias = flat_gradient.sum(0) if ctx.has_bias and needs_bias_grad else None
    grad_attention_output = grad_output_a2a.wait() if grad_output_a2a is not None else None
    return grad_attention_output, grad_weight, grad_bias, grad_gate, None

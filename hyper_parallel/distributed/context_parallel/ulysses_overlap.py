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
"""Paired Ulysses input transfers shared by attention modules.

Inputs use [batch, local_sequence, heads, head_dim] and are exchanged into
[batch, global_sequence, local_heads, head_dim]. Every forward or checkpoint
replay owns separate state; handles and buffers must not be shared across calls.

Paired launch/wait nodes require a complete first-order training backward.
Partial autograd.grad calls can prune the consumer and are not supported.
Model projections, rotary math and parameter management remain caller-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.distributed.context_parallel.collectives import (
    AsyncCPCollective, async_ulysses_seq_to_head_launch,
)


@dataclass
class UlyssesInputA2AState:
    """Own one input A2A and its inverse for a single forward invocation.

    Attributes:
        mesh: Ulysses device mesh.
        forward_a2a_handle: Q/K/V transfer from token-local to head-local layout.
        backward_a2a_handle: Gradient transfer from head-local to token-local layout.
        needs_send_grad: Whether the original input requires its inverse transfer.

    Each handle owns the communication work and its buffers. The paired consumer
    waits for it exactly once before reading the result, then clears the field.
    This shared state passes directly between autograd nodes, not through the
    managed projection's forward arguments and FSDP input conversion.
    """

    mesh: Any
    forward_a2a_handle: AsyncCPCollective | None = None
    backward_a2a_handle: AsyncCPCollective | None = None
    needs_send_grad: bool = False


class _InputA2ALaunch(torch.autograd.Function):
    """Keep independent native compute between a send and its gradient wait."""

    @staticmethod
    def forward(ctx: Any, send_tensor: torch.Tensor, compute_input: torch.Tensor,
                state: UlyssesInputA2AState) -> torch.Tensor:
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
                raise RuntimeError("Ulysses input consumer did not submit its inverse A2A")
            send_gradient = state.backward_a2a_handle.wait()
            state.backward_a2a_handle = None
        return send_gradient, compute_gradient, None


@dataclass
class UlyssesQKVA2AState:
    """Own the Q/K/V transfers for one Ulysses forward invocation."""

    query: UlyssesInputA2AState
    key: UlyssesInputA2AState
    value: UlyssesInputA2AState

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


class _QKVInputA2AWait(torch.autograd.Function):
    """Finish Q/K/V inputs and release Q backward before waiting for K/V."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, state: UlyssesQKVA2AState) -> tuple:
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
                     a2a_state: UlyssesInputA2AState, *, overlap_backward: bool = False) -> torch.Tensor:
    """Submit an input A2A with either native or paired collective backward.

    Args:
        send_tensor: Final token-local Q, K or V ready to send.
        compute_input: Input of the independent branch to execute next.
        a2a_state: State receiving this forward communication handle.
        overlap_backward: Attach a paired gradient dependency. The caller must
            finish these inputs with the matching paired wait and execute a
            complete first-order backward, as in ``finish_qkv_inputs``.

    Returns:
        compute_input, with the paired dependency when requested.
    """
    if overlap_backward:
        return _InputA2ALaunch.apply(send_tensor, compute_input, a2a_state)
    a2a_state.forward_a2a_handle = async_ulysses_seq_to_head_launch(send_tensor, 1, 2, a2a_state.mesh)
    return compute_input


def finish_qkv_inputs(query: torch.Tensor, state: UlyssesQKVA2AState,
                           overlap_backward: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finish Q/K/V inputs using paired or native collective gradients.

    Args:
        query: Final token-local query, after V and K have been submitted.
        state: This invocation's Q/K/V communication state.
        overlap_backward: Pair input autograd nodes for the pure-Ulysses backward.

    Returns:
        Completed query, key and value tensors in head-local layout.
    """
    if overlap_backward:
        return _QKVInputA2AWait.apply(query, state)
    state.query.forward_a2a_handle = async_ulysses_seq_to_head_launch(query, 1, 2, state.query.mesh)
    return state.wait()

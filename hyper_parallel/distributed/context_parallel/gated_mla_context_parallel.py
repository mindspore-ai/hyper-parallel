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
"""Context-parallel scheduling for NoPE MLA with a required sigmoid gate."""

from __future__ import annotations

from typing import Any, Mapping

# Model-owned gate math and CP scheduling use native Torch.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.collectives import async_ulysses_seq_to_head_launch
from hyper_parallel.distributed.context_parallel.gated_mla_backward_overlap import (
    GatedMLAInputState, GatedMLAOutputProjectionState,
)
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLAContextParallel


def gated_mla_forward(module: nn.Module, runtime: MLAContextParallel, hidden_states: torch.Tensor,
                      attention_mask: torch.Tensor | None, options: Mapping[str, Any], *,
                      overlap_output_backward: bool = False) -> torch.Tensor:
    """Run shared MLA inputs and keep the gate at its chosen overlap boundary.

    Args:
        module: Gated MLA owning all projection parameters.
        runtime: Shared MLA CP planner and attention execution.
        hidden_states: Chronological local tokens in BSH layout.
        attention_mask: Optional mask supported by the runtime.
        options: Packed lengths, positions and attention options.
        overlap_output_backward: Cover dO A2A with output wgrad. Input backward
            additionally requires pure Ulysses; hybrid inputs retain native autograd.

    Returns:
        Gated and projected hidden states in the original local-token order.
    """
    # NoPE does not consume caller frequencies, including their layout metadata.
    # The module-owned identity callback also governs both CP projection paths.
    options = {name: value for name, value in options.items() if name != "position_embeddings"}
    # Input Q/K/V pairing is implemented only for pure Ulysses. Hybrid
    # paths can still overlap output dO A2A independently of their input adjoints.
    overlap_input_backward = overlap_output_backward and runtime.plan.all_gather_degree == 1
    input_gate = _prepare_input_gate(module, runtime, hidden_states, overlap_input_backward)
    if input_gate is not None:
        overlap_input_backward = input_gate.defer_gate_gradient
    attention_output, _ = runtime(
        module, *module.project_latent_inputs(hidden_states), attention_mask=attention_mask,
        restore_output=False, overlap_input_backward=overlap_input_backward, input_gate=input_gate, **options,
    )
    output_a2a_handle = None
    if input_gate is None and runtime.plan.overlap and runtime.plan.ulysses_degree > 1:
        # Launch before the native gate so its autograd graph and FSDP hooks
        # stay outside the output Function while O communication is in flight.
        output_a2a_handle = async_ulysses_seq_to_head_launch(attention_output, 2, 1, runtime.ulysses_mesh)
    if input_gate is None:
        gate = module.g_proj(hidden_states).sigmoid()
    else:
        gate, input_gate.value = input_gate.value, None
    if overlap_output_backward:
        deferred_gate = input_gate if input_gate is not None and input_gate.defer_gate_gradient else None
        state = GatedMLAOutputProjectionState(
            runtime.ulysses_mesh, module.v_head_dim, input_gate=deferred_gate, forward_a2a_handle=output_a2a_handle,
        )
        return module.o_proj(attention_output, mla_output_state=state, mla_output_gate=gate)
    output = output_a2a_handle.wait() if output_a2a_handle is not None else runtime.restore_output(attention_output)
    output = output.flatten(2)
    return module.o_proj(output * gate)


def _prepare_input_gate(module, runtime, hidden_states, overlap_input_backward):
    """Use input-side gating only for the expanded pure-Ulysses schedule."""
    plan = runtime.plan
    if not plan.overlap or plan.ulysses_degree <= 1:
        return None
    if plan.strategy != "expanded_ulysses" or plan.all_gather_degree != 1:
        return None
    # Gate-gradient handoff needs the input dependency. Frozen inputs use
    # native gate backward; inference does not build a backward graph.
    defer_gate_gradient = overlap_input_backward and hidden_states.requires_grad
    return GatedMLAInputState(
        hidden_states, lambda source: module.g_proj(source).sigmoid(), defer_gate_gradient,
    )

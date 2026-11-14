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
from hyper_parallel.components.modules.attention.gated_multi_latent_attention.cp_gated_mla_backward_overlap import (
    CPGatedMLAOutputLinear, GatedMLAInputState, GatedMLAOutputProjectionState, finish_gated_expanded_inputs,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.cp_multi_latent_attention import (
    CPMultiLatentAttentionModule, MLAContextParallel,
)
from hyper_parallel.components.modules.attention.gated_multi_latent_attention.gated_multi_latent_attention import (
    GatedMultiLatentAttentionModule,
)


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


class GatedMLAContextParallel(MLAContextParallel):
    """Gated input completion belongs to the gated family, not ordinary MLA."""

    def finish_input_exchange(self, query: torch.Tensor, transfers: Any, overlap_backward: bool,
                              input_gate: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep gate computation and its gradient dependency between Q launch and waits."""
        if input_gate is None:
            return super().finish_input_exchange(query, transfers, overlap_backward, None)
        return finish_gated_expanded_inputs(query, transfers, input_gate)


class CPGatedMultiLatentAttentionModule(GatedMultiLatentAttentionModule, CPMultiLatentAttentionModule):
    """NoPE and mandatory gate math with the same transport strategies as ordinary MLA."""

    def prepare_output_projection(self) -> None:
        """Install the gated output Function inside its managed projection Module."""
        if self.cp_plan.tp_degree != 1:
            raise ValueError("Gated MLA CP currently supports TP1 only")
        if self.cp_plan.overlap and self.cp_plan.ulysses_degree > 1:
            self.o_proj = CPGatedMLAOutputLinear(self.o_proj)

    def make_cp_execution(self, context: Any) -> GatedMLAContextParallel:
        """Borrow CP resources; select only the gated completion boundary."""
        return GatedMLAContextParallel(self.cp_plan, context.mesh)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                actual_seq_len: Any = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Run NoPE gated CP with per-invocation gate/communication state."""
        del position_embeddings
        if past_key_values is not None:
            raise ValueError("Gated MLA CP training does not support a KV cache")
        if self.cp_runtime is None:
            raise RuntimeError("MLA CP resources must be bound before forward")
        hidden_states = self.prepare_input(hidden_states)
        kwargs["actual_seq_len"] = actual_seq_len
        plan = self.cp_plan
        overlap_output = plan.overlap and plan.ulysses_degree > 1 and torch.is_grad_enabled()
        return gated_mla_forward(self, self.cp_runtime, hidden_states, attention_mask, kwargs,
                                 overlap_output_backward=overlap_output), None

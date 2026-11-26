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

from dataclasses import dataclass
from typing import Any

# Model-owned gate math and CP scheduling use native Torch.
import torch

from hyper_parallel.distributed.context_parallel.collectives import async_ulysses_seq_to_head_launch
from hyper_parallel.components.modules.attention.gated_multi_latent_attention.cp_gated_mla_backward_overlap import (
    CPGatedMLAOutputLinear, GatedMLAInputState, GatedMLAOutputProjectionState, finish_gated_expanded_inputs,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.cp_multi_latent_attention import (
    CPMultiLatentAttentionModule,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import MLALatentInputs
from hyper_parallel.components.modules.attention.gated_multi_latent_attention.gated_multi_latent_attention import (
    GatedMultiLatentAttentionModule,
)


@dataclass(frozen=True)
class CPGatedMLAOutputState:
    """Carry the ready local gate and any deferred joint O-exchange/Wo operation."""

    gate: torch.Tensor
    projection: GatedMLAOutputProjectionState | None = None


class CPGatedMultiLatentAttentionModule(CPMultiLatentAttentionModule, GatedMultiLatentAttentionModule):
    """Use shared MLA forward with NoPE/gate math and gated CP stage overrides."""

    def prepare_output_projection(self) -> None:
        """Install the existing gated output derivative inside its managed Linear."""
        if self.cp_plan.tp_degree != 1:
            raise ValueError("Gated MLA CP currently supports TP1 only")
        if self.cp_plan.cp_comm_overlap and self.cp_plan.ulysses_cp_size > 1:
            self.o_proj = CPGatedMLAOutputLinear(self.o_proj)

    def _prepare_input_gate(self, hidden_states: torch.Tensor) -> GatedMLAInputState | None:
        """Reserve a gate in the expanded pure-Ulysses input window, when enabled."""
        plan = self.cp_plan
        if not plan.cp_comm_overlap or plan.ulysses_cp_size <= 1:
            return None
        if plan.ulysses_cp_strategy != "expanded_ulysses" or plan.all_gather_degree != 1:
            return None
        # Frozen inputs keep native gate backward; the paired consumer otherwise
        # releases it after the input gradient collectives have been submitted.
        defer_gate = self._overlap_output_backward() and hidden_states.requires_grad
        return GatedMLAInputState(hidden_states, self.project_gate, defer_gate)

    def prepare_attention_inputs(
        self, hidden_states: torch.Tensor,
    ) -> tuple[MLALatentInputs, GatedMLAInputState | None]:
        """Attach the input-side gate dependency to the shared latent projection stage.

        Args:
            hidden_states: Local tokens supplying down projections and the algorithm-owned gate.

        Returns:
            Normalized latents and an optional gate state for expanded pure Ulysses.
        """
        input_gate = self._prepare_input_gate(hidden_states)
        state, _ = super().prepare_attention_inputs(hidden_states)
        return state, input_gate

    def finish_input_exchange(self, query: torch.Tensor, transfers: Any, overlap_backward: bool,
                              input_gate: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Place algorithm-owned gate computation between Q submission and input waits.

        Args:
            query: Ready query tensor in BSND layout.
            transfers: Pending input collective state shared with paired backward nodes.
            overlap_backward: Whether to retain the paired input adjoints.
            input_gate: This invocation's deferred gate, or None for ordinary scheduling.
        """
        if input_gate is None:
            return super().finish_input_exchange(query, transfers, overlap_backward, None)
        return finish_gated_expanded_inputs(query, transfers, input_gate)

    def restore_attention(self, output: torch.Tensor, hidden_states: torch.Tensor,
                          output_state: GatedMLAInputState | None) -> tuple[torch.Tensor, CPGatedMLAOutputState]:
        """Schedule token restoration and gate readiness before the managed output projection.

        Args:
            output: Head-partitioned FA output in BSND layout.
            hidden_states: Original local tokens used by the gate algorithm.
            output_state: Optional input-side gate state from expanded pure Ulysses.

        Returns:
            Restored output, or raw FA output when joint backward requires deferred
            restoration; the typed state carries the gate and the deferred operation.
        """
        input_gate = output_state
        output_a2a_handle = None
        if input_gate is None and self.cp_plan.cp_comm_overlap and self.cp_plan.ulysses_cp_size > 1:
            # The gate remains a native child-module call, with its original
            # FSDP hooks/autograd, while O communication is in flight.
            output_a2a_handle = async_ulysses_seq_to_head_launch(output, 2, 1, self.ulysses_mesh)
        if input_gate is None:
            gate = self.project_gate(hidden_states)
        else:
            gate, input_gate.value = input_gate.value, None
        if self._overlap_output_backward():
            deferred_gate = input_gate if input_gate is not None and input_gate.defer_gate_gradient else None
            state = GatedMLAOutputProjectionState(
                self.ulysses_mesh, self.v_head_dim, input_gate=deferred_gate, forward_a2a_handle=output_a2a_handle,
            )
            return output, CPGatedMLAOutputState(gate, state)
        if output_a2a_handle is not None:
            output = output_a2a_handle.wait()
        else:
            output, _ = super().restore_attention(output, hidden_states, None)
        return output, CPGatedMLAOutputState(gate)

    def project_output(self, attention_output: torch.Tensor, hidden_states: torch.Tensor,
                       output_state: CPGatedMLAOutputState) -> torch.Tensor:
        """Apply the ready gate through the algorithm or the existing joint output Function.

        Args:
            attention_output: Restored token-local output, or raw FA output for the joint Function.
            hidden_states: Original local tokens defining the output shape.
            output_state: Gate and optional O-exchange/Wo state produced by restore_attention.
        """
        if output_state.projection is not None:
            return self.o_proj(attention_output, mla_output_state=output_state.projection,
                               mla_output_gate=output_state.gate)
        return super().project_output(attention_output, hidden_states, output_state.gate)

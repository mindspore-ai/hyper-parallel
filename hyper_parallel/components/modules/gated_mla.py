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
"""NoPE MLA with a required sigmoid output gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# The high-performance attention modules use native Torch.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.mla_attention import MLAAttention


class GatedMLAAttention(MLAAttention):
    """Define NoPE attention and its mandatory output gate in one math module.

    The shared key features retain the source projection/checkpoint dimensions;
    NoPE disables their rotation. Model adapters only preserve forward APIs.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Validate the source math and retain its managed projection modules."""
        if not getattr(module, "use_nope", False):
            raise ValueError("GatedMLAAttention requires a NoPE source module")
        if not getattr(module, "use_output_gate", False):
            raise ValueError("GatedMLAAttention requires an enabled output gate")
        if type(module.g_proj) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Gated MLA requires a plain Linear output gate")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.use_nope = True
        self.g_proj = module.g_proj

    @staticmethod
    def apply_nope(tensor: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                   *, interleaved: bool = False) -> torch.Tensor:
        """Keep Q/shared-K features unchanged through the CP rotary callback.

        Args:
            tensor: Query or shared-key features in the runtime's BSND layout.
            position_embeddings: Ignored by NoPE attention.
            interleaved: Ignored because no positional rotation is applied.

        Returns:
            The original tensor, without a copy or a positional transform.
        """
        del position_embeddings, interleaved
        return tensor

    def _project_attention_inputs(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None, past_key_values: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the common projections while enforcing the NoPE math locally."""
        del position_embeddings
        return super()._project_attention_inputs(hidden_states, None, past_key_values)

    def project_output(self, attention_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the source gate before Wo, without an additional output Norm.

        Args:
            attention_output: Flattened token-local attention output.
            hidden_states: Original token-local input for the gate projection.

        Returns:
            Gated and projected hidden states.
        """
        attention_output = attention_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attention_output)

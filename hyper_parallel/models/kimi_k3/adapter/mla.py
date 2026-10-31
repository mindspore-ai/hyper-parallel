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
"""Gated NoPE MLA replacement preserving the Kimi attention contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Model adapters use native Torch, as required by the shared MLA replacement.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class KimiMLAAttention(MLAAttention):
    """Reuse MLA projections while retaining the source gate and tensor return."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Preserve the released NoPE and full-rank output-gate semantics."""
        if not getattr(module, "use_nope", False):
            raise ValueError("Kimi MLA replacement requires the NoPE source path")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.use_nope = True
        self.use_output_gate = bool(module.use_output_gate)
        if self.use_output_gate:
            if type(module.g_proj) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
                raise ValueError("Kimi MLA requires a plain Linear output gate")
            self.g_proj = module.g_proj

    def project_output(self, attention_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the source gate before Wo, without an additional output Norm.

        Args:
            attention_output: Flattened token-local attention output.
            hidden_states: Original token-local input for the gate projection.

        Returns:
            Gated and projected hidden states.
        """
        if self.use_output_gate:
            attention_output = attention_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attention_output)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, past_key_values: Any | None = None,
                **kwargs: Any) -> torch.Tensor:
        """Run local gated MLA with the source tensor-only return contract.

        Args:
            hidden_states: Local input in BSH layout.
            attention_mask: Optional mask forwarded to the attention backend.
            position_ids: Unused by NoPE attention.
            past_key_values: Cache is unsupported by this training replacement.
            **kwargs: Packed lengths and attention options.

        Returns:
            Local hidden states after gated attention and output projection.
        """
        del position_ids
        kwargs.pop("position_embeddings", None)
        if past_key_values is not None:
            raise ValueError("Kimi MLA training replacement does not support a KV cache")
        output, _ = super().forward(hidden_states, attention_mask=attention_mask, **kwargs)
        return output

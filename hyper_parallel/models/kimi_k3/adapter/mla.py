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

from typing import Any

# Model adapters use native Torch, as required by the shared MLA replacement.
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.gated_mla import GatedMLAAttention
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class KimiMLAAttention(GatedMLAAttention):
    """Adapt the gated NoPE math module to Kimi's forward API and Tensor return."""

    # Preserve Kimi's native (hidden, mask, position_ids, cache) API and Tensor return, not the HF base signature.
    # pylint: disable-next=arguments-differ
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
        if past_key_values is not None:
            raise ValueError("Kimi MLA training replacement does not support a KV cache")
        output, _ = super().forward(hidden_states, attention_mask=attention_mask, **kwargs)
        return output

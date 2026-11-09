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
"""Complete causal attention algorithm with explicit operand stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from examples.torch.module_family.config import DecoderConfig


@dataclass(frozen=True)
class AttentionOperands:
    """Projected operands and the first query's global sequence coordinate."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    offset: int = 0


class CausalAttentionModule(nn.Module):
    """Ordinary multi-head causal attention; exchange and restore are identity."""

    def __init__(self, *, config: DecoderConfig | None = None, module: nn.Module | None = None,
                 context: Any = None, backend: str = "torch") -> None:
        """Register logical parameters without acquiring parallel resources.

        Args:
            config: Complete global algorithm dimensions for direct construction.
            module: Compatible existing member for source conversion.
            context: Job selection facts, consumed by the family entry.
            backend: Selected numerical implementation.
        """
        super().__init__()
        del context, backend
        self.config = config if module is None else module.config
        self.qkv = nn.Linear(self.config.hidden_size, 3 * self.config.hidden_size, bias=False)
        self.output = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        if module is not None:
            self.qkv, self.output = module.qkv, module.output
            self.train(module.training)

    def exchange_attention(self, operands: AttentionOperands) -> AttentionOperands:
        """Preserve local projected operands.

        Args:
            operands: Local Q/K/V.
        """
        return operands

    def restore_attention(self, output: torch.Tensor) -> torch.Tensor:
        """Preserve output ownership for the local-query algorithm.

        Args:
            output: Attention output in local-query order.
        """
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute causal attention with class-defined operand stages.

        Args:
            hidden_states: Local sequence [B,S,H].
        """
        batch, length, width = hidden_states.shape
        heads = self.config.num_heads
        query, key, value = self.qkv(hidden_states).reshape(batch, length, 3, heads, width // heads).unbind(2)
        operands = self.exchange_attention(AttentionOperands(query, key, value))
        scores = torch.einsum("bshd,bthd->bhst", operands.query, operands.key) / ((width // heads) ** 0.5)
        queries = torch.arange(length, device=hidden_states.device) + operands.offset
        keys = torch.arange(operands.key.shape[1], device=hidden_states.device)
        scores = scores.masked_fill((keys[None, :] > queries[:, None])[None, None, :, :], float("-inf"))
        values = torch.einsum("bhst,bthd->bshd", scores.softmax(dim=-1), operands.value)
        return self.output(self.restore_attention(values).reshape(batch, length, width))

    def state_contract(self) -> dict[str, str]:
        """Declare the common representation and causal semantics."""
        return {"representation": "qkv_output_v1", "mask": "causal_contiguous"}

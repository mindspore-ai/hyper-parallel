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
"""DeepSeek-V3-style MTP layers and teacher-forced training orchestration.

Fusion uses the released-weight convention [normalized embedding, normalized
hidden], as in Transformers MtpLayer. The training loop uses ground-truth future
tokens, not the candidate-token generation loop in Transformers MtpModel.
Model adapters own decoder construction, attention/position metadata, distributed
layouts and checkpoint key mapping. No model-specific routing policy lives here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn

__all__ = ["DeepseekV3MTPLayer", "DeepseekV3MTP", "MTPOutput"]


class _MTPRMSNorm(nn.Module):
    """Match the HF DeepSeek normalization cast order, including FP32 weights."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        """Create a direct scale with the requested normalization epsilon."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32 and preserve the reference multiplication order."""
        input_dtype = hidden_states.dtype
        states = hidden_states.float()
        states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * states.to(input_dtype)


class DeepseekV3MTPLayer(nn.Module):
    """Fuse a future embedding with the previous depth's hidden state.

    Args:
        hidden_size: Width of both inputs and the decoder output.
        decoder: Model-owned causal Transformer block returning a Tensor.
        rms_norm_eps: Epsilon for the default DeepSeek-style RMSNorms.
        norm_factory: Optional factory accepting hidden_size. It must return a
            fresh module for each of enorm, hnorm and post_norm.

    Note:
        All three norms have independent parameters. The returned post-normalized
        state is used by both the shared head and the next MTP depth. A model with
        different fusion/normalization semantics should supply its own layer to
        DeepseekV3MTP, rather than changing this released-weight convention.
    """

    def __init__(self, hidden_size: int, decoder: nn.Module, *, rms_norm_eps: float = 1e-6,
                 norm_factory: Callable[[int], nn.Module] | None = None) -> None:
        """Build independent fusion and output norms around the supplied decoder."""
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive and finite")
        factory = norm_factory or partial(_MTPRMSNorm, eps=rms_norm_eps)
        self.enorm = factory(hidden_size)
        self.hnorm = factory(hidden_size)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.mtp_block = decoder
        self.post_norm = factory(hidden_size)

    def forward(self, hidden_states: torch.Tensor, inputs_embeds: torch.Tensor,
                **decoder_kwargs: Any) -> torch.Tensor:
        """Return [B, S, H] states; kwargs must describe this depth's sequence."""
        if hidden_states.shape != inputs_embeds.shape:
            raise ValueError("MTP hidden_states and inputs_embeds must have identical shapes")
        projection_input = torch.cat((self.enorm(inputs_embeds), self.hnorm(hidden_states)), dim=-1)
        return self.post_norm(self.mtp_block(self.eh_proj(projection_input), **decoder_kwargs))


@dataclass
class MTPOutput:
    """Depth k (one-based) has S-k positions; logits are empty if head is None."""

    hidden_states: tuple[torch.Tensor, ...]
    logits: tuple[torch.Tensor, ...]


class DeepseekV3MTP(nn.Module):
    """Run independent MTP depths sequentially with ground-truth future tokens.

    Args:
        layers: Independent layers with forward(hidden_states, inputs_embeds,
            **decoder_kwargs) -> Tensor. Normally DeepseekV3MTPLayer instances;
            adapters may supply another layer with the same contract.

    Note:
        This loop consumes complete, unsharded rows. It does not implement CP halo
        exchange, packed-document boundaries, generation/KV caching or automatic
        Trainer integration. Those belong to the model adapter. A distributed
        adapter may also call individual layers with already-aligned inputs.
    """

    def __init__(self, layers: Sequence[nn.Module]) -> None:
        """Register the prediction layers without copying external shared weights."""
        super().__init__()
        if not layers:
            raise ValueError("MTP requires at least one prediction layer")
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, *,
                embedding: nn.Module, head: nn.Module | None = None,
                decoder_kwargs: Sequence[Mapping[str, Any]] | None = None) -> MTPOutput:
        """Predict future tokens without detaching the main-model computation.

        Args:
            hidden_states: Main trunk output BEFORE its final norm, [B, S, H].
            input_ids: Ground-truth tokens [B, S], not next-token labels.
            embedding: Main model's embedding, reused without registering a copy.
            head: Main model's output projection, or None for external chunk loss.
            decoder_kwargs: One mapping per depth. For depth k the caller must
                provide attention masks/positions for S-k query positions. None
                is suitable only for decoders that construct causal metadata.

        Returns:
            Per-depth states and optional logits. At depth k, position i fuses
            the prior state at i with token i+k, and predicts token i+k+1.
        """
        if hidden_states.ndim != 3 or input_ids.ndim != 2 or hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("MTP expects aligned hidden_states [B, S, H] and input_ids [B, S]")
        if input_ids.shape[0] == 0 or input_ids.shape[1] <= len(self.layers):
            raise ValueError("MTP needs a nonempty batch and sequence length greater than its depth")
        if decoder_kwargs is not None and len(decoder_kwargs) != len(self.layers):
            raise ValueError("decoder_kwargs must contain one mapping per MTP depth")
        states, logits = [], []
        for index, layer in enumerate(self.layers):
            kwargs = {} if decoder_kwargs is None else decoder_kwargs[index]
            future_embeddings = embedding(input_ids[:, index + 1:])
            hidden_states = layer(hidden_states[:, :-1], future_embeddings, **kwargs)
            states.append(hidden_states)
            if head is not None:
                logits.append(head(hidden_states))
        return MTPOutput(tuple(states), tuple(logits))

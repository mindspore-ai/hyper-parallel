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
"""K3 kernel and right-padding contract on the shared KDA CP schedules."""

from __future__ import annotations

# Native Torch training adapters intentionally use the backend directly.
# pylint: disable=forbidden-backend-import
from typing import Any

import torch
from torch import nn

from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    KimiDeltaAttentionLayerCP,
    KimiDeltaAttentionLayerP2PCP,
    KimiDeltaAttentionLayerUlyssesCP,
)
from hyper_parallel.components.functional.linear_attention_packed import packed_causal_conv1d
from hyper_parallel.models.kimi_k3.adapter.kda import kda, short_convolution


class _RightPadding(nn.Module):
    """Mask projections and outputs after global right-padding validation.

    Padding may still advance the discarded recurrent tail. No state is returned
    or reused: valid tokens must precede that tail across the entire CP group.
    Zero raw decay logits are deliberately not treated as an identity update.
    """

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        """Retain the model's zero-input/zero-output padding semantics.

        Args:
            hidden_states: Local activations [batch, local_sequence, hidden_size].
            attention_mask: Optional local token mask, already validated as globally
                right padded at the model boundary.
            **kwargs: Attention and packed-document options passed to the CP executor.

        Returns:
            Local KDA outputs with invalid token positions zeroed.
        """
        if attention_mask is None:
            return super().forward(hidden_states, **kwargs)
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("K3 CP attention_mask must match the local [batch,sequence] shard")
        valid = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return super().forward(hidden_states * valid, **kwargs) * valid


class KimiK3UlyssesCP(_RightPadding, KimiDeltaAttentionLayerUlyssesCP):
    """Use shared all-to-all and K3's exact local normalization/conv kernels."""

    def _run_local_attention(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Retain K3 normalization after the shared token/head exchange."""
        return kda(*args, a_log=kwargs["a_log"], dt_bias=kwargs["dt_bias"],
                   lower_bound=self.lower_bound, backend=self.backend, sequence_info=kwargs.get("sequence_info"))

    def _local_short_conv(self, projected: torch.Tensor, convolution: nn.Conv1d,
                          sequence_info: Any = None) -> torch.Tensor:
        channels = projected.shape[-1]
        weight = convolution.weight.narrow(0, self.cp_rank * channels, channels)
        if sequence_info is not None:
            return packed_causal_conv1d(
                projected, convolution, sequence_info, weight=weight,
                backend=self.module.config.causal_conv_backend,
            )
        return short_convolution(projected, weight, backend=self.module.config.causal_conv_backend)


class _NormalizedStateCore(nn.Module):
    """Preserve K3 reciprocal rounding before shared fused state communication."""

    def __init__(self, core: nn.Module) -> None:
        """Retain a parameter-free shared schedule with no per-call tensor state."""
        super().__init__()
        self.core = core

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Normalize Q/K once, before both local state summaries and local outputs.

        Args:
            query: Unnormalized local query heads [batch, local_sequence, heads, head_dim].
            key: Unnormalized matching key heads; this wrapper normalizes Q/K once
                using K3 reciprocal rounding.
            value: Local value heads passed to the shared state schedule.
            gate: Decay-gate input passed unchanged to the shared state core.
            beta: Delta-update strength input passed unchanged to the shared core.
            **kwargs: State-core options, including packed sequence_info when local
                document fragments need not align to the 64-token chunk size.

        Returns:
            The shared state core's local attention output. This wrapper always
            applies K3 rsqrt normalization with the reciprocal cast to the input
            dtype, then sets use_qk_l2norm_in_kernel=False to avoid a second pass.
        """
        if query.shape[1] == 0 or (query.shape[1] % 64 and kwargs.get("sequence_info") is None):
            raise ValueError("K3 fused state CP requires local sequence length divisible by 64; "
                             "padding between ranks would change recurrent states")
        query, key = (tensor * torch.rsqrt(tensor.float().square().sum(-1, keepdim=True) + 1e-6).to(tensor.dtype)
                      for tensor in (query, key))
        return self.core(query, key, value, gate, beta, use_qk_l2norm_in_kernel=False, **kwargs)


class KimiK3StateCP(_RightPadding, KimiDeltaAttentionLayerCP):
    """Retain differentiable halos and use K3 normalization for fused states."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Add only a mathematical boundary around the existing state core."""
        super().__init__(*args, **kwargs)
        if self.backend == "triton":
            if self.core.state.chunk_size != 64:
                raise ValueError("K3 fused state CP requires chunk_size=64")
            self.core = _NormalizedStateCore(self.core)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        """Reject rank-internal chunk padding before the convolution halo exchange.

        Args:
            hidden_states: Local [batch, sequence, hidden_size] activations. Fused
                nonpacked execution requires a nonempty multiple of 64 tokens.
            attention_mask: Optional local mask from global right-padding validation.
            **kwargs: Shared CP executor options, including packed_sequence_info
                for explicit document boundaries and state resets.

        Returns:
            Local attention outputs after halo exchange, state communication,
            output projection and any padding mask.
        """
        if (self.backend == "triton" and kwargs.get("packed_sequence_info") is None
                and (hidden_states.shape[1] == 0 or hidden_states.shape[1] % 64)):
            raise ValueError("K3 fused state CP requires local sequence length divisible by 64")
        return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)


class KimiK3P2PCP(KimiK3StateCP):
    """Use the same state implementation with the historical P2P defaults."""


def select_k3_executor(executor_class: type[nn.Module], backend: str,
                       causal_conv_backend: str) -> type[nn.Module]:
    """Choose a model-owned mathematical boundary for an existing CP schedule.

    Args:
        executor_class: Shared Ulysses, state-CP or historical P2P executor class.
        backend: Local KDA implementation, either eager or triton.
        causal_conv_backend: Short-convolution implementation; state CP requires
            eager, while Ulysses can use the configured fused convolution.

    Returns:
        K3 executor class preserving the shared schedule and parameter ownership.
    """
    if backend not in ("eager", "triton"):
        raise ValueError("K3 KDA CP backend must be eager or triton")
    if executor_class is KimiDeltaAttentionLayerUlyssesCP:
        return KimiK3UlyssesCP
    if causal_conv_backend != "eager":
        raise ValueError("K3 state CP currently requires eager conv; use Ulysses for fused convolution")
    return {KimiDeltaAttentionLayerCP: KimiK3StateCP,
            KimiDeltaAttentionLayerP2PCP: KimiK3P2PCP}[executor_class]

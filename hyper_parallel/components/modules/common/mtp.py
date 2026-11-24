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
from contextlib import nullcontext
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules.attention.deepseek_sparse_attention.mtp_index_share import (
    MTPIndexShareContext,
)

__all__ = ["MTPLayer", "MultiTokenPrediction", "MTPOutput", "MTPIndexShareContext", "build_mtp"]


def _mtp_index_share_paths(layers: Sequence[nn.Module]) -> tuple[str, ...]:
    """Resolve matching enabled DSA paths without tying any layer parameters."""
    per_layer = [
        {
            name: module for name, module in layer.named_modules()
            if bool(getattr(module, "mtp_index_share_enabled", False))
        }
        for layer in layers
    ]
    if not any(per_layer):
        return ()
    if any(not modules for modules in per_layer):
        raise ValueError("MTP IndexShare requires DSA attention at every MTP iteration")
    paths = tuple(per_layer[0])
    if any(tuple(modules) != paths for modules in per_layer[1:]):
        raise ValueError("MTP IndexShare requires matching DSA attention paths at every iteration")
    for modules in per_layer:
        for path, module in modules.items():
            module.mtp_index_share_key = path
    return paths


def _requires_explicit_mtp_positions(layers: Sequence[nn.Module]) -> bool:
    """Whether any enabled DSA path is configured for context parallelism."""
    return any(
        bool(getattr(module, "mtp_index_share_enabled", False))
        and getattr(module, "kv_cp_size", 1) > 1
        for layer in layers for module in layer.modules()
    )


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


class MTPLayer(nn.Module):
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
        MultiTokenPrediction, rather than changing this released-weight convention.
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


class MultiTokenPrediction(nn.Module):
    """Run independent MTP depths sequentially with ground-truth future tokens.

    Args:
        layers: Independent layers with forward(hidden_states, inputs_embeds,
            **decoder_kwargs) -> Tensor. Normally MTPLayer instances;
            adapters may supply another layer with the same contract.

    Note:
        This loop consumes complete, unsharded rows. It does not implement CP halo
        exchange, serving-time KV caching or automatic Trainer integration. Packed
        document shifts are supported when the attention adapter accepts flattened
        TND endpoints. Those model-specific paths belong to the adapter. A
        distributed adapter may also call individual layers with aligned inputs.
    """

    def __init__(self, layers: Sequence[nn.Module], *, index_share_for_mtp_iteration: bool | None = None) -> None:
        """Register the prediction layers without copying external shared weights."""
        super().__init__()
        if not layers:
            raise ValueError("MTP requires at least one prediction layer")
        paths = _mtp_index_share_paths(layers)
        inferred = bool(paths)
        if index_share_for_mtp_iteration is not None and not isinstance(index_share_for_mtp_iteration, bool):
            raise ValueError("index_share_for_mtp_iteration must be a boolean or None")
        enabled = inferred if index_share_for_mtp_iteration is None else index_share_for_mtp_iteration
        if enabled and not paths:
            raise ValueError("MTP IndexShare requires DSA attention modules in every MTP iteration")
        if enabled != inferred:
            raise ValueError("MTP IndexShare config must match the DSA attention configs")
        self.index_share_for_mtp_iteration = enabled
        self._mtp_index_share_paths = paths if enabled else ()
        self._requires_explicit_positions = _requires_explicit_mtp_positions(layers) if enabled else False
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, *,
                embedding: nn.Module, head: nn.Module | None = None,
                decoder_kwargs: Sequence[Mapping[str, Any]] | None = None,
                packed_seq_lengths: Sequence[int] | None = None,
                checkpoint_layers: bool = False) -> MTPOutput:
        """Predict future tokens without detaching the main-model computation.

        Args:
            hidden_states: Main trunk output BEFORE its final norm, [B, S, H].
            input_ids: Ground-truth tokens [B, S], not next-token labels.
            embedding: Main model's embedding, reused without registering a copy.
            head: Main model's output projection, or None for external chunk loss.
            decoder_kwargs: One mapping per depth. For depth k the caller must
                provide attention masks/positions for S-k query positions. None
                is suitable only for decoders that construct causal metadata.
            packed_seq_lengths: Optional document lengths over the flattened
                batch. Each MTP depth shifts independently within each document.
            checkpoint_layers: Recompute each complete MTP layer with non-reentrant
                checkpointing. MTP IndexShare state remains invocation-local and is
                read-only during checkpoint replay.

        Returns:
            Per-depth states and optional logits. At depth k, position i fuses
            the prior state at i with token i+k, and predicts token i+k+1.
        """
        if hidden_states.ndim != 3 or input_ids.ndim != 2 or hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("MTP expects aligned hidden_states [B, S, H] and input_ids [B, S]")
        if input_ids.shape[0] == 0 or (packed_seq_lengths is None and input_ids.shape[1] <= len(self.layers)):
            raise ValueError("MTP needs a nonempty batch and sequence length greater than its depth")
        if decoder_kwargs is not None and len(decoder_kwargs) != len(self.layers):
            raise ValueError("decoder_kwargs must contain one mapping per MTP depth")
        if not isinstance(checkpoint_layers, bool):
            raise ValueError("checkpoint_layers must be a boolean")
        if packed_seq_lengths is not None:
            return self._forward_packed(
                hidden_states, input_ids, embedding=embedding, head=head,
                decoder_kwargs=decoder_kwargs, packed_seq_lengths=packed_seq_lengths,
                checkpoint_layers=checkpoint_layers,
            )
        states, logits = [], []
        mtp_share_context = (
            MTPIndexShareContext(expected_keys=self._mtp_index_share_paths)
            if self.index_share_for_mtp_iteration else None
        )
        for index, layer in enumerate(self.layers):
            kwargs = {} if decoder_kwargs is None else decoder_kwargs[index]
            kwargs = dict(kwargs)
            if mtp_share_context is not None:
                if "mtp_index_share_context" in kwargs or "mtp_iteration" in kwargs:
                    raise ValueError("MTP IndexShare context and iteration are owned by MultiTokenPrediction")
                if self._requires_explicit_positions and "position_ids" not in kwargs:
                    raise ValueError("MTP IndexShare with CP requires adapter-provided position_ids for every step")
                kwargs["mtp_index_share_context"] = mtp_share_context
                kwargs["mtp_iteration"] = index
            future_embeddings = embedding(input_ids[:, index + 1:])
            layer_inputs = hidden_states[:, :-1]
            if mtp_share_context is not None and "position_ids" not in kwargs:
                positions = torch.arange(
                    index + 1, index + 1 + layer_inputs.shape[1], device=layer_inputs.device,
                ).expand(layer_inputs.shape[0], -1)
                kwargs["position_ids"] = positions
            if checkpoint_layers and self.training:
                checkpoint_options = {}
                if mtp_share_context is not None:
                    checkpoint_options["context_fn"] = lambda: (nullcontext(), mtp_share_context.recomputing())
                hidden_states = checkpoint(
                    layer, layer_inputs, future_embeddings, use_reentrant=False,
                    **checkpoint_options, **kwargs,
                )
            else:
                hidden_states = layer(layer_inputs, future_embeddings, **kwargs)
            states.append(hidden_states)
            if head is not None:
                logits.append(head(hidden_states))
        if mtp_share_context is not None:
            mtp_share_context.validate_complete()
        return MTPOutput(tuple(states), tuple(logits))

    def _forward_packed(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor, *, embedding: nn.Module,
        head: nn.Module | None, decoder_kwargs: Sequence[Mapping[str, Any]] | None,
        packed_seq_lengths: Sequence[int], checkpoint_layers: bool,
    ) -> MTPOutput:
        """Shift within packed documents and preserve document-local token positions."""
        lengths = tuple(packed_seq_lengths)
        total_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        if (not lengths or any(type(length) is not int or length <= len(self.layers) for length in lengths)
                or sum(lengths) != total_tokens):
            raise ValueError("packed_seq_lengths must cover the flattened input with each length greater than MTP depth")
        starts, offset = [], 0
        for length in lengths:
            starts.append(offset)
            offset += length
        original_positions = tuple(range(total_tokens))
        state_positions = original_positions
        state = hidden_states.reshape(1, total_tokens, hidden_states.shape[-1])
        flat_ids = input_ids.reshape(-1)
        states, logits = [], []
        context = (
            MTPIndexShareContext(expected_keys=self._mtp_index_share_paths)
            if self.index_share_for_mtp_iteration else None
        )

        for index, layer in enumerate(self.layers):
            target_lengths = tuple(length - index - 1 for length in lengths)
            query_positions = [
                position
                for start, length in zip(starts, lengths)
                for position in range(start, start + length - index - 1)
            ]
            previous_rows = {position: row for row, position in enumerate(state_positions)}
            query_rows = torch.tensor(
                [previous_rows[position] for position in query_positions], dtype=torch.long, device=state.device,
            )
            query_hidden = state.index_select(1, query_rows)
            future_rows = torch.tensor(
                [position + index + 1 for position in query_positions], dtype=torch.long, device=flat_ids.device,
            )
            future_embeddings = embedding(flat_ids.index_select(0, future_rows)).unsqueeze(0)
            kwargs = {} if decoder_kwargs is None else dict(decoder_kwargs[index])
            if context is not None:
                if "mtp_index_share_context" in kwargs or "mtp_iteration" in kwargs:
                    raise ValueError("MTP IndexShare context and iteration are owned by MultiTokenPrediction")
                if self._requires_explicit_positions and "position_ids" not in kwargs:
                    raise ValueError("MTP IndexShare with CP requires adapter-provided position_ids for every step")
                kwargs["mtp_index_share_context"] = context
                kwargs["mtp_iteration"] = index
            cumulative = []
            running = 0
            for length in target_lengths:
                running += length
                cumulative.append(running)
            kwargs.setdefault("actual_seq_len", cumulative)
            if "position_ids" not in kwargs:
                local_positions = [
                    position - start + index + 1
                    for start, length in zip(starts, lengths)
                    for position in range(start, start + length - index - 1)
                ]
                kwargs["position_ids"] = torch.tensor(
                    local_positions, dtype=torch.long, device=query_hidden.device,
                ).unsqueeze(0)
            if checkpoint_layers and self.training:
                checkpoint_options = {}
                if context is not None:
                    checkpoint_options["context_fn"] = lambda: (nullcontext(), context.recomputing())
                state = checkpoint(
                    layer, query_hidden, future_embeddings, use_reentrant=False,
                    **checkpoint_options, **kwargs,
                )
            else:
                state = layer(query_hidden, future_embeddings, **kwargs)
            if state.shape[:2] != (1, len(query_positions)):
                raise ValueError("Packed MTP decoder must preserve the compact query-token layout")
            state_positions = tuple(query_positions)
            states.append(state)
            if head is not None:
                logits.append(head(state))
        if context is not None:
            context.validate_complete()
        return MTPOutput(tuple(states), tuple(logits))


def build_mtp(
    mtp_num_layers: int | None = None, *, layer_factory: Callable[[int], nn.Module],
    index_share_for_mtp_iteration: bool | None = None,
) -> MultiTokenPrediction | None:
    """Build the configured MTP depths, or return None when disabled.

    Args:
        mtp_num_layers: Megatron-style depth count. None or zero disables MTP;
            a positive integer creates that many independent prediction layers.
        layer_factory: Model-owned factory called with each zero-based depth.
            It must return a fresh MTP layer, including its causal decoder.
        index_share_for_mtp_iteration: Optional model-config value. When true,
            every constructed layer must contain a DSA attention configured for
            MTP IndexShare.

    Returns:
        Registered MTP stack, or None without invoking the factory when disabled.

    Note:
        Model builders pass their YAML mtp_num_layers value here, attach the
        returned module, and include its output in forward/loss when enabled.
        This factory does not attach MTP to arbitrary models automatically.
    """
    if index_share_for_mtp_iteration is not None and not isinstance(index_share_for_mtp_iteration, bool):
        raise ValueError("index_share_for_mtp_iteration must be a boolean or None")
    if mtp_num_layers is None:
        if index_share_for_mtp_iteration is True:
            raise ValueError("MTP IndexShare cannot be enabled when MTP is disabled")
        return None
    if isinstance(mtp_num_layers, bool) or not isinstance(mtp_num_layers, int) or mtp_num_layers < 0:
        raise ValueError("mtp_num_layers must be None or a nonnegative integer")
    if mtp_num_layers == 0:
        if index_share_for_mtp_iteration is True:
            raise ValueError("MTP IndexShare cannot be enabled when MTP is disabled")
        return None
    return MultiTokenPrediction(
        [layer_factory(depth) for depth in range(mtp_num_layers)],
        index_share_for_mtp_iteration=index_share_for_mtp_iteration,
    )

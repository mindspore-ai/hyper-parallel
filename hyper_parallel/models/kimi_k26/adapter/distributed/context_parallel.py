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
"""Context-parallel input sharding for the Kimi-K2.5/K2.6 VLM.

Split choice ("replicated vision, sharded text"):

* the dataloader shards samples by ``dp_rank``, so the CP peers of one DP group
  receive the *same* sample; every rank therefore runs the vision tower and the
  media ``masked_scatter`` on the full sequence (the full-sequence placeholder
  contract of ``Kimi_K25Model.forward`` is what keeps this simple and correct);
* immediately before the text tower, ``inputs_embeds`` is sliced to this rank's
  ``[cp_rank*L, (cp_rank+1)*L)`` window, RoPE ``position_ids`` are rebuilt with
  the *global* window, and a 4D ``(B, 1, L, S)`` offset-aware causal+padding
  mask is handed to the language model. HF returns an already-4D mask verbatim
  (``masking_utils._preprocess_mask_arguments`` early-exit), which is required
  because the HF 5.x causal builder is index-based with ``q_offset=0`` and
  cannot express a CP offset itself.

The attention-side collectives are the shared registry wrapper ``sdpa_hf``
(K/V all-gather over the CP mesh, see
``distributed/context_parallel/wrappers.py``), declared by the model config's
``plan_overrides`` ``when: cp`` entry. This module only installs the
model-level input contract; it performs no communication.
"""

from __future__ import annotations

import functools
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
    _commit_forward_rewrite,
)
from hyper_parallel.distributed.context_parallel.attention import (
    _cp_offset_causal_mask,
)
from hyper_parallel.models.kimi_k26.adapter import KIMI_MODEL_TYPES

_CP_SHARDED_FLAG = "_hp_kimi_k26_cp_sharded"


def _resolve_language_model(model: nn.Module) -> nn.Module:
    """Return the nested text tower that the CP input slice is applied to."""
    inner = getattr(model, "model", None)
    if not isinstance(inner, nn.Module):
        raise TypeError(
            "Kimi-K2.6 CP input sharding requires a decoder module at model.model"
        )
    language_model = getattr(inner, "language_model", None)
    if not isinstance(language_model, nn.Module):
        raise TypeError(
            "Kimi-K2.6 CP input sharding requires model.model.language_model"
        )
    return language_model


def build_cp_attention_mask(
        attention_mask: torch.Tensor | None,
        *,
        q_len: int,
        kv_len: int,
        query_offset: int,
        device: Any,
) -> torch.Tensor:
    """Build the ``(B, 1, q_len, kv_len)`` CP offset causal + padding mask.

    ``attention_mask`` is the 2D ``[B, S]`` padding mask covering the complete
    (unsharded) sequence; it is required — a padding-free fallback would let
    query rows attend pad tokens that the all-gather brings in from every rank.
    """
    if attention_mask is None:
        raise ValueError(
            "Kimi-K2.6 CP input sharding requires a 2D attention_mask covering "
            "the complete sequence (the K/V all-gather exposes every rank's pad "
            "tokens, so the padding must be masked explicitly)"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            "Kimi-K2.6 CP input sharding expects a 2D [batch, seq] attention "
            f"mask, got shape {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[-1] != kv_len:
        raise ValueError(
            "Kimi-K2.6 CP input sharding expects the attention mask to cover "
            f"the complete sequence: mask length={attention_mask.shape[-1]}, "
            f"sequence length={kv_len}"
        )
    allowed = _cp_offset_causal_mask(q_len, kv_len, query_offset, device)
    allowed = allowed.view(1, 1, q_len, kv_len)
    padding = attention_mask.to(device=device, dtype=torch.bool).view(
        attention_mask.shape[0], 1, 1, kv_len
    )
    return allowed & padding


def bind_context_parallel(model: nn.Module, mesh_context: Any) -> None:
    """Install CP input sharding on the VLM's text tower.

    A no-op when ``cp_size <= 1`` (the sharded window degenerates to the full
    sequence), so callers can invoke it unconditionally after the model build.

    Args:
        model: The ``KimiK25ForConditionalGeneration`` instance.
        mesh_context: Trainer mesh exposing ``cp_size`` and ``cp_rank``.

    Raises:
        TypeError: If the model does not expose the expected Kimi-K2.5/K2.6 structure.
    """
    cp_size = int(getattr(mesh_context, "cp_size", 1))
    if cp_size <= 1:
        return
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in KIMI_MODEL_TYPES:
        raise TypeError(
            "Kimi-K2.6 CP input sharding requires config.model_type in "
            f"{KIMI_MODEL_TYPES}, got {model_type!r}"
        )
    language_model = _resolve_language_model(model)
    if getattr(language_model, _CP_SHARDED_FLAG, False):
        return
    cp_rank = int(getattr(mesh_context, "cp_rank", 0))
    original_forward = language_model.forward

    @functools.wraps(original_forward)
    def _cp_sharded_forward(*args: Any, **kwargs: Any) -> Any:
        """Slice this rank's sequence window and rebuild positions/mask."""
        if args:
            raise TypeError(
                "Kimi-K2.6 CP input sharding passes language-model inputs by keyword"
            )
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            raise RuntimeError(
                "Kimi-K2.6 CP input sharding requires inputs_embeds: the "
                "multimodal forward must embed and scatter media features before "
                "the text tower is entered"
            )
        seq_len = inputs_embeds.shape[1]
        if seq_len % cp_size:
            raise ValueError(
                "Kimi-K2.6 CP input sharding requires the padded sequence length "
                f"to be divisible by cp_size: seq_len={seq_len}, cp_size={cp_size} "
                "(the collator pads to data_transform.max_seq_len)"
            )
        local_len = seq_len // cp_size
        query_offset = cp_rank * local_len
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        call_kwargs = dict(kwargs)
        call_kwargs["inputs_embeds"] = inputs_embeds[
            :, query_offset:query_offset + local_len
        ]
        call_kwargs["position_ids"] = torch.arange(
            query_offset, query_offset + local_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        call_kwargs["attention_mask"] = build_cp_attention_mask(
            kwargs.get("attention_mask"),
            q_len=local_len,
            kv_len=seq_len,
            query_offset=query_offset,
            device=device,
        )
        call_kwargs.pop("input_ids", None)
        call_kwargs["past_key_values"] = None
        call_kwargs["use_cache"] = False
        return original_forward(**call_kwargs)

    _commit_forward_rewrite(
        _ForwardRewriteRequest(
            language_model,
            _cp_sharded_forward,
            companion_attrs={_CP_SHARDED_FLAG: True},
        )
    )


__all__ = ["bind_context_parallel", "build_cp_attention_mask"]

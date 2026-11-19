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
"""Training attention metadata, CP ownership and the shared V4 interface."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.collectives import flex_cp_allgather
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from rl.unified_execution.core.batch_invariant import normalize_policy, verify_batch_invariant
from rl.unified_execution.core.validation import validate_attention_execution_path
from rl.unified_execution.ops.attention.flash_attention_npu_v4 import (
    dense_attention,
    require_attention_provider,
    varlen_attention,
)

_ATTENTION_INTERFACE = "hyper_parallel_training_v4"


def _verify_training_numerics(unused_module: nn.Module, unused_inputs: tuple, *, policy: dict) -> None:
    """Fail before a training forward when another caller changes process numerics."""
    del unused_module, unused_inputs
    verify_batch_invariant(policy)


def guard_training_numerics(model: nn.Module, policy: dict) -> None:
    """Bind a read-only pre-forward check without changing parameters or HP wrappers."""
    normalized = normalize_policy(policy)
    verify_batch_invariant(normalized)
    previous = getattr(model, "unified_numeric_policy", None)
    if previous is not None:
        if previous != normalized:
            raise ValueError("Training model already owns a different numerical policy")
        return
    model.unified_numeric_policy = dict(normalized)
    model.register_forward_pre_hook(partial(_verify_training_numerics, policy=normalized))


@dataclass(frozen=True)
class TrainingAttentionMetadata:
    """Original token coordinates and packed sequence boundaries for one forward."""

    batch_size: int
    sequence_length: int
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


@dataclass(frozen=True)
class ContextParallelAttentionMetadata:
    """Local query rows and complete document prefixes in global CP coordinates."""

    batch_size: int
    local_length: int
    global_length: int
    query_indices: torch.Tensor
    key_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def prepare_cp_attention_metadata(attention_mask: torch.Tensor | None, batch_size: int,
                                  local_length: int, device: torch.device, cp_mesh: Any,
                                  document_lengths: tuple[int, ...] | None = None,
                                  ) -> ContextParallelAttentionMetadata:
    """Pack each local Q interval with its document's KV prefix once per forward.

    Args:
        attention_mask: Global [B, S] binary mask, or the supported causal 4D
            equivalent. S = local_length * CP; padding can differ by example.
        batch_size: B, the number of examples owned by this logical DP replica.
        local_length: S/CP physical query positions on this CP rank.
        device: Device receiving packed indices and cumulative lengths.
        cp_mesh: HP's ordered CP mesh; None denotes a single logical shard.
        document_lengths: Optional packed document lengths for one global row.

    Returns:
        Local-query indices into [B*S/CP], global-key indices into [B*S], and
        separate cumulative Q/KV lengths. Only documents with local queries
        contribute a packed segment; each KV segment ends at its last query.
    """
    size, rank = (cp_mesh.size(), cp_mesh.get_local_rank()) if cp_mesh is not None else (1, 0)
    global_length = local_length * size
    valid = (torch.ones((batch_size, global_length), dtype=torch.bool) if attention_mask is None
             else _padding_mask(attention_mask, batch_size, global_length))
    if document_lengths is not None:
        if (batch_size != 1 or not isinstance(document_lengths, tuple) or not document_lengths
                or any(not isinstance(length, int) or isinstance(length, bool) or length <= 0
                       for length in document_lengths)
                or sum(document_lengths) != global_length):
            raise ValueError("Packed document lengths must be positive and cover the single global input row")
    start, end = rank * local_length, (rank + 1) * local_length
    query_indices, key_indices, q_lengths, k_lengths = [], [], [], []
    for batch in range(batch_size):
        offset = 0
        for length in document_lengths or (global_length,):
            positions = valid[batch, offset:offset + length].nonzero().flatten() + offset
            selected = ((positions >= start) & (positions < end)).nonzero().flatten()
            if selected.numel():
                # A local Q interval uses its document prefix through the last
                # query. Bottom-right causal alignment then preserves global positions.
                prefix = positions[:int(selected[-1]) + 1]
                query_indices.extend((batch * local_length + positions[selected] - start).tolist())
                key_indices.extend((batch * global_length + prefix).tolist())
                q_lengths.append(selected.numel())
                k_lengths.append(prefix.numel())
            offset += length
    return ContextParallelAttentionMetadata(
        batch_size, local_length, global_length,
        torch.tensor(query_indices, dtype=torch.int64, device=device),
        torch.tensor(key_indices, dtype=torch.int64, device=device),
        torch.tensor([0, *q_lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32),
        torch.tensor([0, *k_lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32),
        max(q_lengths, default=0), max(k_lengths, default=0),
    )


def _run_cp_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      metadata: ContextParallelAttentionMetadata, cp_mesh: Any,
                      scaling: float | None) -> torch.Tensor:
    """Reuse HP K/V communication, then run the same native causal V4 provider.

    Inputs are Q [B, Nq, L, D] and K/V [B, Nkv, L, D], where L = S/CP.
    HP gathers K/V to sequence length S; packing produces [Tq, Nq, D] and
    [Tk, Nkv, D]. The restored output is [B, L, Nq, D], with zero padding.
    """
    if (query.shape[0], query.shape[2]) != (metadata.batch_size, metadata.local_length):
        raise ValueError("CP attention metadata does not match the local query coordinates")
    if cp_mesh is not None:
        # HP's differentiable gather retains rank-ordered global K/V and owns
        # gradient communication; no distributed softmax merge is introduced.
        key, value = flex_cp_allgather(key.contiguous(), value.contiguous(), 2, cp_mesh)
    if key.shape[2] != metadata.global_length:
        raise ValueError("CP attention requires globally ordered K/V after gather")
    states = [tensor.transpose(1, 2).flatten(0, 1) for tensor in (query, key, value)]
    if metadata.query_indices.numel() == 0:
        # Empty query ranks still participate in K/V forward and backward collectives.
        return query.transpose(1, 2) * 0 + (key.sum() + value.sum()) * 0
    packed = [states[0].index_select(0, metadata.query_indices),
              *(state.index_select(0, metadata.key_indices) for state in states[1:])]
    output = varlen_attention(
        *packed, cu_seqlens_q=metadata.cu_seqlens_q, cu_seqlens_k=metadata.cu_seqlens_k,
        max_seqlen_q=metadata.max_seqlen_q, max_seqlen_k=metadata.max_seqlen_k, scale=scaling,
    )
    restored = output.new_zeros((metadata.batch_size * metadata.local_length, *output.shape[1:]))
    return restored.index_copy(0, metadata.query_indices, output).view(
        metadata.batch_size, metadata.local_length, *output.shape[1:])


def _padding_mask(mask: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    """Validate a training mask once at the model boundary on the host."""
    mask = mask.detach().cpu()
    if mask.ndim == 2:
        if tuple(mask.shape) != (batch_size, sequence_length) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Unified training requires a binary [batch, sequence] padding mask")
        return mask.bool()
    if mask.ndim != 4 or tuple(mask.shape) != (batch_size, 1, sequence_length, sequence_length):
        raise ValueError("Unified training accepts only 2D padding or 4D causal-plus-padding masks")
    if mask.dtype == torch.bool:
        allowed = mask[:, 0]
    elif mask.is_floating_point():
        blocked = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
        if not torch.all((mask == 0) | blocked):
            raise ValueError("Unified training does not support additive attention biases")
        allowed = (mask == 0)[:, 0]
    else:
        raise ValueError("A 4D training mask must use HF boolean or floating additive semantics")
    valid = allowed.any(dim=1)
    causal = torch.ones((sequence_length, sequence_length), dtype=torch.bool).tril()
    expected = causal.unsqueeze(0) & valid.unsqueeze(1)
    if not torch.equal(allowed, expected):
        raise ValueError("Unified training does not support arbitrary 4D masks or noncausal attention")
    return valid


def prepare_training_attention_metadata(
    attention_mask: torch.Tensor | None, batch_size: int, sequence_length: int, device: torch.device,
) -> TrainingAttentionMetadata | None:
    """Prepare padding coordinates once before entering the decoder stack.

    Args:
        attention_mask: HF 2D binary padding or standard 4D causal-plus-padding mask.
        batch_size: Model input batch size.
        sequence_length: Padded model input length.
        device: Device receiving packed index and cumulative length tensors.

    Returns:
        Packed metadata, or None when every token is valid.

    Note:
        Mask validation intentionally performs one host transfer at the training
        model boundary. Decoder layers and checkpoint recomputation reuse the
        resulting metadata. This preparation is not part of inference graphs.
    """
    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError("Unified training requires a tensor attention mask")
    valid = _padding_mask(attention_mask, batch_size, sequence_length)
    if bool(valid.all()):
        return None
    lengths = valid.sum(dim=1, dtype=torch.int32)
    indices = valid.flatten().nonzero(as_tuple=False).flatten()
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(dim=0, dtype=torch.int32)))
    return TrainingAttentionMetadata(
        batch_size, sequence_length, indices.to(device), cumulative.to(device), int(lengths.max()),
    )


def run_training_attention(
    module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Adapt HF/HP BNSD Q/K/V to native V4 and return BSND output.

    Padding is removed using metadata prepared by the owning model's pre-hook.
    The gather/scatter operations retain autograd and restore original token
    coordinates; padding output rows are zero and carry no attention gradient.

    Args:
        module: Attention owner with optional CP mesh installed by the HP recipe.
        query: [B, Nq, L, D] projected and position-encoded local queries.
        key: [B, Nkv, L, D] local keys before the optional CP gather.
        value: [B, Nkv, L, D] local values.
        attention_mask: Must be None; the model boundary already prepared metadata.
        dropout: Must be zero for the shared deterministic execution contract.
        scaling: Attention scale passed unchanged to the V4 provider.
        **kwargs: Includes immutable unified_attention_metadata for this forward.

    Returns:
        ([B, L, Nq, D] attention output, None). Nq/Nkv are local head counts;
        CP changes sequence ownership, while TP owns head partitioning.
    """
    if dropout != 0.0 or not getattr(module, "is_causal", True):
        raise ValueError("Unified training attention requires causal attention and zero dropout")
    if kwargs.get("sliding_window") is not None or getattr(module, "sliding_window", None) is not None:
        raise ValueError("Unified training attention does not support sliding windows")
    if kwargs.get("actual_seq_len") is not None:
        raise ValueError("Unified training uses explicit padding metadata, not actual_seq_len packing")
    if attention_mask is not None:
        raise ValueError("Unified training masks must be prepared once by install_training_attention")
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("Unified training attention requires BNSD Q/K/V")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("Unified training attention requires equal query/key sequence lengths without a cache")
    metadata = kwargs.get("unified_attention_metadata")
    cp_mesh = getattr(module, "unified_cp_mesh", None)
    if isinstance(metadata, ContextParallelAttentionMetadata):
        return _run_cp_attention(query, key, value, metadata, cp_mesh, scaling), None
    if cp_mesh is not None:
        raise ValueError("CP attention requires global input metadata prepared at the decoder boundary")

    # CP prepares its own packed views after gathering K/V; only the non-CP
    # branch needs these BSND views. CP=1 keeps the existing dense/varlen path.
    states = [tensor.transpose(1, 2) for tensor in (query, key, value)]
    if metadata is None:
        return dense_attention(*states, scale=scaling), None
    if not isinstance(metadata, TrainingAttentionMetadata):
        raise ValueError("Invalid unified training attention metadata")
    batch_size, sequence_length = query.shape[0], query.shape[2]
    if (metadata.batch_size, metadata.sequence_length) != (batch_size, sequence_length):
        raise ValueError("Training attention metadata does not match the current token coordinates")
    packed = [state.flatten(0, 1).index_select(0, metadata.indices) for state in states]
    if metadata.indices.numel() == 0:
        # Keep all three gradients connected for a loss-masked, entirely padded batch.
        return states[0] * 0 + (key.sum() + value.sum()) * 0, None
    output = varlen_attention(
        *packed, cu_seqlens_q=metadata.cu_seqlens, cu_seqlens_k=metadata.cu_seqlens,
        max_seqlen_q=metadata.max_seqlen, max_seqlen_k=metadata.max_seqlen, scale=scaling,
    )
    restored = output.new_zeros((batch_size * sequence_length, output.shape[1], output.shape[2]))
    return restored.index_copy(0, metadata.indices, output).view(batch_size, sequence_length, *output.shape[1:]), None


def _prepare_training_forward(
    module: nn.Module, args: tuple, kwargs: dict, *, signature: inspect.Signature,
) -> tuple[tuple, dict]:
    """Carry immutable per-forward metadata through decoder and AC kwargs."""
    bound = signature.bind_partial(*args, **kwargs)
    arguments = bound.arguments
    if arguments.get("past_key_values") is not None or arguments.get("use_cache"):
        raise ValueError("Unified training attention does not support a KV cache")
    inputs = arguments.get("input_ids")
    if inputs is None:
        inputs = arguments.get("inputs_embeds")
    if inputs is None or inputs.ndim not in (2, 3):
        raise ValueError("Unified training requires batched input_ids or inputs_embeds")
    extra = arguments.setdefault("kwargs", {})
    cp_mesh = getattr(module, "unified_cp_mesh", None)
    document_lengths = extra.pop("unified_document_lengths", None)
    if cp_mesh is not None or document_lengths is not None:
        metadata = prepare_cp_attention_metadata(
            arguments.get("attention_mask"), inputs.shape[0], inputs.shape[1], inputs.device,
            cp_mesh, document_lengths,
        )
        if arguments.get("position_ids") is None:
            raise ValueError("CP/packed training requires explicit original position_ids")
    else:
        metadata = prepare_training_attention_metadata(
            arguments.get("attention_mask"), inputs.shape[0], inputs.shape[1], inputs.device,
        )
    if "unified_attention_metadata" in extra:
        raise ValueError("Training attention metadata is owned by the model input adapter")
    extra["unified_attention_metadata"] = metadata
    arguments["attention_mask"] = {"full_attention": None}
    arguments["use_cache"] = False

    # HF decorators populate keyword defaults; avoid duplicating those parameters
    # by rebuilding a positional prefix that includes use_cache.
    forwarded = {name: value for name, value in arguments.items() if name != "kwargs"}
    forwarded.update(extra)
    return (), forwarded


def install_training_attention(model: nn.Module) -> None:
    """Install native V4 only on this Actor's existing Qwen3 attention callables.

    Args:
        model: Built Qwen3 causal language model or its Qwen3 decoder model.

    Raises:
        ValueError: The model or its attention options are unsupported.

    Note:
        Registration adds a named HF interface; only this model's configuration
        selects it. No module or Parameter is replaced, preserving HP plans,
        parameter identities, optimizer references and existing AC wrappers.
    """
    decoder, attentions = validate_attention_execution_path(model)
    if getattr(decoder, "unified_attention_installed", False):
        return
    if not attentions:
        raise ValueError("Unified training attention requires at least one decoder layer")
    for attention in attentions:
        if (getattr(attention, "sliding_window", None) is not None
                or getattr(attention, "attention_dropout", 0.0) != 0.0
                or not getattr(attention, "is_causal", True)):
            raise ValueError("Unified training requires causal full attention with zero dropout")
    context = getattr(getattr(attentions[0], "o_proj", None), "pik_context", None)
    cp_mesh = getattr(attentions[0], "unified_cp_mesh", None)
    if any(getattr(attention, "unified_cp_mesh", None) is not cp_mesh for attention in attentions):
        raise ValueError("Every training attention layer must use the same CP mesh")
    if cp_mesh is not None:
        decoder.unified_cp_mesh = cp_mesh
        model.unified_cp_mesh = cp_mesh
    if context is not None:
        model.unified_sequence_multiple = context.world_size if context.sequence_parallel else 1
    if cp_mesh is not None:
        model.unified_sequence_multiple = getattr(model, "unified_sequence_multiple", 1) * cp_mesh.size()
    require_attention_provider()
    ALL_ATTENTION_FUNCTIONS.register(_ATTENTION_INTERFACE, run_training_attention)
    for attention in attentions:
        if hasattr(attention, "attention_interface"):
            attention.attention_interface = run_training_attention
        attention.config._attn_implementation = _ATTENTION_INTERFACE  # pylint: disable=protected-access
    decoder.config._attn_implementation = _ATTENTION_INTERFACE  # pylint: disable=protected-access
    decoder.register_forward_pre_hook(
        partial(_prepare_training_forward, signature=inspect.signature(decoder.forward)), with_kwargs=True,
    )
    decoder.unified_attention_installed = True


@inner_wrapper
def unified_cp_attention_wrapper(target_module: Any, mesh: Any, tp_mesh: Any,
                                 cp_mesh: Any, ep_mesh: Any) -> _ForwardRewriteRequest:
    """Attach CP ownership through HP's atomic companion-attribute installation.

    The training adapter owns V4 metadata and calls the existing differentiable
    K/V gather. Retaining the original forward preserves projections and TP/SP.
    """
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Unified CP attention requires an active HP CP mesh")
    if not callable(getattr(target_module, "attention_interface", None)):
        raise ValueError("Unified CP requires the shared HP GQAAttention replacement")
    if getattr(target_module, "unified_cp_mesh", None) is not None:
        raise ValueError("Unified CP attention is already bound")
    return _ForwardRewriteRequest(target_module, target_module.forward,
                                  companion_attrs={"unified_cp_mesh": cp_mesh})

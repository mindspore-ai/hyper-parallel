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
"""V4 attention_interface for the operands prepared by HP grouped-query attention."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import accumulate, pairwise
from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from rl.unified_execution.modules.flash_attention_v4 import (
    dense_attention, require_attention_provider, varlen_attention,
)
from rl.unified_execution.models.structure import training_sequence_multiple, validate_gqa_execution_path

_GQA_V4_BACKEND = "hyper_parallel_training_v4"


@dataclass(frozen=True)
class _V4Layout:
    """Device-local V4 lengths and optional row maps for removing padding."""
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    query_rows: torch.Tensor | None
    key_rows: torch.Tensor | None


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


def prepare_v4_batch(mask: torch.Tensor | None, batch: int, local_length: int,
                     device: torch.device, cp_mesh: Any = None) -> Callable:
    """Cache V4 metadata for this invocation, shared by layers and checkpoint replay."""
    size, rank = (cp_mesh.size(), cp_mesh.get_local_rank()) if cp_mesh is not None else (1, 0)
    width = local_length * size
    valid = None if mask is None else _padding_mask(mask, batch, width)
    if valid is not None and bool(valid.all()):
        valid = None

    @lru_cache(maxsize=1)
    def layout(q_ends: tuple[int, ...], k_ends: tuple[int, ...]) -> _V4Layout | None:
        # HP supplies physical document ends after CP exchange. Ordinary dense
        # inputs with no padding or document boundaries keep the dense V4 call.
        if not q_ends:
            if valid is None:
                return None
            q_ends = k_ends = tuple(range(local_length, batch * local_length + 1, local_length))
        q_bounds, k_bounds = (0, *q_ends), (0, *k_ends)
        q_rows = k_rows = None
        if valid is not None:
            # Each exchanged KV fragment ends at the last physical Q of that
            # document. Use HP's actual fragment sizes to select its valid rows.
            source = (torch.arange(batch)[:, None] * width + rank * local_length + torch.arange(local_length)).flatten()
            q_mask = valid.flatten()[source]
            q_rows = q_mask.nonzero().flatten().to(device)
            keys, q_counts, k_counts = [], [], []
            for (qa, qb), (ka, kb) in zip(pairwise(q_bounds), pairwise(k_bounds), strict=True):
                count = int(q_mask[qa:qb].sum())
                if count:
                    end = int(source[qb - 1]) + 1
                    keep = valid.flatten()[end - (kb - ka):end].nonzero().flatten()
                    keys.append(keep + ka)
                    q_counts.append(count)
                    k_counts.append(keep.numel())
            k_rows = torch.cat(keys).to(device) if keys else q_rows
            q_bounds = (0, *accumulate(q_counts))
            k_bounds = (0, *accumulate(k_counts))

        def lengths(bounds: tuple[int, ...]) -> tuple[torch.Tensor, int]:
            return (torch.tensor(bounds, dtype=torch.int32, device=device),
                    max((end - start for start, end in pairwise(bounds)), default=0))

        cu_q, max_q = lengths(q_bounds)
        cu_k, max_k = lengths(k_bounds) if k_bounds != q_bounds else (cu_q, max_q)
        return _V4Layout(cu_q, cu_k, max_q, max_k, q_rows, k_rows)

    return layout


def gqa_v4_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   attention_mask: torch.Tensor | None, dropout: float = 0.0,
                   scaling: float | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run V4 on ready HP operands [B, heads, sequence, D] and return [B, Q, heads, D]."""
    if attention_mask is not None or dropout or not kwargs.get("is_causal", getattr(module, "is_causal", True)):
        raise ValueError("Training V4 expects a prepared causal mask and zero dropout")
    if kwargs.get("sliding_window") is not None:
        raise ValueError("Training V4 requires full causal attention")
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    layout_for = kwargs.get("v4_batch")
    common = kwargs.get("actual_seq_len")
    common = () if common is None else common
    q_ends = kwargs.get("actual_seq_qlen", common)
    k_ends = kwargs.get("actual_seq_kvlen", common)
    if layout_for is None and len(q_ends):
        layout_for = prepare_v4_batch(None, q.shape[0], q.shape[1], q.device)
    layout = None
    if layout_for is not None:
        layout = layout_for(tuple(q_ends.tolist() if isinstance(q_ends, torch.Tensor) else q_ends),
                            tuple(k_ends.tolist() if isinstance(k_ends, torch.Tensor) else k_ends))
    if layout is None:
        return dense_attention(q, k, v, scale=scaling), None
    if not layout.max_q:
        # Empty query ranks still participate in HP's KV gradient exchanges.
        return q * 0 + (k.sum() + v.sum()) * 0, None

    q_flat, k_flat, v_flat = (t.flatten(0, 1) for t in (q, k, v))
    if layout.query_rows is not None:
        q_flat = q_flat.index_select(0, layout.query_rows)
        k_flat, v_flat = (t.index_select(0, layout.key_rows) for t in (k_flat, v_flat))
    result = varlen_attention(q_flat, k_flat, v_flat, cu_seqlens_q=layout.cu_q, cu_seqlens_k=layout.cu_k,
                              max_seqlen_q=layout.max_q, max_seqlen_k=layout.max_k, scale=scaling)
    if layout.query_rows is not None:
        result = result.new_zeros((q.shape[0] * q.shape[1], *result.shape[1:])).index_copy(
            0, layout.query_rows, result)
    return result.reshape(q.shape), None


def _prepare_v4_inputs(module: nn.Module, args: tuple, kwargs: dict, *, signature: inspect.Signature):
    """Forward the original document boundaries through HF and share padding metadata across layers."""
    arguments = signature.bind_partial(*args, **kwargs).arguments
    if arguments.get("past_key_values") is not None or arguments.get("use_cache"):
        raise ValueError("Unified training attention does not support a KV cache")
    inputs = arguments.get("input_ids")
    if inputs is None:
        inputs = arguments["inputs_embeds"]
    batch, local_length = inputs.shape[:2]
    extra = arguments.pop("kwargs", {})
    mesh = getattr(module, "unified_cp_mesh", None)
    documents = extra.pop("unified_document_lengths", None)
    if documents is not None:
        width = local_length * (mesh.size() if mesh is not None else 1)
        if batch != 1 or not documents or min(documents) <= 0 or sum(documents) != width:
            raise ValueError("Packed document lengths must cover one global input row")
        extra["actual_seq_len"] = tuple(accumulate(documents))
    if (mesh is not None or documents is not None) and arguments.get("position_ids") is None:
        raise ValueError("CP/packed training requires explicit original position_ids")
    mask = arguments.get("attention_mask")
    if mask is not None or mesh is not None or extra.get("actual_seq_len") is not None:
        extra["v4_batch"] = prepare_v4_batch(mask, batch, local_length, inputs.device, mesh)
    arguments.update(extra, attention_mask={"full_attention": None}, use_cache=False)
    return (), arguments


def bind_gqa_v4_backend(model: nn.Module) -> None:
    """Select the V4 interface on the built decoder and prepare padding once per batch."""
    decoder, gqa_layers = validate_gqa_execution_path(model)
    if getattr(decoder, "_rl_gqa_v4_bound", False):
        return
    if not gqa_layers:
        raise ValueError("Unified training attention requires at least one decoder layer")
    for gqa in gqa_layers:
        if (getattr(gqa, "sliding_window", None) is not None
                or getattr(gqa, "attention_dropout", 0.0) != 0.0
                or not getattr(gqa, "is_causal", True)):
            raise ValueError("Unified training requires causal full attention with zero dropout")
    pik_context = getattr(getattr(gqa_layers[0], "o_proj", None), "pik_context", None)
    cp_mesh = getattr(getattr(gqa_layers[0], "cp_context", None), "mesh", None)
    if any(getattr(getattr(gqa, "cp_context", None), "mesh", None) is not cp_mesh for gqa in gqa_layers):
        raise ValueError("Every training attention layer must use the same CP mesh")
    if cp_mesh is not None:
        decoder.unified_cp_mesh = cp_mesh
        model.unified_cp_mesh = cp_mesh
    model.unified_sequence_multiple = training_sequence_multiple(
        model, tp_size=pik_context.world_size if pik_context is not None else 1,
        cp_size=cp_mesh.size() if cp_mesh is not None else 1,
        sequence_parallel=pik_context.sequence_parallel if pik_context is not None else False,
    )
    require_attention_provider()
    ALL_ATTENTION_FUNCTIONS.register(_GQA_V4_BACKEND, gqa_v4_forward)
    for gqa in gqa_layers:
        if hasattr(gqa, "attention_interface"):
            gqa.attention_interface = gqa_v4_forward
        gqa.config._attn_implementation = _GQA_V4_BACKEND  # pylint: disable=protected-access
    decoder.config._attn_implementation = _GQA_V4_BACKEND  # pylint: disable=protected-access
    decoder.register_forward_pre_hook(
        partial(_prepare_v4_inputs, signature=inspect.signature(decoder.forward)), with_kwargs=True,
    )
    decoder._rl_gqa_v4_bound = True  # pylint: disable=protected-access

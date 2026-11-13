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
"""Training masks, attention installation and process numerical guards."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import inspect
from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from rl.unified_execution.ops.attention.flash_attention_npu_v4 import (
    dense_attention, require_attention_provider, varlen_attention,
)
from rl.unified_execution.batch_invariant import normalize_policy, verify_batch_invariant
from rl.unified_execution.validation import validate_attention_execution_path


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
    states = [tensor.transpose(1, 2) for tensor in (query, key, value)]
    metadata = kwargs.get("unified_attention_metadata")
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
    del module
    bound = signature.bind_partial(*args, **kwargs)
    arguments = bound.arguments
    if arguments.get("past_key_values") is not None or arguments.get("use_cache"):
        raise ValueError("Unified training attention does not support a KV cache")
    inputs = arguments.get("input_ids")
    if inputs is None:
        inputs = arguments.get("inputs_embeds")
    if inputs is None or inputs.ndim not in (2, 3):
        raise ValueError("Unified training requires batched input_ids or inputs_embeds")
    metadata = prepare_training_attention_metadata(
        arguments.get("attention_mask"), inputs.shape[0], inputs.shape[1], inputs.device,
    )
    extra = arguments.setdefault("kwargs", {})
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
    if context is not None:
        model.unified_sequence_multiple = context.world_size if context.sequence_parallel else 1
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

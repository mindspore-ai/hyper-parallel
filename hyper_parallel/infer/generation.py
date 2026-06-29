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
"""Prefill + decode generation loop."""
import inspect
from typing import Optional

import torch
import torch.distributed as dist

from hyper_parallel.infer.kv_cache import (
    ContextParallelKVCache,
    KVCache,
    detach_and_validate_past_key_values,
)
from hyper_parallel.infer.sampler import sample_next_token
from hyper_parallel.infer.utils import (
    GenerationConfig,
    append_attention_mask,
    apply_logits_processors,
    build_causal_mask,
    build_position_ids,
    prepare_logits_for_sampling,
    should_stop_generation,
)


def _get_output(outputs, name: str):
    """Read an output field from dict-like or object-like model outputs."""
    if isinstance(outputs, dict):
        return outputs.get(name)
    return getattr(outputs, name, None)


def _model_forward(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values,
    use_cache: bool,
    sequence_shard_info=None,
    global_seq_len: Optional[int] = None,
):
    """Call model.forward with only the keyword arguments it accepts."""
    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "sequence_shard_info": sequence_shard_info,
        "global_seq_len": global_seq_len,
    }
    forward = getattr(model, "forward", model)
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return forward(**kwargs)
    parameters = signature.parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    if not accepts_kwargs:
        for name in list(kwargs):
            if name not in parameters:
                kwargs.pop(name)
    return forward(**kwargs)


def _resolve_context_parallel_rank_world(config: GenerationConfig) -> tuple[int, int]:
    if config.context_parallel_rank is not None:
        return config.context_parallel_rank, config.context_parallel_world_size
    if not dist.is_available() or not dist.is_initialized():
        raise ValueError(
            "context_parallel_cache requires initialized torch.distributed "
            "or explicit context_parallel_rank/context_parallel_world_size",
        )
    return (
        dist.get_rank(group=config.context_process_group),
        dist.get_world_size(group=config.context_process_group),
    )


def _init_cache(config: GenerationConfig) -> KVCache:
    if not config.context_parallel_cache:
        return KVCache()
    rank, world_size = _resolve_context_parallel_rank_world(config)
    return ContextParallelKVCache(rank=rank, world_size=world_size)


def _cache_shard_info(cache: KVCache):
    return cache.shard_info if isinstance(cache, ContextParallelKVCache) else None


def _cache_seq_len(past_key_values) -> Optional[int]:
    """Resolve cached sequence length from tuple or opaque HF-style cache."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length") and not isinstance(
        past_key_values, (list, tuple),
    ):
        return int(past_key_values.get_seq_length())
    values = detach_and_validate_past_key_values(past_key_values)
    if not values:
        return 0
    return int(values[0][0].shape[-2])


def _cache_batch_size(past_key_values) -> Optional[int]:
    """Resolve cache batch size when cache tensors are inspectable."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length") and not isinstance(
        past_key_values, (list, tuple),
    ):
        return None
    values = detach_and_validate_past_key_values(past_key_values)
    if not values:
        return None
    return int(values[0][0].shape[0])


def _resolve_prefix_length(config: GenerationConfig) -> int:
    """Validate and resolve reusable prefix cache length."""
    if config.prefix_past_key_values is None:
        return 0
    candidates = []
    if config.prefix_cache_length is not None:
        candidates.append(int(config.prefix_cache_length))
    if config.prefix_attention_mask is not None:
        candidates.append(int(config.prefix_attention_mask.shape[-1]))
    if config.prefix_sequence_shard_info is not None:
        candidates.append(int(config.prefix_sequence_shard_info.global_seq_len))
    seq_len = _cache_seq_len(config.prefix_past_key_values)
    if seq_len is not None and config.prefix_sequence_shard_info is None:
        candidates.append(seq_len)
    if not candidates:
        raise ValueError(
            "prefix_past_key_values requires prefix_cache_length for opaque caches",
        )
    prefix_len = candidates[0]
    if any(length != prefix_len for length in candidates):
        raise ValueError("prefix cache length metadata is inconsistent")
    return prefix_len


def _prepare_prefix_attention_mask(
    config: GenerationConfig,
    input_ids: torch.Tensor,
    device,
) -> tuple[Optional[torch.Tensor], int]:
    """Prepare a 2-D attention mask for reusable prefix cache."""
    prefix_len = _resolve_prefix_length(config)
    if prefix_len == 0:
        return None, 0
    cache_batch_size = _cache_batch_size(config.prefix_past_key_values)
    if cache_batch_size is not None and cache_batch_size != input_ids.size(0):
        raise ValueError("prefix cache batch size must match input_ids batch size")
    prefix_attention_mask = config.prefix_attention_mask
    if prefix_attention_mask is None:
        return torch.ones(
            input_ids.size(0),
            prefix_len,
            device=device,
            dtype=torch.long,
        ), prefix_len
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape (batch, prefix_seq)")
    if prefix_attention_mask.shape != (input_ids.size(0), prefix_len):
        raise ValueError("prefix_attention_mask batch/sequence length mismatch")
    return prefix_attention_mask.to(device=device), prefix_len


def _init_cache_with_prefix(config: GenerationConfig) -> tuple[KVCache, int]:
    """Create the generation cache and preload prefix cache when present."""
    cache = _init_cache(config)
    prefix_len = _resolve_prefix_length(config)
    if prefix_len == 0:
        return cache, prefix_len
    if isinstance(cache, ContextParallelKVCache):
        if config.prefix_sequence_shard_info is None:
            cache.update_full(config.prefix_past_key_values)
        else:
            cache.update_local(
                config.prefix_past_key_values,
                config.prefix_sequence_shard_info,
            )
        return cache, prefix_len
    cache.update(config.prefix_past_key_values)
    return cache, prefix_len


def _update_cache(cache: KVCache, outputs) -> None:
    """Update normal or context-parallel KV cache from model outputs."""
    past_key_values = _get_output(outputs, "past_key_values")
    if not isinstance(cache, ContextParallelKVCache):
        cache.update(past_key_values)
        return
    if past_key_values is None:
        return
    sequence_shard_info = _get_output(outputs, "sequence_shard_info")
    if sequence_shard_info is not None:
        cache.update_local(past_key_values, sequence_shard_info)
        return
    if cache.is_empty:
        cache.update_full(past_key_values)
        return
    raise ValueError(
        "context-parallel cached decode requires model output sequence_shard_info",
    )


def _resolve_mask_dtype(model, config: GenerationConfig) -> torch.dtype:
    """Choose additive-mask dtype from config or model floating state."""
    if config.mask_dtype is not None:
        return config.mask_dtype
    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(model, iterator_name, None)
        if iterator is None:
            continue
        for tensor in iterator():
            if tensor.is_floating_point():
                return tensor.dtype
    return torch.float32


def _build_decode_key_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build additive key padding mask for one-token cached decode."""
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (batch, seq)")
    batch_size, seq_len = attention_mask.shape
    mask = torch.zeros(
        batch_size,
        1,
        1,
        seq_len,
        device=attention_mask.device,
        dtype=dtype,
    )
    padding = attention_mask == 0
    return mask.masked_fill(padding.view(batch_size, 1, 1, seq_len), float("-inf"))


def _combined_attention_mask(
    prefix_attention_mask: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if prefix_attention_mask is None:
        return attention_mask
    if attention_mask is None:
        return prefix_attention_mask
    return torch.cat([prefix_attention_mask, attention_mask], dim=-1)


def _build_prefill_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    prefix_attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the prefill causal mask, including optional prefix keys."""
    if prefix_attention_mask is None:
        return build_causal_mask(input_ids, attention_mask, dtype=dtype)
    batch_size, query_len = input_ids.shape
    prefix_len = prefix_attention_mask.shape[-1]
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones(
            batch_size,
            query_len,
            device=device,
            dtype=prefix_attention_mask.dtype,
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    mask = torch.zeros(
        batch_size,
        1,
        query_len,
        prefix_len + query_len,
        device=device,
        dtype=dtype,
    )
    causal = torch.triu(
        torch.full((query_len, query_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    mask[:, :, :, prefix_len:] = causal.view(1, 1, query_len, query_len)
    current_padding = attention_mask == 0
    current_key_padding = torch.cat([prefix_attention_mask == 0, current_padding], dim=-1)
    mask = mask.masked_fill(
        current_key_padding.view(batch_size, 1, 1, prefix_len + query_len),
        float("-inf"),
    )
    mask = mask.masked_fill(current_padding.view(batch_size, 1, query_len, 1), 0.0)
    return mask


def _build_prefill_position_ids(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    prefix_attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build position ids for prefill with optional prefix offset."""
    position_ids = build_position_ids(input_ids, attention_mask)
    if prefix_attention_mask is None:
        return position_ids
    prefix_lengths = prefix_attention_mask.long().sum(dim=-1).view(-1, 1)
    return position_ids + prefix_lengths


def _prompt_lengths(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    """Count valid prompt tokens per batch row."""
    if attention_mask is None:
        return torch.full(
            (input_ids.size(0),),
            input_ids.size(1),
            device=input_ids.device,
            dtype=torch.long,
        )
    return attention_mask.long().sum(dim=-1)


def _finalize_sequences(
    sequences: torch.Tensor,
    initial_attention_mask: Optional[torch.Tensor],
    prompt_lengths: torch.Tensor,
    generated_counts: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Strip left padding and right-pad finalized generated sequences."""
    rows = []
    max_len = 0
    for batch_idx in range(sequences.size(0)):
        if initial_attention_mask is None:
            start = 0
        else:
            starts = torch.nonzero(
                initial_attention_mask[batch_idx].bool(), as_tuple=False,
            )
            if starts.numel() == 0:
                raise ValueError("attention_mask row must contain at least one valid token")
            start = int(starts[0].item())
        total_len = int(prompt_lengths[batch_idx].item() + generated_counts[batch_idx].item())
        row = sequences[batch_idx, start:start + total_len]
        rows.append(row)
        max_len = max(max_len, row.numel())
    output = sequences.new_full((len(rows), max_len), pad_token_id)
    for idx, row in enumerate(rows):
        output[idx, :row.numel()] = row
    return output


def _validate_generate_inputs(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, seq)")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    if attention_mask is not None and torch.any(attention_mask.long().sum(dim=-1) == 0):
        raise ValueError("attention_mask rows must contain at least one valid token")


def _finalize_zero_new_tokens(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    config: GenerationConfig,
) -> torch.Tensor:
    """Finalize left-padded prompts when no new tokens are requested."""
    prompt_lengths = _prompt_lengths(input_ids, attention_mask)
    generated_counts = torch.zeros(
        input_ids.size(0),
        device=input_ids.device,
        dtype=torch.long,
    )
    return _finalize_sequences(
        input_ids.clone(),
        initial_attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        generated_counts=generated_counts,
        pad_token_id=config.pad_token_id,
    )


def _prepare_generation_context(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    config: GenerationConfig,
):
    """Create the mutable generation context used by the decode loop."""
    mask_dtype = _resolve_mask_dtype(model, config)
    sequences = input_ids.clone()
    prefix_attention_mask, prefix_len = _prepare_prefix_attention_mask(
        config,
        input_ids,
        input_ids.device,
    )
    current_attention_mask = attention_mask.clone() if attention_mask is not None else None
    if prefix_attention_mask is not None and current_attention_mask is None:
        current_attention_mask = torch.ones_like(sequences, dtype=torch.long)
    prompt_lengths = _prompt_lengths(input_ids, current_attention_mask)
    prefix_valid_lengths = (
        prefix_attention_mask.long().sum(dim=-1)
        if prefix_attention_mask is not None
        else torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.long)
    )
    cache, prefix_len = _init_cache_with_prefix(config)
    return {
        "mask_dtype": mask_dtype,
        "sequences": sequences,
        "prefix_attention_mask": prefix_attention_mask,
        "current_attention_mask": current_attention_mask,
        "initial_attention_mask": (
            current_attention_mask.clone()
            if current_attention_mask is not None
            else None
        ),
        "prompt_lengths": prompt_lengths,
        "generated_counts": torch.zeros(
            input_ids.size(0), device=input_ids.device, dtype=torch.long,
        ),
        "unfinished": torch.ones(
            input_ids.size(0), device=input_ids.device, dtype=torch.bool,
        ),
        "prefix_valid_lengths": prefix_valid_lengths,
        "cache": cache,
        "prefix_len": prefix_len,
    }


def _prefill(
    model,
    config: GenerationConfig,
    context: dict,
):
    """Run the initial full-prompt forward pass."""
    position_ids = _build_prefill_position_ids(
        context["sequences"],
        context["current_attention_mask"],
        context["prefix_attention_mask"],
    )
    attention_mask = _build_prefill_mask(
        context["sequences"],
        context["current_attention_mask"],
        context["prefix_attention_mask"],
        dtype=context["mask_dtype"],
    )
    cache = context["cache"]
    return _model_forward(
        model,
        input_ids=context["sequences"],
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None if cache.is_empty else cache.past_key_values,
        use_cache=config.use_cache,
        sequence_shard_info=_cache_shard_info(cache),
        global_seq_len=context["prefix_len"] + context["sequences"].shape[-1],
    )


def _required_logits(outputs) -> torch.Tensor:
    logits = _get_output(outputs, "logits")
    if logits is None:
        raise ValueError("model output must contain logits")
    return logits


def _finalize_prefill_outputs(
    config: GenerationConfig,
    context: dict,
    outputs,
) -> tuple[torch.Tensor, bool]:
    """Validate prefill output and decide whether cached decode can be used."""
    logits = _required_logits(outputs)
    if (
        config.prefix_past_key_values is not None
        and _get_output(outputs, "past_key_values") is None
    ):
        raise ValueError("prefix_past_key_values requires model to return past_key_values")
    _update_cache(context["cache"], outputs)
    return logits, config.use_cache and not context["cache"].is_empty


def _append_next_token(context: dict, next_tokens: torch.Tensor, config: GenerationConfig):
    """Append sampled tokens and advance per-row generation metadata."""
    if config.eos_token_id is not None:
        next_tokens = torch.where(
            context["unfinished"].view(-1, 1),
            next_tokens,
            torch.full_like(next_tokens, config.pad_token_id),
        )
    context["sequences"] = torch.cat([context["sequences"], next_tokens], dim=-1)
    context["generated_counts"] = (
        context["generated_counts"] + context["unfinished"].long()
    )
    context["current_attention_mask"] = append_attention_mask(
        context["current_attention_mask"],
        next_tokens,
    )
    if config.eos_token_id is not None:
        context["unfinished"] = (
            context["unfinished"] & (next_tokens.squeeze(-1) != config.eos_token_id)
        )
    return next_tokens


def _should_finish_generation(
    context: dict,
    logits: torch.Tensor,
    config: GenerationConfig,
    step: int,
) -> bool:
    """Check EOS, custom stopping criteria, and max token limit."""
    if config.eos_token_id is not None and not context["unfinished"].any():
        return True
    if should_stop_generation(context["sequences"], logits, config):
        return True
    return step == config.max_new_tokens - 1


def _decode(
    model,
    context: dict,
    next_tokens: torch.Tensor,
    use_cached_decode: bool,
):
    """Run one cached or no-cache decode step."""
    model_attention_mask = _combined_attention_mask(
        context["prefix_attention_mask"],
        context["current_attention_mask"],
    )
    if use_cached_decode:
        decode_pos = (
            context["prefix_valid_lengths"]
            + context["prompt_lengths"]
            + context["generated_counts"]
            - 1
        )
        return _model_forward(
            model,
            input_ids=next_tokens,
            position_ids=decode_pos.view(-1, 1),
            attention_mask=_build_decode_key_mask(
                model_attention_mask,
                context["mask_dtype"],
            ),
            past_key_values=context["cache"].past_key_values,
            use_cache=True,
            sequence_shard_info=_cache_shard_info(context["cache"]),
            global_seq_len=context["prefix_len"] + context["sequences"].shape[-1],
        )
    decode_pos = _build_prefill_position_ids(
        context["sequences"],
        context["current_attention_mask"],
        context["prefix_attention_mask"],
    )
    decode_mask = _build_prefill_mask(
        context["sequences"],
        context["current_attention_mask"],
        context["prefix_attention_mask"],
        dtype=context["mask_dtype"],
    )
    return _model_forward(
        model,
        input_ids=context["sequences"],
        position_ids=decode_pos,
        attention_mask=decode_mask,
        past_key_values=None,
        use_cache=False,
        sequence_shard_info=_cache_shard_info(context["cache"]),
        global_seq_len=context["prefix_len"] + context["sequences"].shape[-1],
    )


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    generation_config: Optional[GenerationConfig] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Generate token ids from a causal language model."""
    if kwargs:
        raise TypeError(f"Unexpected generate kwargs: {sorted(kwargs)}")
    config = generation_config or GenerationConfig()
    _validate_generate_inputs(input_ids, attention_mask)
    if config.max_new_tokens == 0:
        return _finalize_zero_new_tokens(input_ids, attention_mask, config)

    was_training = getattr(model, "training", False)
    model.eval()
    try:
        context = _prepare_generation_context(model, input_ids, attention_mask, config)
        outputs = _prefill(model, config, context)
        logits, use_cached_decode = _finalize_prefill_outputs(
            config,
            context,
            outputs,
        )

        for step in range(config.max_new_tokens):
            next_logits = prepare_logits_for_sampling(logits[:, -1, :], config)
            next_logits = apply_logits_processors(
                context["sequences"],
                next_logits,
                config,
            )
            next_tokens = sample_next_token(next_logits, context["sequences"], config)
            next_tokens = _append_next_token(context, next_tokens, config)
            if _should_finish_generation(context, next_logits, config, step):
                break

            outputs = _decode(model, context, next_tokens, use_cached_decode)
            if use_cached_decode:
                _update_cache(context["cache"], outputs)
            logits = _required_logits(outputs)

        return _finalize_sequences(
            context["sequences"],
            initial_attention_mask=context["initial_attention_mask"],
            prompt_lengths=context["prompt_lengths"],
            generated_counts=context["generated_counts"],
            pad_token_id=config.pad_token_id,
        )
    finally:
        if was_training:
            model.train()

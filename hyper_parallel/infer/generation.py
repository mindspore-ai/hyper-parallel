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

from hyper_parallel.infer.kv_cache import KVCache
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
):
    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
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


def _prompt_lengths(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
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
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, seq)")
    config = generation_config or GenerationConfig()
    if config.max_new_tokens == 0:
        return input_ids.clone()
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    if attention_mask is not None and torch.any(attention_mask.long().sum(dim=-1) == 0):
        raise ValueError("attention_mask rows must contain at least one valid token")

    was_training = getattr(model, "training", False)
    model.eval()
    try:
        mask_dtype = _resolve_mask_dtype(model, config)
        sequences = input_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        initial_attention_mask = (
            current_attention_mask.clone()
            if current_attention_mask is not None
            else None
        )
        prompt_lengths = _prompt_lengths(input_ids, current_attention_mask)
        generated_counts = torch.zeros(
            input_ids.size(0), device=input_ids.device, dtype=torch.long,
        )
        unfinished = torch.ones(
            input_ids.size(0), device=input_ids.device, dtype=torch.bool,
        )

        cache = KVCache()
        position_ids = build_position_ids(sequences, current_attention_mask)
        prefill_mask = build_causal_mask(
            sequences,
            current_attention_mask,
            dtype=mask_dtype,
        )
        outputs = _model_forward(
            model,
            input_ids=sequences,
            position_ids=position_ids,
            attention_mask=prefill_mask,
            past_key_values=None,
            use_cache=config.use_cache,
        )
        logits = _get_output(outputs, "logits")
        if logits is None:
            raise ValueError("model output must contain logits")
        cache.update(_get_output(outputs, "past_key_values"))
        use_cached_decode = config.use_cache and not cache.is_empty

        for step in range(config.max_new_tokens):
            next_logits = prepare_logits_for_sampling(logits[:, -1, :], config)
            next_logits = apply_logits_processors(sequences, next_logits, config)
            next_tokens = sample_next_token(next_logits, sequences, config)
            if config.eos_token_id is not None:
                next_tokens = torch.where(
                    unfinished.view(-1, 1),
                    next_tokens,
                    torch.full_like(next_tokens, config.pad_token_id),
                )
            sequences = torch.cat([sequences, next_tokens], dim=-1)
            generated_counts = generated_counts + unfinished.long()
            current_attention_mask = append_attention_mask(
                current_attention_mask,
                next_tokens,
            )
            if config.eos_token_id is not None:
                unfinished = unfinished & (next_tokens.squeeze(-1) != config.eos_token_id)
                if not unfinished.any():
                    break
            if should_stop_generation(sequences, next_logits, config):
                break
            if step == config.max_new_tokens - 1:
                break

            if use_cached_decode:
                decode_input = next_tokens
                decode_pos = prompt_lengths + generated_counts - 1
                decode_mask = _build_decode_key_mask(current_attention_mask, mask_dtype)
                outputs = _model_forward(
                    model,
                    input_ids=decode_input,
                    position_ids=decode_pos.view(-1, 1),
                    attention_mask=decode_mask,
                    past_key_values=cache.past_key_values,
                    use_cache=True,
                )
                cache.update(_get_output(outputs, "past_key_values"))
            else:
                decode_pos = build_position_ids(sequences, current_attention_mask)
                decode_mask = build_causal_mask(
                    sequences,
                    current_attention_mask,
                    dtype=mask_dtype,
                )
                outputs = _model_forward(
                    model,
                    input_ids=sequences,
                    position_ids=decode_pos,
                    attention_mask=decode_mask,
                    past_key_values=None,
                    use_cache=False,
                )
            logits = _get_output(outputs, "logits")
            if logits is None:
                raise ValueError("model output must contain logits")

        return _finalize_sequences(
            sequences,
            initial_attention_mask=initial_attention_mask,
            prompt_lengths=prompt_lengths,
            generated_counts=generated_counts,
            pad_token_id=config.pad_token_id,
        )
    finally:
        if was_training:
            model.train()

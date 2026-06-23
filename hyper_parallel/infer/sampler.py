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
"""Sampling helpers for autoregressive generation."""
import torch
from torch.nn import functional as F

from hyper_parallel.infer.utils import GenerationConfig


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Select the highest-logit token for each batch item."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab)")
    return logits.argmax(dim=-1, keepdim=True)


def top_k_sample(
    logits: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample from the top-k logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab)")
    if top_k <= 0 or top_k >= logits.size(-1):
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    probs = F.softmax(values / temperature, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return indices.gather(dim=-1, index=sampled)


def top_p_sample(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample from the nucleus token set."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab)")
    if top_p >= 1.0:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits / temperature, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative - sorted_probs > top_p
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(filtered / temperature, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(dim=-1, index=sampled)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply per-item repetition penalty to seen token ids."""
    if penalty == 1.0:
        return logits
    if logits.ndim != 2 or input_ids.ndim != 2:
        raise ValueError("logits and input_ids must be 2-D tensors")
    adjusted = logits.clone()
    vocab_size = logits.size(-1)
    valid = (input_ids >= 0) & (input_ids < vocab_size)
    seen_mask = torch.zeros_like(adjusted, dtype=torch.bool)
    token_ids = input_ids.to(dtype=torch.long).clamp(min=0, max=max(vocab_size - 1, 0))
    seen_mask.scatter_(dim=1, index=token_ids, src=valid)
    penalized = torch.where(adjusted < 0, adjusted * penalty, adjusted / penalty)
    return torch.where(seen_mask, penalized, adjusted)


def sample_next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Apply repetition penalty and select the next token."""
    logits = apply_repetition_penalty(
        logits,
        input_ids=input_ids,
        penalty=config.repetition_penalty,
    )
    if not config.do_sample:
        return greedy_sample(logits)
    if config.top_p < 1.0:
        return top_p_sample(logits, top_p=config.top_p, temperature=config.temperature)
    return top_k_sample(logits, top_k=config.top_k, temperature=config.temperature)

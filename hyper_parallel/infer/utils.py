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
"""Generation configuration and mask helpers."""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist


@dataclass
class GenerationConfig:
    """Runtime options for autoregressive generation."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: Optional[int] = 2
    pad_token_id: int = 0
    repetition_penalty: float = 1.0
    use_cache: bool = True
    prefix_past_key_values: Optional[Any] = None
    prefix_attention_mask: Optional[torch.Tensor] = None
    prefix_sequence_shard_info: Optional[Any] = None
    prefix_cache_length: Optional[int] = None
    context_parallel_cache: bool = False
    context_parallel_rank: Optional[int] = None
    context_parallel_world_size: Optional[int] = None
    # context_logits_rank is local to context_process_group.
    context_logits_rank: Optional[Any] = None
    context_process_group: Optional[Any] = None
    gather_logits: bool = False
    logits_process_group: Optional[Any] = None
    logits_gather_dim: int = -1
    mask_dtype: Optional[torch.dtype] = None
    logits_processor: Optional[List[Callable]] = None
    stopping_criteria: Optional[List[Callable]] = None

    def __post_init__(self):
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.prefix_past_key_values is None:
            if self.prefix_attention_mask is not None:
                raise ValueError("prefix_attention_mask requires prefix_past_key_values")
            if self.prefix_sequence_shard_info is not None:
                raise ValueError("prefix_sequence_shard_info requires prefix_past_key_values")
            if self.prefix_cache_length is not None:
                raise ValueError("prefix_cache_length requires prefix_past_key_values")
        else:
            if not self.use_cache:
                raise ValueError("prefix_past_key_values requires use_cache=True")
            if self.prefix_cache_length is not None and self.prefix_cache_length < 0:
                raise ValueError("prefix_cache_length must be >= 0")
            if self.prefix_sequence_shard_info is not None and not self.context_parallel_cache:
                raise ValueError("prefix_sequence_shard_info requires context_parallel_cache=True")
        if self.context_parallel_cache and not self.use_cache:
            raise ValueError("context_parallel_cache requires use_cache=True")
        if (self.context_parallel_rank is None) != (self.context_parallel_world_size is None):
            raise ValueError(
                "context_parallel_rank and context_parallel_world_size must be set together",
            )
        if self.context_parallel_world_size is not None:
            if self.context_parallel_world_size <= 0:
                raise ValueError("context_parallel_world_size must be > 0")
            if (
                self.context_parallel_rank < 0
                or self.context_parallel_rank >= self.context_parallel_world_size
            ):
                raise ValueError("context_parallel_rank must be in [0, context_parallel_world_size)")
        if self.logits_gather_dim >= 0:
            raise ValueError("logits_gather_dim must be negative")
        if self.mask_dtype is not None and not isinstance(self.mask_dtype, torch.dtype):
            raise ValueError("mask_dtype must be a torch.dtype")
        self._validate_callables(self.logits_processor, "logits_processor")
        self._validate_callables(self.stopping_criteria, "stopping_criteria")

    @staticmethod
    def _validate_callables(values: Optional[List[Callable]], field_name: str) -> None:
        """Validate optional generation extension hooks."""
        if values is None:
            return
        if not isinstance(values, list):
            raise ValueError(f"{field_name} must be a list of callables")
        if not all(callable(item) for item in values):
            raise ValueError(f"{field_name} must contain only callables")


def build_position_ids(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build left-padding aware position ids."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, seq)")
    if attention_mask is None:
        seq_len = input_ids.size(1)
        return torch.arange(
            seq_len, device=input_ids.device, dtype=torch.long,
        ).view(1, -1).expand(input_ids.size(0), -1)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def gather_context_parallel_logits(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    """Select final-token logits from the owning CP rank before sampling."""
    if config.context_logits_rank is None:
        return logits
    if not dist.is_available() or not dist.is_initialized():
        return logits
    world_size = dist.get_world_size(group=config.context_process_group)
    if world_size == 1:
        return logits
    gathered = [torch.empty_like(logits) for _ in range(world_size)]
    dist.all_gather(gathered, logits, group=config.context_process_group)
    stacked = torch.stack(gathered, dim=0)
    owner = torch.as_tensor(
        config.context_logits_rank,
        device=logits.device,
        dtype=torch.long,
    )
    if owner.ndim == 0:
        owner_rank = int(owner.item())
        if owner_rank < 0 or owner_rank >= world_size:
            raise ValueError("context_logits_rank contains an invalid rank")
        return stacked[owner_rank]
    if owner.shape != (logits.shape[0],):
        raise ValueError("context_logits_rank must be a scalar or a batch-sized tensor")
    if torch.any((owner < 0) | (owner >= world_size)):
        raise ValueError("context_logits_rank contains an invalid rank")
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return stacked[owner, batch_indices]


def gather_tensor_parallel_logits(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    """Gather vocab-sharded logits before sampling when TP inference is active."""
    if not config.gather_logits:
        return logits
    if not dist.is_available() or not dist.is_initialized():
        return logits
    world_size = dist.get_world_size(group=config.logits_process_group)
    if world_size == 1:
        return logits
    gather_dim = logits.ndim + config.logits_gather_dim
    if gather_dim < 0 or gather_dim >= logits.ndim:
        raise ValueError("logits_gather_dim is out of range for logits")
    local_shard = torch.tensor(
        [logits.shape[gather_dim]],
        device=logits.device,
        dtype=torch.long,
    )
    shard_sizes = [torch.empty_like(local_shard) for _ in range(world_size)]
    dist.all_gather(shard_sizes, local_shard, group=config.logits_process_group)
    shard_sizes = torch.cat(shard_sizes)
    if torch.any(shard_sizes != shard_sizes[0]):
        raise ValueError(
            "tensor-parallel logits gather requires equal local vocab shard sizes; "
            "pad vocab shards before generation",
        )
    gathered = [torch.empty_like(logits) for _ in range(world_size)]
    dist.all_gather(gathered, logits, group=config.logits_process_group)
    return torch.cat(gathered, dim=config.logits_gather_dim)


def prepare_logits_for_sampling(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    """Apply distributed logits handoffs before sampling."""
    logits = gather_context_parallel_logits(logits, config)
    return gather_tensor_parallel_logits(logits, config)


def apply_logits_processors(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Apply user-supplied logits processors in order."""
    if config.logits_processor is None:
        return logits
    processed = logits
    for processor in config.logits_processor:
        processed = processor(input_ids, processed)
        if not isinstance(processed, torch.Tensor):
            raise ValueError("logits_processor must return a tensor")
    return processed


def should_stop_generation(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    config: GenerationConfig,
) -> bool:
    """Return whether any configured stopping criterion requests termination."""
    if config.stopping_criteria is None:
        return False
    for criterion in config.stopping_criteria:
        result = criterion(input_ids, logits)
        if isinstance(result, torch.Tensor):
            if result.numel() != 1:
                raise ValueError("stopping_criteria tensor output must be scalar")
            result = bool(result.item())
        if bool(result):
            return True
    return False


def build_causal_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build additive causal + padding mask for prefill."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, seq)")
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    mask = torch.zeros(
        batch_size, 1, seq_len, seq_len, device=device, dtype=dtype,
    )
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    mask = mask + causal.view(1, 1, seq_len, seq_len)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        padding = attention_mask.to(device=device) == 0
        mask = mask.masked_fill(padding.view(batch_size, 1, 1, seq_len), float("-inf"))
        mask = mask.masked_fill(padding.view(batch_size, 1, seq_len, 1), 0.0)
    return mask


def append_attention_mask(
    attention_mask: Optional[torch.Tensor],
    next_tokens: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Append valid-token mask entries for generated tokens."""
    if attention_mask is None:
        return None
    ones = torch.ones(
        next_tokens.shape,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat([attention_mask, ones], dim=-1)

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
"""Static request encoding and padded-token-budget batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch

from hyper_parallel.predict.types import PredictionRequest, PredictionSettings


@dataclass
class EncodedRequest:
    """Request identity paired with the complete encoded prompt."""

    request: PredictionRequest
    token_ids: list[int]


def encode_requests(
        requests: list[PredictionRequest], tokenizer: Any,
        chat_template: str | None = None, chat_template_kwargs: dict[str, Any] | None = None,
) -> list[EncodedRequest]:
    """Encode plain prompts or explicitly templated histories without truncation."""
    kwargs = dict(chat_template_kwargs or {})
    reserved = {"chat_template", "tokenize", "add_generation_prompt", "return_dict", "conversation"}
    if reserved.intersection(kwargs):
        raise ValueError("chat_template_kwargs cannot override prediction rendering arguments")
    encoded = []
    for request in requests:
        if request.messages is None:
            tokens = tokenizer.encode(request.prompt, add_special_tokens=True)
        else:
            if chat_template is None:
                raise ValueError("messages require an explicit model_assets.chat_template")
            tokens = tokenizer.apply_chat_template(
                request.messages, chat_template=chat_template, tokenize=True,
                add_generation_prompt=True, return_dict=False, **kwargs,
            )
        if not tokens:
            raise ValueError(f"request {request.input_id!r} encodes to an empty prompt")
        encoded.append(EncodedRequest(request, list(tokens)))
    return encoded


def plan_batches(requests: list[EncodedRequest], config: PredictionSettings) -> Iterator[list[EncodedRequest]]:
    """Yield batches bounded by both request count and padded prompt token count."""
    ordered = sorted(requests, key=lambda value: len(value.token_ids)) if config.bucket_by_length else requests
    batch = []
    width = 0
    for request in ordered:
        length = len(request.token_ids)
        if config.max_batch_tokens is not None and length > config.max_batch_tokens:
            raise ValueError(f"request {request.request.input_id!r} exceeds prediction.max_batch_tokens")
        new_width = max(width, length)
        over_budget = config.max_batch_tokens is not None and new_width * (len(batch) + 1) > config.max_batch_tokens
        if batch and (len(batch) == config.batch_size or over_budget):
            yield batch
            batch, width = [], 0
        batch.append(request)
        width = max(width, length)
    if batch:
        yield batch


def pad_batch(batch: list[EncodedRequest], pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Create left-padded model inputs without changing the tokenizer's settings."""
    width = max(len(request.token_ids) for request in batch)
    ids, masks = [], []
    for request in batch:
        padding = width - len(request.token_ids)
        ids.append([pad_token_id] * padding + request.token_ids)
        masks.append([0] * padding + [1] * len(request.token_ids))
    return {
        "input_ids": torch.tensor(ids, device=device, dtype=torch.long),
        "attention_mask": torch.tensor(masks, device=device, dtype=torch.long),
    }

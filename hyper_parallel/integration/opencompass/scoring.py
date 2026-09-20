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
"""OpenCompass-compatible causal-language-model PPL scoring."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as functional


def _validate_inputs(logits: torch.Tensor, input_ids: torch.Tensor) -> None:
    """Validate the aligned causal-LM logits and token IDs."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocabulary)")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence)")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must have identical batch and sequence dimensions")
    if input_ids.shape[1] < 2:
        raise ValueError("PPL scoring requires at least two tokens per sequence")


def _build_score_mask(
    shift_labels: torch.Tensor,
    mask_length: Sequence[int],
    valid_token_counts: torch.Tensor,
) -> torch.Tensor:
    """Build the OpenCompass prefix-exclusion mask for shifted labels."""
    if len(mask_length) != shift_labels.shape[0]:
        raise ValueError("mask_length must contain one value per input sequence")
    lengths = torch.as_tensor(mask_length, device=shift_labels.device, dtype=torch.long)
    if torch.any(lengths < 1):
        raise ValueError("mask_length values must be at least 1")
    if torch.any(lengths >= valid_token_counts):
        raise ValueError("mask_length must leave at least one scored token in every sequence")
    positions = torch.arange(shift_labels.shape[1], device=shift_labels.device)
    return positions.unsqueeze(0) >= (lengths - 1).unsqueeze(1)


def causal_lm_ppl_scores(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: Optional[int],
    mask_length: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Return OpenCompass-compatible normalized causal-LM NLL scores.

    OpenCompass calls these values PPL scores, but its HuggingFace backend
    returns mean cross entropy rather than exponentiated perplexity. This
    function intentionally matches that behavior so MMLU candidate ordering
    and native-backend parity use the same denominator.

    Args:
        logits: Full-vocabulary logits with shape ``(batch, sequence, vocab)``.
        input_ids: Token IDs aligned with ``logits``.
        pad_token_id: Token ID excluded from loss and length counts, or ``None``
            when no padding exists.
        mask_length: Optional number of leading input tokens excluded per row.

    Returns:
        One normalized negative-log-likelihood score per input sequence.

    Raises:
        ValueError: If shapes or mask lengths cannot define a valid score.
    """
    _validate_inputs(logits, input_ids)
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = input_ids[..., 1:].contiguous()
    ignore_index = pad_token_id if pad_token_id is not None else -100
    losses = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(shift_labels)

    if pad_token_id is None:
        valid_token_counts = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )
    else:
        valid_token_counts = (input_ids != pad_token_id).sum(dim=-1)

    denominators = valid_token_counts
    if mask_length is not None:
        score_mask = _build_score_mask(shift_labels, mask_length, valid_token_counts)
        losses = losses * score_mask
        denominators = valid_token_counts - torch.as_tensor(
            mask_length,
            device=valid_token_counts.device,
            dtype=valid_token_counts.dtype,
        )
    if torch.any(denominators <= 0):
        raise ValueError("every input must contain at least one scored token")
    return losses.sum(dim=-1) / denominators

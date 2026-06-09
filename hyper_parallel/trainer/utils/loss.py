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
"""Token-weighted global loss normalization utilities.

"""
from typing import Any, Dict

from hyper_parallel import get_platform

platform = get_platform()


def count_loss_token(batch: Dict[str, Any]) -> int:
    """Count non-padding tokens in a micro-batch.

    A token is considered valid (non-padding) when its label is not -100,
    which is the conventional ignore index used in cross-entropy loss.

    Args:
        batch: Dictionary containing at least a ``"labels"`` tensor with
               shape ``(batch_size, seq_len)``.

    Returns:
        Integer count of tokens where ``labels != -100``.
    """
    labels = batch.get("labels")
    if labels is None:
        return 0
    return int((labels != -100).sum().item())


def mean_global_loss(
    loss: Any,
    micro_batch_tokens: int,
    total_tokens: int,
    fsdp_size: int,
) -> Any:
    """Compute token-weighted, globally normalised loss for one micro-batch.

    Each micro-batch contributes a fraction proportional to how many of the
    total global tokens it contains.  Multiplying by ``fsdp_size`` corrects
    for the fact that FSDP averages gradients across data-parallel ranks,
    while token counts are *per-rank* (not global).

    Formula::

        normalised_loss = raw_loss * (micro_tokens / global_tokens) * fsdp_size

    Args:
        loss: Raw loss scalar returned by the model (may be a DTensor partial).
        micro_batch_tokens: Number of non-padding tokens in this micro-batch.
        total_tokens: Total non-padding tokens across **all** micro-batches and
                      all data-parallel ranks in this global step.
        fsdp_size: Number of data-parallel (FSDP) ranks participating in
                   gradient reduction.

    Returns:
        Scaled loss with the same type as ``loss``.  If ``total_tokens`` is
        zero, returns ``loss`` unchanged to avoid division by zero.

    Raises:
        ValueError: If ``fsdp_size`` is not a positive integer.
    """
    if fsdp_size <= 0:
        raise ValueError(f"fsdp_size must be a positive integer, got {fsdp_size}")
    if total_tokens <= 0:
        return loss
    return loss * (micro_batch_tokens / total_tokens) * fsdp_size

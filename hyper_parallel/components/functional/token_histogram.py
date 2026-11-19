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
"""Exact token counts with a bounded FP32 histogram fast path."""

from __future__ import annotations

import torch


_FP32_EXACT_INTEGER_LIMIT = 1 << 24
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def token_histogram(
        indices: torch.Tensor,
        num_bins: int,
        valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Count integer IDs without reading device values on the host.

    Args:
        indices: Integer IDs of any shape. Every unmasked ID must be in
            ``[0, num_bins)``; callers guarantee this range without a device sync.
        num_bins: Positive, statically known number of output bins.
        valid_mask: Optional boolean tensor with the same shape and device as
            indices. Masked IDs are ignored and may be outside the valid range.

    Returns:
        INT64 counts of shape ``[num_bins]`` on the indices device.

    Raises:
        ValueError: Bin count, index dtype, or mask metadata is invalid.

    Note:
        FP32 exactly represents all IDs and partial counts when the bin count
        and number of entries are bounded by ``2**24``. Larger shapes retain
        INT64 scatter accumulation. Masked entries use a discarded extra bin
        so the fast path does not need a dynamic-size selection or host sync.
    """
    if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")
    if indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("indices must have an integer dtype")
    if valid_mask is not None and (
            valid_mask.shape != indices.shape
            or valid_mask.dtype != torch.bool
            or valid_mask.device != indices.device
    ):
        raise ValueError("valid_mask must be boolean and match the indices shape and device")
    flat = indices.reshape(-1)
    histogram_bins = num_bins + int(valid_mask is not None)
    if histogram_bins <= _FP32_EXACT_INTEGER_LIMIT and flat.numel() <= _FP32_EXACT_INTEGER_LIMIT:
        values = flat.float()
        if valid_mask is not None:
            values = values.masked_fill(~valid_mask.reshape(-1), num_bins)
        return torch.histc(values, bins=histogram_bins, min=0, max=histogram_bins)[:num_bins].long()

    flat = flat.long()
    weights = torch.ones_like(flat) if valid_mask is None else valid_mask.reshape(-1).long()
    if valid_mask is not None:
        flat = flat.masked_fill(~valid_mask.reshape(-1), 0)
    counts = torch.zeros(num_bins, dtype=torch.int64, device=indices.device)
    return counts.scatter_add_(0, flat, weights)

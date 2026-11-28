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
"""Masked hidden-state aggregation function."""

import torch

from hyper_parallel.components import ops

__all__ = ["aggregate_hidden"]


def aggregate_hidden(input_tensor: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aggregate each hidden channel over a causal window and mask output positions.

    For a window of size K, the output is::

        output[t, b, h] = mask[b, t] * sum(
            weight[k, h] * input_tensor[t - K + 1 + k, b, h]
            for k in range(K)
        )

    Input positions before the sequence start contribute zero. The last filter
    tap multiplies the current token. The mask gates outputs only: a masked
    token can still contribute to later unmasked outputs. Channels are filtered
    independently, without adding a bias or a residual connection.

    Args:
        input_tensor: Floating hidden states of shape [sequence, batch, hidden].
        weight: Filter taps of shape [window, hidden], with a positive window
            size and the same dtype and device as input_tensor.
        mask: Boolean output mask of shape [batch, sequence] on the input device.

    Returns:
        Aggregated hidden states with the input shape, dtype and device. False
        mask positions are zero. Empty sequences or batches return empty states
        while retaining gradient connections to the input and filter weights.

    Raises:
        ValueError: Input shapes, dtypes or devices do not match this contract.
    """
    return ops.aggregate_hidden(input_tensor, weight, mask)

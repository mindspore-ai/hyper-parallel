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
"""Manifold hyper-connection post-processing functions."""

import torch

from hyper_parallel.components.ops import mhc_post as selected_mhc_post


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    num_stream: int,
) -> torch.Tensor:
    """Combine transformed and residual streams after an MHC-wrapped block.

    Args:
        x: Output of the wrapped block.
        residual: Flattened residual streams.
        h_post: Per-stream output mixing coefficients.
        h_res: Residual stream mixing matrix.
        num_stream: Number of residual streams.

    Returns:
        Flattened mixed residual streams.

    Notes:
        Backend selection follows the process-wide ops configuration.
    """
    return selected_mhc_post(x, residual, h_post, h_res, num_stream)

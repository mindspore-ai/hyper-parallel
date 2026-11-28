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
"""Sinkhorn normalization functions used by manifold hyper-connections."""

import torch

from hyper_parallel.components.ops import sinkhorn as selected_sinkhorn
from hyper_parallel.components.ops.impl.sinkhorn import native_sinkhorn


def sinkhorn(h_res: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a square mixing matrix through the configured OpSelect backend.

    Args:
        h_res: Residual mixing matrix.
        sinkhorn_iters: Number of normalization iterations.
        eps: Numerical stability epsilon.

    Returns:
        The normalized residual mixing matrix.
    """
    return selected_sinkhorn(h_res, sinkhorn_iters, eps)


def sinkhorn_knopps(h_res: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a square mixing matrix with the portable PyTorch algorithm.

    Args:
        h_res: Residual mixing logits.
        sinkhorn_iters: Number of normalization iterations.
        eps: Numerical stability epsilon.

    Returns:
        The normalized residual mixing matrix.
    """
    return native_sinkhorn(h_res, sinkhorn_iters, eps)

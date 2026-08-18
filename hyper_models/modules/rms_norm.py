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
"""Root mean square normalization module."""

import torch  # pylint: disable=forbidden-backend-import

from hyper_models.ops import rms_norm


class RMSNorm(torch.nn.Module):
    """NPU-accelerated root mean square normalization module."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        """Initialize the RMSNorm weight.

        Args:
            hidden_size: Size of the normalized dimension.
            eps: Epsilon added to the variance.
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply NPU-accelerated RMS normalization."""
        return rms_norm(x, self.weight, self.eps)

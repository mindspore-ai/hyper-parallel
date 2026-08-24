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
"""Root mean square normalization function."""

import torch  # pylint: disable=forbidden-backend-import
import torch_npu


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply NPU-accelerated RMS normalization.

    Args:
        x: Input tensor.
        weight: RMSNorm scale tensor.
        eps: Positive epsilon added to the variance.

    Returns:
        The normalized tensor.
    """
    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]

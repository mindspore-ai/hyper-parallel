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
"""Root Mean Square Layer Normalization — shared building block."""
import os

import torch
from torch import nn
from torch.nn import functional as F

def _use_npu_rms_norm() -> bool:
    """True when ``HYPER_USE_V1_KERNELS=1`` and ``torch_npu`` is importable.

    Selects the fused ``torch_npu.npu_rms_norm`` kernel; the eager fp32 path
    is used otherwise.
    """
    if os.environ.get("HYPER_USE_V1_KERNELS", "0") != "1":
        return False
    try:
        import torch_npu  # pylint: disable=C0415,W0611
        return True
    except ImportError:
        return False

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Forward dispatches to ``torch_npu.npu_rms_norm`` when
    ``HYPER_USE_V1_RMSNORM=1`` is set in the environment. Otherwise the eager
    fp32 path is used (fp32 normalize then bf16 multiply).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if _use_npu_rms_norm() and hidden_states.device.type == "npu":
            # pylint: disable=C0415
            import torch_npu
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

class RMSNormGated(nn.Module):
    """RMSNorm with multiplicative SiLU-gated path — used by Qwen3.5 / Qwen3-Next
    Gated DeltaNet to fuse the per-head normalization and the residual gate.

    normalize input → multiply by
    ``weight`` → multiply by ``F.silu(gate)``. fp32 internal compute, returns
    in input dtype.

    Args:
        hidden_size: Channel dim that gets normalized.
        eps: RMS epsilon.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **_unused):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

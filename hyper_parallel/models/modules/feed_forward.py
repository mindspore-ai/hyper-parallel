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
"""SwiGLU feed-forward network — shared building block.

Standard naming ``gate_proj`` / ``up_proj`` / ``down_proj`` so a model-level
``_tp_plan`` with ``*.gate_proj``-style patterns plugs in without per-MLP
overrides.
"""
import torch
from torch import nn
from torch.nn import functional as F

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP (Llama / Qwen / Mistral convention).

    Standard naming: ``gate_proj`` / ``up_proj`` / ``down_proj``, matched by
    typical ``*.gate_proj`` / ``*.up_proj`` / ``*.down_proj`` TP plan patterns.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

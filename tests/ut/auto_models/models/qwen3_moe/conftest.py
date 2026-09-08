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
"""Qwen3-MoE expert-layout contract fixtures (Gate-1).

The Qwen3-MoE support matrix spans two Transformers layouts: the 4.57 legacy
``ModuleList`` of per-expert MLPs and the newer batched
``gate_up_proj``/``down_proj`` tensors. Both are faked here so tests assert
the supported matrix explicitly instead of depending on whichever
Transformers version happens to be installed on the development machine.
"""

import pytest
import torch
from torch import nn

from tests.ut.auto_models.models.model_fixtures import TinyMLP, tiny_moe_config


class LegacyExpertsModule(nn.Module):
    """Transformers 4.57 layout: ``experts`` is a ModuleList of MLPs."""

    def __init__(self, hidden_size=8, intermediate_size=16, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        )


class BatchedExpertsModule(nn.Module):
    """Newer layout: experts fused into batched ``gate_up_proj``/``down_proj``.

    Shapes follow the Transformers batched-expert convention:
    ``gate_up_proj`` is ``(num_experts, hidden, 2 * intermediate)`` and
    ``down_proj`` is ``(num_experts, intermediate, hidden)``.
    """

    def __init__(self, hidden_size=8, intermediate_size=16, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, 2 * intermediate_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )


@pytest.fixture
def moe_config():
    """The shared tiny MoE config for both expert layouts."""
    return tiny_moe_config()


@pytest.fixture
def legacy_experts(moe_config):
    """A deterministic legacy (ModuleList) expert block."""
    torch.manual_seed(0)
    return LegacyExpertsModule(
        moe_config.hidden_size, moe_config.intermediate_size, moe_config.num_experts
    )


@pytest.fixture
def batched_experts(moe_config):
    """A deterministic batched (gate_up_proj/down_proj) expert block."""
    torch.manual_seed(0)
    return BatchedExpertsModule(
        moe_config.hidden_size, moe_config.intermediate_size, moe_config.num_experts
    )

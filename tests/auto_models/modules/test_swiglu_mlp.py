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
"""CPU contracts for the high-performance SwiGLU MLP replacement."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from hyper_parallel.auto_models.components.checkpoint import ConcatenateWithSections
from hyper_parallel.auto_models.components.model_transform import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.auto_models.modules import SwiGLUMLP


class SwiGLUSource(nn.Module):
    """Minimal Transformers-style SwiGLU module."""

    def __init__(self) -> None:
        """Build separate Gate, Up, and Down projections."""
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        self.gate_proj = nn.Linear(8, 12, bias=True)
        self.up_proj = nn.Linear(8, 12, bias=True)
        self.down_proj = nn.Linear(12, 8, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the standard SwiGLU MLP formula."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def test_swiglu_mlp_replacement_matches_source() -> None:
    """Packed projections must preserve the source forward result."""
    torch.manual_seed(7)
    source = SwiGLUSource()
    model = nn.Sequential(source)
    inputs = torch.randn(2, 3, 8)
    expected = source(inputs)
    spec = ModuleReplacementSpec(
        match=("0",),
        factory=SwiGLUMLP,
        module_type=SwiGLUSource,
        exact_type=True,
    )

    plan = compile_module_replacements(model, [spec])
    weights_mapping = []
    apply_module_replacements(model, plan, weights_mapping=weights_mapping)

    torch.testing.assert_close(model(inputs), expected)
    assert len(weights_mapping) == 4
    operation = model[0].make_transforms()[0].operations[0]
    assert isinstance(operation, ConcatenateWithSections)

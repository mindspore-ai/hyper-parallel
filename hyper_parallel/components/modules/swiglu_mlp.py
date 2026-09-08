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
"""Transformers-compatible SwiGLU MLP with a fused Gate/Up projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# This package provides PyTorch-specific high-performance modules.
# pylint: disable=forbidden-backend-import
import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F
from torch import nn
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
)

from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.functional import swiglu

@module_replacement
class SwiGLUMLP(nn.Module):
    """Transformers-compatible SwiGLU MLP using one fused Gate/Up matmul."""

    @staticmethod
    def _source_projections(module: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Validate and return the source Gate, Up, and Down projections."""
        required = ("gate_proj", "up_proj", "down_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"SwiGLUMLP source module is missing: {missing}")
        projections = (module.gate_proj, module.up_proj, module.down_proj)
        if not all(isinstance(projection, nn.Linear) for projection in projections):
            raise TypeError("SwiGLUMLP requires Gate, Up, and Down projections to be nn.Linear")
        return projections

    @staticmethod
    def _validate_projection_contract(
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
    ) -> None:
        """Validate source projection shapes, placement, and training policy."""
        if gate_proj.in_features != up_proj.in_features:
            raise ValueError("Gate and Up projections must have the same input size")
        if gate_proj.out_features != up_proj.out_features:
            raise ValueError("Gate and Up projections must have the same output size")
        if down_proj.in_features != gate_proj.out_features:
            raise ValueError("Down projection input size must equal the SwiGLU intermediate size")
        if down_proj.out_features != gate_proj.in_features:
            raise ValueError("Down projection output size must equal the MLP hidden size")
        if gate_proj.weight.device != up_proj.weight.device or gate_proj.weight.dtype != up_proj.weight.dtype:
            raise ValueError("Gate and Up projection weights must share device and dtype")
        if gate_proj.weight.device != down_proj.weight.device or gate_proj.weight.dtype != down_proj.weight.dtype:
            raise ValueError("Gate, Up, and Down projection weights must share device and dtype")
        if gate_proj.weight.requires_grad != up_proj.weight.requires_grad:
            raise ValueError("Gate and Up projection weights must share a training policy")
        if (gate_proj.bias is None) != (up_proj.bias is None):
            raise ValueError("Gate and Up projections must use the same bias policy")
        if gate_proj.bias is not None and gate_proj.bias.requires_grad != up_proj.bias.requires_grad:
            raise ValueError("Gate and Up projection biases must share a training policy")

    def _initialize_fc1(self, gate_proj: nn.Linear, up_proj: nn.Linear) -> None:
        """Create the packed Gate/Up projection with source parameter values."""
        pack = ConcatenateWithSections((self.intermediate_size, self.intermediate_size))
        packed_weight = pack.convert(
            {"gate": gate_proj.weight.detach(), "up": up_proj.weight.detach()},
            ["gate", "up"],
            ["linear_fc1.weight"],
        )["linear_fc1.weight"]
        self.linear_fc1 = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=gate_proj.bias is not None,
            device=gate_proj.weight.device,
            dtype=gate_proj.weight.dtype,
        )
        self.linear_fc1.weight = nn.Parameter(packed_weight, requires_grad=gate_proj.weight.requires_grad)
        if gate_proj.bias is not None:
            packed_bias = pack.convert(
                {"gate": gate_proj.bias.detach(), "up": up_proj.bias.detach()},
                ["gate", "up"],
                ["linear_fc1.bias"],
            )["linear_fc1.bias"]
            self.linear_fc1.bias = nn.Parameter(packed_bias, requires_grad=gate_proj.bias.requires_grad)

    def _initialize_fc2(self, down_proj: nn.Linear) -> None:
        """Create the Down projection and reuse its source parameters."""
        self.linear_fc2 = nn.Linear(
            down_proj.in_features,
            down_proj.out_features,
            bias=down_proj.bias is not None,
            device=down_proj.weight.device,
            dtype=down_proj.weight.dtype,
        )
        self.linear_fc2.weight = down_proj.weight
        self.linear_fc2.bias = down_proj.bias

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the high-performance MLP from separate source projections.

        Args:
            module: Source MLP exposing ``gate_proj``, ``up_proj``, and ``down_proj``.
            module_fqn: Fully qualified source-module name supplied by replacement.
            context: Replacement context supplied by Trainer.

        Raises:
            TypeError: If the source projection modules are missing or unsupported.
            ValueError: If projection layouts, training policies, or activation differ.
        """
        super().__init__()
        del module_fqn, context
        gate_proj, up_proj, down_proj = self._source_projections(module)
        self._validate_projection_contract(gate_proj, up_proj, down_proj)

        config = getattr(module, "config", None)
        hidden_act = getattr(config, "hidden_act", getattr(config, "hidden_activation", None))
        if hidden_act is not None and hidden_act not in ("silu", "swiglu"):
            raise ValueError(f"SwiGLUMLP requires a SiLU activation, but got {hidden_act}")
        self.config = config
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features

        self._initialize_fc1(gate_proj, up_proj)
        self._initialize_fc2(down_proj)
        self.train(module.training)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        transforms: list[WeightRenaming | WeightConverter] = [
            WeightConverter(
                source_patterns=["gate_proj.weight", "up_proj.weight"],
                target_patterns="linear_fc1.weight",
                operations=[
                    ConcatenateWithSections((self.intermediate_size, self.intermediate_size))
                ],
            )
        ]
        if self.linear_fc1.bias is not None:
            transforms.append(
                WeightConverter(
                    source_patterns=["gate_proj.bias", "up_proj.bias"],
                    target_patterns="linear_fc1.bias",
                    operations=[
                        ConcatenateWithSections((self.intermediate_size, self.intermediate_size))
                    ],
                )
            )
        transforms.append(WeightRenaming("down_proj.weight", "linear_fc2.weight"))
        if self.linear_fc2.bias is not None:
            transforms.append(WeightRenaming("down_proj.bias", "linear_fc2.bias"))
        return transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fused Gate/Up projection, SwiGLU, and Down projection."""
        intermediate_parallel = self.linear_fc1(x)
        if intermediate_parallel.device.type == "npu":
            intermediate_parallel = swiglu(intermediate_parallel)
        else:
            gate, up = intermediate_parallel.chunk(2, dim=-1)
            intermediate_parallel = F.silu(gate) * up
        return self.linear_fc2(intermediate_parallel)

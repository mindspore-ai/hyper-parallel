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
from torch import nn
import torch.nn.functional as F
from transformers.core_model_loading import (
    ConversionOps,
    WeightConverter,
    WeightRenaming,
)

from hyper_models.ops import swiglu

try:
    import torch_npu  # pylint: disable=unused-import

    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def _one_tensor(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Return the only tensor collected for one checkpoint pattern."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("SwiGLU projection conversion expects one tensor per source pattern")
        return value[0]
    return value


class _PackGateUpProjection(ConversionOps):
    """Pack separate Gate and Up projections into the high-performance layout."""

    def __init__(self, intermediate_size: int) -> None:
        """Record the size of each projection in the packed dimension."""
        self.intermediate_size = intermediate_size

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Concatenate Gate and Up tensors along their output dimension."""
        del kwargs
        if len(source_patterns) != 2:
            raise ValueError("packing SwiGLU projections requires Gate and Up source patterns")
        if len(target_patterns) != 1:
            raise ValueError("packing SwiGLU projections requires exactly one target pattern")
        gate = _one_tensor(input_dict[source_patterns[0]])
        up = _one_tensor(input_dict[source_patterns[1]])
        if gate.shape[0] != self.intermediate_size or up.shape[0] != self.intermediate_size:
            raise ValueError("Gate and Up tensors must match the configured intermediate size")
        return {target_patterns[0]: torch.cat((gate, up), dim=0).contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the inverse projection-layout conversion."""
        return _UnpackGateUpProjection(self.intermediate_size)


class _UnpackGateUpProjection(ConversionOps):
    """Restore separate Gate and Up projections from the packed layout."""

    def __init__(self, intermediate_size: int) -> None:
        """Record the size of each restored projection."""
        self.intermediate_size = intermediate_size

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Split a packed Gate/Up tensor along its output dimension."""
        del kwargs
        if len(source_patterns) != 1:
            raise ValueError("unpacking SwiGLU projections requires exactly one source pattern")
        if len(target_patterns) != 2:
            raise ValueError("unpacking SwiGLU projections requires Gate and Up target patterns")
        packed = _one_tensor(input_dict[source_patterns[0]])
        if packed.shape[0] != 2 * self.intermediate_size:
            raise ValueError("packed Gate/Up tensor must contain two intermediate-size projections")
        gate, up = torch.split(packed, self.intermediate_size, dim=0)
        return {target_patterns[0]: gate.contiguous(), target_patterns[1]: up.contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the inverse projection-layout conversion."""
        return _PackGateUpProjection(self.intermediate_size)


class SwiGLUMLP(nn.Module):
    """Transformers-compatible SwiGLU MLP using one fused Gate/Up matmul."""

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
        required = ("gate_proj", "up_proj", "down_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"SwiGLUMLP source module is missing: {missing}")

        gate_proj = module.gate_proj
        up_proj = module.up_proj
        down_proj = module.down_proj
        if not all(isinstance(projection, nn.Linear) for projection in (gate_proj, up_proj, down_proj)):
            raise TypeError("SwiGLUMLP requires Gate, Up, and Down projections to be nn.Linear")
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

        config = getattr(module, "config", None)
        hidden_act = getattr(config, "hidden_act", getattr(config, "hidden_activation", None))
        if hidden_act is not None and hidden_act not in ("silu", "swiglu"):
            raise ValueError(f"SwiGLUMLP requires a SiLU activation, but got {hidden_act}")
        self.config = config
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features

        pack = _PackGateUpProjection(self.intermediate_size)
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
        self.linear_fc1.weight = nn.Parameter(
            packed_weight,
            requires_grad=gate_proj.weight.requires_grad,
        )
        if gate_proj.bias is not None:
            packed_bias = pack.convert(
                {"gate": gate_proj.bias.detach(), "up": up_proj.bias.detach()},
                ["gate", "up"],
                ["linear_fc1.bias"],
            )["linear_fc1.bias"]
            self.linear_fc1.bias = nn.Parameter(
                packed_bias,
                requires_grad=gate_proj.bias.requires_grad,
            )

        self.linear_fc2 = nn.Linear(
            down_proj.in_features,
            down_proj.out_features,
            bias=down_proj.bias is not None,
            device=down_proj.weight.device,
            dtype=down_proj.weight.dtype,
        )
        self.linear_fc2.weight = down_proj.weight
        self.linear_fc2.bias = down_proj.bias
        self.train(module.training)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        transforms: list[WeightRenaming | WeightConverter] = [
            WeightConverter(
                source_patterns=["gate_proj.weight", "up_proj.weight"],
                target_patterns="linear_fc1.weight",
                operations=[_PackGateUpProjection(self.intermediate_size)],
            )
        ]
        if self.linear_fc1.bias is not None:
            transforms.append(
                WeightConverter(
                    source_patterns=["gate_proj.bias", "up_proj.bias"],
                    target_patterns="linear_fc1.bias",
                    operations=[_PackGateUpProjection(self.intermediate_size)],
                )
            )
        transforms.append(WeightRenaming("down_proj.weight", "linear_fc2.weight"))
        if self.linear_fc2.bias is not None:
            transforms.append(WeightRenaming("down_proj.bias", "linear_fc2.bias"))
        return transforms

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the fused Gate/Up projection, SwiGLU, and Down projection."""
        intermediate_parallel = torch.matmul(
            hidden_states,
            self.linear_fc1.weight.t().contiguous(),
        )
        if self.linear_fc1.bias is not None:
            intermediate_parallel = intermediate_parallel + self.linear_fc1.bias
        if HAS_NPU and intermediate_parallel.device.type != "cpu":
            intermediate_parallel = swiglu(intermediate_parallel)
        else:
            intermediate_parallel = torch.chunk(intermediate_parallel, 2, dim=-1)
            intermediate_parallel = F.silu(intermediate_parallel[0]) * intermediate_parallel[1]
        output = torch.matmul(
            intermediate_parallel,
            self.linear_fc2.weight.t().contiguous(),
        )
        if self.linear_fc2.bias is not None:
            output = output + self.linear_fc2.bias
        return output

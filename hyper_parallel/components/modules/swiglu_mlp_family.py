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
from math import isfinite
from typing import Any

# This package provides PyTorch-specific high-performance modules.
import torch
import torch.nn.functional as F
from torch import nn
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
)

from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.modules.module_family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    build_family,
    register_module_family,
)


def _validate_mlp_config(config: Any) -> None:
    """Validate projection settings supplied by an arbitrary model config."""
    for name in ("hidden_size", "intermediate_size"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(config.bias, bool):
        raise TypeError("bias must be bool")
    limit = getattr(config, "swiglu_limit", 0.0)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not isfinite(limit):
        raise ValueError("swiglu_limit must be finite; nonpositive values disable clamping")


def swiglu_activation(gate: torch.Tensor, up: torch.Tensor, *, limit: float = 0.0,
                       activation_dtype: str = "input") -> torch.Tensor:
    """Apply declared separate-projection activation math without changing its return dtype.

    FP32 activation remains FP32 even when clamp is disabled. The projection
    owner decides when to cast back before its down projection.

    Args:
        gate: Gate projection activations.
        up: Matching up-projection activations.
        limit: Positive clamp bound; nonpositive values disable clamping.
        activation_dtype: Compute in the input dtype or float32.
    """
    if activation_dtype not in ("input", "float32"):
        raise ValueError("activation_dtype must be input or float32")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not isfinite(limit):
        raise ValueError("SwiGLU limit must be finite")
    if activation_dtype == "float32":
        gate, up = gate.float(), up.float()
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return F.silu(gate) * up


def _resolve_activation_dtype(activation_dtype: str | None, limit: float) -> str:
    """Resolve the shared input/FP32 activation policy without changing projection dtype."""
    if activation_dtype is None:
        activation_dtype = "float32" if limit > 0 else "input"
    if activation_dtype not in ("input", "float32"):
        raise ValueError("activation_dtype must be input or float32")
    return activation_dtype


def local_swiglu(x: torch.Tensor, *, limit: float = 0.0, activation_dtype: str | None = None) -> torch.Tensor:
    """Apply packed SwiGLU, retaining the historical FP32 clamp default.

    Args:
        x: Packed gate/up activations ending in twice the intermediate width.
        limit: Positive clamp bound; nonpositive values disable clamping.
        activation_dtype: Explicit compute dtype policy or the clamp-dependent default.
    """
    activation_dtype = _resolve_activation_dtype(activation_dtype, limit)
    if x.device.type == "npu" and limit <= 0 and activation_dtype == "input":
        # Optional NPU operators are required only by NPU execution.
        from hyper_parallel.components.functional import swiglu  # pylint: disable=C0415
        return swiglu(x)
    gate, up = x.chunk(2, dim=-1)
    return swiglu_activation(gate, up, limit=limit, activation_dtype=activation_dtype).to(x.dtype)


def _activation_settings(source: Any, *, limit: float | None = None,
                         activation_dtype: str | None = None) -> tuple[float, str]:
    """Resolve explicit source metadata without guessing custom forward behavior."""
    config = getattr(source, "config", None)
    source_type = type(source)
    hf_source = source_type.__module__.startswith("transformers.models.")
    hf_clamped_source = (source_type.__module__.startswith("transformers.models.deepseek_v4.")
                         and source_type.__name__ == "DeepseekV4MLP")
    literal_source_limit = limit is None and hf_clamped_source
    if limit is None:
        limit = source.limit if hf_clamped_source else getattr(
            source, "swiglu_limit", 0.0 if hf_source else getattr(config, "swiglu_limit", 0.0))
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not isfinite(limit):
        raise ValueError("swiglu_limit must be finite")
    if literal_source_limit and limit <= 0:
        raise ValueError("HF DeepseekV4 literal nonpositive clamp is not a disabled-clamp source contract")
    if activation_dtype is None:
        activation_dtype = getattr(source, "activation_dtype",
                                   None if hf_source else getattr(config, "activation_dtype", None))
    if activation_dtype is None and hf_clamped_source:
        activation_dtype = "input"
    return limit, _resolve_activation_dtype(activation_dtype, limit)


class SwiGLUMLPModule(nn.Module):
    """Transformers-compatible SwiGLU MLP using one fused Gate/Up matmul."""

    @staticmethod
    def _source_projections(module: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Validate and return the source Gate, Up, and Down projections."""
        required = ("gate_proj", "up_proj", "down_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"SwiGLUMLPModule source module is missing: {missing}")
        projections = (module.gate_proj, module.up_proj, module.down_proj)
        if not all(isinstance(projection, nn.Linear) for projection in projections):
            raise TypeError("SwiGLUMLPModule requires Gate, Up, and Down projections to be nn.Linear")
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
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
        activation_dtype: str | None = None,
        swiglu_limit: float | None = None,
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
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        source = config if config is not None else module
        self.swiglu_limit, self.activation_dtype = _activation_settings(
            source, limit=swiglu_limit, activation_dtype=activation_dtype)
        if config is not None:
            _validate_mlp_config(config)
            self.config = config
            self.hidden_size = config.hidden_size
            self.intermediate_size = config.intermediate_size
            self.linear_fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=config.bias)
            self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
            return
        if isinstance(module, SwiGLUMLPModule):
            self.config = module.config
            self.hidden_size, self.intermediate_size = module.hidden_size, module.intermediate_size
            self.linear_fc1, self.linear_fc2 = module.linear_fc1, module.linear_fc2
            self._identity_source = True
            self.train(module.training)
            return
        if all(isinstance(getattr(module, name, None), nn.Linear) for name in ("w1", "w2")):
            source_config = getattr(module, "config", None)
            if (not hasattr(module, "swiglu_limit")
                    and getattr(source_config, "hidden_act", None) not in ("silu", "swiglu")):
                raise TypeError("packed w1/w2 source must explicitly declare SwiGLU activation semantics")
            if module.w1.out_features != 2 * module.w2.in_features or module.w2.out_features != module.w1.in_features:
                raise ValueError("packed SwiGLU source must contain [gate, up] rows followed by a down projection")
            if module.w1.weight.device != module.w2.weight.device or module.w1.weight.dtype != module.w2.weight.dtype:
                raise ValueError("packed SwiGLU projections must share device and dtype")
            self.config = getattr(module, "config", None)
            self.hidden_size, self.intermediate_size = module.w1.in_features, module.w2.in_features
            self.linear_fc1, self.linear_fc2 = module.w1, module.w2
            self._packed_source = True
            self.train(module.training)
            return
        gate_proj, up_proj, down_proj = self._source_projections(module)
        self._validate_projection_contract(gate_proj, up_proj, down_proj)

        config = getattr(module, "config", None)
        hidden_act = getattr(config, "hidden_act", getattr(config, "hidden_activation", None))
        if hidden_act is not None and hidden_act not in ("silu", "swiglu"):
            raise ValueError(f"SwiGLUMLPModule requires a SiLU activation, but got {hidden_act}")
        self.config = config
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features

        self._initialize_fc1(gate_proj, up_proj)
        self._initialize_fc2(down_proj)
        self.train(module.training)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        if getattr(self, "_identity_source", False):
            return []
        if getattr(self, "_packed_source", False):
            return [WeightRenaming(f"{source}.{name}", f"{target}.{name}")
                    for source, target in (("w1", "linear_fc1"), ("w2", "linear_fc2"))
                    for name, _ in getattr(self, target).named_parameters()]
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

    def forward(self, x: torch.Tensor | None = None, *, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the fused Gate/Up projection, SwiGLU, and Down projection.

        Args:
            x: Token activations, exclusive with hidden_states.
            hidden_states: Equivalent keyword spelling for token activations.
        """
        if (x is None) == (hidden_states is None):
            raise ValueError("provide exactly one of x or hidden_states")
        x = hidden_states if x is None else x
        intermediate_parallel = self.linear_fc1(x)
        intermediate_parallel = local_swiglu(intermediate_parallel, limit=self.swiglu_limit,
                                             activation_dtype=self.activation_dtype)
        return self.linear_fc2(intermediate_parallel)

    def reset_parameters(self) -> None:
        """Initialize owned projections after meta materialization."""
        self.linear_fc1.reset_parameters()
        self.linear_fc2.reset_parameters()


class W1W2SwiGLUMLPModule(nn.Module):
    """Packed gate/up member preserving the vision source's w1/w2 state layout."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 activation_dtype: str | None = None, swiglu_limit: float | None = None) -> None:
        """Construct vision projections or retain a declared packed source unchanged."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self.swiglu_limit, self.activation_dtype = _activation_settings(
            config if config is not None else module, limit=swiglu_limit, activation_dtype=activation_dtype)
        if config is not None:
            _validate_mlp_config(config)
            self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=config.bias)
            self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        else:
            if not all(isinstance(getattr(module, name, None), nn.Linear) for name in ("w1", "w2")):
                raise TypeError("w1_w2 source requires two Linear projections")
            if (module.w1.out_features != 2 * module.w2.in_features
                    or module.w1.in_features != module.w2.out_features):
                raise ValueError("w1_w2 source must contain gate/up rows and a compatible down projection")
            if module.w1.weight.dtype != module.w2.weight.dtype or module.w1.weight.device != module.w2.weight.device:
                raise ValueError("w1/w2 must share dtype and device")
            self.w1, self.w2 = module.w1, module.w2
            self.train(module.training)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the shared SwiGLU math without changing source checkpoint names.

        Args:
            hidden_states: Token activations ending in the input hidden width.
        """
        return self.w2(local_swiglu(self.w1(hidden_states), limit=self.swiglu_limit,
                                   activation_dtype=self.activation_dtype))

    def reset_parameters(self) -> None:
        """Initialize both projections after meta materialization."""
        self.w1.reset_parameters()
        self.w2.reset_parameters()

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Packed source and target share the same logical state names."""
        return []


@module_replacement
def SwiGLUMLP(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    activation_dtype: str | None = None, swiglu_limit: float | None = None,
    state_layout: str = "linear_fc",
) -> nn.Module:
    """Construct a tracked MLP from complete config or separate source projections."""
    return build_family(SwiGLUMLP, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, activation_dtype=activation_dtype,
                        swiglu_limit=swiglu_limit, state_layout=state_layout)


register_module_family(ModuleFamily(
    name="swiglu_mlp", baseline=SwiGLUMLP,
    variants={"local": "hyper_parallel.components.modules.swiglu_mlp_family.SwiGLUMLPModule",
              "w1_w2": "hyper_parallel.components.modules.swiglu_mlp_family.W1W2SwiGLUMLPModule"},
    selector_specs={"state_layout": SelectorSpec(str, "linear_fc")},
    variant_rules=(VariantRule({"state_layout": "w1_w2", "parallel": False}, "w1_w2"),),
    default_variant="local",
))

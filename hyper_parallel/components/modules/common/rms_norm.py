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

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch
from torch._subclasses.fake_tensor import is_fake
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter

from hyper_parallel.components.checkpoint.conversion_ops import AddScalar
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.modules.family import SelectorSpec, VariantRule
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family


def _validate_rms_settings(hidden_size: int, eps: float) -> None:
    """Validate normalization semantics without requiring a concrete config class."""
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("RMSNormModule hidden_size must be a positive integer")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("RMSNormModule eps must be a real scalar")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("RMSNormModule eps must be finite and positive")


def _hf_scale_kind(source: Any) -> str | None:
    """Recognize audited HF parameter representations in the source conversion owner."""
    source_type = type(source)
    if not source_type.__module__.startswith("transformers.models."):
        return None
    if source_type.__name__ in ("Qwen3_5RMSNorm", "Qwen3_5MoeRMSNorm"):
        return "offset"
    if source_type.__name__ in ("Qwen3RMSNorm", "Qwen3MoeRMSNorm", "Qwen2MoeRMSNorm",
                                "Qwen3VLMoeTextRMSNorm", "DeepseekV3RMSNorm", "DeepseekV32RMSNorm",
                                "DeepseekV4RMSNorm"):
        return "direct"
    return None


class _RMSNormPolicy(torch.nn.Module):
    """Declare rounding order and optional FP32 scale ownership at construction."""

    def _set_policy(self, source: Any, cast_order: str | None, fp32_weight: bool | None) -> None:
        """Resolve source rounding order and FP32 scale ownership."""
        if cast_order is None:
            cast_order = getattr(source, "rms_norm_cast_order", None)
        if cast_order is None:
            # HF direct-scale text norms cast normalized activations before multiplication.
            cast_order = "before_weight" if _hf_scale_kind(source) == "direct" else "after_weight"
        if cast_order not in ("before_weight", "after_weight"):
            raise ValueError("cast_order must be before_weight or after_weight")
        if fp32_weight is None:
            fp32_weight = getattr(source, "fp32_weight", False)
        if not isinstance(fp32_weight, bool):
            raise TypeError("fp32_weight must be bool")
        self.rms_norm_cast_order = cast_order
        self.fp32_weight = fp32_weight
        self.fp32_parameter_names = ("weight",) if fp32_weight else ()

    def _apply(self, fn: Any, recurse: bool = True) -> _RMSNormPolicy:
        """Preserve declared FP32 parameters and gradients during conversion."""
        if not self.fp32_weight:
            return super()._apply(fn, recurse=recurse)
        protected = {id(self.weight)}
        if self.weight.grad is not None:
            protected.add(id(self.weight.grad))

        def preserve_scale(tensor: torch.Tensor) -> torch.Tensor:
            converted = fn(tensor)
            if id(tensor) not in protected or converted.dtype == torch.float32:
                return converted
            return (converted.float() if tensor.is_meta
                    else tensor.to(device=converted.device, dtype=torch.float32))

        return super()._apply(preserve_scale, recurse=recurse)


class RMSNormModule(_RMSNormPolicy):
    """NPU-fused direct-scale RMSNorm with source-compatible precision and state."""

    def __init__(
        self,
        hidden_size: int | None = None,
        eps: float = 1e-5,
        *,
        module: torch.nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
        cast_order: str | None = None,
        fp32_weight: bool | None = None,
    ) -> None:
        """Build a direct-scale RMSNormModule or replace an existing module.

        Args:
            hidden_size: Size of the normalized dimension for direct construction.
            eps: Epsilon added to the variance.
            module: Optional source module whose parameter layout is preserved.
            module_fqn: Fully qualified source-module name supplied by replacement.
            context: Replacement context supplied by Trainer.
            config: Complete direct construction configuration, exclusive with source arguments.
            cast_order: Rounding boundary relative to scale multiplication.
            fp32_weight: Keep the scale FP32 during model dtype conversion.

        Raises:
            TypeError: If the source module does not expose a supported RMSNormModule contract.
            ValueError: If neither a source module nor a positive hidden size is supplied.
        """
        super().__init__()
        del module_fqn, context
        self._set_policy(config if config is not None else module, cast_order, fp32_weight)
        if config is not None:
            if module is not None or hidden_size is not None:
                raise ValueError("config cannot be combined with module or hidden_size")
            _validate_rms_settings(config.hidden_size, config.eps)
            hidden_size, eps = config.hidden_size, config.eps
        if module is not None:
            if _hf_scale_kind(module) == "offset" or isinstance(module, OffsetRMSNormModule):
                raise ValueError("unit-offset source requires OffsetRMSNorm, not direct-scale RMSNorm")
            if not isinstance(getattr(module, "weight", None), torch.nn.Parameter):
                raise TypeError("RMSNormModule source module must expose an nn.Parameter weight")
            source_eps = getattr(module, "eps", None)
            if source_eps is None:
                source_eps = getattr(module, "variance_epsilon", None)
            if source_eps is None:
                raise TypeError(
                    "RMSNormModule source module must expose eps or variance_epsilon"
                )
            self.eps = float(source_eps)
            if module.weight.ndim != 1:
                raise ValueError("RMSNormModule source weight must be one-dimensional")
            _validate_rms_settings(module.weight.numel(), self.eps)
            if self.fp32_weight and module.weight.dtype != torch.float32:
                raise ValueError("fp32_weight source must already own an FP32 scale")
            self.weight = module.weight
            self.train(module.training)
            return
        _validate_rms_settings(hidden_size, eps)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32 if self.fp32_weight else None))

    def _normalize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return _npu_rms_norm(x, scale, self.eps, self.rms_norm_cast_order)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the concrete member with the declared precision contract."""
        return self._normalize(hidden_states, self.weight)

    def reset_parameters(self) -> None:
        """Restore the direct scale after meta materialization."""
        torch.nn.init.ones_(self.weight)

    def make_transforms(self) -> list:
        """Declare that this member keeps parameter names and layouts unchanged.

        Returns:
            An empty converter list, matching members whose initialization is
            fully owned by ``reset_parameters``.
        """
        return []


class OffsetRMSNormModule(_RMSNormPolicy):
    """NPU-fused unit-offset RMSNorm preserving offset checkpoint state."""

    def __init__(self, *, config: Any = None, module: torch.nn.Module | None = None,
                 module_fqn: str = "", context: Any = None,
                 cast_order: str | None = None, fp32_weight: bool | None = None,
                 source_representation: str = "offset") -> None:
        """Retain offset Parameters; checkpoints use the source's offset representation."""
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self._set_policy(config if config is not None else module, cast_order, fp32_weight)
        if source_representation not in ("offset", "scale"):
            raise ValueError("source_representation must be offset or scale")
        if config is not None and source_representation != "offset":
            raise ValueError("direct construction stores offsets; scale conversion requires a source")
        self.source_representation = source_representation
        if _hf_scale_kind(module) == "direct" and source_representation != "scale":
            raise ValueError("direct-scale HF source requires RMSNorm, not OffsetRMSNorm")
        if config is not None:
            self.eps = config.eps
            _validate_rms_settings(config.hidden_size, self.eps)
            self.weight = torch.nn.Parameter(torch.zeros(
                config.hidden_size, dtype=torch.float32 if self.fp32_weight else None))
        else:
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 1:
                raise TypeError("offset source must expose a one-dimensional Parameter weight")
            self.eps = module.eps
            _validate_rms_settings(weight.numel(), self.eps)
            if self.fp32_weight and weight.dtype != torch.float32:
                raise ValueError("fp32_weight source must already own an FP32 offset")
            if source_representation == "scale":
                if isinstance(module, OffsetRMSNormModule) or _hf_scale_kind(module) == "offset":
                    raise ValueError("offset source must not be declared as a direct scale")
                weight = torch.nn.Parameter(weight.detach() - 1.0, requires_grad=weight.requires_grad)
            self.weight = weight
            self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize the stored offset, not the effective scale."""
        torch.nn.init.zeros_(self.weight)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe only an explicitly requested legacy direct-scale conversion."""
        if self.source_representation == "scale":
            return [WeightConverter(source_patterns=["weight"], target_patterns="weight",
                                    operations=[AddScalar(-1.0)])]
        return []

    def _normalize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return _npu_rms_norm(x, scale, self.eps, self.rms_norm_cast_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the offset in FP32 before applying the declared rounding order."""
        return self._normalize(x, 1.0 + self.weight.float())


def _npu_rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float, cast_order: str) -> torch.Tensor:
    """Execute fused NPU normalization with the declared source rounding boundary."""
    # Census traces installed fused modules with CPU FakeTensors and operator stand-ins.
    if x.device.type != "npu" and not is_fake(x):
        raise ValueError("NPU RMSNorm requires NPU tensors; select use_fused_ops=False for Torch execution")
    # Native members must not load the optional NPU package.
    from hyper_parallel.components.functional.rms_norm import rms_norm  # pylint: disable=import-outside-toplevel

    if cast_order == "before_weight":
        # The NPU kernel accumulates internally; retain the input dtype for saved activations.
        normalized = rms_norm(x, torch.ones_like(scale, dtype=x.dtype), eps)
        return scale * normalized
    return rms_norm(x.float(), scale.float(), eps).to(x.dtype)


def _native_rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float, cast_order: str) -> torch.Tensor:
    """Preserve FP32 accumulation and the source rounding boundary using Torch operations."""
    values = x.float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    if cast_order == "before_weight":
        return scale * normalized.to(x.dtype)
    return (normalized * scale.float()).to(x.dtype)


class NativeRMSNormModule(RMSNormModule):
    """Ordinary Torch direct-scale member sharing the fused member's parameter contract."""

    def _normalize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return _native_rms_norm(x, scale, self.eps, self.rms_norm_cast_order)


class NativeOffsetRMSNormModule(OffsetRMSNormModule):
    """Ordinary Torch unit-offset member retaining offset parameters without conversion."""

    def _normalize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return _native_rms_norm(x, scale, self.eps, self.rms_norm_cast_order)


@module_replacement
def RMSNorm(  # pylint: disable=invalid-name
    *, config: Any = None, module: torch.nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    cast_order: str | None = None, fp32_weight: bool | None = None, use_fused_ops: bool = False,
) -> torch.nn.Module:
    """Build a tracked RMSNormModule family member from config or a direct-scale source.

    ``use_fused_ops=False`` selects the ordinary Torch member by default;
    ``use_fused_ops=True`` explicitly selects the NPU-fused member without fallback.
    Existing RMSNormModule class construction remains available for type compatibility.
    Offset source conversion continues through OffsetRMSNormModule.
    """
    return build_family(RMSNorm, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, cast_order=cast_order, fp32_weight=fp32_weight,
                        use_fused_ops=use_fused_ops)


register_module_family(ModuleFamily(
    name="rms_norm", baseline=RMSNorm,
    variants={"local": "hyper_parallel.components.modules.common.rms_norm.NativeRMSNormModule",
              "npu": "hyper_parallel.components.modules.common.rms_norm.RMSNormModule"},
    selector_specs={"use_fused_ops": SelectorSpec(bool, False)},
    variant_rules=(VariantRule({"use_fused_ops": True, "parallel": False}, "npu"),),
    default_variant="local",
))


@module_replacement
def OffsetRMSNorm(  # pylint: disable=invalid-name
    *, config: Any = None, module: torch.nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    cast_order: str | None = None, fp32_weight: bool | None = None,
    source_representation: str = "offset", use_fused_ops: bool = False,
) -> torch.nn.Module:
    """Select native/NPU execution while preserving the offset checkpoint representation."""
    return build_family(OffsetRMSNorm, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, cast_order=cast_order, fp32_weight=fp32_weight,
                        source_representation=source_representation, use_fused_ops=use_fused_ops)


register_module_family(ModuleFamily(
    name="offset_rms_norm", baseline=OffsetRMSNorm,
    variants={"local": "hyper_parallel.components.modules.common.rms_norm.NativeOffsetRMSNormModule",
              "npu": "hyper_parallel.components.modules.common.rms_norm.OffsetRMSNormModule"},
    selector_specs={"use_fused_ops": SelectorSpec(bool, False)},
    variant_rules=(VariantRule({"use_fused_ops": True, "parallel": False}, "npu"),),
    default_variant="local",
))

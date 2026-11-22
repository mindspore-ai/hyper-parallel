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
"""Gate and token-choice routing protocols with shared family registration."""

from __future__ import annotations
from collections.abc import Mapping
from functools import lru_cache
from math import isfinite
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.family import SelectorSpec, VariantRule
from hyper_parallel.models.replacement import module_replacement


def routing_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Apply nonnegative expert scoring without mixing in correction bias or model statistics."""
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()  # pylint: disable=not-callable
    if scoring_func == "softmax":
        return logits.softmax(dim=-1)
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"unsupported router scoring_func: {scoring_func!r}")



def select_topk_scores(scores: torch.Tensor, *, top_k: int, correction_bias: torch.Tensor | None = None,
                       routed_scaling_factor: float = 1.0, normalize_topk: bool = True,
                       sorted_indices: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Select with correction bias, then gather unbiased weights and optionally normalize.

    """
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating [tokens, experts] tensor")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= scores.shape[-1]:
        raise ValueError("top_k must be an integer in [1, num_experts]")
    if (isinstance(routed_scaling_factor, bool) or not isinstance(routed_scaling_factor, (int, float))
            or not isfinite(routed_scaling_factor)):
        raise ValueError("routed_scaling_factor must be finite")
    if not isinstance(normalize_topk, bool) or not isinstance(sorted_indices, bool):
        raise TypeError("normalize_topk and sorted_indices must be bool")
    selection = scores
    if correction_bias is not None:
        if (correction_bias.device != scores.device or not correction_bias.is_floating_point()
                or tuple(correction_bias.shape) not in (
                    (scores.shape[-1],), (1, scores.shape[-1]), tuple(scores.shape))):
            raise ValueError("correction_bias must share the device and broadcast as [experts] or [tokens, experts]")
        selection = scores + correction_bias.float()
    indices = selection.topk(top_k, dim=-1, sorted=sorted_indices).indices
    weights = scores.gather(1, indices)
    if normalize_topk:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * routed_scaling_factor, indices



class GateModule(nn.Module):
    """Own common gate configuration, projection state, initialization and routing math.

    Projection still follows the enclosing Torch autocast policy, as the model
    source does. Correction state and fused execution belong to derived gates.
    """

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Construct routing math or retain a public member; model state is not a replacement boundary."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            self._validate_source(module)
        source = config if config is not None else module
        self._set_routing_settings(source)
        if module is None:
            self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
            # Derived gates create their correction state after the common projection.
            GateModule.reset_parameters(self)
        else:
            weight = getattr(module, "weight", None)
            if (not isinstance(weight, nn.Parameter) or not weight.is_floating_point()
                    or tuple(weight.shape) != (self.num_experts, self.hidden_size)):
                raise ValueError("source weight must be a floating Parameter [num_experts, hidden_size]")
            self.weight = weight
            self.train(module.training)

    def _set_routing_settings(self, source: Any) -> None:
        """Validate dimensions, scoring, TopK normalization and initialization policy."""
        for name in ("hidden_size", "num_experts", "top_k"):
            value = getattr(source, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.scoring_func = source.scoring_func
        if self.scoring_func not in ("sqrtsoftplus", "softmax", "sigmoid"):
            raise ValueError("unsupported scoring_func")
        self.routed_scaling_factor = source.routed_scaling_factor
        if (isinstance(self.routed_scaling_factor, bool)
                or not isinstance(self.routed_scaling_factor, (int, float))
                or not isfinite(self.routed_scaling_factor)):
            raise ValueError("routed_scaling_factor must be finite")
        self.normalize_topk = getattr(source, "normalize_topk", self.top_k > 1)
        self.sorted_indices = getattr(source, "sorted_indices", False)
        if not isinstance(self.normalize_topk, bool) or not isinstance(self.sorted_indices, bool):
            raise TypeError("normalize_topk and sorted_indices must be bool")
        self.initializer_range = source.initializer_range
        if (isinstance(self.initializer_range, bool) or not isinstance(self.initializer_range, (int, float))
                or not isfinite(self.initializer_range) or self.initializer_range < 0):
            raise ValueError("initializer_range must be finite and nonnegative")

    @classmethod
    def _validate_source(cls, module: nn.Module) -> None:
        if module.__class__ is not cls:
            raise TypeError("model routers own extra state; reuse routing math rather than replacing the whole router")

    def reset_parameters(self) -> None:
        """Initialize the projection after direct/meta construction."""
        nn.init.normal_(self.weight, std=self.initializer_range)

    def make_transforms(self) -> list:
        """Declare that this member keeps parameter names and layouts unchanged.

        Returns:
            An empty converter list, matching members whose initialization is
            fully owned by ``reset_parameters``.
        """
        return []

    def forward(self, hidden_states: torch.Tensor,
                correction_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """Return flattened logits, scaled weights and selected expert IDs."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must end in hidden_size and include a token dimension")
        logits = F.linear(  # pylint: disable=not-callable
            hidden_states.reshape(-1, self.hidden_size).float(), self.weight.float())
        weights, indices = select_topk_scores(
            routing_scores(logits, self.scoring_func), top_k=self.top_k, correction_bias=correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            normalize_topk=self.normalize_topk, sorted_indices=self.sorted_indices)
        return logits, weights, indices



@lru_cache(maxsize=None)
def _plan_for_device(device: torch.device) -> Any:
    # HyperMulticore is optional and its runtime is needed only for native NPU calls.
    from hyper_multicore.modules.mega_gate.plan import build_mega_gate_plan  # pylint: disable=import-outside-toplevel

    return build_mega_gate_plan(device)



class MegaGateModule(GateModule):
    """Extend the ordinary gate with correction biases and a fused NPU route.

    The Gate base owns the projection and common routing contract. HyperMulticore
    supplies the native function and plan, without owning this module's state.
    """

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 module_fqn: str = "", context: Mapping[str, Any] | None = None) -> None:
        """Construct or adopt the public MegaGate without accepting model-specific subclasses."""
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            self._validate_source(module)
        source = config if config is not None else module
        if (getattr(source, "normalize_topk", source.top_k > 1) != (source.top_k > 1)
                or getattr(source, "sorted_indices", False)):
            raise ValueError("MegaGate requires normalize_topk=(top_k > 1) and unsorted indices")
        vision_enabled = (getattr(source, "vision_enabled", False)
                          if config is not None else getattr(source, "bias_vl", None) is not None)
        if not isinstance(vision_enabled, bool):
            raise TypeError("vision_enabled must be bool")
        super().__init__(config=config, module=module)
        for name in ("bias", "bias_vl"):
            if name == "bias_vl" and not vision_enabled:
                self.register_parameter(name, None)
                continue
            if module is None:
                parameter = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32, device=self.weight.device))
            else:
                parameter = getattr(module, name, None)
                if (not isinstance(parameter, nn.Parameter) or tuple(parameter.shape) != (self.num_experts,)
                        or parameter.device != self.weight.device or parameter.dtype != torch.float32):
                    raise ValueError(f"source {name} must be an FP32 Parameter [num_experts] on the weight device")
            setattr(self, name, parameter)

    @classmethod
    def _validate_source(cls, module: nn.Module) -> None:
        if module.__class__ is cls:
            return
        if not module.__class__.__module__.startswith("hyper_multicore.modules.mega_gate"):
            raise TypeError("MegaGate family accepts only the public MegaGate; use a model adapter for other routers")
        # Existing HyperMulticore modules remain valid sources; this is not a base class.
        from hyper_multicore.modules.mega_gate import MegaGate as SourceMegaGate  # pylint: disable=import-outside-toplevel

        if module.__class__ is not SourceMegaGate:
            raise TypeError("MegaGate family accepts only the public MegaGate; use a model adapter for other routers")

    def reset_parameters(self) -> None:
        """Reset the common projection and MegaGate-specific correction state."""
        super().reset_parameters()
        nn.init.zeros_(self.bias)
        if self.bias_vl is not None:
            nn.init.zeros_(self.bias_vl)

    def make_transforms(self) -> list:
        """Declare that this member keeps parameter names and layouts unchanged.

        Returns:
            An empty converter list, matching members whose initialization is
            fully owned by ``reset_parameters``.
        """
        return []

    def _apply(self, fn: Any, recurse: bool = True) -> MegaGateModule:
        """Preserve declared FP32 parameters and gradients during conversion."""
        # Keep correction values and their gradients FP32 through model dtype conversion.
        protected = {id(parameter) for parameter in (self.bias, self.bias_vl) if parameter is not None}
        protected.update(id(parameter.grad) for parameter in self.parameters() if parameter.grad is not None
                         and id(parameter) in protected)

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            converted = fn(tensor)
            if id(tensor) not in protected or converted.dtype == torch.float32:
                return converted
            return converted.float() if tensor.is_meta else tensor.to(device=converted.device, dtype=torch.float32)

        return super()._apply(apply, recurse=recurse)

    def _torch_forward(self, hidden_states: torch.Tensor,
                       image_mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Use base gate math while keeping optional modality bias owned by MegaGate."""
        correction_bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            correction_bias = torch.where(image_mask.reshape(-1, 1), self.bias_vl, self.bias)
        return super().forward(hidden_states, correction_bias)

    def _native_forward(self, hidden_states: torch.Tensor,
                        image_mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        # Only the fused execution path requires the optional native package.
        from hyper_multicore.modules.mega_gate.function import mega_gate  # pylint: disable=import-outside-toplevel

        return mega_gate(hidden_states, self.weight, self.bias, _plan_for_device(hidden_states.device),
                         vision_bias=self.bias_vl if image_mask is not None else None,
                         image_mask=None if image_mask is None else image_mask.view(-1),
                         top_k=self.top_k, routed_scaling_factor=self.routed_scaling_factor)

    # The fused NPU API takes a modality mask in the second argument.
    def forward(self, hidden_states: torch.Tensor,  # pylint: disable=arguments-renamed
                image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """Use shared Gate math or the equivalent native sqrt-softplus NPU route."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must end in hidden_size and include a token dimension")
        if image_mask is not None:
            if self.bias_vl is None:
                raise ValueError("image_mask requires vision_enabled=True when constructing MegaGate")
            if (image_mask.shape != hidden_states.shape[:2] or image_mask.dtype != torch.bool
                    or image_mask.device != hidden_states.device):
                raise ValueError("image_mask must be BOOL [batch, sequence] on the hidden_states device")
        if self.scoring_func != "sqrtsoftplus" or hidden_states.device.type != "npu" or hidden_states.numel() == 0:
            return self._torch_forward(hidden_states, image_mask)
        if image_mask is not None and not image_mask.is_contiguous():
            raise ValueError("image_mask must be contiguous for native MegaGate")
        return self._native_forward(hidden_states, image_mask)

    @classmethod
    def from_module(cls, module: nn.Module, **configuration: Any) -> MegaGateModule:
        """Retain the source's declared settings; overrides require a separate explicit construction."""
        if configuration:
            raise ValueError("family source construction preserves source settings; overrides are unsupported")
        return cls(module=module)



@module_replacement
def Gate(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
         context: Any = None, mesh: Any = None, implementation: str = "local") -> nn.Module:
    """Build a local gate or the compatible MegaGate execution variant."""
    return build_family(Gate, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, implementation=implementation)



register_module_family(ModuleFamily(
    name="gate", baseline=Gate,
    variants={"local": "hyper_parallel.components.modules.moe.router.GateModule",
              "mega": "hyper_parallel.components.modules.moe.router.MegaGateModule"},
    default_variant="local", selector_specs={"implementation": SelectorSpec(str, "local")},
    variant_rules=(VariantRule({"implementation": "mega", "parallel": False}, "mega"),),
))


__all__ = [
    "Gate", "GateModule", "MegaGateModule",
    "routing_scores", "select_topk_scores",
]

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
"""Model-facing Torch MegaGate router module."""

from __future__ import annotations

__all__ = ["GateBackend", "GateOutput", "MegaGate"]

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any, NamedTuple

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.modules._precision import preserve_fp32_parameters
from hyper_parallel.core.multicore.modules.mega_gate.function import mega_gate
from hyper_parallel.core.multicore.modules.mega_gate.plan import MegaGatePlan, build_mega_gate_plan


class GateOutput(NamedTuple):
    """Flattened token routing results, retaining positional unpacking compatibility."""

    logits: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor


class GateBackend(NamedTuple):
    """Selected Route implementation and its reason, without initializing native resources."""

    name: str
    reason: str


def _reset_mega_gate_parameters(
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter,
    bias_vl: torch.nn.Parameter | None,
    initializer_range: float,
) -> None:
    """Initialize one MegaGate parameter set."""
    torch.nn.init.normal_(weight, mean=0.0, std=initializer_range)
    torch.nn.init.zeros_(bias)
    if bias_vl is not None:
        torch.nn.init.zeros_(bias_vl)


def _create_mega_gate_parameters(
    hidden_size: int,
    num_experts: int,
    vision_enabled: bool,
    initializer_range: float,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter | None]:
    """Create independently initialized projection and correction biases."""
    weight = torch.nn.Parameter(torch.empty(num_experts, hidden_size))
    bias = torch.nn.Parameter(torch.empty(num_experts, dtype=torch.float32))
    bias_vl = (
        torch.nn.Parameter(torch.empty(num_experts, dtype=torch.float32))
        if vision_enabled
        else None
    )
    _reset_mega_gate_parameters(weight, bias, bias_vl, initializer_range)
    return weight, bias, bias_vl


@lru_cache(maxsize=None)
def _plan_for_device(device: torch.device) -> MegaGatePlan:
    """Build one immutable MegaGate plan per process and NPU device."""
    return build_mega_gate_plan(device)


class MegaGate(torch.nn.Module):
    """Project hidden states and select experts with MegaGate routing.

    The module owns its projection and correction-bias parameters. NPU
    sqrt-softplus calls use HyperMegaGate Route; other scoring functions and
    non-NPU tensors use the equivalent Torch implementation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        scoring_func: str,
        routed_scaling_factor: float,
        vision_enabled: bool = False,
        initializer_range: float = 0.02,
    ) -> None:
        """Initialize a standalone MegaGate router.

        Args:
            hidden_size: Size of the final hidden-state dimension.
            num_experts: Number of routed experts.
            top_k: Number of experts selected for each token.
            scoring_func: ``sqrtsoftplus``, ``softmax`` or ``sigmoid``.
            routed_scaling_factor: Finite multiplier applied to routing weights.
            vision_enabled: Create a separate correction bias for visual tokens.
            initializer_range: Standard deviation of the normal weight initialization.

        Raises:
            ValueError: An extent, scoring function, scale or initializer is invalid.
        """
        super().__init__()
        if hidden_size <= 0 or num_experts <= 0:
            raise ValueError("hidden_size and num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if scoring_func not in ("sqrtsoftplus", "softmax", "sigmoid"):
            raise ValueError(f"unsupported scoring_func: {scoring_func!r}")
        if not math.isfinite(routed_scaling_factor):
            raise ValueError("routed_scaling_factor must be finite")
        if not math.isfinite(initializer_range) or initializer_range < 0.0:
            raise ValueError("initializer_range must be finite and nonnegative")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.scoring_func = scoring_func
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.initializer_range = float(initializer_range)
        self.weight, self.bias, bias_vl = _create_mega_gate_parameters(
            self.hidden_size,
            self.num_experts,
            vision_enabled,
            self.initializer_range,
        )
        if bias_vl is None:
            self.register_parameter("bias_vl", None)
        else:
            self.bias_vl = bias_vl

    def reset_parameters(self) -> None:
        """Reset projection and correction-bias parameters."""
        _reset_mega_gate_parameters(
            self.weight,
            self.bias,
            self.bias_vl,
            self.initializer_range,
        )

    @classmethod
    def from_module(cls, module: torch.nn.Module, **configuration: Any) -> MegaGate:
        """Bind an existing router's projection and correction-bias Parameters.

        Args:
            module: Router with weight [E, H], optional correction bias [E] and bias_vl [E].
                Its bias is a TopK correction, not a linear-projection bias.
            **configuration: top_k, scoring_func and routed_scaling_factor, when not present on module.

        Returns:
            Router sharing the original Parameters. Construct the optimizer after module replacement.
        """
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
            raise ValueError("source router requires a weight Parameter shaped [num_experts, hidden_size]")
        options = {}
        for name in ("top_k", "scoring_func", "routed_scaling_factor"):
            value = configuration.pop(name, getattr(module, name, None))
            if value is None:
                raise ValueError(f"Provide {name} or expose it on the source router")
            options[name] = value
        bias_vl = getattr(module, "bias_vl", None)
        result = cls(hidden_size=weight.shape[1], num_experts=weight.shape[0],
                     vision_enabled=bias_vl is not None, **options, **configuration).to(device=weight.device)
        result.weight = weight
        for name in ("bias", "bias_vl"):
            value = getattr(module, name, None)
            if value is not None:
                if (not isinstance(value, torch.nn.Parameter) or value.shape != (weight.shape[0],)
                        or value.dtype != torch.float32 or value.device != weight.device):
                    raise ValueError(f"source {name} must be an FP32 Parameter [num_experts] on the weight device")
                setattr(result, name, value)
        return result

    def _apply(self, fn: Callable, recurse: bool = True) -> MegaGate:
        """Keep correction biases FP32 under enclosing model dtype conversion."""
        return super()._apply(preserve_fp32_parameters(self, fn, ("bias", "bias_vl")), recurse=recurse)

    def backend_for(self, hidden_states: torch.Tensor) -> GateBackend:
        """Query Route selection for this input without loading or executing native code.

        Args:
            hidden_states: Input whose device and emptiness determine execution support.

        Returns:
            Selected backend name and reason; scoring semantics are never changed to select a backend.
        """
        if self.scoring_func != "sqrtsoftplus":
            return GateBackend("torch", f"{self.scoring_func} uses the Torch implementation")
        if hidden_states.device.type != "npu":
            return GateBackend("torch", "native Route requires NPU input")
        if hidden_states.numel() == 0:
            return GateBackend("torch", "empty input uses the Torch implementation")
        return GateBackend("native", "NPU sqrtsoftplus Route")

    def _plan(self, hidden_states: torch.Tensor) -> MegaGatePlan:
        """Return the process-level plan for an NPU tensor."""
        return _plan_for_device(hidden_states.device)

    def _torch_forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the DeepSeek V4.1 reference operation order with Torch ops."""
        flattened = hidden_states.reshape(-1, self.hidden_size)
        logits = functional.linear(  # pylint: disable=not-callable
            flattened.float(), self.weight.float()
        )
        if self.scoring_func == "sqrtsoftplus":
            scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
        elif self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise ValueError(f"Unsupported V4.1 router scoring function: {self.scoring_func!r}")

        correction_bias = self.bias
        if image_mask is not None:
            if self.bias_vl is not None:
                correction_bias = torch.where(
                    image_mask.reshape(-1, 1),
                    self.bias_vl.unsqueeze(0),
                    self.bias.unsqueeze(0),
                )
        expert_indices = torch.topk(
            scores + correction_bias,
            self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        routing_weights = scores.gather(1, expert_indices)
        if self.top_k > 1:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1.0e-20
            )
        return logits, routing_weights * self.routed_scaling_factor, expert_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> GateOutput:
        """Project hidden states and select experts for every flattened token.

        Args:
            hidden_states: Tensor shaped ``[batch, sequence, hidden_size]``.
                The projection follows the active autocast policy. The
                logits are converted to FP32 before the native NPU Route.
            image_mask: Optional BOOL tensor shaped ``[batch, sequence]``.
                When visual bias is enabled, true entries select ``bias_vl``
                and false entries select ``bias``. Native visual routing
                requires a contiguous mask on the same NPU as
                ``hidden_states``. Passing a mask without vision_enabled=True raises ValueError.

        Returns:
            A GateOutput containing logits ``[tokens, num_experts]`` in the
            projection output dtype, FP32 routing weights
            ``[tokens, top_k]`` and INT64 expert indices ``[tokens, top_k]``.

        Raises:
            ValueError: ``image_mask`` does not have shape
                ``[batch, sequence]`` or a native visual-routing mask is not
                contiguous.
            RuntimeError: Native loading, dtype/device validation or execution
                fails.

        Note:
            The native NPU Route supports first-order differentiation only.
            Differentiating through RouteGrad is unsupported.
        """
        if image_mask is not None and self.bias_vl is None:
            raise ValueError("image_mask requires vision_enabled=True when constructing MegaGate")
        if image_mask is not None and image_mask.shape != hidden_states.shape[:2]:
            raise ValueError("image_mask must have shape [batch, sequence]")
        if self.backend_for(hidden_states).name == "torch":
            return GateOutput(*self._torch_forward(hidden_states, image_mask))

        vision_bias = self.bias_vl
        use_vision_bias = image_mask is not None and vision_bias is not None
        flat_image_mask = None
        if use_vision_bias:
            if not image_mask.is_contiguous():
                raise ValueError("image_mask must be contiguous for native MegaGate")
            flat_image_mask = image_mask.view(-1)
        return GateOutput(*mega_gate(
            hidden_states,
            self.weight,
            self.bias,
            self._plan(hidden_states),
            vision_bias=vision_bias if use_vision_bias else None,
            image_mask=flat_image_mask,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
        ))

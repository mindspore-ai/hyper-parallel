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
"""DeepSeek packed-expert adapter backed by A5 HiFloat8 GMMs."""

from collections.abc import Mapping
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_models.components.model_transform.replacement import module_replacement
from hyper_models.components.training.low_precision.functional.hifloat8_gmm_func import (
    hifloat8_grouped_linear,
)
from hyper_models.components.training.low_precision.ops.npu_hifloat8 import (
    validate_hifloat8_gmm_runtime,
)
from hyper_models.components.training.low_precision.quantizers.hifloat8 import (
    GRADIENT_FORMAT_MAX,
    INPUT_WEIGHT_FORMAT_MAX,
    HiFloat8Quantizer,
)

_EXPERT_PARAMETER_NAMES = ("gate_up_proj", "down_proj")


class HiFloat8GroupedExperts(nn.Module):
    """Preserve packed expert parameters and execute their projections in HiF8."""

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        fqn: str = "",
        input_quantizer: Optional[HiFloat8Quantizer] = None,
        weight_quantizer: Optional[HiFloat8Quantizer] = None,
        grad_output_quantizer: Optional[HiFloat8Quantizer] = None,
    ) -> None:
        """Create an unbound packed-expert module."""

        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim)
        )
        self.act_fn = nn.SiLU()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.fqn = fqn
        self._initialize_quantizers(
            input_quantizer,
            weight_quantizer,
            grad_output_quantizer,
        )

    def _initialize_quantizers(
        self,
        input_quantizer: Optional[HiFloat8Quantizer],
        weight_quantizer: Optional[HiFloat8Quantizer],
        grad_output_quantizer: Optional[HiFloat8Quantizer],
    ) -> None:
        self.input_quantizer = input_quantizer or HiFloat8Quantizer(
            fp8_max=INPUT_WEIGHT_FORMAT_MAX
        )
        self.weight_quantizer = weight_quantizer or HiFloat8Quantizer(
            fp8_max=INPUT_WEIGHT_FORMAT_MAX
        )
        self.grad_output_quantizer = grad_output_quantizer or HiFloat8Quantizer(
            fp8_max=GRADIENT_FORMAT_MAX
        )

    @classmethod
    def from_module(
        cls,
        source: nn.Module,
        *,
        fqn: str,
        input_quantizer: Optional[HiFloat8Quantizer] = None,
        weight_quantizer: Optional[HiFloat8Quantizer] = None,
        grad_output_quantizer: Optional[HiFloat8Quantizer] = None,
    ) -> "HiFloat8GroupedExperts":
        """Create a no-allocation shell retaining source registrations."""

        parameter_names = tuple(source._parameters)  # pylint: disable=protected-access
        if parameter_names != _EXPERT_PARAMETER_NAMES:
            raise TypeError(
                f"{fqn!r} must register packed expert parameters "
                f"{_EXPERT_PARAMETER_NAMES}, got {parameter_names}."
            )
        gate_up_proj = source._parameters["gate_up_proj"]  # pylint: disable=protected-access
        down_proj = source._parameters["down_proj"]  # pylint: disable=protected-access
        if gate_up_proj is None or down_proj is None:
            raise ValueError(f"{fqn!r} packed expert parameters cannot be None.")
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError(
                f"{fqn!r} packed expert weights must be three-dimensional."
            )
        expected_down_shape = (
            gate_up_proj.shape[0],
            gate_up_proj.shape[2],
            gate_up_proj.shape[1] // 2,
        )
        if gate_up_proj.shape[1] % 2 or tuple(down_proj.shape) != expected_down_shape:
            raise ValueError(
                f"{fqn!r} has incompatible gate/up and down expert shapes: "
                f"gate_up={tuple(gate_up_proj.shape)}, "
                f"down={tuple(down_proj.shape)}."
            )

        converted = cls.__new__(cls)
        nn.Module.__init__(converted)
        for name, parameter in source._parameters.items():  # pylint: disable=protected-access
            converted.register_parameter(name, parameter)
        for name, buffer in source._buffers.items():  # pylint: disable=protected-access
            converted.register_buffer(
                name,
                buffer,
                persistent=name not in source._non_persistent_buffers_set,  # pylint: disable=protected-access
            )
        for name, module in source._modules.items():  # pylint: disable=protected-access
            converted.add_module(name, module)

        converted.num_experts = gate_up_proj.shape[0]
        converted.hidden_dim = gate_up_proj.shape[2]
        converted.intermediate_dim = gate_up_proj.shape[1] // 2
        converted.fqn = fqn
        converted._initialize_quantizers(
            input_quantizer,
            weight_quantizer,
            grad_output_quantizer,
        )
        if hasattr(source, "config"):
            converted.config = source.config
        converted.training = source.training
        return converted

    def _grouped_forward(
        self,
        sorted_inputs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run expert-major tokens through gate/up and down HiFloat8 GMMs."""

        gate_up = hifloat8_grouped_linear(
            sorted_inputs,
            self.gate_up_proj,
            tokens_per_expert,
            self.input_quantizer,
            self.weight_quantizer,
            self.grad_output_quantizer,
            group_list_type=1,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
        return hifloat8_grouped_linear(
            intermediate,
            self.down_proj,
            tokens_per_expert,
            self.input_quantizer,
            self.weight_quantizer,
            self.grad_output_quantizer,
            group_list_type=1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sort routes, execute packed experts, and restore token order."""

        if hidden_states.ndim != 2:
            raise ValueError(
                "Packed HiFloat8 experts require two-dimensional hidden_states, "
                f"got shape {tuple(hidden_states.shape)}."
            )
        if top_k_index.shape != top_k_weights.shape or top_k_index.ndim != 2:
            raise ValueError(
                "top_k_index and top_k_weights must have the same 2D shape."
            )
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "The routed token count must match hidden_states: "
                f"routes={top_k_index.shape[0]}, "
                f"tokens={hidden_states.shape[0]}."
            )

        token_count = hidden_states.shape[0]
        top_k = top_k_index.shape[1]
        source_token_indices = torch.arange(
            token_count,
            device=hidden_states.device,
        ).unsqueeze(1).expand(-1, top_k).reshape(-1)
        flattened_expert_indices = top_k_index.reshape(-1)
        expert_order = flattened_expert_indices.argsort()
        sorted_inputs = hidden_states[source_token_indices[expert_order]]
        tokens_per_expert = torch.bincount(
            flattened_expert_indices,
            minlength=self.num_experts,
        )
        sorted_outputs = self._grouped_forward(sorted_inputs, tokens_per_expert)
        sorted_weights = top_k_weights.reshape(-1)[expert_order]
        sorted_outputs = (
            sorted_outputs * sorted_weights.unsqueeze(-1)
        ).to(hidden_states.dtype)

        inverse_order = torch.empty_like(expert_order)
        inverse_order[expert_order] = torch.arange(
            expert_order.shape[0],
            device=expert_order.device,
        )
        return sorted_outputs[inverse_order].view(
            token_count,
            top_k,
            self.hidden_dim,
        ).sum(dim=1)


@module_replacement
def replace_hifloat8_grouped_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> HiFloat8GroupedExperts:
    """Replace one EP=1 packed expert container with HiFloat8 GMMs."""

    active_model_parallel_axes = [
        axis.upper()
        for axis in ("tp", "cp", "ep", "pp")
        if context.get(axis)
    ]
    if active_model_parallel_axes:
        raise NotImplementedError(
            "HiFloat8 grouped experts currently require TP=CP=EP=PP=1; "
            f"active axes: {active_model_parallel_axes}."
        )
    validate_hifloat8_gmm_runtime()
    return HiFloat8GroupedExperts.from_module(module, fqn=module_fqn)


__all__ = ["HiFloat8GroupedExperts", "replace_hifloat8_grouped_experts"]

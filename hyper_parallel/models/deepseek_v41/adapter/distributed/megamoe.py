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
"""Declarative MegaMoe expert replacement for DeepSeek-V4.1 training."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, TYPE_CHECKING

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.activations import ACT2FN

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.models.runtime_resources import ModelRuntimeModule

if TYPE_CHECKING:
    from hyper_parallel.core.multicore import MegaMoeExperts


def _swiglu_limit(value: float) -> float:
    """Require the positive clamp used by the source DeepSeek experts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("DSV4.1 swiglu_limit must be finite and positive")
    return float(value)


@module_replacement
class DeepseekV41MegaMoeExperts(nn.Module, ModelRuntimeModule):
    """Own native-layout parameters while retaining the HF checkpoint names."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None, local_num_tokens: int = 4096,
                 expert_capacity_factor: float | None = None, initializer_range: float = 0.02) -> None:
        """Convert a full HF holder before EP partitioning and FSDP construction.

        Args:
            module: HF DeepSeek routed experts with a supported SwiGLU limit.
            module_fqn: Checkpoint scope assigned by the replacement executor.
            context: Active parallel axes; binding occurs after structure replacement.
            local_num_tokens: Fixed, 128-aligned expert execution size; shorter inputs are padded.
            expert_capacity_factor: Receive capacity factor; None reserves the worst case.
            initializer_range: Standard deviation used for meta-model random initialization.
        """
        super().__init__()
        del module_fqn
        if any((context or {}).get(axis, False) for axis in ("tp", "cp", "pp")):
            raise ValueError("DSV4.1 MegaMoe requires TP=CP=PP=1")
        if isinstance(local_num_tokens, bool) or not isinstance(local_num_tokens, int) or (
                local_num_tokens <= 0 or local_num_tokens % 128):
            raise ValueError("local_num_tokens must be a positive multiple of 128")
        self.local_num_tokens = local_num_tokens
        self.expert_capacity_factor = expert_capacity_factor
        source_limit = _swiglu_limit(module.limit)
        if not isinstance(module.act_fn, (nn.SiLU, type(ACT2FN["silu"]))):
            raise ValueError("MegaMoe Trainer replacement requires SiLU")
        self.global_experts = self.num_experts = module.num_experts
        self.hidden_size = module.hidden_dim
        self.intermediate_size = module.intermediate_dim
        self.is_transposed = True
        self.initializer_range = initializer_range
        self.swiglu_limit = source_limit
        self.gate_up_proj = nn.Parameter(module.gate_up_proj.detach().transpose(1, 2).contiguous(),
                                         requires_grad=module.gate_up_proj.requires_grad)
        self.down_proj = nn.Parameter(module.down_proj.detach().transpose(1, 2).contiguous(),
                                      requires_grad=module.down_proj.requires_grad)
        self._executor: MegaMoeExperts | None = None
        self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize native-layout weights after meta materialization."""
        nn.init.normal_(self.gate_up_proj, std=self.initializer_range)
        nn.init.normal_(self.down_proj, std=self.initializer_range)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe the reversible HF-to-native checkpoint layout conversion."""
        return [WeightConverter(source_patterns=name, target_patterns=name,
                                operations=[Transpose(dim0=-2, dim1=-1)])
                for name in ("gate_up_proj", "down_proj")]

    def configure(self, ep_group: Any, ep_size: int, top_k: int) -> None:
        """Bind the EP mesh and routing width after model replacement.

        Args:
            ep_group: Whole-world EP group in global rank order.
            ep_size: Number of EP ranks.
            top_k: Number of selected experts per token.
        """
        if self._executor is not None:
            raise RuntimeError("Cannot reconfigure an active MegaMoe expert executor")
        # Multicore is optional for the default native-EP replacement provider.
        from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=C0415

        self._executor = MegaMoeExperts(
            local_num_tokens=self.local_num_tokens, hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size, num_experts=self.global_experts,
            top_k=top_k, ep_group=ep_group, ep_size=ep_size,
            expert_capacity_factor=self.expert_capacity_factor, swiglu_limit=self.swiglu_limit,
            create_parameters=False,
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Execute with the current parameters supplied by FSDP unshard hooks.

        Args:
            hidden_states: Token states before any EP dispatch.
            top_k_index: Global expert IDs selected by the unchanged router.
            top_k_weights: Differentiable router weights, including source scaling.

        Returns:
            Routed output with the same shape as hidden_states.
        """
        if self._executor is None:
            raise RuntimeError("Configure MegaMoe EP execution before forward")
        weights = tuple(parameter.to_local() if isinstance(parameter, DTensor) else parameter
                        for parameter in (self.gate_up_proj, self.down_proj))
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_flat.shape[0]
        top_k = self._executor.top_k
        if top_k_index.shape != (num_tokens, top_k) or top_k_weights.shape != (num_tokens, top_k):
            raise ValueError("MegaMoe routing tensors must match the unpadded token count and top_k")
        padding = self._executor.local_num_tokens - num_tokens
        if padding < 0:
            raise ValueError("Packed token count exceeds configured MegaMoe capacity; increase the packing bound")
        if padding:
            # Balanced synthetic routes avoid overloading expert zero; zero weights remove their gradients.
            padding_ids = torch.arange(padding * top_k, device=top_k_index.device, dtype=top_k_index.dtype)
            padding_ids = padding_ids.remainder(self.global_experts).reshape(padding, top_k)
            hidden_flat = torch.cat((hidden_flat, hidden_flat.new_zeros((padding, self.hidden_size))))
            top_k_index = torch.cat((top_k_index, padding_ids))
            top_k_weights = torch.cat((top_k_weights, top_k_weights.new_zeros((padding, top_k))))
        output = self._executor(hidden_flat, top_k_index, top_k_weights, expert_weights=weights)
        return output[:num_tokens].reshape(original_shape)

    @staticmethod
    def prepare_runtime_group(experts: list[DeepseekV41MegaMoeExperts]) -> None:
        """Share transient storage across configured, serial decoder layers.

        Args:
            experts: Experts with compatible shapes and the same EP process group.
        """
        if not experts:
            return
        if any(module._executor is None for module in experts):
            raise RuntimeError("Configure every expert before sharing execution resources")
        # Import only when explicitly sharing configured MegaMoe executors.
        from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=C0415

        MegaMoeExperts.share_execution_resources(module._executor for module in experts)

    def close_runtime(self) -> None:
        """Release native resources after the final backward has completed."""
        if self._executor is not None:
            self._executor.close()
            self._executor = None


__all__ = ["DeepseekV41MegaMoeExperts"]

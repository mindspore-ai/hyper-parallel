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
"""Checkpoint-convertible DeepSeek-V4.1 MegaMoe experts for Trainer EP/FSDP."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.activations import ACT2FN

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.deepseek_v41.adapter.expert_parallel import _router
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class DeepseekV41TrainingExperts(nn.Module):
    """Own native-layout parameters while retaining the HF checkpoint names."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None, initializer_range: float = 0.02) -> None:
        """Convert a full HF holder before EP partitioning and FSDP construction.

        Args:
            module: Unclamped HF DeepSeek routed experts.
            module_fqn: Checkpoint scope assigned by the replacement executor.
            context: Replacement context; model-parallel binding occurs later.
            initializer_range: Standard deviation used for meta-model random initialization.
        """
        super().__init__()
        del module_fqn, context
        if module.limit != 0:
            raise ValueError("MegaMoe Trainer replacement requires swiglu_limit=0")
        if not isinstance(module.act_fn, (nn.SiLU, type(ACT2FN["silu"]))):
            raise ValueError("MegaMoe Trainer replacement requires SiLU")
        self.global_experts = self.num_experts = module.num_experts
        self.hidden_size = module.hidden_dim
        self.intermediate_size = module.intermediate_dim
        self.is_transposed = True
        self.initializer_range = initializer_range
        self.gate_up_proj = nn.Parameter(module.gate_up_proj.detach().transpose(1, 2).contiguous(),
                                         requires_grad=module.gate_up_proj.requires_grad)
        self.down_proj = nn.Parameter(module.down_proj.detach().transpose(1, 2).contiguous(),
                                      requires_grad=module.down_proj.requires_grad)
        self._kernel = None
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

    def configure(self, ep_group: Any, ep_size: int, local_num_tokens: int,
                  dispatch_mode: str, top_k: int,
                  expert_capacity_factor: float | None = None) -> None:
        """Bind the actual EP group before any native resources are acquired.

        Args:
            ep_group: Explicit expert process group, or None for EP1.
            ep_size: Size of that group.
            local_num_tokens: Fixed number of tokens at this MoE boundary.
            dispatch_mode: Native push or pull transport.
            top_k: Number of global experts selected per token.
            expert_capacity_factor: Optional bounded receive-capacity multiplier.
        """
        if self._kernel is not None:
            raise RuntimeError("Cannot reconfigure an active MegaMoe expert executor")
        if ep_size <= 0 or self.global_experts % ep_size:
            raise ValueError("Expert count must be divisible by EP size")
        if dispatch_mode not in ("push", "pull"):
            raise ValueError("dispatch_mode must be push or pull")
        # Register every layer before the first forward sizes the fixed SHMEM heap.
        # Native allocation remains lazy and no FSDP parameter is retained here.
        self._kernel = MegaMoeExperts(
            local_num_tokens=local_num_tokens, hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size, num_experts=self.global_experts,
            top_k=top_k, ep_group=ep_group, ep_size=ep_size,
            dispatch_mode=dispatch_mode, create_parameters=False,
            expert_capacity_factor=expert_capacity_factor,
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
        if self._kernel is None:
            raise RuntimeError("Configure MegaMoe EP execution before forward")
        weights = tuple(parameter.to_local() if isinstance(parameter, DTensor) else parameter
                        for parameter in (self.gate_up_proj, self.down_proj))
        return self._kernel(hidden_states, top_k_index, top_k_weights, expert_weights=weights)

    @staticmethod
    def share_execution_resources(experts: list[DeepseekV41TrainingExperts]) -> None:
        """Share transient storage across configured, serial decoder layers.

        Args:
            experts: Experts with compatible shapes and the same EP process group.
        """
        if not experts:
            return
        if any(module._kernel is None for module in experts):
            raise RuntimeError("Configure every expert before sharing execution resources")
        MegaMoeExperts.share_execution_resources(module._kernel for module in experts)

    def close(self) -> None:
        """Release native resources after the final backward has completed."""
        if self._kernel is not None:
            self._kernel.close()
            self._kernel = None


@local_compute
def deepseek_v41_megamoe_compute_fn(
        *, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
        local_num_tokens: int = 128, dispatch_mode: str = "push",
        expert_capacity_factor: float | None = None,
) -> Callable:
    """Replace the entire routed branch while preserving its nested FSDP call.

    Args:
        module: DSV4.1 MoE containing DeepseekV41TrainingExperts.
        mesh: Dense mesh context.
        tp_mesh: Tensor-parallel mesh; only degree one is currently supported.
        cp_mesh: Context-parallel mesh; only degree one is currently supported.
        ep_mesh: Derived expert mesh, with group-local ownership.
        local_num_tokens: Fixed local token count after boundary transformations.
        dispatch_mode: Native push or pull transport.
        expert_capacity_factor: Optional bounded receive-capacity multiplier.

    Returns:
        A text/multimodal-compatible routed-plus-shared compute function.
    """
    del mesh
    if any(axis is not None and axis.size() > 1 for axis in (tp_mesh, cp_mesh)):
        raise ValueError("DSV4.1 MegaMoe Trainer currently requires TP=CP=1")
    if module.is_hash or module.shared_experts.limit != 0:
        raise ValueError("DSV4.1 MegaMoe requires learned routing and unclamped shared experts")
    if not isinstance(module.experts, DeepseekV41TrainingExperts):
        raise TypeError("Apply DeepseekV41TrainingExperts replacement before the MegaMoe EP factory")
    group = None if ep_mesh is None else ep_mesh.get_group("ep")
    size = 1 if ep_mesh is None else ep_mesh["ep"].size()
    module.experts.configure(
        group, size, local_num_tokens, dispatch_mode, module.gate.top_k,
        expert_capacity_factor,
    )

    if "image_mask" not in inspect.signature(module.forward).parameters:
        def text_compute_fn(module: Any, hidden_states: torch.Tensor,
                            input_ids: torch.Tensor | None = None) -> torch.Tensor:
            """Preserve the original text-only signature.

            Args:
                module: Configured DSV4.1 routed block.
                hidden_states: Local un-dispatched token states.
                input_ids: Unused token IDs for the learned router.
            """
            del input_ids
            indices, weights = _router(module, hidden_states)
            return module.experts(hidden_states, indices, weights) + module.shared_experts(hidden_states)

        return text_compute_fn

    def compute_fn(module: Any, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
                   image_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run global routing once, then call experts through the FSDP boundary.

        Args:
            module: Configured DSV4.1 routed block.
            hidden_states: Local un-dispatched token states.
            input_ids: Unused token IDs for the learned router.
            image_mask: Optional modality mask for the unchanged router.
        """
        del input_ids
        indices, weights = _router(module, hidden_states, image_mask)
        return module.experts(hidden_states, indices, weights) + module.shared_experts(hidden_states)

    return compute_fn

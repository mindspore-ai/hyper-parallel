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
"""Model-facing Torch MegaMoe expert module."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem

from ..module import MulticoreModule
from .function import execute_mega_moe_with_permutation
from .plan import build_mega_moe_plan
from .route import prepare_topk_route, restore_topk_output
from .spec import _COMMUNICATION_SPLIT, _balanced_communication_split, bind_mega_moe_spec
from .workspace import MegaMoeWorkspace, configure_symmetric_heap

__all__ = ["MegaMoeExperts"]


def _create_mega_moe_parameters(
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Create independently initialized local expert parameters."""
    gate_up_weight = torch.nn.Parameter(
        torch.empty((local_experts, hidden_size, intermediate_size * 2))
    )
    down_weight = torch.nn.Parameter(
        torch.empty((local_experts, intermediate_size, hidden_size))
    )
    for expert_index in range(local_experts):
        torch.nn.init.xavier_uniform_(gate_up_weight[expert_index])
        torch.nn.init.xavier_uniform_(down_weight[expert_index])
    return gate_up_weight, down_weight


def _validate_resource_layout(specifications: tuple[Any, ...], tensor: torch.Tensor, spec: Any) -> None:
    """Agree on every symmetric allocation before initializing the heap."""
    if spec.ep_size == 1:
        return
    layout = (
        str(tensor.dtype),
        os.getenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE"),
        tuple(tuple(sorted((key, value) for key, value in item.items() if key != "ep_group"))
              for item in specifications),
    )
    layouts = [None] * spec.ep_size
    dist.all_gather_object(layouts, layout, group=spec.ep_group)
    if any(peer_layout != layout for peer_layout in layouts):
        raise ValueError("MegaMoe static shapes, heap configuration and allocation order must match on all EP ranks.")


def _select_plan(resources: Any, route: Any) -> Any:
    """Keep coarse GET tasks only while the global receive load stays nearly uniform."""
    balanced = resources.balanced_plan
    if balanced is None:
        return resources.plan
    maximum_received = route.maximum_received_slots
    if 64 * maximum_received < 65 * resources.spec.routed_slots:
        return balanced
    moderate = resources.moderate_plan
    if maximum_received <= 2 * resources.spec.routed_slots:
        return moderate
    return resources.plan


def _alternative_plans(spec: Any, device: Any, plan: Any) -> tuple[Any, Any]:
    """Keep the pull scheduling policy independent of push resource allocation."""
    balanced_split = _balanced_communication_split(spec.local_num_tokens)
    if spec.dispatch_mode != "pull" or balanced_split <= _COMMUNICATION_SPLIT:
        return None, None
    balanced = build_mega_moe_plan(replace(spec, dispatch_split=balanced_split, combine_split=balanced_split), device)
    moderate_split = math.gcd(spec.local_num_tokens, 512)
    moderate = (build_mega_moe_plan(replace(spec, combine_split=moderate_split), device)
                if moderate_split > _COMMUNICATION_SPLIT else plan)
    return balanced, moderate


class _MegaMoeExecutionResources:
    """Own one shape-bound plan and workspace in the shared SHMEM lifecycle."""

    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        self.spec = bind_mega_moe_spec(specification, tensor)
        _validate_resource_layout(active_specifications, tensor, self.spec)
        configure_symmetric_heap(active_specifications, tensor)
        shmem.acquire(self.spec.ep_group)
        try:
            self.plan = build_mega_moe_plan(self.spec, tensor.device)
            self.balanced_plan, self.moderate_plan = _alternative_plans(self.spec, tensor.device, self.plan)
            self.workspace = MegaMoeWorkspace(shared=shared)
        except Exception:
            shmem.release()
            raise
        self._closed = False

    def close(self) -> None:
        """Release the workspace and leave the shared SHMEM lifecycle."""
        if self._closed:
            return
        self.workspace.close()
        shmem.release()
        self._closed = True


class MegaMoeExperts(MulticoreModule):
    """Execute Router-selected local experts with the Torch MegaMoe kernel.

    Serial model layers can call :meth:`share_execution_resources` before
    their first forward to share one lossless-capacity workspace while keeping
    independent parameters and optimizer state.
    """

    def __init__(
        self,
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float | None = None,
        ep_size: int = 1,
        ep_group: Any | None = None,
        dispatch_mode: str = "push",
    ) -> None:
        """Initialize local expert parameters and a lazy execution owner.

        Args:
            local_num_tokens: Static token rows supplied to this rank.
            hidden_size: Input and output hidden dimension.
            intermediate_size: SwiGLU intermediate dimension per expert.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            expert_capacity_factor: Optional bounded receive-capacity multiplier.
                ``None`` reserves the maximum lossless capacity. A finite value
                of at least 1.0 reserves that multiple of the local routed rows
                and raises a clear error if a route exceeds it.
            dispatch_mode: Dispatch transport, either "push" (default) or "pull".
                Construct separate modules to switch modes; sharing requires equal modes.
            ep_size: Expert-parallel degree. The current SHMEM path requires it
                to cover the complete Torch distributed world.
            ep_group: Torch expert-parallel process group with the same rank
                ordering as the complete distributed world.
        """
        if dispatch_mode not in ("push", "pull"):
            raise ValueError("dispatch_mode must be push or pull")
        self._validate_topology(
            local_num_tokens=local_num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor,
            ep_size=ep_size,
        )
        if expert_capacity_factor is not None:
            expert_capacity_factor = float(expert_capacity_factor)
        specification = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "expert_capacity_factor": expert_capacity_factor,
            "ep_size": ep_size,
            "ep_group": ep_group,
            "dispatch_mode": dispatch_mode,
        }
        compatibility_key = (
            local_num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            expert_capacity_factor,
            ep_size,
            id(ep_group),
            dispatch_mode,
        )
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=compatibility_key,
            resource_scope_key=("mega_moe", id(ep_group)),
        )
        self.local_num_tokens = local_num_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.dispatch_mode = dispatch_mode
        self.ep_size = ep_size
        self.local_experts = num_experts // ep_size
        self._ep_group = ep_group
        self.gate_up_weight, self.down_weight = _create_mega_moe_parameters(
            self.local_experts,
            hidden_size,
            intermediate_size,
        )

    @staticmethod
    def _validate_topology(
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float | None,
        ep_size: int,
    ) -> None:
        """Validate static shape and topology values before allocation."""
        values = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "ep_size": ep_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})."
            )
        if num_experts % ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
            )
        if local_num_tokens % _COMMUNICATION_SPLIT:
            raise ValueError(
                "local_num_tokens must be divisible by the fixed communication "
                f"split {_COMMUNICATION_SPLIT}, got {local_num_tokens}."
            )
        if expert_capacity_factor is None:
            return
        valid_factor_type = isinstance(
            expert_capacity_factor,
            (int, float),
        ) and not isinstance(expert_capacity_factor, bool)
        try:
            valid_factor_value = valid_factor_type and math.isfinite(
                expert_capacity_factor
            )
        except OverflowError:
            valid_factor_value = False
        if not valid_factor_value or expert_capacity_factor < 1.0:
            raise ValueError(
                "expert_capacity_factor must be None or a finite number at least 1.0, "
                f"got {expert_capacity_factor!r}."
            )

    def _validate_tensors(self, hidden_states: torch.Tensor) -> None:
        """Validate activation and parameter metadata before acquiring resources."""
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        weight1 = self.gate_up_weight
        weight2 = self.down_weight
        expected_weight1 = (
            self.local_experts,
            self.hidden_size,
            self.intermediate_size * 2,
        )
        expected_weight2 = (
            self.local_experts,
            self.intermediate_size,
            self.hidden_size,
        )
        if tuple(weight1.shape) != expected_weight1:
            raise ValueError(
                f"gate_up_weight must have shape {expected_weight1}, got {tuple(weight1.shape)}."
            )
        if tuple(weight2.shape) != expected_weight2:
            raise ValueError(
                f"down_weight must have shape {expected_weight2}, got {tuple(weight2.shape)}."
            )
        if (
            weight1.device != hidden_states.device
            or weight2.device != hidden_states.device
        ):
            raise ValueError(
                "expert weights and hidden states must be on the same NPU."
            )
        if weight1.dtype != hidden_states.dtype or weight2.dtype != hidden_states.dtype:
            raise TypeError("expert weights and hidden states must all use BF16.")

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor | None,
    ) -> torch.Tensor:
        """Validate input shapes and return flattened local token states."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have at least two dimensions and configured "
                f"hidden size {self.hidden_size}, got {tuple(hidden_states.shape)}."
            )
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        if hidden_flat.shape[0] != self.local_num_tokens:
            raise ValueError(
                f"MegaMoeExperts expects {self.local_num_tokens} local tokens, "
                f"got {hidden_flat.shape[0]}."
            )
        route_shape = (self.local_num_tokens, self.top_k)
        if tuple(topk_ids.shape) != route_shape:
            raise ValueError(
                f"topk_ids must have shape {route_shape}, got {tuple(topk_ids.shape)}."
            )
        if tuple(topk_weights.shape) != route_shape:
            raise ValueError(
                f"topk_weights must have shape {route_shape}, "
                f"got {tuple(topk_weights.shape)}."
            )
        if tokens_per_expert is not None and tuple(tokens_per_expert.shape) != (
            self.num_experts,
        ):
            raise ValueError(
                f"tokens_per_expert must have shape ({self.num_experts},), "
                f"got {tuple(tokens_per_expert.shape)}."
            )
        self._validate_tensors(hidden_flat)
        return hidden_flat

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[local_num_tokens, top_k]``.
            topk_weights: Router weights with the same shape as ``topk_ids``.
            tokens_per_expert: Optional trusted exact global-expert histogram.
                Supplying Router-produced counts skips histogram recomputation.

        Returns:
            Tensor matching the shape and dtype of ``hidden_states``.

        Note:
            ``tokens_per_expert`` must describe the current ``topk_ids`` exactly.
            The steady-state path validates metadata but deliberately does not
            rebuild and compare the histogram.
        """
        hidden_flat = self._validate_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
        )
        resources = self._get_execution_resources(hidden_flat)
        # The expert autograd bridge consumes permutation gradients before the
        # workspace can be reused, so route preparation needs no separate node.
        with torch.no_grad():
            route = prepare_topk_route(
                hidden_flat,
                topk_ids,
                topk_weights,
                resources.spec,
                tokens_per_expert,
                workspace=resources.workspace,
            )
        expert_output = execute_mega_moe_with_permutation(
            hidden_flat,
            topk_ids,
            self.gate_up_weight,
            self.down_weight,
            route,
            _select_plan(resources, route),
            resources.workspace,
            topk_weights=topk_weights if self.dispatch_mode == "pull" else None,
        )
        if self.dispatch_mode == "pull":
            return expert_output.reshape_as(hidden_states)
        output = restore_topk_output(
            expert_output,
            route.unpermute_mapping,
            topk_weights,
        )
        return output.reshape(hidden_states.shape)

    def _create_execution_resources(
        self,
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
    ) -> _MegaMoeExecutionResources:
        """Create the plan and workspace owned by this resource group."""
        return _MegaMoeExecutionResources(
            self._resource_group.specification,
            tensor,
            shared=shared,
            active_specifications=active_specifications,
        )

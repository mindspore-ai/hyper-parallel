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
"""Differentiable autograd function for the low-level MegaMoe ABI."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.core.multicore.torch import ops as multicore_ops

from .plan import MegaMoePlan
from .route import RouteMetadata
from .workspace import MegaMoeWorkspace



def _workspace_tensor(tensor: Any | None, name: str) -> Any:
    """Return an initialized workspace tensor or raise an invariant error."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe workspace {name} is not initialized.")
    return tensor


def _allocate_forward_intermediates(
    spec: Any,
    capacity: int,
    routed_tokens: Any,
    dispatch: Any,
) -> tuple[Any, Any, Any]:
    """Allocate forward tensors outside the autograd bridge."""
    up_proj = torch.zeros(
        (capacity, spec.intermediate_size * 2),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    activation = torch.zeros(
        (capacity, spec.intermediate_size),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    return up_proj, activation, torch.zeros_like(dispatch)


def _allocate_backward_intermediates(
    spec: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Allocate zero-safe backward tensors outside the autograd bridge."""
    grad_weight2 = torch.zeros_like(weight2)
    act_grad = torch.zeros(
        (capacity, spec.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = torch.zeros(
        (capacity, spec.intermediate_size * 2),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    gate_dx = torch.zeros(
        (capacity, spec.hidden_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    # Empty experts must retain exact zero gradients on the baseline kernel.
    grad_weight1 = torch.zeros_like(weight1)
    return grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx


def _save_forward_state(
    ctx: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    dispatch: Any,
    up_proj: Any,
    activation: Any,
    weight1: Any,
    weight2: Any,
    metadata: RouteMetadata,
) -> None:
    """Save owned forward tensors and route state for backward."""
    ctx.plan = plan
    ctx.workspace = workspace
    ctx.save_for_backward(
        dispatch.clone(),
        up_proj,
        activation,
        weight1,
        weight2,
        metadata.group_list,
        metadata.dispatch_src_off,
        metadata.dispatch_target_off,
        metadata.dispatch_size,
        metadata.combine_src_off,
        metadata.combine_target_off,
        metadata.combine_size,
    )


# Torch declares variadic autograd hooks; concrete functions use operator-specific signatures.
class _MegaMoeFunction(  # pylint: disable=abstract-method,arguments-differ
    torch.autograd.Function
):
    """Autograd bridge for the in-place forward/backward custom ops."""

    @staticmethod
    def forward(
        ctx: Any,
        routed_tokens: Any,
        weight1: Any,
        weight2: Any,
        route: RouteMetadata,
        plan: MegaMoePlan,
        workspace: MegaMoeWorkspace,
    ) -> Any:
        """Launch the legacy forward op and save owned backward inputs.

        Args:
            ctx: Autograd context for saved backward state.
            routed_tokens: Expert-major token rows.
            weight1: Local gate and up-projection weights.
            weight2: Local down-projection weights.
            route: Route offsets, sizes and group metadata.
            plan: Shape-specific forward and backward descriptors.
            workspace: Reusable directional execution buffers.

        Returns:
            Owned expert-major output rows.
        """
        spec = plan.spec
        metadata = route
        workspace.ensure(spec, routed_tokens.dtype, routed_tokens.device)
        workspace.claim()
        try:
            dispatch = _workspace_tensor(workspace.expert_buffer, "expert_buffer")
            combine = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
            events = _workspace_tensor(
                workspace.forward_event_counters,
                "forward_event_counters",
            )
            gmm_workspace = _workspace_tensor(
                workspace.gmm_workspace, "gmm_workspace"
            )
            dispatch.zero_()
            combine.zero_()
            events.zero_()
            capacity = workspace.expert_capacity
            up_proj, activation, down_proj = _allocate_forward_intermediates(
                spec,
                capacity,
                routed_tokens,
                dispatch,
            )
            workspace.symmetric_memory.barrier()
            multicore_ops.mega_moe(
                dispatch,
                metadata.dispatch_target_off * spec.hidden_size,
                routed_tokens.contiguous(),
                metadata.dispatch_src_off * spec.hidden_size,
                metadata.dispatch_size * spec.hidden_size,
                weight1.contiguous(),
                metadata.group_list,
                up_proj,
                activation,
                weight2.contiguous(),
                metadata.group_list,
                down_proj,
                combine,
                metadata.combine_target_off * spec.hidden_size,
                metadata.combine_src_off * spec.hidden_size,
                metadata.combine_size * spec.hidden_size,
                gmm_workspace,
                plan.up_proj_tiling,
                plan.swiglu_tiling,
                plan.down_proj_tiling,
                plan.fwd_runtime_config,
                events,
                spec.rank_id,
                spec.ep_size,
                spec.num_experts,
                spec.hidden_size,
                spec.local_num_tokens,
            )
            output = combine.clone()
            _save_forward_state(
                ctx,
                plan,
                workspace,
                dispatch,
                up_proj,
                activation,
                weight1,
                weight2,
                metadata,
            )
            return output
        finally:
            workspace.release()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        """Launch the legacy backward op into zero-safe gradient buffers.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_output: Expert-major output gradient rows.

        Returns:
            Gradients aligned with the differentiable forward inputs.
        """
        plan = ctx.plan
        workspace = ctx.workspace
        spec = plan.spec
        (
            saved_dispatch,
            up_proj,
            activation,
            weight1,
            weight2,
            group_list,
            dispatch_src_off,
            dispatch_target_off,
            dispatch_size,
            combine_src_off,
            combine_target_off,
            combine_size,
        ) = ctx.saved_tensors
        workspace.claim()
        try:
            dispatch = _workspace_tensor(workspace.expert_buffer, "expert_buffer")
            grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
            events = _workspace_tensor(
                workspace.backward_event_counters,
                "backward_event_counters",
            )
            gmm_workspace = _workspace_tensor(
                workspace.gmm_workspace, "gmm_workspace"
            )
            swiglu_workspace = _workspace_tensor(
                workspace.swiglu_grad_workspace,
                "swiglu_grad_workspace",
            )
            dispatch.zero_()
            grad_x.zero_()
            events.zero_()
            capacity = workspace.expert_capacity
            (
                grad_weight1,
                grad_weight2,
                act_grad,
                swiglu_grad,
                gate_dx,
            ) = _allocate_backward_intermediates(
                spec,
                capacity,
                grad_output,
                weight1,
                weight2,
            )
            workspace.symmetric_memory.barrier()
            multicore_ops.mega_moe_grad(
                dispatch,
                dispatch_target_off * spec.hidden_size,
                grad_output.contiguous(),
                dispatch_src_off * spec.hidden_size,
                dispatch_size * spec.hidden_size,
                activation,
                grad_weight2,
                weight2,
                act_grad,
                up_proj,
                swiglu_grad,
                weight1,
                gate_dx,
                grad_x,
                combine_target_off * spec.hidden_size,
                combine_src_off * spec.hidden_size,
                combine_size * spec.hidden_size,
                saved_dispatch,
                grad_weight1,
                group_list,
                plan.act_grad_tiling,
                plan.gate_grad_tiling,
                plan.w1_grad_tiling,
                plan.w2_grad_tiling,
                plan.swiglu_grad_tiling,
                gmm_workspace,
                swiglu_workspace,
                plan.bwd_runtime_config,
                events,
                spec.rank_id,
                spec.ep_size,
                spec.num_experts,
                spec.hidden_size,
                spec.local_num_tokens,
            )
            return grad_x.clone(), grad_weight1, grad_weight2, None, None, None
        finally:
            workspace.release()



def execute_mega_moe(
    routed_tokens: Any,
    weight1: Any,
    weight2: Any,
    route: RouteMetadata,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
) -> Any:
    """Execute expert-major rows through the differentiable legacy kernels.

    Args:
        routed_tokens: Expert-major token rows.
        weight1: Local gate and up-projection weights.
        weight2: Local down-projection weights.
        route: Route offsets, sizes and group metadata.
        plan: Shape-specific forward and backward descriptors.
        workspace: Reusable directional execution buffers.

    Returns:
        Expert-major output rows with independent storage.
    """
    return _MegaMoeFunction.apply(routed_tokens, weight1, weight2, route, plan, workspace)

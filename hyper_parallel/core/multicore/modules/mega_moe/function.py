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

from hyper_parallel.core.multicore.profiler.profiler import prepare_mega_kernel_call
from hyper_parallel.core.multicore.torch import ops as multicore_ops

from .plan import MegaMoePlan
from .route import PreparedTopKRoute, RouteMetadata
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
    """Allocate outputs whose received rows are overwritten before use."""
    # GMM and SwiGLU cover the received prefix; an empty rank reads no rows.
    up_proj = torch.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    activation = torch.empty(
        (capacity, spec.intermediate_size),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    return up_proj, activation, torch.empty_like(dispatch[:capacity])


def _allocate_backward_intermediates(
    spec: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
    reusable_dispatch: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Allocate overwritten activations and zero-safe expert gradients."""
    grad_weight2 = torch.zeros_like(weight2)
    act_grad = torch.empty(
        (capacity, spec.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = torch.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    gate_dx = reusable_dispatch if reusable_dispatch is not None else torch.empty(
        (capacity, spec.hidden_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    # Empty experts must retain exact zero gradients on the baseline kernel.
    grad_weight1 = torch.zeros_like(weight1)
    return grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx


def _restore_input_gradient(ctx: Any, grad_x: Any, permutation_inputs: tuple[Any, ...]) -> Any:
    """Return an owned input gradient while the workspace lease is held."""
    if not ctx.needs_input_grad[0]:
        return None
    if not ctx.has_permutation:
        return grad_x.clone()
    (unpermute_mapping,) = permutation_inputs
    spec = ctx.plan.spec
    # Consume the shared gradient before release records completion.
    # The permutation gradient owns its reduced [T, H] output.
    return multicore_ops.moe_token_permute_grad(
        grad_x, unpermute_mapping, spec.local_num_tokens, spec.top_k
    )


def _save_forward_state(
    ctx: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    saved_dispatch: Any,
    up_proj: Any,
    activation: Any,
    weight1: Any,
    weight2: Any,
    metadata: RouteMetadata,
    permutation_inputs: tuple[Any, ...] = (),
) -> None:
    """Save owned forward tensors and route state for backward."""
    ctx.plan = plan
    ctx.workspace = workspace
    ctx.save_for_backward(
        saved_dispatch,
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
        *permutation_inputs,
    )


def _launch_forward_kernel(
    plan: MegaMoePlan,
    metadata: RouteMetadata,
    routed_tokens: Any,
    weight1: Any,
    weight2: Any,
    dispatch: Any,
    combine: Any,
    intermediates: tuple[Any, Any, Any],
    profile_call: Any,
    gmm_workspace: Any,
) -> None:
    """Launch the internal forward ABI with prepared buffers and metadata."""
    spec = plan.spec
    up_proj, activation, down_proj = intermediates
    multicore_ops.mega_moe_with_profile_buffer(
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
        profile_call.runtime_config,
        profile_call.event_counters,
        profile_call.profile_buffer,
        spec.rank_id,
        spec.ep_size,
        spec.num_experts,
        spec.hidden_size,
        spec.local_num_tokens,
    )


def _launch_backward_kernel(
    plan: MegaMoePlan,
    saved_tensors: tuple[Any, ...],
    grad_output: Any,
    dispatch: Any,
    grad_x: Any,
    gradients: tuple[Any, Any, Any, Any, Any],
    profile_call: Any,
    gmm_workspace: Any,
    swiglu_workspace: Any,
) -> None:
    """Launch the internal backward ABI with prepared buffers and metadata."""
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
    ) = saved_tensors
    grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx = gradients
    multicore_ops.mega_moe_grad_with_profile_buffer(
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
        profile_call.runtime_config,
        profile_call.event_counters,
        profile_call.profile_buffer,
        spec.rank_id,
        spec.ep_size,
        spec.num_experts,
        spec.hidden_size,
        spec.local_num_tokens,
    )


# Torch declares variadic autograd hooks; concrete functions use operator-specific signatures.
class _MegaMoeFunction(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Autograd bridge for the in-place forward/backward custom ops."""

    @staticmethod
    # pylint: disable-next=arguments-differ
    def forward(
        ctx: Any,
        routed_tokens: Any,
        weight1: Any,
        weight2: Any,
        route: RouteMetadata,
        plan: MegaMoePlan,
        workspace: MegaMoeWorkspace,
        permutation: tuple[Any, Any, Any] | None,
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
            permutation: Optional routed rows, expert IDs and inverse mapping
                when the first tensor argument contains original token rows.

        Returns:
            Owned expert-major output rows.
        """
        spec = plan.spec
        metadata = route
        permutation_inputs = ()
        if permutation is not None:
            routed_tokens, _, unpermute_mapping = permutation
            if ctx.needs_input_grad[0]:
                permutation_inputs = (unpermute_mapping,)
        ctx.has_permutation = permutation is not None
        workspace.ensure(spec, routed_tokens.dtype, routed_tokens.device)
        workspace.claim()
        profile_call = None
        try:
            dispatch = _workspace_tensor(workspace.expert_buffer, "expert_buffer")
            combine = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
            events = workspace.prepare_event_counters(forward=True)
            profile_call = prepare_mega_kernel_call(
                plan.fwd_runtime,
                direction="forward",
                fallback_event_counters=events,
            )
            gmm_workspace = _workspace_tensor(
                workspace.gmm_workspace, "gmm_workspace"
            )
            # Dispatch and combine overwrite disjoint route ranges before consumers run.
            capacity = metadata.expert_capacity
            up_proj, activation, down_proj = _allocate_forward_intermediates(
                spec,
                capacity,
                routed_tokens,
                dispatch,
            )
            _launch_forward_kernel(
                plan,
                metadata,
                routed_tokens,
                weight1,
                weight2,
                dispatch,
                combine,
                (up_proj, activation, down_proj),
                profile_call,
                gmm_workspace,
            )
            profile_call.complete()
            output = combine.clone()
            # Combine has consumed down_proj on this stream. Retain its owned
            # storage for backward before the next call reuses SHMEM dispatch.
            down_proj.copy_(dispatch[:capacity])
            _save_forward_state(
                ctx,
                plan,
                workspace,
                down_proj,
                up_proj,
                activation,
                weight1,
                weight2,
                metadata,
                permutation_inputs,
            )
            return output
        finally:
            if profile_call is not None:
                profile_call.cancel()
            workspace.release()

    @staticmethod
    # pylint: disable-next=arguments-differ
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
        saved_tensors = ctx.saved_tensors
        saved_dispatch = saved_tensors[0]
        weight1 = saved_tensors[3]
        weight2 = saved_tensors[4]
        permutation_inputs = saved_tensors[12:]
        kernel_saved_tensors = saved_tensors[:12]
        workspace.claim()
        profile_call = None
        try:
            dispatch = _workspace_tensor(workspace.expert_buffer, "expert_buffer")
            grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
            events = workspace.prepare_event_counters(forward=False)
            profile_call = prepare_mega_kernel_call(
                plan.bwd_runtime,
                direction="backward",
                fallback_event_counters=events,
            )
            gmm_workspace = _workspace_tensor(
                workspace.gmm_workspace, "gmm_workspace"
            )
            swiglu_workspace = _workspace_tensor(
                workspace.swiglu_grad_workspace,
                "swiglu_grad_workspace",
            )
            capacity = saved_dispatch.shape[0]
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
                dispatch[:capacity] if plan.reuse_backward_dispatch else None,
            )
            _launch_backward_kernel(
                plan,
                kernel_saved_tensors,
                grad_output,
                dispatch,
                grad_x,
                (grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx),
                profile_call,
                gmm_workspace,
                swiglu_workspace,
            )
            profile_call.complete()
            # Both kernels use the current stream. Release ordinary scratch
            # before allocating the owned token gradient; SHMEM stays leased.
            del act_grad, swiglu_grad, gate_dx
            grad_input = _restore_input_gradient(ctx, grad_x, permutation_inputs)
            return grad_input, grad_weight1, grad_weight2, None, None, None, None
        finally:
            if profile_call is not None:
                profile_call.cancel()
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
    return _MegaMoeFunction.apply(routed_tokens, weight1, weight2, route, plan, workspace, None)


def execute_mega_moe_with_permutation(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    route: PreparedTopKRoute,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
) -> torch.Tensor:
    """Include input permutation backward within the workspace lease.

    Args:
        hidden_states: Original flattened token rows.
        topk_ids: Expert IDs used to prepare the routed rows.
        weight1: Local gate and up-projection weights.
        weight2: Local down-projection weights.
        route: Rows and metadata prepared without recording autograd operations.
        plan: Shape-specific native descriptors.
        workspace: Reusable communication buffers.

    Returns:
        Owned expert-major output rows, differentiable with respect to the
        original token rows and local expert weights.
    """
    return _MegaMoeFunction.apply(
        hidden_states,
        weight1,
        weight2,
        route.metadata,
        plan,
        workspace,
        (route.routed_tokens, topk_ids, route.unpermute_mapping),
    )

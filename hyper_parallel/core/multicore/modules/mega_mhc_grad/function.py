# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Functional entry point for the BF16 HyperMegaMhcGrad mega kernel."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from hyper_parallel.core.multicore.profiler.profiler import prepare_mega_kernel_call
from hyper_parallel.core.multicore.torch import ops as multicore_ops

from .golden import MegaMhcGradCache, MegaMhcGradOutputs
from .graph import DEFAULT_GRAD_TOKEN_TILE
from .plan import MegaMhcGradPlan, build_mega_mhc_grad_plan


def _resolve_core_counts(device_index: int) -> tuple[int, int]:
    """Read and validate the physical core counts used by the 1:2 mixed launch."""
    limits = torch.npu.get_device_limit(device_index)
    num_cube_cores = limits.get("cube_core_num")
    num_vector_cores = limits.get("vector_core_num")
    if not isinstance(num_cube_cores, int) or num_cube_cores <= 0:
        raise RuntimeError(
            "torch.npu.get_device_limit() returned an invalid cube_core_num "
            f"for NPU {device_index}: {num_cube_cores!r}."
        )
    if not isinstance(num_vector_cores, int) or num_vector_cores <= 0:
        raise RuntimeError(
            "torch.npu.get_device_limit() returned an invalid vector_core_num "
            f"for NPU {device_index}: {num_vector_cores!r}."
        )
    if num_vector_cores != 2 * num_cube_cores:
        raise RuntimeError(
            "HyperMegaMhcGrad requires the KERNEL_TYPE_MIX_AIC_1_2 core ratio, "
            f"but NPU {device_index} reports cube_core_num={num_cube_cores} and "
            f"vector_core_num={num_vector_cores}."
        )
    return num_cube_cores, num_vector_cores


@lru_cache(maxsize=16)
def _get_cached_plan(
    token_count: int,
    hidden_size: int,
    device_index: int,
    num_cube_cores: int,
    num_vector_cores: int,
    token_tile: int,
) -> MegaMhcGradPlan:
    """Reuse immutable RuntimeConfig tensors for recurring static shapes."""
    return build_mega_mhc_grad_plan(
        token_count,
        hidden_size,
        torch.device("npu", device_index),
        num_cube_cores,
        num_vector_cores,
        token_tile,
    )


def _launch_grad_kernel(
    kernel_inputs: tuple[torch.Tensor, ...],
    profile_call: Any,
    hc_eps: float,
) -> tuple[torch.Tensor, ...]:
    """Launch the native backward kernel with profiler-owned runtime buffers.

    Args:
        kernel_inputs: Shape-normalized operator inputs.
        profile_call: Prepared runtime and profiling buffers.
        hc_eps: RMSNorm epsilon.

    Returns:
        Native operator outputs in ACLNN order.
    """
    return multicore_ops.hyper_mega_mhc_grad(
        *kernel_inputs,
        profile_call.runtime_config,
        profile_call.event_counters,
        profile_call.profile_buffer,
        hc_eps=hc_eps,
    )


def _restore_output_shapes(
    outputs: tuple[torch.Tensor, ...],
    leading_shape: tuple[int, ...],
    hidden_size: int,
) -> MegaMhcGradOutputs:
    """Restore public leading dimensions and reorder native outputs.

    Args:
        outputs: Native operator outputs in ACLNN order.
        leading_shape: Original activation leading dimensions.
        hidden_size: Hidden dimension of every token.

    Returns:
        Public gradient tuple in forward-input order.
    """
    (
        grad_residual,
        grad_phi,
        grad_alpha,
        grad_bias,
        grad_previous_output,
        grad_previous_pre,
        grad_previous_post,
        grad_previous_residual,
        grad_norm_weight,
    ) = outputs
    return (
        grad_previous_output.reshape(*leading_shape, hidden_size),
        grad_residual.reshape(*leading_shape, 4, hidden_size),
        grad_previous_pre.reshape(*leading_shape, 4),
        grad_previous_post.reshape(*leading_shape, 4),
        grad_previous_residual.reshape(*leading_shape, 4, 4),
        grad_phi,
        grad_alpha,
        grad_bias,
        grad_norm_weight,
    )


def hyper_mega_mhc_grad(
    grad_new_residual: torch.Tensor,
    grad_next_pre_mix: torch.Tensor,
    grad_next_post_mix: torch.Tensor,
    grad_next_residual_mix: torch.Tensor,
    grad_block_input: torch.Tensor,
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    cache: MegaMhcGradCache,
    *,
    hc_eps: float = 1e-6,
    token_tile: int = DEFAULT_GRAD_TOKEN_TILE,
) -> MegaMhcGradOutputs:
    """Execute the fused Prepare/PhiRms/PrevXPost backward pipeline.

    Args:
        grad_new_residual: Gradient of the updated residual branches.
        grad_next_pre_mix: Gradient of the next pre-mix coefficients.
        grad_next_post_mix: Gradient of the next post-mix coefficients.
        grad_next_residual_mix: Gradient of the next residual-mix coefficients.
        grad_block_input: Gradient of the normalized block input.
        previous_output: Previous block output activation.
        residual: Current residual branches.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weight.
        alpha: Projection scale.
        bias: Projection bias.
        norm_weight: RMSNorm weight.
        cache: Forward intermediates required by the backward kernel.
        hc_eps: RMSNorm epsilon.
        token_tile: Number of tokens assigned to each AIV task.

    Returns:
        Gradients for the five block inputs and four trainable tensors.
    """
    if not residual.is_npu or residual.dtype != torch.bfloat16:
        raise TypeError("hyper_mega_mhc_grad requires BF16 NPU activations.")
    leading_shape, hidden_size = residual.shape[:-2], residual.shape[-1]
    token_count = residual.numel() // (4 * hidden_size)
    device_index = residual.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    num_cube_cores, num_vector_cores = _resolve_core_counts(device_index)
    if token_count < num_vector_cores:
        raise ValueError(
            "hyper_mega_mhc_grad requires at least one token per vector core: "
            f"tokens={token_count}, vector_cores={num_vector_cores}."
        )
    plan = _get_cached_plan(
        token_count,
        hidden_size,
        device_index,
        num_cube_cores,
        num_vector_cores,
        token_tile,
    )
    profile_call = prepare_mega_kernel_call(
        plan.runtime,
        direction="backward",
        fallback_event_counters=plan.event_counters,
    )
    try:
        kernel_inputs = (
            grad_block_input.reshape(1, token_count, hidden_size).contiguous(),
            grad_next_post_mix.reshape(1, token_count, 4).contiguous(),
            grad_next_residual_mix.reshape(1, token_count, 4, 4).contiguous(),
            cache.new_residual.reshape(1, token_count, 4, hidden_size).contiguous(),
            phi.contiguous(),
            alpha.contiguous(),
            bias.contiguous(),
            previous_pre_mix.reshape(1, token_count, 4).contiguous(),
            cache.hc_before_norm.contiguous(),
            cache.inv_rms.contiguous(),
            cache.sum_out.contiguous(),
            cache.norm_out.contiguous(),
            grad_next_pre_mix.reshape(1, token_count, 4).contiguous(),
            cache.mixed_input.reshape(1, token_count, hidden_size).contiguous(),
            cache.rms_rstd.contiguous(),
            norm_weight.contiguous(),
            grad_new_residual.reshape(1, token_count, 4, hidden_size).contiguous(),
            residual.reshape(1, token_count, 4, hidden_size).contiguous(),
            previous_output.reshape(1, token_count, hidden_size).contiguous(),
            previous_post_mix.reshape(1, token_count, 4).contiguous(),
            previous_residual_mix.reshape(1, token_count, 4, 4).contiguous(),
        )
        outputs = _launch_grad_kernel(kernel_inputs, profile_call, hc_eps)
        # The cached counters start at zero. Clear them after the asynchronous
        # launch so stream ordering prepares the same buffer for its next use.
        profile_call.clear_event_counters.zero_()
        profile_call.complete()
    finally:
        profile_call.cancel()
    return _restore_output_shapes(outputs, leading_shape, hidden_size)

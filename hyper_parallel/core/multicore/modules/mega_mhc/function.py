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
"""Functional entry point for the BF16 HyperMegaMhc scheduler kernel."""

from __future__ import annotations

from functools import lru_cache

import torch

from hyper_parallel.core.multicore.profiler.profiler import prepare_mega_kernel_call
from hyper_parallel.core.multicore.torch import ops as multicore_ops

from .golden import MHC_SINKHORN_ITERS, MegaMhcOutputs, _validate_common
from .graph import DEFAULT_TOKEN_TILE
from .plan import MegaMhcPlan, build_mega_mhc_plan


def _resolve_num_cube_cores(device_index: int) -> int:
    """Read the physical AIC count used by the 1:2 mixed launch."""
    limits = torch.npu.get_device_limit(device_index)
    num_cube_cores = limits.get("cube_core_num")
    if not isinstance(num_cube_cores, int) or num_cube_cores <= 0:
        raise RuntimeError(
            "torch.npu.get_device_limit() returned an invalid cube_core_num "
            f"for NPU {device_index}: {num_cube_cores!r}."
        )
    return num_cube_cores


@lru_cache(maxsize=16)
def _get_cached_plan(
    token_count: int,
    hidden_size: int,
    device_index: int,
    num_cube_cores: int,
    token_tile: int,
) -> MegaMhcPlan:
    """Reuse immutable RuntimeConfig tensors for recurring static shapes."""
    return build_mega_mhc_plan(
        token_count,
        hidden_size,
        torch.device("npu", device_index),
        num_cube_cores,
        token_tile,
    )


def hyper_mega_mhc(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = MHC_SINKHORN_ITERS,
    need_backward: bool = True,
    token_tile: int = DEFAULT_TOKEN_TILE,
) -> MegaMhcOutputs:
    """Run one shifted boundary through pure AIC/AIV token-tile tasks.

    Args:
        previous_output: Previous block output.
        residual: Previous residual streams.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weights.
        alpha: Mapping scale parameters.
        bias: Mapping bias parameters.
        norm_weight: Shifted RMSNorm weight.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
        need_backward: Whether to retain native backward caches.
        token_tile: Number of tokens assigned to each logical task.
    """
    _validate_common(
        previous_output,
        residual,
        previous_pre_mix,
        previous_post_mix,
        previous_residual_mix,
        phi,
        alpha,
        bias,
        norm_weight,
        num_iters,
    )
    if not residual.is_npu or residual.dtype != torch.bfloat16:
        raise TypeError("hyper_mega_mhc requires BF16 NPU activations.")
    leading_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    token_count = residual.numel() // (4 * hidden_size)
    device_index = residual.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    num_cube_cores = _resolve_num_cube_cores(device_index)
    plan = _get_cached_plan(
        token_count,
        hidden_size,
        device_index,
        num_cube_cores,
        token_tile,
    )
    profile_call = prepare_mega_kernel_call(
        plan.runtime,
        direction="forward",
        fallback_event_counters=plan.event_counters,
    )
    try:
        outputs = multicore_ops.hyper_mega_mhc(
            previous_output.reshape(1, token_count, hidden_size).contiguous(),
            residual.reshape(1, token_count, 4, hidden_size).contiguous(),
            previous_pre_mix.reshape(1, token_count, 4).contiguous(),
            previous_post_mix.reshape(1, token_count, 4).contiguous(),
            previous_residual_mix.reshape(1, token_count, 4, 4).contiguous(),
            phi.contiguous(),
            alpha.contiguous(),
            bias.contiguous(),
            norm_weight.contiguous(),
            profile_call.runtime_config,
            profile_call.event_counters,
            profile_call.profile_buffer,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            num_iters=num_iters,
            need_backward=need_backward,
        )
        # The counters are initially zero. Reset them after the asynchronous
        # launch so Host preparation for zero_ overlaps the long-running mega
        # kernel; stream ordering makes the buffer ready before its next use.
        profile_call.clear_event_counters.zero_()
        profile_call.complete()
    finally:
        profile_call.cancel()
    new_residual, next_pre, next_post, next_res, block_input = outputs
    return (
        new_residual.reshape(*leading_shape, 4, hidden_size),
        next_pre.reshape(*leading_shape, 4),
        next_post.reshape(*leading_shape, 4),
        next_res.reshape(*leading_shape, 4, 4),
        block_input.reshape(*leading_shape, hidden_size),
    )

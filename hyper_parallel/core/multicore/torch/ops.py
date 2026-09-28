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
"""
hyper_parallel.core.multicore.torch.ops
=============================================
Out-of-tree PyTorch operator registration for MoE-FFN operators.

Registers into the ``hyper_parallel`` PyTorch namespace — does NOT modify
op-plugin or any PyTorch source. The operators are accessible via:

    torch.ops.hyper_parallel.mega_moe(...)
    torch.ops.hyper_parallel.mega_moe_grad(...)

Or via the Python wrappers in this module:

    from hyper_parallel.core.multicore.torch.ops import mega_moe, mega_moe_grad

Forward and backward ACLNN symbols are packaged in one component-owned
``hyper_parallel_multicore_nn`` vendor. Source the packaged ``set_env.bash``
before starting the application or framework Python process so CANN can discover that vendor.
"""
__all__ = [
    "cann_mhc_pre_sinkhorn",
    "cann_mhc_pre_sinkhorn_with_cache",
    "hyper_mega_mhc",
    "hyper_mega_mhc_grad",
    "mega_moe",
    "mega_moe_grad",
]

from functools import lru_cache
import torch
import torch_npu  # pylint: disable=unused-import  # Registers native NPU operators.

from hyper_parallel.core.multicore._loader import (
    NativeComponentUnavailableError,
    get_multicore_paths,
    preload_vendor_library,
)


@lru_cache(maxsize=1)
def _load_native() -> None:
    """Register the ABI-specific native adapter once on first operation."""
    vendor_root, adapter_path = get_multicore_paths()
    preload_vendor_library(vendor_root)
    try:
        torch.ops.load_library(str(adapter_path))
    except (OSError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED] component=multicore framework=torch "
            f"library={adapter_path} error={error}. "
            "Check the Python/Torch/torch_npu/CANN version combination and rebuild the Torch adapter."
        ) from error


# ---------------------------------------------------------------------------


# Python wrappers — thin pass-through to the registered C++ ops
# ---------------------------------------------------------------------------


def cann_mhc_pre_sinkhorn(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call CANN MhcPreSinkhorn while retaining mappings needed by shifting.

    Args:
        x: Token-major residual input.
        phi: Projection weights.
        alpha: MhcPre scaling parameters.
        bias: MhcPre bias parameters.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
    """
    _load_native()
    if x.ndim != 3:
        raise ValueError(f"x must be token-major [M, 4, H], got {tuple(x.shape)}.")
    token_count, hc_mult, hidden_size = x.shape
    hc_mix = hc_mult * hc_mult + 2 * hc_mult
    x_bsnd = x.unsqueeze(0)
    hin = torch.empty((1, token_count, hidden_size), dtype=x.dtype, device=x.device)
    h_post = torch.empty((1, token_count, hc_mult), dtype=torch.float32, device=x.device)
    h_res = torch.empty((1, token_count, hc_mult * hc_mult), dtype=torch.float32, device=x.device)
    h_pre = torch.empty((1, token_count, hc_mult), dtype=torch.float32, device=x.device)
    hc_before_norm = torch.empty((1, token_count, hc_mix), dtype=torch.float32, device=x.device)
    inv_rms = torch.empty((1, token_count, 1), dtype=torch.float32, device=x.device)
    sum_out = torch.empty((2 * num_iters, 1, token_count, hc_mult), dtype=torch.float32, device=x.device)
    norm_out = torch.empty(
        (2 * num_iters, 1, token_count, hc_mult, hc_mult),
        dtype=torch.float32,
        device=x.device,
    )
    torch.ops.hyper_parallel.cann_mhc_pre_sinkhorn(
        x_bsnd,
        phi,
        alpha,
        bias,
        hin,
        h_post,
        h_res,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps,
        norm_eps,
        num_iters,
    )
    return (
        hin.squeeze(0),
        h_post.squeeze(0),
        h_res.squeeze(0).reshape(token_count, hc_mult, hc_mult),
        h_pre.squeeze(0),
    )


def cann_mhc_pre_sinkhorn_with_cache(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = 20,
) -> tuple[torch.Tensor, ...]:
    """Call CANN MhcPreSinkhorn and expose all native backward caches.

    Args:
        x: Token-major residual input.
        phi: Projection weights.
        alpha: MhcPre scaling parameters.
        bias: MhcPre bias parameters.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
    """
    _load_native()
    if x.ndim != 3:
        raise ValueError(f"x must be token-major [M, 4, H], got {tuple(x.shape)}.")
    token_count, hc_mult, hidden_size = x.shape
    hc_mix = hc_mult * hc_mult + 2 * hc_mult
    x_bsnd = x.unsqueeze(0)
    hin = torch.empty((1, token_count, hidden_size), dtype=x.dtype, device=x.device)
    h_post = torch.empty((1, token_count, hc_mult), dtype=torch.float32, device=x.device)
    h_res = torch.empty((1, token_count, hc_mult * hc_mult), dtype=torch.float32, device=x.device)
    h_pre = torch.empty((1, token_count, hc_mult), dtype=torch.float32, device=x.device)
    hc_before_norm = torch.empty((1, token_count, hc_mix), dtype=torch.float32, device=x.device)
    inv_rms = torch.empty((1, token_count, 1), dtype=torch.float32, device=x.device)
    sum_out = torch.empty((2 * num_iters, 1, token_count, hc_mult), dtype=torch.float32, device=x.device)
    norm_out = torch.empty(
        (2 * num_iters, 1, token_count, hc_mult, hc_mult), dtype=torch.float32, device=x.device
    )
    torch.ops.hyper_parallel.cann_mhc_pre_sinkhorn(
        x_bsnd,
        phi,
        alpha,
        bias,
        hin,
        h_post,
        h_res,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps,
        norm_eps,
        num_iters,
    )
    return hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out


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
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = 20,
    need_backward: bool,
) -> tuple[torch.Tensor, ...]:
    """Allocate outputs and launch the single-kernel shifted mHC implementation.

    Args:
        previous_output: Previous block output used by MhcPost.
        residual: Previous block residual state.
        previous_pre_mix: Previous block pre-mix coefficients.
        previous_post_mix: Previous block post-mix coefficients.
        previous_residual_mix: Previous block residual-mix coefficients.
        phi: Projection weights.
        alpha: MhcPre scaling parameters.
        bias: MhcPre bias parameters.
        norm_weight: Shifted RMSNorm weight.
        runtime_config: Serialized multicore schedule.
        all_event_counters: Device event-counter storage.
        profile_buffer: Device profiling storage.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
        need_backward: Whether MhcPre should populate backward caches.

    Returns:
        Five public forward outputs followed by the six tensors consumed by
        ``HyperMegaMhcGrad``.
    """
    _load_native()
    if residual.ndim != 4:
        raise ValueError(f"residual must be BSND [B, S, 4, H], got {tuple(residual.shape)}.")
    batch_size, sequence_length, hc_mult, hidden_size = residual.shape
    new_residual = torch.empty_like(residual)
    next_pre_mix = torch.empty(
        (batch_size, sequence_length, hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )
    next_post_mix = torch.empty_like(next_pre_mix)
    next_residual_mix = torch.empty(
        (batch_size, sequence_length, hc_mult * hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )
    block_input = torch.empty(
        (batch_size, sequence_length, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    hc_mix = hc_mult * hc_mult + 2 * hc_mult
    sinkhorn_steps = 2 * num_iters
    cache_outputs = (
        torch.empty(
            (batch_size, sequence_length, hc_mix),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (batch_size, sequence_length, 1),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (sinkhorn_steps, batch_size, sequence_length, hc_mult),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (sinkhorn_steps, batch_size, sequence_length, hc_mult, hc_mult),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (batch_size, sequence_length, hidden_size),
            dtype=residual.dtype,
            device=residual.device,
        ),
        torch.empty(
            (batch_size, sequence_length, 1),
            dtype=torch.float32,
            device=residual.device,
        ),
    )
    torch.ops.hyper_parallel.mega_mhc(
        previous_output,
        residual,
        previous_pre_mix,
        previous_post_mix,
        previous_residual_mix,
        phi,
        alpha,
        bias,
        norm_weight,
        runtime_config,
        all_event_counters,
        profile_buffer,
        new_residual,
        next_pre_mix,
        next_post_mix,
        next_residual_mix,
        block_input,
        *cache_outputs,
        hc_eps,
        norm_eps,
        num_iters,
        need_backward,
    )
    return (
        new_residual,
        next_pre_mix,
        next_post_mix,
        next_residual_mix.reshape(batch_size, sequence_length, hc_mult, hc_mult),
        block_input,
        *cache_outputs,
    )


def _allocate_hyper_mega_mhc_grad_outputs(
    previous_residual: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    previous_output: torch.Tensor,
    previous_pre: torch.Tensor,
    previous_post: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Allocate output gradients required by the native backward ABI."""
    return (
        torch.empty_like(previous_residual),
        torch.empty_like(phi),
        torch.empty_like(alpha),
        torch.empty_like(bias),
        torch.empty_like(previous_output),
        torch.empty_like(previous_pre),
        torch.empty_like(previous_post),
        torch.empty_like(previous_residual_mix),
        torch.empty(norm_weight.shape, dtype=torch.float32, device=norm_weight.device),
    )


def hyper_mega_mhc_grad(
    grad_hin_placeholder: torch.Tensor,
    grad_h_post: torch.Tensor,
    grad_h_res: torch.Tensor,
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    previous_pre: torch.Tensor,
    hc_before_norm: torch.Tensor,
    inv_rms: torch.Tensor,
    sum_out: torch.Tensor,
    norm_out: torch.Tensor,
    grad_current_pre: torch.Tensor,
    mixed_input: torch.Tensor,
    rms_rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    direct_grad_x: torch.Tensor,
    previous_residual: torch.Tensor,
    previous_output: torch.Tensor,
    previous_post: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    """Allocate gradients and launch the single-kernel shifted mHC backward.

    Args:
        grad_hin_placeholder: Placeholder gradient matching native ABI order.
        grad_h_post: Gradient of current post-mix coefficients.
        grad_h_res: Gradient of current residual-mix coefficients.
        x: Forward MhcPre input.
        phi: Projection weights.
        alpha: MhcPre scaling parameters.
        bias: MhcPre bias parameters.
        previous_pre: Previous pre-mix coefficients.
        hc_before_norm: Cached pre-normalization coefficients.
        inv_rms: Cached coefficient inverse RMS.
        sum_out: Cached Sinkhorn reductions.
        norm_out: Cached Sinkhorn normalization values.
        grad_current_pre: Gradient of current pre-mix coefficients.
        mixed_input: Cached shifted input mix.
        rms_rstd: Cached shifted RMSNorm reciprocal standard deviation.
        norm_weight: Shifted RMSNorm weight.
        direct_grad_x: Direct gradient of the current residual state.
        previous_residual: Previous residual state.
        previous_output: Previous block output.
        previous_post: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        runtime_config: Serialized multicore schedule.
        all_event_counters: Device event-counter storage.
        profile_buffer: Device profiling storage.
        hc_eps: Sinkhorn numerical-stability epsilon.
    """
    _load_native()
    grad_outputs = _allocate_hyper_mega_mhc_grad_outputs(
        previous_residual,
        phi,
        alpha,
        bias,
        previous_output,
        previous_pre,
        previous_post,
        previous_residual_mix,
        norm_weight,
    )
    return torch.ops.hyper_parallel.mega_mhc_grad(
        grad_hin_placeholder,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        previous_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        grad_current_pre,
        mixed_input,
        rms_rstd,
        norm_weight,
        direct_grad_x,
        previous_residual,
        previous_output,
        previous_post,
        previous_residual_mix,
        runtime_config,
        all_event_counters,
        profile_buffer,
        *grad_outputs,
        hc_eps,
    )


def mega_moe(
    dispatch_target: torch.Tensor,
    dispatch_target_off: torch.Tensor,
    dispatch_src: torch.Tensor,
    dispatch_src_off: torch.Tensor,
    dispatch_size: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_glist: torch.Tensor,
    up_proj_y: torch.Tensor,
    swiglu_out: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_glist: torch.Tensor,
    down_proj_y: torch.Tensor,
    combine_target: torch.Tensor,
    combine_target_off: torch.Tensor,
    combine_src_off: torch.Tensor,
    combine_size: torch.Tensor,
    gmm_workspace: torch.Tensor,
    up_proj_tiling: torch.Tensor,
    swiglu_tiling: torch.Tensor,
    down_proj_tiling: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> None:
    """
    MoE-FFN forward operator.

    Args:
        dispatch_target: First operator tensor; the detailed ABI is documented below.

    Writes in-place to: dispatch_target, up_proj_y, swiglu_out, down_proj_y,
                        combine_target.
    All output tensors must be pre-allocated with correct shapes.

    Args:
        dispatch_target: First tensor in the fixed flat forward ABI; the complete parameter groups are documented
            below and retain their registered schema order.

    Parameters
    ----------
    dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off,
    dispatch_size :
        AllToAll dispatch buffers — dispatch_target written in-place.
    up_proj_weight, up_proj_glist :
        Expert weight and cumulative group sizes for GMM1 (up-projection).
    up_proj_y, swiglu_out :
        GMM1 output and SwiGLU output — written in-place.
    down_proj_weight, down_proj_glist, down_proj_y :
        Expert weight, cumulative group sizes, and output for GMM2 (down-projection).
    combine_target, combine_target_off, combine_src_off, combine_size :
        AllToAll combine buffers — combine_target written in-place.
    gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling :
        Pre-computed tiling tensors (from gen_runtime_data.py).
    runtime_config :
        Per-rank runtime config tensor (from gen_runtime_data.py).
    all_event_counters :
        Event synchronization counter tensor.
    rank_id, ep, expert_num, hidden_size, seq_size :
        Topology / shape attributes.
    """
    _load_native()
    torch.ops.hyper_parallel.mega_moe(
        dispatch_target,
        dispatch_target_off,
        dispatch_src,
        dispatch_src_off,
        dispatch_size,
        up_proj_weight,
        up_proj_glist,
        up_proj_y,
        swiglu_out,
        down_proj_weight,
        down_proj_glist,
        down_proj_y,
        combine_target,
        combine_target_off,
        combine_src_off,
        combine_size,
        gmm_workspace,
        up_proj_tiling,
        swiglu_tiling,
        down_proj_tiling,
        runtime_config,
        all_event_counters,
        all_event_counters,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
    )


def mega_moe_grad(
    dispatch_target: torch.Tensor,
    dispatch_target_off: torch.Tensor,
    dy: torch.Tensor,
    dispatch_src_off: torch.Tensor,
    dispatch_size: torch.Tensor,
    hidden: torch.Tensor,
    hidden_dw: torch.Tensor,
    w2: torch.Tensor,
    act_grad_y: torch.Tensor,
    gate: torch.Tensor,
    grad_gate: torch.Tensor,
    w1: torch.Tensor,
    gate_dx: torch.Tensor,
    grad_x: torch.Tensor,
    combine_target_off: torch.Tensor,
    combine_src_off: torch.Tensor,
    combine_size: torch.Tensor,
    permute_out: torch.Tensor,
    gate_dw: torch.Tensor,
    group_list: torch.Tensor,
    act_grad_tiling: torch.Tensor,
    gate_grad_tiling: torch.Tensor,
    w1_grad_tiling: torch.Tensor,
    w2_grad_tiling: torch.Tensor,
    swiglu_grad_tiling: torch.Tensor,
    gmm_workspace: torch.Tensor,
    swiglu_grad_workspace: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> None:
    """Launch the public MoE-FFN backward operator.

    Tensor arguments describe dispatch/combine buffers, saved activations,
    output gradients, weights, tiling data, workspaces, RuntimeConfig and
    event counters. Integer arguments describe the rank-local topology and
    shape. Output and workspace tensors must be pre-allocated; the operator
    writes gradients and communication results in place.

    Args:
        dispatch_target: First tensor in the fixed flat backward ABI; remaining tensor and topology arguments retain
            their registered schema order.
    """
    _load_native()
    torch.ops.hyper_parallel.mega_moe_grad(
        dispatch_target,
        dispatch_target_off,
        dy,
        dispatch_src_off,
        dispatch_size,
        hidden,
        hidden_dw,
        w2,
        act_grad_y,
        gate,
        grad_gate,
        w1,
        gate_dx,
        grad_x,
        combine_target_off,
        combine_src_off,
        combine_size,
        permute_out,
        gate_dw,
        group_list,
        act_grad_tiling,
        gate_grad_tiling,
        w1_grad_tiling,
        w2_grad_tiling,
        swiglu_grad_tiling,
        gmm_workspace,
        swiglu_grad_workspace,
        runtime_config,
        all_event_counters,
        all_event_counters,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
    )


def mega_moe_with_profile_buffer(
    dispatch_target: torch.Tensor,
    dispatch_target_off: torch.Tensor,
    dispatch_src: torch.Tensor,
    dispatch_src_off: torch.Tensor,
    dispatch_size: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_glist: torch.Tensor,
    up_proj_y: torch.Tensor,
    swiglu_out: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_glist: torch.Tensor,
    down_proj_y: torch.Tensor,
    combine_target: torch.Tensor,
    combine_target_off: torch.Tensor,
    combine_src_off: torch.Tensor,
    combine_size: torch.Tensor,
    gmm_workspace: torch.Tensor,
    up_proj_tiling: torch.Tensor,
    swiglu_tiling: torch.Tensor,
    down_proj_tiling: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> None:
    """
    Launch the internal forward ABI with a profiler-owned ordinary NPU buffer.

    Args:
        dispatch_target: First operator tensor; the detailed ABI is documented below.

    Writes in-place to: dispatch_target, up_proj_y, swiglu_out, down_proj_y,
                        combine_target.
    All output tensors must be pre-allocated with correct shapes.

    Args:
        dispatch_target: First tensor in the fixed profiled-forward ABI; the complete parameter groups are documented
            below and retain their registered schema order.

    Parameters
    ----------
    dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off,
    dispatch_size :
        AllToAll dispatch buffers — dispatch_target written in-place.
    up_proj_weight, up_proj_glist :
        Expert weight and cumulative group sizes for GMM1 (up-projection).
    up_proj_y, swiglu_out :
        GMM1 output and SwiGLU output — written in-place.
    down_proj_weight, down_proj_glist, down_proj_y :
        Expert weight, cumulative group sizes, and output for GMM2 (down-projection).
    combine_target, combine_target_off, combine_src_off, combine_size :
        AllToAll combine buffers — combine_target written in-place.
    gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling :
        Pre-computed tiling tensors (from gen_runtime_data.py).
    runtime_config :
        Per-rank runtime config tensor (from gen_runtime_data.py).
    all_event_counters :
        Event synchronization counter tensor.
    profile_buffer :
        Ordinary NPU memory receiving per-worker cycle records.
    rank_id, ep, expert_num, hidden_size, seq_size :
        Topology / shape attributes.
    """
    _load_native()
    torch.ops.hyper_parallel.mega_moe(
        dispatch_target,
        dispatch_target_off,
        dispatch_src,
        dispatch_src_off,
        dispatch_size,
        up_proj_weight,
        up_proj_glist,
        up_proj_y,
        swiglu_out,
        down_proj_weight,
        down_proj_glist,
        down_proj_y,
        combine_target,
        combine_target_off,
        combine_src_off,
        combine_size,
        gmm_workspace,
        up_proj_tiling,
        swiglu_tiling,
        down_proj_tiling,
        runtime_config,
        all_event_counters,
        profile_buffer,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
    )


def mega_moe_grad_with_profile_buffer(
    dispatch_target: torch.Tensor,
    dispatch_target_off: torch.Tensor,
    dy: torch.Tensor,
    dispatch_src_off: torch.Tensor,
    dispatch_size: torch.Tensor,
    hidden: torch.Tensor,
    hidden_dw: torch.Tensor,
    w2: torch.Tensor,
    act_grad_y: torch.Tensor,
    gate: torch.Tensor,
    grad_gate: torch.Tensor,
    w1: torch.Tensor,
    gate_dx: torch.Tensor,
    grad_x: torch.Tensor,
    combine_target_off: torch.Tensor,
    combine_src_off: torch.Tensor,
    combine_size: torch.Tensor,
    permute_out: torch.Tensor,
    gate_dw: torch.Tensor,
    group_list: torch.Tensor,
    act_grad_tiling: torch.Tensor,
    gate_grad_tiling: torch.Tensor,
    w1_grad_tiling: torch.Tensor,
    w2_grad_tiling: torch.Tensor,
    swiglu_grad_tiling: torch.Tensor,
    gmm_workspace: torch.Tensor,
    swiglu_grad_workspace: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> None:
    """Launch the profiled MoE-FFN backward ABI.

    The argument contract matches :func:`mega_moe_grad` with one additional
    ordinary-NPU-memory ``profile_buffer``. Device workers write cycle records
    directly into that buffer. All output and workspace tensors remain
    caller-owned and are written in place.

    Args:
        dispatch_target: First tensor in the fixed profiled-backward ABI; remaining tensor and topology arguments
            retain their registered schema order.
    """
    _load_native()
    torch.ops.hyper_parallel.mega_moe_grad(
        dispatch_target,
        dispatch_target_off,
        dy,
        dispatch_src_off,
        dispatch_size,
        hidden,
        hidden_dw,
        w2,
        act_grad_y,
        gate,
        grad_gate,
        w1,
        gate_dx,
        grad_x,
        combine_target_off,
        combine_src_off,
        combine_size,
        permute_out,
        gate_dw,
        group_list,
        act_grad_tiling,
        gate_grad_tiling,
        w1_grad_tiling,
        w2_grad_tiling,
        swiglu_grad_tiling,
        gmm_workspace,
        swiglu_grad_workspace,
        runtime_config,
        all_event_counters,
        profile_buffer,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
    )

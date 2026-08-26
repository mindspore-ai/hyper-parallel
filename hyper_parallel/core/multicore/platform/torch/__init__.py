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
hyper_parallel.core.multicore.platform.torch
=============================================
Out-of-tree PyTorch operator registration for MoE-FFN operators.

Registers into the ``hyper_parallel`` PyTorch namespace — does NOT modify
op-plugin or any PyTorch source. The operators are accessible via:

    torch.ops.hyper_parallel.mega_moe(...)
    torch.ops.hyper_parallel.mega_moe_grad(...)

Or via the Python wrappers in this module:

    from hyper_parallel.core.multicore.platform.torch import mega_moe, mega_moe_grad

Forward and backward ACLNN symbols are packaged in one component-owned
``hyper_parallel_multicore_nn`` vendor. Source the packaged ``set_env.bash``
before starting the application or framework Python process so CANN can discover that vendor.
"""
__all__ = ["mega_moe", "mega_moe_grad"]

from functools import lru_cache
import importlib

from hyper_parallel.core.multicore._loader import (
    NativeComponentUnavailableError,
    get_multicore_paths,
    preload_vendor_library,
)


@lru_cache(maxsize=1)
def _get_torch():
    """Load torch and register the ABI-specific multicore adapter once."""
    vendor_root, adapter_path = get_multicore_paths("torch")
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    preload_vendor_library(vendor_root)
    try:
        torch.ops.load_library(str(adapter_path))
    except (OSError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED] component=multicore framework=torch "
            f"library={adapter_path} error={error}. "
            "Check the Python/Torch/torch_npu/CANN version combination and rebuild the Torch adapter."
        ) from error
    return torch

# ---------------------------------------------------------------------------
# Python wrappers — thin pass-through to the registered C++ ops
# ---------------------------------------------------------------------------


def mega_moe(
    dispatch_target, dispatch_target_off,
    dispatch_src, dispatch_src_off, dispatch_size,
    up_proj_weight, up_proj_glist,
    up_proj_y, swiglu_out,
    down_proj_weight, down_proj_glist, down_proj_y,
    combine_target, combine_target_off, combine_src_off, combine_size,
    gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
    runtime_config, all_event_counters,
    rank_id: int, ep: int, expert_num: int,
    hidden_size: int, seq_size: int,
):
    """
    MoE-FFN forward operator.

    Writes in-place to: dispatch_target, up_proj_y, swiglu_out, down_proj_y,
                        combine_target.
    All output tensors must be pre-allocated with correct shapes.

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
    _get_torch().ops.hyper_parallel.mega_moe(
        dispatch_target, dispatch_target_off,
        dispatch_src, dispatch_src_off, dispatch_size,
        up_proj_weight, up_proj_glist,
        up_proj_y, swiglu_out,
        down_proj_weight, down_proj_glist, down_proj_y,
        combine_target, combine_target_off, combine_src_off, combine_size,
        gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
        runtime_config, all_event_counters,
        rank_id, ep, expert_num, hidden_size, seq_size,
    )


def mega_moe_grad(
    dispatch_target, dispatch_target_off,
    dy, dispatch_src_off, dispatch_size,
    hidden, hidden_dw,
    w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
    combine_target_off, combine_src_off, combine_size,
    permute_out, gate_dw, group_list,
    act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
    swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
    runtime_config, all_event_counters,
    rank_id: int, ep: int, expert_num: int,
    hidden_size: int, seq_size: int,
):
    """
    MoE-FFN backward operator.

    Writes in-place to: dispatch_target, hidden_dw, act_grad_y, grad_gate,
                        gate_dx, grad_x, permute_out, gate_dw.
    All output tensors must be pre-allocated with correct shapes.

    Parameters
    ----------
    dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size :
        AllToAll dispatch buffers — dispatch_target written in-place with
        the dispatched gradient.  dy is the source gradient tensor.
    hidden :
        SwiGLU output saved from the forward pass (used by W2-grad, GMM4).
    hidden_dw :
        W2 weight gradient — written in-place.
    w2 :
        W2 weight (= down_proj_weight from forward).
    act_grad_y :
        Activation gradient output from GMM1 bwd (target @ W2.T) — written in-place.
    gate :
        up_proj_y saved from the forward pass (SwiGLU input).
    grad_gate :
        SwiGLU gradient output — written in-place.
    w1 :
        W1 weight (= up_proj_weight from forward).
    gate_dx :
        GMM2 bwd output (grad_gate @ W1.T), before AllToAll combine — written in-place.
    grad_x :
        AllToAll combine output (final activation gradient) — written in-place.
    combine_target_off, combine_src_off, combine_size :
        AllToAll combine buffer descriptors.
    permute_out :
        In-place intermediate buffer for W1-grad (GMM4).
    gate_dw :
        W1 weight gradient — written in-place.
    group_list :
        Cumulative expert token counts ([E] int64).
    act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
    swiglu_grad_tiling :
        Pre-computed tiling tensors (from gen_runtime_data.py bwd).
    gmm_workspace, swiglu_grad_workspace :
        Workspace tensors.
    runtime_config :
        Per-rank runtime config tensor (from gen_runtime_data.py bwd).
    all_event_counters :
        Event synchronization counter tensor.
    rank_id, ep, expert_num, hidden_size, seq_size :
        Topology / shape attributes.
    """
    _get_torch().ops.hyper_parallel.mega_moe_grad(
        dispatch_target, dispatch_target_off,
        dy, dispatch_src_off, dispatch_size,
        hidden, hidden_dw,
        w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
        combine_target_off, combine_src_off, combine_size,
        permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_config, all_event_counters,
        rank_id, ep, expert_num, hidden_size, seq_size,
    )

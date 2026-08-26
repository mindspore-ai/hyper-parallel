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
hyper_parallel.core.multicore.platform.mindspore
==================================================
MindSpore platform adapter for MoE-FFN multicore operators.

Exposes ``mega_moe`` and ``mega_moe_grad`` backed by the
``hyper_parallel_mega_moe_ms`` pybind11 extension.

.. note::
    **Only MindSpore PyNative mode is supported.**
    Graph mode (``ms.GRAPH_MODE``) is not yet implemented — the operator
    YAML definitions set ``function: disable: True`` and
    ``dispatch: enable: False``, so the MindSpore compiler cannot trace
    or lower these ops into a static graph.  Calling either function
    inside a ``@ms.jit``-decorated function or when
    ``ms.set_context(mode=ms.GRAPH_MODE)`` is active will raise a
    ``RuntimeError`` at call time.

Forward and backward ACLNN symbols are packaged in one component-owned
``hyper_parallel_multicore_nn`` vendor. Source the packaged ``set_env.bash``
before starting the application or framework Python process so CANN can discover that vendor.
"""
__all__ = ["mega_moe", "mega_moe_grad"]

from hyper_parallel.core.multicore._loader import (
    get_multicore_paths,
    load_cpython_extension,
    preload_vendor_library,
)

# Lazily import the compiled MindSpore extension on first use to avoid a
# circular import triggered by MindSpore's custom-op scan at load time.
_ms_ops = None


def _check_pynative_mode(fn_name: str) -> None:
    """Raise RuntimeError if the current MindSpore execution mode is not PyNative.

    Graph mode (GRAPH_MODE) is not yet implemented for MoE-FFN multicore ops.
    The operator YAML definitions disable function-level dispatch and static
    graph compilation, so these ops cannot be traced or lowered by the
    MindSpore compiler.
    """
    import mindspore as ms  # pylint: disable=import-outside-toplevel
    if ms.get_context("mode") != ms.PYNATIVE_MODE:
        raise RuntimeError(
            f"'{fn_name}' only supports MindSpore PyNative mode. "
            "Graph mode (ms.GRAPH_MODE) is not yet implemented for MoE-FFN "
            "multicore operators. "
            "Please call ms.set_context(mode=ms.PYNATIVE_MODE) before using "
            "this operator, and do not wrap it inside @ms.jit functions."
        )


def _get_ms_ops():
    """Lazily load and return the compiled MindSpore extension module."""
    global _ms_ops
    if _ms_ops is None:
        vendor_root, adapter_path = get_multicore_paths("mindspore")
        __import__("mindspore")
        preload_vendor_library(vendor_root)
        _ms_ops = load_cpython_extension("hyper_parallel_mega_moe_ms", adapter_path)
    return _ms_ops


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
    """MoE-FFN forward operator (MindSpore).

    .. note:: Only PyNative mode is supported. Raises ``RuntimeError`` if
        called in Graph mode or inside an ``@ms.jit`` function.
    """
    _check_pynative_mode("mega_moe")
    _get_ms_ops().mega_moe(
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
    """MoE-FFN backward operator (MindSpore).

    .. note:: Only PyNative mode is supported. Raises ``RuntimeError`` if
        called in Graph mode or inside an ``@ms.jit`` function.
    """
    _check_pynative_mode("mega_moe_grad")
    _get_ms_ops().mega_moe_grad(
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

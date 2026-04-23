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

Exposes ``moe_ffn_fwd`` and ``moe_ffn_bwd`` backed by the
``hyper_parallel_multicore_moe_ffn_ms`` pybind11 extension.

.. note::
    **Only MindSpore PyNative mode is supported.**
    Graph mode (``ms.GRAPH_MODE``) is not yet implemented — the operator
    YAML definitions set ``function: disable: True`` and
    ``dispatch: enable: False``, so the MindSpore compiler cannot trace
    or lower these ops into a static graph.  Calling either function
    inside a ``@ms.jit``-decorated function or when
    ``ms.set_context(mode=ms.GRAPH_MODE)`` is active will raise a
    ``RuntimeError`` at call time.

Vendor library resolution order (highest priority first):

  1. ``CANN_VENDOR_FWD_LIBDIR`` / ``CANN_VENDOR_BWD_LIBDIR``
  2. ``HP_MULTICORE_DIR``  (vendors root; fwd/bwd derived automatically)
  3. ``CANN_VENDOR_LIBDIR``  (legacy single-lib fallback)
  4. Auto-detect from ``prebuild/multicore_moe_ffn/vendors/``
"""
import ctypes
import os
import re
import sys

_ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH",
                               "/usr/local/Ascend/ascend-toolkit/latest")

_CANN_VENDOR_FWD_LIBDIR = os.environ.get("CANN_VENDOR_FWD_LIBDIR", "")
_CANN_VENDOR_BWD_LIBDIR = os.environ.get("CANN_VENDOR_BWD_LIBDIR", "")

if not (_CANN_VENDOR_FWD_LIBDIR or _CANN_VENDOR_BWD_LIBDIR):
    _HP_MULTICORE_DIR = os.environ.get("HP_MULTICORE_DIR", "")
    if _HP_MULTICORE_DIR:
        _CANN_VENDOR_FWD_LIBDIR = os.path.join(
            _HP_MULTICORE_DIR, "multicore_moe_ffn_nn", "op_api", "lib")
        _CANN_VENDOR_BWD_LIBDIR = os.path.join(
            _HP_MULTICORE_DIR, "multicore_moe_ffn_grad_nn", "op_api", "lib")
    elif os.environ.get("CANN_VENDOR_LIBDIR", ""):
        _CANN_VENDOR_FWD_LIBDIR = os.environ["CANN_VENDOR_LIBDIR"]
        _CANN_VENDOR_BWD_LIBDIR = os.environ["CANN_VENDOR_LIBDIR"]
    else:
        _PREBUILD_DIR = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../prebuild/multicore_moe_ffn"))
        if os.path.isdir(_PREBUILD_DIR):
            _vendors = os.path.join(_PREBUILD_DIR, "vendors")
            _fwd = os.path.join(_vendors, "multicore_moe_ffn_nn", "op_api", "lib")
            _bwd = os.path.join(_vendors, "multicore_moe_ffn_grad_nn", "op_api", "lib")
            if os.path.isdir(_fwd):
                _CANN_VENDOR_FWD_LIBDIR = _fwd
            if os.path.isdir(_bwd):
                _CANN_VENDOR_BWD_LIBDIR = _bwd

# ---------------------------------------------------------------------------
# Set ASCEND_CUSTOM_OPP_PATH at module load time — BEFORE any MindSpore import.
#
# GetOpApiFuncAddr (called by LAUNCH_ACLNN_FUNC at op invocation) scans
# ASCEND_CUSTOM_OPP_PATH to populate g_custom_lib_path, which may be cached
# as a static variable on first call.  MindSpore's own initialization can
# trigger GetOpApiFuncAddr for standard CANN ops during `import mindspore`,
# which would cache an empty g_custom_lib_path if we set the env var too late
# (e.g., inside _get_ms_ops()).  Setting it here ensures the correct paths are
# present before any such call.
# ---------------------------------------------------------------------------
for _libdir in (_CANN_VENDOR_FWD_LIBDIR, _CANN_VENDOR_BWD_LIBDIR):
    if _libdir:
        _vendor_root = re.sub(r'/op_api/lib/?$', '', _libdir)
        _cur_opp = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
        if _vendor_root and _vendor_root not in _cur_opp:
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = (
                f"{_vendor_root}:{_cur_opp}" if _cur_opp else _vendor_root)

# Pre-load CANN base libs with RTLD_GLOBAL.
for _ascendcl_path in [
    os.path.join(_ASCEND_HOME, "lib64", "libascendcl.so"),
    os.path.join(_ASCEND_HOME, "fwkacllib", "lib64", "libascendcl.so"),
]:
    if os.path.exists(_ascendcl_path):
        ctypes.CDLL(_ascendcl_path, mode=ctypes.RTLD_GLOBAL)
        break

_opapi_path = os.path.join(_ASCEND_HOME, "lib64", "libopapi.so")
if os.path.exists(_opapi_path):
    ctypes.CDLL(_opapi_path, mode=ctypes.RTLD_GLOBAL)

# Pre-load both libcust_opapi.so with RTLD_GLOBAL so LAUNCH_ACLNN_FUNC can
# resolve aclnnMulticoreMoeFfn* / aclnnMulticoreMoeFfnGrad* from either package.
for _vendor_libdir in (_CANN_VENDOR_FWD_LIBDIR, _CANN_VENDOR_BWD_LIBDIR):
    if _vendor_libdir:
        _cust_opapi = os.path.join(_vendor_libdir, "libcust_opapi.so")
        if os.path.exists(_cust_opapi):
            ctypes.CDLL(_cust_opapi, mode=ctypes.RTLD_GLOBAL)

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
        build_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "lib")
        if build_lib not in sys.path:
            sys.path.insert(0, build_lib)
        import hyper_parallel_multicore_moe_ffn_ms as _m  # noqa: E402  # pylint: disable=import-outside-toplevel
        _ms_ops = _m
    return _ms_ops


def moe_ffn_fwd(
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
    _check_pynative_mode("moe_ffn_fwd")
    _get_ms_ops().moe_ffn_fwd(
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


def moe_ffn_bwd(
    dispatch_target, dispatch_target_off,
    dy, dispatch_src_off, dispatch_size,
    hidden, hidden_dw,
    w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
    combine_target_off, combine_src_off, combine_size,
    permute_out, gate_dw, group_list,
    act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling,
    swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
    runtime_config, all_event_counters,
    rank_id: int, ep: int, expert_num: int,
    hidden_size: int, seq_size: int,
):
    """MoE-FFN backward operator (MindSpore).

    .. note:: Only PyNative mode is supported. Raises ``RuntimeError`` if
        called in Graph mode or inside an ``@ms.jit`` function.
    """
    _check_pynative_mode("moe_ffn_bwd")
    _get_ms_ops().moe_ffn_bwd(
        dispatch_target, dispatch_target_off,
        dy, dispatch_src_off, dispatch_size,
        hidden, hidden_dw,
        w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
        combine_target_off, combine_src_off, combine_size,
        permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_config, all_event_counters,
        rank_id, ep, expert_num, hidden_size, seq_size,
    )


__all__ = ["moe_ffn_fwd", "moe_ffn_bwd"]

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
Out-of-tree PyTorch operator registration for MoE-FFN mega kernels.

Registers into the ``hyper_parallel`` PyTorch namespace — does NOT modify
op-plugin or any PyTorch source. The operators are accessible via:

    torch.ops.hyper_parallel.moe_ffn_fwd(...)
    torch.ops.hyper_parallel.moe_ffn_bwd(...)

Or via the Python wrappers in this module:

    from hyper_parallel.core.multicore.platform.torch import moe_ffn_fwd, moe_ffn_bwd

Build prerequisites
-------------------
No manual env-var setup is required if ``prebuild/multicore_moe_ffn.tar.gz``
exists alongside this package.  The tarball is auto-extracted and the correct
``CANN_VENDOR_*`` paths are derived automatically.

Alternatively, set one of the following environment variables before importing:

    # Preferred: per-op explicit paths
    export CANN_VENDOR_FWD_LIBDIR=/path/to/mega_kernel_gmm_nn/op_api/lib
    export CANN_VENDOR_BWD_LIBDIR=/path/to/mega_kernel_gmm_grad_nn/op_api/lib

    # Or: legacy single-lib (used for both fwd and bwd)
    export CANN_VENDOR_LIBDIR=/path/to/mega_kernel_gmm_nn/op_api/lib

Symbol lookup (new op-plugin API)
----------------------------------
GetOpApiFuncAddr (in op_api_common.cpp) resolves aclnnMegaKernelGmm* at
runtime via dlopen.  It searches each entry in ``g_custom_lib_path`` (a C++
file-scope ``const`` global in libopapi.so) for
``op_api/lib/libcust_opapi.so``.  ``g_custom_lib_path`` is initialised
**once** from ``ASCEND_CUSTOM_OPP_PATH`` when ``libopapi.so`` is first loaded
(triggered by ``import torch_npu``); after that it is frozen and cannot be
updated from Python.

Our extension links against ``libopapi.so`` and shares its
``g_custom_lib_path`` — the extension does **not** carry its own copy.

This module therefore sets ``ASCEND_CUSTOM_OPP_PATH`` **before** its own
``import torch`` call (line 127) so that, when ``import torch_npu`` on
line 145 causes ``libopapi.so`` to load, the variable is already present.

**Limitation — if** ``torch_npu`` **was imported before this module**:
``libopapi.so`` will have been loaded (and ``g_custom_lib_path`` frozen)
before we can set ``ASCEND_CUSTOM_OPP_PATH``.  In that scenario the vendor
libs will not be found via ``GetOpApiFuncAddr``.  The only reliable fix is to
set ``ASCEND_CUSTOM_OPP_PATH`` at the shell level (or at the very top of the
Python entry-point script) **before** any ``import torch_npu``.
"""
import ctypes
import os
import re as _re

# ---------------------------------------------------------------------------
# Step 0: locate both CANN vendor lib dirs (forward + backward packages).
#
# Two vendor packages provide separate libcust_opapi.so files:
#   Forward : mega_kernel_gmm_nn      → aclnnMegaKernelGmm*
#   Backward: mega_kernel_gmm_grad_nn → aclnnMegaKernelGmmGrad*
#
# Resolution order (highest priority first):
#   1. CANN_VENDOR_FWD_LIBDIR / CANN_VENDOR_BWD_LIBDIR  (explicit per-op)
#   2. CANN_VENDOR_LIBDIR  (legacy single-lib; used for both fwd and bwd)
#   3. Auto-detect from prebuild tarball / directory (no env vars needed)
#
# This block uses only pure-Python (os.path / tarfile) so it is safe to run
# before any C++ .so is loaded.
# ---------------------------------------------------------------------------
_ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH",
                               "/usr/local/Ascend/ascend-toolkit/latest")

_CANN_VENDOR_FWD_LIBDIR = os.environ.get("CANN_VENDOR_FWD_LIBDIR", "")
_CANN_VENDOR_BWD_LIBDIR = os.environ.get("CANN_VENDOR_BWD_LIBDIR", "")

if not (_CANN_VENDOR_FWD_LIBDIR or _CANN_VENDOR_BWD_LIBDIR):
    _legacy = os.environ.get("CANN_VENDOR_LIBDIR", "")
    if _legacy:
        _CANN_VENDOR_FWD_LIBDIR = _legacy
        _CANN_VENDOR_BWD_LIBDIR = _legacy
    else:
        # Auto-detect: prebuild directory is 2 levels up from platform/torch/
        _PREBUILD_DIR = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../prebuild/multicore_moe_ffn"))
        _TARBALL = _PREBUILD_DIR + ".tar.gz"
        if not os.path.isdir(_PREBUILD_DIR) and os.path.isfile(_TARBALL):
            import tarfile as _tarfile
            with _tarfile.open(_TARBALL) as _tf:
                _tf.extractall(os.path.dirname(_PREBUILD_DIR))
        if os.path.isdir(_PREBUILD_DIR):
            _vendors = os.path.join(_PREBUILD_DIR, "vendors")
            _fwd = os.path.join(_vendors, "mega_kernel_gmm_nn", "op_api", "lib")
            _bwd = os.path.join(_vendors, "mega_kernel_gmm_grad_nn", "op_api", "lib")
            if os.path.isdir(_fwd):
                _CANN_VENDOR_FWD_LIBDIR = _fwd
            if os.path.isdir(_bwd):
                _CANN_VENDOR_BWD_LIBDIR = _bwd

# ---------------------------------------------------------------------------
# Step 1: set ASCEND_CUSTOM_OPP_PATH from the detected vendor lib dirs.
#
# GetOpApiFuncAddr searches each entry in ASCEND_CUSTOM_OPP_PATH for
# op_api/lib/libcust_opapi.so to resolve aclnnMegaKernelGmm* symbols.
# g_custom_lib_path (a C++ static-duration global in op_api_common.cpp) is
# populated from ASCEND_CUSTOM_OPP_PATH once — when the .so that contains
# op_api_common.cpp is first loaded into the process.  Depending on the
# build, that .so may be libopapi.so (loaded transitively by torch or
# torch_npu) or our own extension.
#
# To cover both cases, ASCEND_CUSTOM_OPP_PATH MUST be set before ANY
# C++ .so import — i.e., before `import torch` below.
# ---------------------------------------------------------------------------
for _libdir in (_CANN_VENDOR_FWD_LIBDIR, _CANN_VENDOR_BWD_LIBDIR):
    if _libdir:
        _vendor_root = _re.sub(r'/op_api/lib/?$', '', _libdir)
        _cur_opp = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
        if _vendor_root and _vendor_root not in _cur_opp:
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = (
                f"{_vendor_root}:{_cur_opp}" if _cur_opp else _vendor_root)

import torch  # pylint: disable=import-self,wrong-import-position  # noqa: E402

# ---------------------------------------------------------------------------
# Step 2: pre-load CANN base libs with RTLD_GLOBAL.
#
# libascendcl.so: belt-and-suspenders — import torch_npu loads it, but
# promoting to GLOBAL ensures any late-loading consumer sees its symbols.
# libopapi.so: provides aclopExecutor which libcust_opapi.so calls without
# DT_NEEDED (CANN cmake allows undefined symbols).  Must be GLOBAL first.
# ---------------------------------------------------------------------------
for _ascendcl_path in [
    os.path.join(_ASCEND_HOME, "lib64", "libascendcl.so"),
    os.path.join(_ASCEND_HOME, "fwkacllib", "lib64", "libascendcl.so"),
]:
    if os.path.exists(_ascendcl_path):
        ctypes.CDLL(_ascendcl_path, mode=ctypes.RTLD_GLOBAL)
        break

import torch_npu  # noqa: F401 — loads libtorch_npu.so  # pylint: disable=wrong-import-position

_opapi_path = os.path.join(_ASCEND_HOME, "lib64", "libopapi.so")
if os.path.exists(_opapi_path):
    ctypes.CDLL(_opapi_path, mode=ctypes.RTLD_GLOBAL)

# ---------------------------------------------------------------------------
# Step 3: pre-load both libcust_opapi.so files with RTLD_GLOBAL.
#
# Forward  (mega_kernel_gmm_nn)      exports: aclnnMegaKernelGmm*
# Backward (mega_kernel_gmm_grad_nn) exports: aclnnMegaKernelGmmGrad*
# Pre-loading with RTLD_GLOBAL ensures symbols from both are globally visible
# so GetOpApiFuncAddr can resolve either operator at runtime.
# ---------------------------------------------------------------------------
for _vendor_libdir in (_CANN_VENDOR_FWD_LIBDIR, _CANN_VENDOR_BWD_LIBDIR):
    if _vendor_libdir:
        _cust_opapi = os.path.join(_vendor_libdir, "libcust_opapi.so")
        if os.path.exists(_cust_opapi):
            ctypes.CDLL(_cust_opapi, mode=ctypes.RTLD_GLOBAL)

# ---------------------------------------------------------------------------
# Step 4: import the compiled extension.
#
# ASCEND_CUSTOM_OPP_PATH is now set (Step 1) and libcust_opapi.so is
# pre-loaded (Step 3), so g_custom_lib_path will be initialized correctly
# when the .so's static initializers run.
# ---------------------------------------------------------------------------
from . import hyper_parallel_multicore_moe_ffn_pta  # noqa: F401, E402  # pylint: disable=wrong-import-position,import-self

# ---------------------------------------------------------------------------
# Python wrappers — thin pass-through to the registered C++ ops
# ---------------------------------------------------------------------------

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
    """
    MoE-FFN forward mega kernel.

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
    torch.ops.hyper_parallel.moe_ffn_fwd(
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
    """
    MoE-FFN backward mega kernel.

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
    act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling,
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
    torch.ops.hyper_parallel.moe_ffn_bwd(
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

# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""cp_wrappers: built-in CP-aware inner attention wrappers (public, explicit injection).

Since the explicit-injection rework the framework NEVER picks a CP wrapper
automatically: the four built-ins below are referenced explicitly — by
registry name (``spec.inner_wrapper="sdpa_hf"``), by callable, or by a YAML
Target pointing at one of these functions:

.. code-block:: yaml

    plan_overrides:
      - match: "*.self_attn"
        when: cp
        inner_wrapper:
          _target_: hyper_models.components.distributed.cp_wrappers.sdpa_hf_cp_wrapper

Wrapper contract (also the Target factory contract — at apply time the
Target is built with its declared context, filled by name)::

    @inner_wrapper
    fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh) -> None

The ``@inner_wrapper`` decorator is MANDATORY (injection discipline,
see injection.py): the mesh family (``mesh``/``tp_mesh``/``cp_mesh``/
``ep_mesh``) plus the anchor ``target_module`` are REQUIRED context
params, ALL filled by the framework at apply time (None for inactive
axes) — the user just uses them.
Undecorated wrappers fail fast in the resolution chain; *args/**kwargs
are forbidden. The wrapper replaces ``target_module.forward`` in place
(K/V all-gather + dual-mode tolerance) and returns None; a wrapper that
instead RETURNS a callable is treated as a custom wrapper (also
@inner_wrapper decorated). The replaced forward must accept the original
forward's params (validated at apply time).

- ``sdpa_qkv_cp_wrapper``: NeMo convention ``forward(q, k, v, ...)`` — explicit
  K/V all-gather + D-04 offset-aware causal mask (fixes the is_causal
  alignment error under CP);
- ``sdpa_hf_cp_wrapper``: HF convention ``forward(hidden_states, ...)`` —
  primitive interception of ``F.scaled_dot_product_attention`` (reuses HF
  projections/RoPE), with misfire detection (raises if no call is
  intercepted);
- ``flex_qkv_cp_wrapper`` / ``flex_hf_cp_wrapper``: the two isomorphic
  FlexAttention entries (block_mask must be built for the global kv length).

Users may register their own named schemes::

    INNER_WRAPPER_REGISTRY["my_flash"] = my_fn

after which ``spec.inner_wrapper="my_flash"`` references it by name.

The ``is_*_attention`` helpers are public utilities for authors of custom
wrappers (choosing a scheme programmatically); the framework itself no
longer dispatches heuristically on them.
"""

import functools
import inspect
import logging

import torch
import torch.nn.functional as F

from hyper_models.components.distributed.cp_utils import (
    _cp_offset_causal_mask,
    flex_cp_allgather,
)
from hyper_models.components.distributed.injection import (
    inner_wrapper,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Style-detection helpers (public utilities for custom wrapper authors)
# ────────────────────────────────────────────────────────────────────────────

def _attn_implementation(module):
    cfg = getattr(module, "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl is None and isinstance(cfg, dict):
        impl = cfg.get("attn_implementation")
    return impl


def is_sdpa_attention(module) -> bool:
    impl = _attn_implementation(module)
    return (impl == "sdpa") or ("SdpaAttention" in type(module).__name__)


def is_flex_attention(module) -> bool:
    impl = _attn_implementation(module)
    return (impl == "flex_attention") or ("FlexAttention" in type(module).__name__)


def is_hf_style_attention(module) -> bool:
    """HF style (forward(hidden_states,...), projections inside forward) -> the primitive-interception path."""
    has_proj = (hasattr(module, "q_proj") and hasattr(module, "k_proj")
                and hasattr(module, "v_proj"))
    if not has_proj:
        return False
    try:
        sig = inspect.signature(module.forward)
        first_param = next(iter(sig.parameters.values()), None)
        return first_param is not None and first_param.name == "hidden_states"
    except (ValueError, TypeError):
        return type(module).__name__.endswith("Attention")


# ────────────────────────────────────────────────────────────────────────────
# CP-aware SDPA core (K/V all-gather + D-04 offset causal mask)
# ────────────────────────────────────────────────────────────────────────────

def _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kwargs):
    """CP-aware SDPA: K/V all-gather + D-04 offset-aware causal mask."""
    cp_dim = 2  # sequence dim of the [B, N, S, H] layout
    global_k, global_v = flex_cp_allgather(
        k.contiguous(), v.contiguous(), cp_dim, cp_mesh)
    if kwargs.get("is_causal") and cp_mesh.size() > 1:
        # D-04: triggered by CP semantics (do NOT use the q_len != kv_len
        # shape comparison as a proxy -- GQA's head-count difference does not
        # affect the sequence dim, but cross-attention/KV-cache q_len != kv_len
        # is unrelated to CP, and shape inference would misapply the lo-offset
        # semantics). With CP active, q is this rank's contiguous chunk and kv
        # is the full sequence; when q_len != kv_len, torch's is_causal aligns
        # top-left (equivalent to assuming Q starts at global 0), so the mask
        # is wrong for rank>0 chunks (G4) -> replace it with an explicit lower-
        # triangular mask offset by this rank's global Q offset lo (on rank0
        # lo=0 degenerates to the standard causal mask, same behavior).
        # Performance note: an explicit attn_mask rules out the SDPA flash
        # backend (falling back to mem_efficient/math); correctness of the
        # CP+causal path takes priority over this (05 §4.4.2).
        cp_rank = cp_mesh.get_local_rank()
        lo = cp_rank * q.shape[cp_dim]
        kwargs = dict(kwargs)
        kwargs.pop("is_causal")
        kwargs["attn_mask"] = _cp_offset_causal_mask(
            q.shape[cp_dim], global_k.shape[cp_dim], lo, q.device)
    return orig_sdpa(q, global_k, global_v, **kwargs)


# ────────────────────────────────────────────────────────────────────────────
# The four built-in wrappers
# ────────────────────────────────────────────────────────────────────────────

@inner_wrapper
def sdpa_qkv_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """NeMo/Megatron SDPA path (registry "sdpa_qkv"): explicit all-gather K/V.

    Assumes the inner_attention.forward(q,k,v,...) signature convention.
    **Local-only** (injection discipline): the dual-mode adapter converts
    DTensor inputs to local and re-wraps the output — this wrapper never
    touches DTensor. Since it wraps an inner submodule, the plan MUST
    declare ``inner_out_src: "first_input"`` (its output layout == q's
    layout). cp_mesh is framework-filled context; fail fast when the plan
    has no cp axis.
    """

    if cp_mesh is None:
        raise ValueError(
            "仓内 CP wrapper 参考实现 'sdpa_qkv' 需要活跃的 cp 轴（K/V all-gather 的通信"
            "域），但框架填入的 cp_mesh 为 None（当前 plan 无 cp 轴）——无 CP "
            "时请改用自定义 @inner_wrapper wrapper（cp_mesh 为 None 语义自"
            "负），或改用 local_compute_fn 通道（见 "
            "examples/distributed/perf_replacement.py）")

    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(q, k, v, **kwargs):
        return _cp_sdpa_call(original_forward, cp_mesh, q, k, v, kwargs)

    target_module.forward = cp_forward


@inner_wrapper
def flex_qkv_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """NeMo/Megatron FlexAttention path (registry "flex_qkv"): explicit all-gather K/V.

    **Local-only** (see sdpa_qkv_cp_wrapper); requires
    ``inner_out_src: "first_input"`` in the plan. Constraint: the
    block_mask must be built for the global kv length. cp_mesh is
    framework-filled context; fail fast when the plan has no cp axis.
    """

    if cp_mesh is None:
        raise ValueError(
            "仓内 CP wrapper 参考实现 'flex_qkv' 需要活跃的 cp 轴（K/V all-gather 的通信"
            "域），但框架填入的 cp_mesh 为 None（当前 plan 无 cp 轴）——无 CP "
            "时请改用自定义 @inner_wrapper wrapper（cp_mesh 为 None 语义自"
            "负），或改用 local_compute_fn 通道（见 "
            "examples/distributed/perf_replacement.py）")

    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(q, k, v, **kwargs):
        global_k, global_v = flex_cp_allgather(
            k.contiguous(), v.contiguous(), 2, cp_mesh)
        return original_forward(q, global_k, global_v, **kwargs)

    target_module.forward = cp_forward


@inner_wrapper
def sdpa_hf_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """HF standard SDPA path: forward(hidden_states,...) -> primitive interception (05 §4.4.2).

    **Local-only** (see sdpa_qkv_cp_wrapper): the adapter handles DTensor
    unwrap/param-unwrap/rewrap; the rewrap uses the boundary's declared
    out_src (this wrapper targets the boundary module itself). The
    primitive interception is a temporary global function replacement
    (restored via try/finally) and is not thread-safe; it is safe under
    single-process SPMD training (consistent with the TorchTitan CP
    implementation).
    Misfire detection: the wrapper may be applied to a module that does not
    call F.sdpa -- if the primitive did not intercept a single call, the K/V
    were not gathered (a silent numerical error), so raise a RuntimeError
    immediately.
    """

    if cp_mesh is None:
        raise ValueError(
            "仓内 CP wrapper 参考实现 'sdpa_hf' 需要活跃的 cp 轴（K/V all-gather 的通信"
            "域），但框架填入的 cp_mesh 为 None（当前 plan 无 cp 轴）——无 CP "
            "时请改用自定义 @inner_wrapper wrapper（cp_mesh 为 None 语义自"
            "负），或改用 local_compute_fn 通道（见 "
            "examples/distributed/perf_replacement.py）")

    original_forward = target_module.forward
    orig_sdpa = F.scaled_dot_product_attention

    @functools.wraps(original_forward)
    def cp_forward(hidden_states, *args, **kwargs):
        fired = {"hit": False}

        def cp_aware_sdpa(q, k, v, **kw):
            fired["hit"] = True
            return _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kw)

        F.scaled_dot_product_attention = cp_aware_sdpa
        try:
            out = original_forward(hidden_states, *args, **kwargs)
        finally:
            F.scaled_dot_product_attention = orig_sdpa
        if not fired["hit"]:
            raise RuntimeError(
                f"CP wrapper 'sdpa_hf' did not intercept any "
                f"F.scaled_dot_product_attention call on "
                f"{type(target_module).__name__} -- the wrapper does "
                f"not match the module implementation (K/V were not "
                f"all-gathered; continuing would produce silent numerical "
                f"errors). Please explicitly set inner_wrapper='sdpa_qkv' "
                f"(the (q,k,v) convention), or provide a custom inner_wrapper "
                f"callable")
        return out

    target_module.forward = cp_forward


@inner_wrapper
def flex_hf_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """HF standard FlexAttention path: intercept flex_attention (same structure as the SDPA path).

    **Local-only** (see sdpa_qkv_cp_wrapper). Constraint:
    score_mod/block_mask pass through verbatim via kwargs -- under CP,
    kv_len changes from S/cp to S, so the block_mask must be built for the
    **global kv length** (constructed on the full sequence in the data
    pipeline / model side), otherwise shapes and semantics are misaligned.
    The wrapper does not validate this.
    Misfire detection is the same as 'sdpa_hf': if no flex_attention call is
    intercepted, raise a RuntimeError.
    """

    if cp_mesh is None:
        raise ValueError(
            "仓内 CP wrapper 参考实现 'flex_hf' 需要活跃的 cp 轴（K/V all-gather 的通信"
            "域），但框架填入的 cp_mesh 为 None（当前 plan 无 cp 轴）——无 CP "
            "时请改用自定义 @inner_wrapper wrapper（cp_mesh 为 None 语义自"
            "负），或改用 local_compute_fn 通道（见 "
            "examples/distributed/perf_replacement.py）")

    original_forward = target_module.forward
    from torch.nn.attention.flex_attention import flex_attention as _orig_flex

    @functools.wraps(original_forward)
    def cp_forward(hidden_states, *args, **kwargs):
        import torch.nn.attention.flex_attention as _flex_mod
        fired = {"hit": False}

        def cp_aware_flex(q, k, v, **kw):
            fired["hit"] = True
            global_k, global_v = flex_cp_allgather(
                k.contiguous(), v.contiguous(), 2, cp_mesh)
            return _orig_flex(q, global_k, global_v, **kw)

        _flex_mod.flex_attention = cp_aware_flex
        try:
            out = original_forward(hidden_states, *args, **kwargs)
        finally:
            _flex_mod.flex_attention = _orig_flex
        if not fired["hit"]:
            raise RuntimeError(
                f"CP wrapper 'flex_hf' did not intercept any flex_attention "
                f"call on {type(target_module).__name__} -- the wrapper "
                f"does not match the module implementation (K/V were "
                f"not all-gathered; continuing would produce silent numerical "
                f"errors). Please explicitly set inner_wrapper='flex_qkv' "
                f"(the (q,k,v) convention), or provide a custom inner_wrapper "
                f"callable")
        return out

    target_module.forward = cp_forward


# {registry_name: wrapper_fn} -- inner-wrapper 命名注册表（05 §4.4.2）。
# 机制不 CP 门控（声明即应用）；四个仓内参考实现是 CP 语义（自检要求活跃 cp
# 轴），用户可注册任意命名方案。
# Contract: @inner_wrapper fn(target_module, mesh, tp_mesh, cp_mesh,
# ep_mesh) replaces target_module.forward in place (K/V all-gather
# + dual-mode tolerance). Users may register their own named schemes:
# INNER_WRAPPER_REGISTRY["my_flash"] = my_fn, after which
# spec.inner_wrapper="my_flash" references it by name.
INNER_WRAPPER_REGISTRY = {
    "sdpa_qkv": sdpa_qkv_cp_wrapper,  # NeMo convention forward(q,k,v,...) + D-04 mask
    "sdpa_hf": sdpa_hf_cp_wrapper,    # HF convention + F.sdpa primitive interception (misfire detection)
    "flex_qkv": flex_qkv_cp_wrapper,
    "flex_hf": flex_hf_cp_wrapper,
}

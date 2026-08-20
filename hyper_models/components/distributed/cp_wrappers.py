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
automatically: the built-ins below are referenced explicitly — by
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
    fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh, <configured params...>) -> None

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
- ``*_ulysses_cp_wrapper``: Pure Ulysses variants that exchange Q/K/V from
  sequence shards to head shards before attention and restore the local
  sequence shard afterward.
- ``*_hybrid_cp_wrapper``: local-tensor Hybrid variants that run Ulysses
  all-to-all inside each subgroup and K/V all-gather across the complementary
  Colossal subgroup.
- ``sdpa_*_load_balance_cp_wrapper``: local-tensor Colossal Head-Tail variants
  for Q exchange, K/V all-gather, dual SDPA execution, output restoration,
  and backward communication.

Users may register their own named schemes::

    INNER_WRAPPER_REGISTRY["my_flash"] = my_fn

after which ``spec.inner_wrapper="my_flash"`` references it by name.

The ``is_*_attention`` helpers are public utilities for authors of custom
wrappers (choosing a scheme programmatically); the framework itself no
longer dispatches heuristically on them.
"""

import functools
import importlib
import inspect
import logging
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F

from hyper_parallel.platform import get_platform
from hyper_models.components.distributed.cp_utils import (
    _build_hybrid_cp_submeshes,
    _ULYSSES_WRAPPED_FLAG,
    _UlyssesContext,
    _cp_offset_causal_mask,
    _dsa_cp_alltoall,
    _mla_cp_alltoall,
    _mome_cp_halo_exchange,
    _slice_sequence,
    async_cp_allgather_launch,
    async_ulysses_seq_to_head_launch,
    flex_cp_allgather,
    head_tail_load_balance_attention,
    hybrid_cp_attention,
    ulysses_head_to_seq,
    ulysses_seq_to_head,
)
from hyper_models.components.distributed.injection import (
    inner_wrapper,
)
from hyper_models.components.models.qwen3_moe_fusions import (
    _fused_rms_norm,
    qwen3_moe_flash_attention_forward,
)

logger = logging.getLogger(__name__)
platform = get_platform()
Module = platform.Module


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


def _ulysses_attention_call(
        attention_fn, cp_mesh, query, key, value, kwargs,
        *, seq_dim=2, head_dim=1):
    """Run attention in the Pure Ulysses full-sequence/head-sharded layout."""
    query, key, value = (
        ulysses_seq_to_head(tensor, seq_dim, head_dim, cp_mesh)
        for tensor in (query, key, value)
    )
    output = attention_fn(query, key, value, **kwargs)
    return ulysses_head_to_seq(output, seq_dim, head_dim, cp_mesh)


def _ulysses_qkv_forward(original_forward, cp_mesh, args, kwargs):
    """Preserve a QKV forward signature while applying Pure Ulysses."""
    signature = inspect.signature(original_forward)
    bound = signature.bind(*args, **kwargs)
    names = list(signature.parameters)
    if len(names) < 3:
        raise TypeError("Ulysses QKV wrapper requires at least three forward inputs")
    try:
        query, key, value = (
            bound.arguments[names[index]] for index in range(3)
        )
    except KeyError as exc:
        raise TypeError(
            "Ulysses QKV wrapper requires query, key, and value inputs"
        ) from exc
    query, key, value = (
        ulysses_seq_to_head(tensor, 2, 1, cp_mesh)
        for tensor in (query, key, value)
    )
    bound.arguments[names[0]] = query
    bound.arguments[names[1]] = key
    bound.arguments[names[2]] = value
    output = original_forward(*bound.args, **bound.kwargs)
    return ulysses_head_to_seq(output, 2, 1, cp_mesh)


def _require_ulysses_cp_mesh(cp_mesh, wrapper_name):
    """Validate the communication context required by Pure Ulysses."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(
            f"Ulysses wrapper {wrapper_name!r} requires an active CP mesh"
        )


def _normalize_hf_sdpa_gqa(
        query: Any, key: Any, value: Any,
        call_kwargs: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Give every CP rank the same explicit KV-head layout before gather.

    HF may keep compact GQA K/V when no mask is present but expand K/V on a
    rank with an explicit padding mask.  CP communication starts after that
    decision, so the rank-local layouts must be normalized first.
    """
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    value_heads = value.shape[1]
    if key_heads != value_heads:
        raise ValueError(
            "HF CP SDPA requires matching K/V head counts, got "
            f"key_heads={key_heads}, value_heads={value_heads}"
        )
    if query_heads % key_heads:
        raise ValueError(
            "HF CP SDPA requires Q heads divisible by KV heads, "
            f"got query_heads={query_heads}, key_heads={key_heads}"
        )
    if query_heads != key_heads:
        groups = query_heads // key_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    normalized_kwargs = dict(call_kwargs)
    normalized_kwargs.pop("enable_gqa", None)
    return query, key, value, normalized_kwargs


def _bind_qkv_invocation(
        original_forward: Callable[..., Any], args: tuple[Any, ...],
        kwargs: dict[str, Any], wrapper_name: str):
    """Bind a QKV call once and return a local-tensor attention callback."""
    signature = inspect.signature(original_forward)
    parameter_names = tuple(signature.parameters)
    if len(parameter_names) < 3:
        raise TypeError(
            f"CP wrapper {wrapper_name!r} requires at least three forward inputs"
        )
    qkv_names = parameter_names[:3]
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    try:
        query, key, value = (
            bound.arguments[name] for name in qkv_names
        )
    except KeyError as exc:
        raise TypeError(
            f"CP wrapper {wrapper_name!r} requires query, key, and value inputs"
        ) from exc

    var_keyword_name = next(
        (
            name for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )
    attention_kwargs = {}
    for name, parameter in signature.parameters.items():
        if name in qkv_names or name not in bound.arguments:
            continue
        value_item = bound.arguments[name]
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            attention_kwargs.update(value_item)
        elif parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
            attention_kwargs[name] = value_item

    base_arguments = dict(bound.arguments)
    if var_keyword_name is not None:
        base_arguments[var_keyword_name] = dict(
            base_arguments.get(var_keyword_name, {})
        )

    def attention_call(
            call_query: Any, call_key: Any, call_value: Any,
            call_kwargs: dict[str, Any]) -> Any:
        """Invoke the original forward without duplicating bound Q/K/V args."""
        call_arguments = dict(base_arguments)
        if var_keyword_name is not None:
            call_arguments[var_keyword_name] = dict(
                base_arguments[var_keyword_name]
            )
        call_arguments[qkv_names[0]] = call_query
        call_arguments[qkv_names[1]] = call_key
        call_arguments[qkv_names[2]] = call_value
        for name, item in call_kwargs.items():
            parameter = signature.parameters.get(name)
            if (parameter is not None
                    and parameter.kind is not inspect.Parameter.VAR_KEYWORD):
                call_arguments[name] = item
            elif var_keyword_name is not None:
                call_arguments[var_keyword_name][name] = item
            else:
                raise TypeError(
                    f"CP wrapper {wrapper_name!r} needs to pass attention "
                    f"argument {name!r}, but the original forward does not "
                    "accept it"
                )
        call_bound = inspect.BoundArguments(signature, call_arguments)
        return original_forward(*call_bound.args, **call_bound.kwargs)

    return query, key, value, attention_kwargs, attention_call


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

    del mesh, tp_mesh, ep_mesh
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

    del mesh, tp_mesh, ep_mesh
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

    del mesh, tp_mesh, ep_mesh
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

    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None:
        raise ValueError(
            "仓内 CP wrapper 参考实现 'flex_hf' 需要活跃的 cp 轴（K/V all-gather 的通信"
            "域），但框架填入的 cp_mesh 为 None（当前 plan 无 cp 轴）——无 CP "
            "时请改用自定义 @inner_wrapper wrapper（cp_mesh 为 None 语义自"
            "负），或改用 local_compute_fn 通道（见 "
            "examples/distributed/perf_replacement.py）")

    original_forward = target_module.forward
    # FlexAttention is optional on older supported PyTorch versions.
    flex_attention_module = importlib.import_module(
        "torch.nn.attention.flex_attention")
    original_flex_attention = flex_attention_module.flex_attention

    @functools.wraps(original_forward)
    def cp_forward(hidden_states, *args, **kwargs):
        fired = {"hit": False}

        def cp_aware_flex(q, k, v, **kw):
            fired["hit"] = True
            global_k, global_v = flex_cp_allgather(
                k.contiguous(), v.contiguous(), 2, cp_mesh)
            return original_flex_attention(q, global_k, global_v, **kw)

        flex_attention_module.flex_attention = cp_aware_flex
        try:
            out = original_forward(hidden_states, *args, **kwargs)
        finally:
            flex_attention_module.flex_attention = original_flex_attention
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


# ────────────────────────────────────────────────────────────────────────────
# Colossal Head-Tail load-balance wrappers (explicit plan_overrides methods)
# ────────────────────────────────────────────────────────────────────────────

@inner_wrapper
def sdpa_qkv_load_balance_cp_wrapper(
        target_module: Module, mesh: Any, tp_mesh: Any,
        cp_mesh: Any, ep_mesh: Any) -> None:
    """Apply Colossal Head-Tail load balancing to a QKV-style SDPA module.

    Args:
        target_module: Attention module whose first three inputs are Q/K/V.
        mesh: Framework-owned root mesh; unused by this wrapper.
        tp_mesh: Framework-owned TP mesh; unused by this wrapper.
        cp_mesh: Framework-owned CP communication mesh.
        ep_mesh: Framework-owned EP mesh; unused by this wrapper.
    """
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(
            "sdpa_qkv_load_balance_cp_wrapper requires an active CP mesh"
        )
    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Route local Q/K/V tensors through Head-Tail communication."""
        query, key, value, call_kwargs, attention_call = (
            _bind_qkv_invocation(
                original_forward,
                args,
                kwargs,
                "sdpa_qkv_load_balance",
            )
        )
        return head_tail_load_balance_attention(
            attention_call,
            query,
            key,
            value,
            call_kwargs,
            cp_mesh,
        )

    target_module.forward = cp_forward


@inner_wrapper
def sdpa_hf_load_balance_cp_wrapper(
        target_module: Module, mesh: Any, tp_mesh: Any,
        cp_mesh: Any, ep_mesh: Any) -> None:
    """Apply Colossal Head-Tail load balancing at an HF SDPA primitive.

    Args:
        target_module: HF-style attention containing an SDPA primitive call.
        mesh: Framework-owned root mesh; unused by this wrapper.
        tp_mesh: Framework-owned TP mesh; unused by this wrapper.
        cp_mesh: Framework-owned CP communication mesh.
        ep_mesh: Framework-owned EP mesh; unused by this wrapper.
    """
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(
            "sdpa_hf_load_balance_cp_wrapper requires an active CP mesh"
        )
    original_forward = target_module.forward
    original_sdpa = F.scaled_dot_product_attention

    @functools.wraps(original_forward)
    def cp_forward(
            hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept the module's SDPA primitive for one forward call."""
        fired = {"hit": False}

        def load_balanced_sdpa(
                q: Any, k: Any, v: Any,
                **call_kwargs: Any) -> Any:
            """Run one intercepted SDPA call with Head-Tail communication."""
            fired["hit"] = True
            q, k, v, call_kwargs = _normalize_hf_sdpa_gqa(
                q, k, v, call_kwargs
            )
            return head_tail_load_balance_attention(
                lambda query, key, value, attention_kwargs: original_sdpa(
                    query, key, value, **attention_kwargs
                ),
                q,
                k,
                v,
                call_kwargs,
                cp_mesh,
            )

        F.scaled_dot_product_attention = load_balanced_sdpa
        try:
            output = original_forward(hidden_states, *args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa
        if not fired["hit"]:
            raise RuntimeError(
                "CP wrapper 'sdpa_hf_load_balance' did not intercept any "
                "F.scaled_dot_product_attention call on "
                f"{type(target_module).__name__}; choose the QKV wrapper or "
                "provide a matching custom inner_wrapper"
            )
        return output

    target_module.forward = cp_forward


# ────────────────────────────────────────────────────────────────────────────
# Pure Ulysses attention wrappers (explicit plan_overrides methods)
# ────────────────────────────────────────────────────────────────────────────

@inner_wrapper
def sdpa_qkv_ulysses_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Apply Pure Ulysses to a separated ``forward(query, key, value, ...)``."""
    del mesh, tp_mesh, ep_mesh
    _require_ulysses_cp_mesh(cp_mesh, "sdpa_qkv_ulysses")
    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(*args, **kwargs):
        return _ulysses_qkv_forward(original_forward, cp_mesh, args, kwargs)

    target_module.forward = cp_forward


@inner_wrapper
def flex_qkv_ulysses_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Apply Pure Ulysses to a separated FlexAttention QKV forward."""
    del mesh, tp_mesh, ep_mesh
    _require_ulysses_cp_mesh(cp_mesh, "flex_qkv_ulysses")
    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(*args, **kwargs):
        return _ulysses_qkv_forward(original_forward, cp_mesh, args, kwargs)

    target_module.forward = cp_forward


@inner_wrapper
def sdpa_hf_ulysses_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Apply Pure Ulysses by intercepting HF SDPA primitive calls."""
    del mesh, tp_mesh, ep_mesh
    _require_ulysses_cp_mesh(cp_mesh, "sdpa_hf_ulysses")
    original_forward = target_module.forward
    original_sdpa = F.scaled_dot_product_attention

    @functools.wraps(original_forward)
    def cp_forward(*args, **kwargs):
        fired = {"hit": False}

        def ulysses_sdpa(query, key, value, **attention_kwargs):
            fired["hit"] = True
            return _ulysses_attention_call(
                original_sdpa,
                cp_mesh,
                query,
                key,
                value,
                attention_kwargs,
            )

        F.scaled_dot_product_attention = ulysses_sdpa
        try:
            output = original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa
        if not fired["hit"]:
            raise RuntimeError(
                "CP wrapper 'sdpa_hf_ulysses' did not intercept any "
                "F.scaled_dot_product_attention call; the selected wrapper "
                "does not match the module implementation"
            )
        return output

    target_module.forward = cp_forward


@inner_wrapper
def flex_hf_ulysses_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Apply Pure Ulysses by intercepting HF FlexAttention primitive calls."""
    del mesh, tp_mesh, ep_mesh
    _require_ulysses_cp_mesh(cp_mesh, "flex_hf_ulysses")
    original_forward = target_module.forward
    flex_attention_module = importlib.import_module(
        "torch.nn.attention.flex_attention"
    )
    original_flex_attention = flex_attention_module.flex_attention

    @functools.wraps(original_forward)
    def cp_forward(*args, **kwargs):
        fired = {"hit": False}

        def ulysses_flex(query, key, value, **attention_kwargs):
            fired["hit"] = True
            return _ulysses_attention_call(
                original_flex_attention,
                cp_mesh,
                query,
                key,
                value,
                attention_kwargs,
            )

        flex_attention_module.flex_attention = ulysses_flex
        try:
            output = original_forward(*args, **kwargs)
        finally:
            flex_attention_module.flex_attention = original_flex_attention
        if not fired["hit"]:
            raise RuntimeError(
                "CP wrapper 'flex_hf_ulysses' did not intercept any "
                "flex_attention call; the selected wrapper does not match "
                "the module implementation"
            )
        return output

    target_module.forward = cp_forward


# ────────────────────────────────────────────────────────────────────────────
# Hybrid attention wrappers (explicit Target with wrapper-local degree)
# ────────────────────────────────────────────────────────────────────────────

def _validate_hybrid_config(
        cp_mesh: Any, ulysses_degree: int, wrapper_name: str) -> None:
    """Validate Hybrid topology at wrapper installation time."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(
            f"Hybrid wrapper {wrapper_name!r} requires an active CP mesh"
        )
    if isinstance(ulysses_degree, bool) or not isinstance(ulysses_degree, int):
        raise TypeError(
            f"Hybrid wrapper {wrapper_name!r} requires integer "
            f"ulysses_degree, got {type(ulysses_degree).__name__}"
        )
    cp_size = cp_mesh.size()
    if not 1 < ulysses_degree < cp_size:
        raise ValueError(
            f"Hybrid wrapper {wrapper_name!r} requires 1 < ulysses_degree "
            f"< cp_size, got ulysses_degree={ulysses_degree}, cp_size={cp_size}"
        )
    if cp_size % ulysses_degree:
        raise ValueError(
            f"cp_size ({cp_size}) must be divisible by ulysses_degree "
            f"({ulysses_degree})"
        )


def _apply_qkv_hybrid_wrapper(
        target_module, cp_mesh, ulysses_degree, wrapper_name):
    """Replace a separated QKV forward with local-tensor Hybrid CP."""
    _validate_hybrid_config(cp_mesh, ulysses_degree, wrapper_name)
    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the original QKV forward through local-tensor Hybrid CP."""
        query, key, value, call_kwargs, attention_call = (
            _bind_qkv_invocation(
                original_forward,
                args,
                kwargs,
                wrapper_name,
            )
        )
        return hybrid_cp_attention(
            attention_call,
            query,
            key,
            value,
            call_kwargs,
            cp_mesh,
            ulysses_degree,
        )

    target_module.forward = cp_forward


@inner_wrapper
def sdpa_qkv_hybrid_cp_wrapper(
        target_module: Module, mesh: Any, tp_mesh: Any, cp_mesh: Any,
        ep_mesh: Any, ulysses_degree: int) -> None:
    """Apply Hybrid CP to a separated SDPA ``forward(query, key, value, ...)``."""
    del mesh, tp_mesh, ep_mesh
    _apply_qkv_hybrid_wrapper(
        target_module,
        cp_mesh,
        ulysses_degree,
        "sdpa_qkv_hybrid",
    )


@inner_wrapper
def flex_qkv_hybrid_cp_wrapper(
        target_module: Module, mesh: Any, tp_mesh: Any, cp_mesh: Any,
        ep_mesh: Any, ulysses_degree: int) -> None:
    """Apply Hybrid CP to a separated FlexAttention QKV forward."""
    del mesh, tp_mesh, ep_mesh
    _apply_qkv_hybrid_wrapper(
        target_module,
        cp_mesh,
        ulysses_degree,
        "flex_qkv_hybrid",
    )


@inner_wrapper
def sdpa_hf_hybrid_cp_wrapper(
        target_module: Module, mesh: Any, tp_mesh: Any, cp_mesh: Any,
        ep_mesh: Any, ulysses_degree: int) -> None:
    """Apply Hybrid CP by intercepting HF SDPA primitive calls."""
    del mesh, tp_mesh, ep_mesh
    _validate_hybrid_config(cp_mesh, ulysses_degree, "sdpa_hf_hybrid")
    original_forward = target_module.forward
    original_sdpa = F.scaled_dot_product_attention

    @functools.wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the HF attention forward with temporary Hybrid SDPA interception."""
        fired = {"hit": False}

        def hybrid_sdpa(
                query: Any, key: Any, value: Any,
                **attention_kwargs: Any) -> Any:
            """Route one intercepted SDPA call through Hybrid CP."""
            fired["hit"] = True
            query, key, value, attention_kwargs = _normalize_hf_sdpa_gqa(
                query, key, value, attention_kwargs
            )
            return hybrid_cp_attention(
                lambda call_query, call_key, call_value, call_kwargs: (
                    original_sdpa(
                        call_query, call_key, call_value, **call_kwargs
                    )
                ),
                query,
                key,
                value,
                attention_kwargs,
                cp_mesh,
                ulysses_degree,
            )

        F.scaled_dot_product_attention = hybrid_sdpa
        try:
            output = original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa
        if not fired["hit"]:
            raise RuntimeError(
                "CP wrapper 'sdpa_hf_hybrid' did not intercept any "
                "F.scaled_dot_product_attention call; the selected wrapper "
                "does not match the module implementation"
            )
        return output

    target_module.forward = cp_forward


@inner_wrapper
def flex_hf_hybrid_cp_wrapper(
        target_module: Module, mesh: Any, tp_mesh: Any, cp_mesh: Any,
        ep_mesh: Any, ulysses_degree: int) -> None:
    """Apply Hybrid CP by intercepting HF FlexAttention primitive calls."""
    del mesh, tp_mesh, ep_mesh
    _validate_hybrid_config(cp_mesh, ulysses_degree, "flex_hf_hybrid")
    original_forward = target_module.forward
    flex_attention_module = importlib.import_module(
        "torch.nn.attention.flex_attention"
    )
    original_flex_attention = flex_attention_module.flex_attention

    @functools.wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the HF attention forward with temporary Hybrid Flex interception."""
        fired = {"hit": False}

        def hybrid_flex(
                query: Any, key: Any, value: Any,
                **attention_kwargs: Any) -> Any:
            """Route one intercepted FlexAttention call through Hybrid CP."""
            fired["hit"] = True
            return hybrid_cp_attention(
                lambda call_query, call_key, call_value, call_kwargs: (
                    original_flex_attention(
                        call_query, call_key, call_value, **call_kwargs
                    )
                ),
                query,
                key,
                value,
                attention_kwargs,
                cp_mesh,
                ulysses_degree,
            )

        flex_attention_module.flex_attention = hybrid_flex
        try:
            output = original_forward(*args, **kwargs)
        finally:
            flex_attention_module.flex_attention = original_flex_attention
        if not fired["hit"]:
            raise RuntimeError(
                "CP wrapper 'flex_hf_hybrid' did not intercept any "
                "flex_attention call; the selected wrapper does not match "
                "the module implementation"
            )
        return output

    target_module.forward = cp_forward


# ────────────────────────────────────────────────────────────────────────────
# MLA/DSA Ulysses wrapper
# ────────────────────────────────────────────────────────────────────────────

def _input_cp_sharding(text_model, context):
    """Configure CP-rank input slicing on a text model."""
    if getattr(text_model, _ULYSSES_WRAPPED_FLAG, False):
        return
    original_forward = text_model.forward

    @functools.wraps(original_forward)
    def forward_with_sequence_sharding(*args, **kwargs):
        if getattr(text_model, "_hyper_cp_inside_forward", False):
            return original_forward(*args, **kwargs)
        call_kwargs = kwargs.copy()
        inputs_embeds = call_kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            return original_forward(*args, **kwargs)
        call_kwargs["inputs_embeds"] = _slice_sequence(inputs_embeds, 1, context)
        position_ids = call_kwargs.get("position_ids")
        if position_ids is not None:
            call_kwargs["position_ids"] = _slice_sequence(
                position_ids, position_ids.ndim - 1, context)
        mome_mask = call_kwargs.get("mome_mask")
        if mome_mask is not None:
            call_kwargs["mome_mask"] = _slice_sequence(mome_mask, 1, context)
        if call_kwargs.get("use_cache"):
            raise ValueError("Ulysses CP requires use_cache=False")
        call_kwargs["use_cache"] = False
        setattr(text_model, "_hyper_cp_inside_forward", True)
        try:
            return original_forward(*args, **call_kwargs)
        finally:
            setattr(text_model, "_hyper_cp_inside_forward", False)

    text_model.forward = forward_with_sequence_sharding
    setattr(text_model, _ULYSSES_WRAPPED_FLAG, True)


def _validate_ulysses_requirements(target_module, cp_size):
    """Validate the model configuration supported by Ulysses CP."""
    config = getattr(target_module, "config", None)
    if config is None:
        raise ValueError(
            "MLA/DSA Ulysses wrapper requires target_module.config")
    text_config = getattr(config, "text_config", config)
    if text_config is None:
        raise ValueError(
            "MLA/DSA Ulysses wrapper requires a non-None text_config")
    for name in ("num_attention_heads", "index_num_attention_heads"):
        value = getattr(text_config, name, None)
        if value is None:
            raise ValueError(
                "MLA/DSA Ulysses wrapper requires "
                f"text_config.{name}")
        count = int(value)
        if count % cp_size:
            raise ValueError(
                f"{name}={count} is not divisible by CP size {cp_size}")
    if getattr(text_config, "dsa_dense_warm_up", False):
        raise ValueError("MLA/DSA CP does not support DSA dense warm-up")
    if not getattr(text_config, "apply_FA_rescale", False):
        raise ValueError("MLA/DSA CP requires apply_FA_rescale=True")
    if getattr(text_config, "use_fused_sink_fa", False):
        raise ValueError("MLA/DSA CP does not support fused sink FA")


@inner_wrapper
def mla_dsa_ulysses_cp_wrapper(
        target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Configure input, MoME, MLA and DSA Ulysses adaptations."""
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(
            "MLA/DSA Ulysses wrapper requires an active CP mesh")
    if getattr(target_module, _ULYSSES_WRAPPED_FLAG, False):
        return
    _validate_ulysses_requirements(target_module, cp_mesh.size())
    context = _UlyssesContext(cp_mesh)

    text_models = [module for name, module in target_module.named_modules()
                   if name.rsplit(".", maxsplit=1)[-1] in {
                       "text_model", "language_model"}]
    if not text_models:
        raise RuntimeError("Cannot find a text or language model")
    for text_model in text_models:
        _input_cp_sharding(text_model, context)

    attention_modules = {
        inspect.getmodule(module) for module in target_module.modules()
        if getattr(module, "attention_type", None) in {"mla", "dsa"}}
    attention_modules.discard(None)
    if len(attention_modules) != 1:
        names = sorted(module.__name__ for module in attention_modules)
        raise RuntimeError(
            f"Expected one MLA/DSA attention module, found {names}")
    attention_module = next(iter(attention_modules))
    attention_registries = [
        value for value in vars(attention_module).values()
        if isinstance(value, dict)
        and {"npu_fa_rescale", "dsa_sparse_attention"} <= value.keys()]
    if len(attention_registries) != 1:
        raise RuntimeError(
            "Expected one attention-function registry containing MLA and DSA backends")
    attention_functions = attention_registries[0]
    _mome_cp_halo_exchange(attention_module, context)
    _mla_cp_alltoall(attention_functions, context)
    _dsa_cp_alltoall(attention_module, attention_functions, context)

    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def forward_with_ulysses_adapters(*args, **kwargs):
        return original_forward(*args, **kwargs)

    target_module.forward = forward_with_ulysses_adapters
    setattr(target_module, "_hyper_ulysses_context", context)
    setattr(target_module, _ULYSSES_WRAPPED_FLAG, True)


# ────────────────────────────────────────────────────────────────────────────
# Qwen3-MoE asynchronous CP wrappers
# ────────────────────────────────────────────────────────────────────────────

_QWEN3_MOE_SEQ_DIM = 2
_QWEN3_MOE_HEAD_DIM = 1


def _require_qwen3_moe_attention(module: Any) -> None:
    required = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "head_dim",
        "scaling",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise TypeError(
            "Qwen3-MoE async CP requires an attention module with attributes "
            f"{required}; missing {missing} on {type(module).__name__}"
        )


def _require_qwen3_moe_training_call(past_key_values: Any | None) -> None:
    if past_key_values is not None:
        raise ValueError(
            "Qwen3-MoE async CP currently supports training without KV cache; "
            "past_key_values must be None"
        )


def _qwen3_moe_position_terms(position_embeddings):
    cos, sin = position_embeddings
    return cos.unsqueeze(1), sin.unsqueeze(1)


def _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin):
    import torch_npu  # pylint: disable=C0415

    query = module.q_proj(hidden_states).view(hidden_shape)
    query = _fused_rms_norm(
        query,
        module.q_norm.weight,
        module.q_norm.variance_epsilon,
    ).transpose(1, 2)
    return torch_npu.npu_rotary_mul(query, cos, sin)


def _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin):
    import torch_npu  # pylint: disable=C0415

    key = module.k_proj(hidden_states).view(hidden_shape)
    key = _fused_rms_norm(
        key,
        module.k_norm.weight,
        module.k_norm.variance_epsilon,
    ).transpose(1, 2)
    return torch_npu.npu_rotary_mul(key, cos, sin)


def _qwen3_moe_project_value(module, hidden_states, hidden_shape):
    return module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


def _prepare_qwen3_moe_attention_mask(
    attention_mask: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
    query_offset: int,
) -> torch.Tensor | None:
    q_len = query.shape[_QWEN3_MOE_SEQ_DIM]
    kv_len = key.shape[_QWEN3_MOE_SEQ_DIM]
    if attention_mask is None:
        if q_len == kv_len and query_offset == 0:
            return None
        return _cp_offset_causal_mask(
            q_len,
            kv_len,
            query_offset,
            query.device,
        )
    if attention_mask.shape[-1] != kv_len:
        raise ValueError(
            "Qwen3-MoE CP attention_mask must cover the global KV sequence: "
            f"mask kv length={attention_mask.shape[-1]}, expected {kv_len}"
        )
    if attention_mask.ndim >= 2 and attention_mask.shape[-2] != q_len:
        if attention_mask.shape[-2] < query_offset + q_len:
            raise ValueError(
                "Qwen3-MoE CP attention_mask does not cover this rank's query "
                f"range [{query_offset}, {query_offset + q_len})"
            )
        attention_mask = attention_mask.narrow(-2, query_offset, q_len)
    return attention_mask


def _run_qwen3_moe_fused_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    kwargs,
):
    return qwen3_moe_flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=module.sliding_window,
        **kwargs,
    )


def _finish_qwen3_moe_attention(module, attention_output, input_shape):
    output = attention_output.reshape(*input_shape, -1).contiguous()
    return module.o_proj(output)


def _qwen3_moe_async_colossal_forward(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    *,
    cp_mesh: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-MoE with async K/V AllGather and local Q."""
    _require_qwen3_moe_training_call(past_key_values)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = _qwen3_moe_position_terms(position_embeddings)

    query = _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin)
    key = _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin)
    key_pending = async_cp_allgather_launch(key, _QWEN3_MOE_SEQ_DIM, cp_mesh)
    value = _qwen3_moe_project_value(module, hidden_states, hidden_shape)
    value_pending = async_cp_allgather_launch(value, _QWEN3_MOE_SEQ_DIM, cp_mesh)

    key = key_pending.wait()
    value = value_pending.wait()
    query_offset = cp_mesh.get_local_rank() * query.shape[_QWEN3_MOE_SEQ_DIM]
    attention_mask = _prepare_qwen3_moe_attention_mask(
        attention_mask, query, key, query_offset
    )
    attention_output, attention_weights = _run_qwen3_moe_fused_attention(
        module, query, key, value, attention_mask, kwargs
    )
    return _finish_qwen3_moe_attention(module, attention_output, input_shape), attention_weights


def _qwen3_moe_async_ulysses_forward(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    *,
    cp_mesh: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-MoE with async Q/K/V sequence-to-head A2A."""
    _require_qwen3_moe_training_call(past_key_values)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = _qwen3_moe_position_terms(position_embeddings)

    query = _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin)
    query_pending = async_ulysses_seq_to_head_launch(
        query, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )
    key = _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin)
    key_pending = async_ulysses_seq_to_head_launch(
        key, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )
    value = _qwen3_moe_project_value(module, hidden_states, hidden_shape)
    value_pending = async_ulysses_seq_to_head_launch(
        value, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )

    query = query_pending.wait()
    key = key_pending.wait()
    value = value_pending.wait()
    attention_mask = _prepare_qwen3_moe_attention_mask(
        attention_mask, query, key, query_offset=0
    )
    attention_output, attention_weights = _run_qwen3_moe_fused_attention(
        module, query, key, value, attention_mask, kwargs
    )
    output_bnsd = attention_output.transpose(1, 2).contiguous()
    output_bnsd = ulysses_head_to_seq(
        output_bnsd, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )
    attention_output = output_bnsd.transpose(1, 2).contiguous()
    return _finish_qwen3_moe_attention(module, attention_output, input_shape), attention_weights


def _qwen3_moe_async_hybrid_forward(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    *,
    cp_mesh: Any,
    ulysses_degree: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-MoE with async Ulysses A2A and Colossal K/V gather."""
    _require_qwen3_moe_training_call(past_key_values)
    ulysses_mesh, colossal_mesh = _build_hybrid_cp_submeshes(
        cp_mesh, ulysses_degree
    )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = _qwen3_moe_position_terms(position_embeddings)

    query = _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin)
    query_a2a = async_ulysses_seq_to_head_launch(
        query, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )
    key = _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin)
    key_a2a = async_ulysses_seq_to_head_launch(
        key, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )
    value = _qwen3_moe_project_value(module, hidden_states, hidden_shape)
    value_a2a = async_ulysses_seq_to_head_launch(
        value, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )

    key = key_a2a.wait()
    key_gather = async_cp_allgather_launch(
        key, _QWEN3_MOE_SEQ_DIM, colossal_mesh
    )
    value = value_a2a.wait()
    value_gather = async_cp_allgather_launch(
        value, _QWEN3_MOE_SEQ_DIM, colossal_mesh
    )
    query = query_a2a.wait()
    key = key_gather.wait()
    value = value_gather.wait()

    query_offset = colossal_mesh.get_local_rank() * query.shape[_QWEN3_MOE_SEQ_DIM]
    attention_mask = _prepare_qwen3_moe_attention_mask(
        attention_mask, query, key, query_offset
    )
    attention_output, attention_weights = _run_qwen3_moe_fused_attention(
        module, query, key, value, attention_mask, kwargs
    )
    output_bnsd = attention_output.transpose(1, 2).contiguous()
    output_bnsd = ulysses_head_to_seq(
        output_bnsd, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )
    attention_output = output_bnsd.transpose(1, 2).contiguous()
    return _finish_qwen3_moe_attention(module, attention_output, input_shape), attention_weights


def _validate_qwen3_moe_cp_mesh(cp_mesh, wrapper_name):
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")


def _validate_qwen3_moe_ulysses_heads(module, degree, wrapper_name):
    config = module.config
    for name in ("num_attention_heads", "num_key_value_heads"):
        count = getattr(config, name)
        if count % degree:
            raise ValueError(
                f"{wrapper_name} requires {name} ({count}) to be divisible "
                f"by Ulysses degree ({degree})"
            )


def _install_qwen3_moe_async_forward(
    target_module: Any,
    forward_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    **forward_config: Any,
) -> None:
    _require_qwen3_moe_attention(target_module)
    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dispatch the original attention signature to the async CP forward."""
        return forward_fn(
            target_module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **forward_config,
            **kwargs,
        )

    target_module.forward = cp_forward


@inner_wrapper
def qwen3_moe_async_colossal_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> None:
    """Install the model-specific async Colossal Qwen3-MoE forward."""
    del mesh, tp_mesh, ep_mesh
    _validate_qwen3_moe_cp_mesh(cp_mesh, "qwen3_moe_async_colossal")
    _install_qwen3_moe_async_forward(
        target_module,
        _qwen3_moe_async_colossal_forward,
        cp_mesh=cp_mesh,
    )


@inner_wrapper
def qwen3_moe_async_ulysses_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> None:
    """Install the model-specific async Pure Ulysses Qwen3-MoE forward."""
    del mesh, tp_mesh, ep_mesh
    _validate_qwen3_moe_cp_mesh(cp_mesh, "qwen3_moe_async_ulysses")
    _validate_qwen3_moe_ulysses_heads(
        target_module, cp_mesh.size(), "qwen3_moe_async_ulysses"
    )
    _install_qwen3_moe_async_forward(
        target_module,
        _qwen3_moe_async_ulysses_forward,
        cp_mesh=cp_mesh,
    )


@inner_wrapper
def qwen3_moe_async_hybrid_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    ulysses_degree: int,
) -> None:
    """Install the model-specific async Hybrid Qwen3-MoE forward."""
    del mesh, tp_mesh, ep_mesh
    _validate_qwen3_moe_cp_mesh(cp_mesh, "qwen3_moe_async_hybrid")
    _build_hybrid_cp_submeshes(cp_mesh, ulysses_degree)
    _validate_qwen3_moe_ulysses_heads(
        target_module, ulysses_degree, "qwen3_moe_async_hybrid"
    )
    _install_qwen3_moe_async_forward(
        target_module,
        _qwen3_moe_async_hybrid_forward,
        cp_mesh=cp_mesh,
        ulysses_degree=ulysses_degree,
    )


# {registry_name: wrapper_fn} -- inner-wrapper 命名注册表（05 §4.4.2）。
# 机制不 CP 门控（声明即应用）；仓内参考实现是 CP 语义（自检要求活跃 cp
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
    "sdpa_qkv_ulysses": sdpa_qkv_ulysses_cp_wrapper,
    "flex_qkv_ulysses": flex_qkv_ulysses_cp_wrapper,
    "sdpa_hf_ulysses": sdpa_hf_ulysses_cp_wrapper,
    "flex_hf_ulysses": flex_hf_ulysses_cp_wrapper,
    "mla_dsa_ulysses": mla_dsa_ulysses_cp_wrapper,
}

# Static requirements for shipped wrappers. Custom registry entries own their
# semantics; built-ins are known to contain CP collectives and therefore must
# run as black-box local regions during placement validation.
INNER_WRAPPER_REQUIREMENTS = {
    name: {
        "requires_cp": True,
        "region_dispatch": False,
        "forward_style": (
            "hf_hidden_states"
            if name.endswith("_hf") or "_hf_" in name
            else "qkv"
        ),
    }
    for name in INNER_WRAPPER_REGISTRY
}

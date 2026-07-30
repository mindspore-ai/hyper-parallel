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
"""sharding_applier: runtime application of ShardingPlan (05 §4 canonical).

apply_sharding_plan: Phase 0 normalization -> A parameter sharding -> B special
handlers -> C entry unpack + tp_grad_info -> C forward wrapping
(production/validate/moe/cp/vocab_embed, five paths) -> D tied weights.

Dual-mode architecture constraint (05 §1.4): production has zero DTensor
dispatch (build-time unpack + PrecompiledBoundary); the only difference between
validate and production is the boundary stitching method -- for any module whose
DTensor dispatch hides data-dependent logic (embedding mask / attention K/V
gather / MoE all-to-all), both modes explicitly reconstruct it with the same
local-region wrapper (D-01''/D-02/D-03').
"""

import functools
import inspect
import logging

import torch
import torch.nn as nn

from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import (
    Placement,
    Replicate,
    Shard,
)
from hyper_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
)
from hyper_models.components.distributed.injection import (
    INNER_WRAPPER,
    LOCAL_COMPUTE,
    fill_context_kwargs,
    require_injection_meta,
    validate_local_compute_signature,
    validate_wrapped_forward,
)
from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_models.components.distributed.sharding.apply import (
    _get_attr_by_path,
    _local_params_context,
    _resolve_module,
    _set_param_by_path,
    _stack_moe_experts,
    _temp_local_params,
)
from hyper_models.components.distributed.sharding_config import (
    PlacementMismatchError,
    _normalize_out_fields,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import SPECIAL_HANDLERS
from hyper_models.components.distributed.tp_grad import build_tp_grad_info
from hyper_models.components.distributed.head_count import (
    maybe_update_head_counts,
)

logger = logging.getLogger(__name__)


def _is_delayed_target(obj) -> bool:
    """Duck-typed check for a config Target (avoids a components -> trainer
    import): a Target carries ``build()`` and ``_target_`` and is itself NOT
    the compute/wrapper callable — it must be built at apply time."""
    return hasattr(obj, "build") and hasattr(obj, "_target_")


def _check_target_config_keys(target, kind):
    """Fail fast on configured Target kwargs the callable would never bind.

    Target kwargs are bound BY NAME at build time
    (``{**configured, **runtime}`` -> ``fn(**kwargs)``). When the target
    callable accepts VAR_KEYWORD (e.g. ``**_context``, there to tolerate the
    framework-filled generic context), a misspelled configured key is
    swallowed SILENTLY — the user's value never takes effect and the
    framework may even auto-fill the intended parameter instead. Guard: any
    configured key that is not an explicitly declared (keyword-bindable)
    parameter fails fast with the valid parameter names. A callable whose
    signature cannot be introspected skips the check (the call itself will
    surface any mismatch).
    """
    configured = getattr(target, "_kwargs", None)
    if not configured:
        return
    fn = getattr(target, "_target_", None)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return
    bindable = {
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = sorted(set(configured) - bindable)
    if unknown:
        raise ValueError(
            f"{kind} Target {getattr(target, '_target_path', fn)!r} 配置了未声明的"
            f"键 {unknown} —— Target 的 kwargs 按名绑定到目标函数的关键字形参，"
            f"这些键不在 {getattr(fn, '__name__', fn)} 的显式形参列表中，会被 "
            f"**kwargs 静默吞掉、不会生效（疑似拼写错误）。"
            f"合法形参: {sorted(bindable) or '(无)'}")


def _preflight_compute_injection(plan, mesh):
    """Fail-fast BEFORE any mutation: CP/EP sharding without an explicit
    compute injection is a silent numerical error (no auto-injection since
    the explicit-injection rework).

    - CP: an attention boundary (``_needs_cp_attn`` metadata from the
      template) under an active cp mesh needs ``inner_wrapper``;
    - EP: a TP-extend-EP boundary (``_ep_size > 0``, expert params already
      destined for ``{EP: Shard(0)}``) needs ``local_compute_fn`` — or an
      explicit ``region_dispatch=False`` when the module's own forward is
      EP-aware (a2a inside forward).
    """
    cp_mesh = _get_cp_submesh(mesh, plan.mesh_dim_names)
    if cp_mesh is not None and cp_mesh.size() > 1:
        for fqn, spec in plan.modules.items():
            if (spec.is_boundary and getattr(spec, "_needs_cp_attn", False)
                    and getattr(spec, "inner_wrapper", None) is None):
                raise ValueError(
                    f"cp_size={cp_mesh.size()} 已生效，attention 边界 {fqn!r} 需要 "
                    "CP-aware 内部 forward（K/V all-gather），但未声明 "
                    "inner_wrapper —— 框架不再启发式自动选择。请显式注入：\n"
                    "  plan_overrides:\n"
                    "    - match: \"*.self_attn\"\n"
                    "      when: cp\n"
                    "      region_dispatch: false   # wrapper 内含通信，不可 dispatch\n"
                    "      inner_wrapper:\n"
                    "        _target_: hyper_models.components.distributed."
                    "cp_wrappers.sdpa_hf_cp_wrapper\n"
                    f"（注册表 {sorted(INNER_WRAPPER_REGISTRY)} 可按 str 名引用；"
                    "NeMo 风格 (q,k,v) 签名用 sdpa_qkv，HF 风格 "
                    "forward(hidden_states) 用 sdpa_hf；或给 callable/"
                    "Target 自定义实现）")
    for fqn, spec in plan.modules.items():
        if (spec.is_boundary and getattr(spec, "_ep_size", 0)
                and getattr(spec, "local_compute_fn", None) is None
                and getattr(spec, "region_dispatch", None) is not False):
            raise ValueError(
                f"ep_size={spec._ep_size} 已生效（专家参数将按 {{EP: Shard(0)}} "
                f"分片），但边界 {fqn!r} 没有 local-region 计算来源 —— 专家计算与 "
                "all-to-all 无人执行，框架不再自动注入任何实现。请选择其一：\n"
                "  ① HF 原生 MoE → 注入仓内参考实现：\n"
                "     plan_overrides:\n"
                "       - match: \"*.mlp\"\n"
                "         when: ep\n"
                "         region_dispatch: false   # a2a 在区域内，不可 dispatch\n"
                "         local_compute_fn:\n"
                "           _target_: hyper_models.components.distributed."
                "ep_compute.hf_native_ep_compute_fn\n"
                "  ② 自研 EP-aware MoE（forward 内已含 all-to-all）→ 声明 "
                "region_dispatch: false\n"
                "  ③ 自定义 compute → local_compute_fn 指向 "
                "fn(module, *local_args)（callable 或工厂 Target）+ 显式 "
                "region_dispatch")


def _require_region_dispatch(spec, *, source):
    """注入纪律：声明注入（local_compute_fn / inner_wrapper）必须显式给出
    region_dispatch（无默认——可 dispatch 的纯算子注入传 True，含通信/
    自定义 kernel 的传 False；教程与示例逐一说明原因）。"""
    rd = getattr(spec, "region_dispatch", None)
    has_injection = (getattr(spec, "local_compute_fn", None) is not None
                     or getattr(spec, "inner_wrapper", None) is not None)
    if rd is None:
        if has_injection:
            raise ValueError(
                f"{source}: 声明了注入但 region_dispatch 未显式声明（无默认值）"
                "——注入物是纯标准算子、validate 可 dispatch 穿透（融合算子/"
                "脚本写法优化）→ region_dispatch=True（区域内策略传播 + "
                "out_src 真校验）；注入物含通信原语/自定义 kernel（CP K/V "
                "all-gather、EP all-to-all、量化 GEMM 等）→ "
                "region_dispatch=False（骨架/适配器黑盒托管 local 执行 + "
                "声明式重包）")
        return
    if rd is True and not has_injection:
        raise ValueError(
            f"{source}: region_dispatch=True 但未声明任何注入——普通边界的 "
            "forward 天然 dispatch 穿透（公理缺省），该声明是冗余的，请删除")


def _log_injection_choice(module_fqn, spec):
    """可观察性补强：注入边界的选择结果即时可见（一行 INFO/边界），形成
    "声明 → 看到后果"的反馈闭环。必须在 _require_region_dispatch 之后调用
    （此处注入 + region_dispatch 组合已合法）。"""
    rd = getattr(spec, "region_dispatch", None)
    has_fn = getattr(spec, "local_compute_fn", None) is not None
    has_wrap = getattr(spec, "inner_wrapper", None) is not None
    if not (has_fn or has_wrap or rd is not None):
        return   # 普通边界：无注入、公理缺省穿透，不刷屏
    what = "+".join(
        [x for x, ok in (("local_compute_fn", has_fn),
                         ("inner_wrapper", has_wrap)) if ok]
    ) or "模块自身 forward（未注入 fn，region_dispatch=False 声明）"
    if rd is True:
        effect = "validate 穿透真校验已启用（区域内策略传播 + out_src 真校验）"
    else:  # False（None+注入已被 _require_region_dispatch 拦截）
        effect = "黑盒托管（区域内 local 执行、跳过传播校验、声明式重包）"
    logger.info(
        "boundary %s: 注入[%s], region_dispatch=%s → %s",
        module_fqn, what, rd, effect)


# ────────────────────────────────────────────────────────────────────────────
# Main entry (05 §4.1)
# ────────────────────────────────────────────────────────────────────────────

def apply_sharding_plan(model, plan, mesh, *, validate_mode=False):
    """Apply a ShardingPlan to any nn.Module (or a list of PP parts), enabling dual-mode DTensor.

    Returns (model, tp_grad_info):
    - production: at the Phase C entry, a one-shot `_local_params_context` permanently
      unwraps DTensor parameters into plain local tensors, and builds tp_grad_info
      for fully_shard to use;
    - validate: no unwrap (parameters remain DTensors); tp_grad_info is None.
    """
    mesh_dim_names = plan.mesh_dim_names
    # Active sub-mesh: the planner strips size=1 axes (plan.mesh_dim_names), but the
    # passed-in mesh may still contain those axes -- placements are resolved against
    # plan.mesh_dim_names, so the dimensionality must align with the mesh, otherwise
    # distribute_tensor will silently shard along the wrong axis.
    full_mesh = mesh   # D-10: deriving the expert mesh requires the full dense region (including dp/cp axes)
    mesh = _get_active_mesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    models = model if isinstance(model, list) else [model]

    # Explicit-injection guard: CP/EP sharding without an explicit compute
    # injection fails fast here, BEFORE any parameter is touched
    _preflight_compute_injection(plan, mesh)

    # D-10: when any spec enables TP-extend-EP, derive the expert mesh
    # (repartition of the full dense region; 05 §6.4.8) ONCE — it is used both
    # for expert parameter sharding (Phase A) and as the ep_mesh context of
    # injected factories/wrappers (Phase C), so the a2a communication domain
    # and the sharding domain are the same object by construction. The
    # derivation is logged (nothing filled silently).
    ep_size = next((getattr(s, "_ep_size", 0) for s in plan.modules.values()
                    if getattr(s, "_ep_size", 0)), 0)
    expert_mesh = None
    if ep_size:
        expert_mesh = _build_expert_mesh(
            full_mesh, full_mesh.mesh_dim_names, ep_size)
        logger.info(
            "expert mesh: framework derived %s from the dense region "
            "(ep_size=%d; shared by parameter sharding and injected compute)",
            dict(zip(tuple(expert_mesh.mesh_dim_names),
                     tuple(expert_mesh.mesh_shape))), ep_size)

    # ====== Phase 0: normalize out_src/out_dst scalar shorthand (idempotent, covers user-injected paths) ======
    for spec in plan.modules.values():
        _normalize_out_fields(spec)

    # ====== Phase A: parameter sharding ======
    for part in models:
        for module_fqn, spec in plan.modules.items():
            module = _resolve_module(part, module_fqn)
            # D-09b: HF native MoE per-expert parameters are first stacked into
            # [E, ...], then sharded as stacked entries (05 §6.4.7)
            if getattr(spec, "_ep_stack", None):
                _stack_moe_experts(module, spec._ep_stack)
            if getattr(spec, "_ep_size", 0):
                # D-10: expert parameters are sharded on the derived expert mesh
                # ({EP: Shard(0)}, only the expert dim is split); all other
                # parameters go through the main mesh
                expert_params = {k: v for k, v in spec.params.items()
                                 if k.startswith("experts.")}
                dense_params = {k: v for k, v in spec.params.items()
                                if not k.startswith("experts.")}
                _shard_module_params(module, expert_params, expert_mesh,
                                     expert_mesh.mesh_dim_names)
                _shard_module_params(module, dense_params, mesh, mesh_dim_names)
            else:
                _shard_module_params(module, spec.params, mesh, mesh_dim_names)
            # D-17: production forwards run on permanently unwrapped local
            # tensors -- rewrite cached head counts to the TP-local value so
            # modeling code that reshapes with an explicit (global) num_heads
            # keeps working. Validate keeps the global counts here: boundary
            # modules run DTensor dispatch on the global logical shape.
            if not validate_mode:
                maybe_update_head_counts(
                    module, spec, module_fqn, mesh, mesh_dim_names)

    # ====== Phase B: special handlers ======
    for part in models:
        for param_ref, handler_name in plan.special_handlers.items():
            handler = SPECIAL_HANDLERS.get(handler_name)
            if handler is None:
                logger.warning("SPECIAL_HANDLERS has no registered handler: %s", handler_name)
                continue
            module_fqn, param_name = param_ref.rsplit(".", 1)
            handler(_resolve_module(part, module_fqn), param_name, mesh)

    # ====== Phase C entry: one-shot unpack at build time (production only) ======
    tp_grad_info = None
    if not validate_mode:
        tp_grad_records = {}
        for part in models:
            tp_grad_records.update(_local_params_context(part))
        if tp_grad_records and tp_mesh is not None:
            tp_grad_info = build_tp_grad_info(plan, tp_mesh)

    # ====== Phase C: wrap forward ======
    for part in models:
        _apply_phase_c(part, plan, mesh, validate_mode, expert_mesh=expert_mesh)

    # ====== Phase D: tied weights ======
    tied_pairs = list(plan.tied_pairs) or detect_tied_weights(models[0])
    for part in models:
        _replicate_tied_weights(part, mesh, tied_pairs)

    return model, tp_grad_info


def _get_active_mesh(mesh, mesh_dim_names):
    """Return the active sub-mesh aligned with plan.mesh_dim_names (the dimension set after stripping size=1 axes)."""
    names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
    if names == tuple(mesh_dim_names):
        return mesh
    if mesh_dim_names and names and all(n in names for n in mesh_dim_names):
        return mesh[tuple(mesh_dim_names)]
    return mesh


def _get_tp_submesh(mesh, mesh_dim_names):
    if "tp" not in mesh_dim_names:
        return None
    return mesh["tp"]


def _get_cp_submesh(mesh, mesh_dim_names):
    if "cp" not in mesh_dim_names:
        return None
    return mesh["cp"]


def _get_ep_submesh(mesh, mesh_dim_names):
    if "ep" not in mesh_dim_names:
        return None
    return mesh["ep"]


def _expert_mesh_layout(mesh, mesh_dim_names, ep_size):
    """(shape, dim_names, rank_list) of the derived expert mesh (pure mapping; no process group is created).

    D-10 TP-extend-EP (05 §6.4.8 / 06 §4.5.1): the expert domain = the full
    dense region (all ranks on the non-pp axes of the mesh, i.e.
    dp_replicate x dp_cp x tp). After a row-major flatten in mesh axis order,
    it is re-sliced as (edp = D/ep_size, ep = ep_size):
    - EP groups (inner, the a2a communication domain): ep_size consecutive
      ranks in flatten order -- tp is usually the innermost axis, so an EP
      group first spans the entire TP group and then extends to adjacent
      dp/cp ranks (isomorphic to MindSpeed TP-extend-EP / Megatron etp=1 with
      ep spanning TP).
      Example: mesh (dp=4, tp=2), ep_size=4 -> EP groups {0,1,2,3} / {4,5,6,7};
    - edp groups (outer): expert data-parallel degree = D/ep_size.
    Expert weights are only Shard(0) on the ep axis (the expert dim); there is
    no second-axis sharding.
    """
    import numpy as np

    if "pp" in mesh_dim_names:
        raise NotImplementedError(
            "D-10 TP-extend-EP v1 does not yet support the pp axis (call after splitting the mesh by stage)")
    arr = np.array(mesh.rank_list).reshape(mesh.mesh_shape)
    domain = int(np.prod(arr.shape))
    if ep_size <= 0 or domain % ep_size != 0:
        raise ValueError(
            f"ep_size ({ep_size}) must divide the dense region ({domain})"
        )
    edp = domain // ep_size
    derived = arr.reshape(edp, ep_size)
    return (edp, ep_size), ("edp", "ep"), tuple(
        int(r) for r in derived.flatten())


def _build_expert_mesh(mesh, mesh_dim_names, ep_size):
    """D-10 (05 §6.4.8 / 06 §4.5.1): repartition the full dense region into the derived expert mesh (edp, ep)."""
    from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

    shape, names, rank_list = _expert_mesh_layout(mesh, mesh_dim_names, ep_size)
    # Propagate the no-backend (metadata-only) mode of the source mesh —
    # a meta mesh has no _dim_group_names (no process groups were created).
    return init_device_mesh(mesh.device_type, shape, mesh_dim_names=names,
                            rank_list=rank_list,
                            init_backend=hasattr(mesh, "_dim_group_names"))


def build_expert_mesh(mesh, ep_size: int):
    """Public: derive the D-10 TP-extend-EP expert mesh (edp, ep) from *mesh*
    (the FULL mesh — the dense region must include dp/cp axes).

    Standalone helper for introspection and for custom code that needs the
    expert domain outside the injection path. Injected factories/wrappers do
    NOT need to call this: the framework derives the expert mesh once at
    apply time (shared by parameter sharding and injected compute) and hands
    it to them as the ``ep_mesh`` context.
    """
    return _build_expert_mesh(mesh, tuple(mesh.mesh_dim_names), ep_size)


# ────────────────────────────────────────────────────────────────────────────
# Phase A: parameter sharding (05 §4.2)
# ────────────────────────────────────────────────────────────────────────────

def _shard_module_params(module, param_specs, mesh, mesh_dim_names):
    """distribute_tensor() converts parameters into DTensors.

    - meta tensor -> DTensor: _local_tensor remains meta (zero-memory path);
    - real tensor -> DTensor: physically split; each rank holds a local shard;
    - already a DTensor: skipped if the placement matches, otherwise raises
      PlacementMismatchError.
    """
    for param_path, named in param_specs.items():
        param = _get_attr_by_path(module, param_path)
        placements = tuple(resolve_placements(named, mesh_dim_names))
        if not placements:
            continue  # no active DTensor axes (all size 1) -- no sharding needed

        if isinstance(param, DTensor):
            if tuple(param.placements) != placements:
                raise PlacementMismatchError(
                    f"{type(module).__name__}.{param_path}",
                    placements, tuple(param.placements), "params",
                )
            continue

        src = param.data if hasattr(param, "data") else param
        dt = distribute_tensor(src, mesh, placements)
        requires_grad = getattr(param, "requires_grad", True)
        _set_param_by_path(module, param_path,
                           nn.Parameter(dt, requires_grad=requires_grad))


# ────────────────────────────────────────────────────────────────────────────
# Phase C: forward wrapping (05 §4.4)
# ────────────────────────────────────────────────────────────────────────────

def _apply_phase_c(model, plan, mesh, validate_mode, expert_mesh=None):
    """Phase C: wrap forward (production/validate/moe/cp/vocab_embed, five paths).

    D-14 invariant 2 (05 §13.3): boundaries are wrapped in post-order (deepest
    FQN first) — an outer boundary's local_compute_fn may cache inner forwards,
    and the unpack-scope exclusion (invariant 3) requires inner wrappers to be
    installed first.
    """
    mesh_dim_names = plan.mesh_dim_names
    cp_mesh = _get_cp_submesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    for module_fqn, spec in sorted(
            plan.modules.items(), key=lambda kv: -kv[0].count(".")):
        if not spec.is_boundary:
            continue
        module = _resolve_module(model, module_fqn)
        boundary = PrecompiledBoundary(spec, mesh, mesh_dim_names)
        _bind_input_indices(boundary, module)
        # 注入纪律：声明注入必须显式 region_dispatch；无注入声明 True 是冗余
        _require_region_dispatch(spec, source=f"boundary {module_fqn!r}")
        # 可观察性：注入选择结果即时可见（声明 → 后果 的反馈闭环）
        _log_injection_choice(module_fqn, spec)

        # Step 1: inner-wrap —— 通用"织入/替换 inner forward"机制
        # (D-01'': production and validate inject the same wrapper, so the
        # in-region computation is instruction-for-instruction identical).
        # NOT gated on cp_mesh since the generalization: the derived gate is
        # the resolution chain itself (explicit inner_wrapper declaration ->
        # applied; nothing declared -> None -> no-op). cp_mesh may be None
        # (no cp axis) — the four shipped CP wrappers self-guard and fail
        # fast then; custom callables/Targets receive None and own their
        # semantics.
        # (_preflight_compute_injection has already failed fast when a CP
        # attention boundary declares nothing)
        _wrap_inner_attention(
            module, cp_mesh, spec=spec, mesh=mesh,
            mesh_dim_names=mesh_dim_names, tp_mesh=tp_mesh,
            ep_mesh=expert_mesh,
        )

        # Step 2: forward wrapping
        # local region path (D-03'): the gate is derived from the compute_fn
        # resolution chain (non-None means take the skeleton) — the remaining
        # sources after the explicit-injection rework are user
        # local_compute_fn (callable or factory Target) and the
        # region_dispatch=False gate; the built-in EP auto-injection link
        # was REMOVED (05 §4.4.3)
        compute_fn = _resolve_local_compute_fn(
            module, spec, mesh, mesh_dim_names, expert_mesh)
        if compute_fn is not None:
            if validate_mode:
                # D-17: inside the local region the module sees local
                # tensors in both modes -- validate rewrites cached head
                # counts only for local-region modules (boundary modules
                # keep global counts for DTensor dispatch)
                maybe_update_head_counts(
                    module, spec, module_fqn, mesh, mesh_dim_names)
            # D-14 invariant 3 (05 §13.3): the region's temp-unwrap scope
            # excludes nested-boundary subtrees — their parameters must stay
            # DTensors for the inner validate islands (dispatch needs
            # __torch_function__)
            _wrap_local_region_forward(
                module, boundary, spec, mesh, mesh_dim_names,
                validate_mode=validate_mode, compute_fn=compute_fn,
                exclude_subtrees=_descendant_boundary_fqns(plan, module_fqn))
        elif validate_mode:
            _wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names)
        else:
            # D-02: production vocab-parallel embedding masked wrapper
            if _is_vocab_parallel_embed(module, spec, tp_mesh):
                _wrap_vocab_parallel_embedding(module, tp_mesh)
            _wrap_production_forward(module, boundary)


def _descendant_boundary_fqns(plan, module_fqn):
    """Relative FQNs of boundaries nested inside *module_fqn* (D-14).

    Returned relative to module_fqn (matching the name space of
    module.named_parameters(recurse=True)); the root spec (fqn "") treats
    every other boundary as a descendant.
    """
    if module_fqn == "":
        return [f for f in plan.modules if f]
    prefix = module_fqn + "."
    return [f[len(prefix):] for f in plan.modules if f.startswith(prefix)]


def _bind_input_indices(boundary, module):
    """Bind the arg_name of in_plan to the positional index of the forward signature.

    Inter-module calls are mostly positional (self.mlp(x) inside a layer), so
    RedistOp's kwargs lookup by name would miss -- the signature index is bound
    at compile time, and at runtime _get_arg checks kwargs first, then args.
    """
    try:
        sig = inspect.signature(module.forward)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        positional = [
            name for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        name_to_idx = {name: i for i, name in enumerate(positional)}
        for op in boundary.in_plan:
            if op.arg_index is None and op.arg_name in name_to_idx:
                op.arg_index = name_to_idx[op.arg_name]
    # Positional fallback: a single-input contract (in_plan has only 1 op) is
    # bound to the first positional parameter -- covers cases where the
    # template key (e.g. "hidden_states") differs from the leaf module's
    # signature (nn.Linear.forward(input)).
    if len(boundary.in_plan) == 1 and boundary.in_plan[0].arg_index is None:
        boundary.in_plan[0].arg_index = 0


def _wrap_production_forward(module, boundary):
    """Production mode: pure local tensor computation + precompiled boundary communication (05 §4.4.1).

    _local_params_context was already invoked at the Phase C entry (parameters
    permanently unpacked).
    """
    original_forward = module.forward

    @functools.wraps(original_forward)
    def production_forward(*args, **kwargs):
        args, kwargs = boundary.redistribute_inputs(args, kwargs)
        outputs = original_forward(*args, **kwargs)
        return boundary.redistribute_outputs(outputs)

    module.forward = production_forward


def _wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names):
    """Validate mode: DTensor propagation end to end -> validate out_src (core) + out_dst (terminal modules only).

    The in-house DTensor is forward-only: validation covers only forward
    placement propagation; backward is local autograd in both modes (05 §1.0),
    and gradient equivalence is guaranteed by testing/grad_equiv.py.
    """
    original_forward = module.forward
    module_name = type(module).__name__

    @functools.wraps(original_forward)
    def validate_forward(*args, **kwargs):
        # D-14 nesting (05 §13.4): detect whether the call arrives from an
        # outer DTensor-propagating boundary BEFORE Step 1 wraps everything
        # into DTensors (Step 1 would make the check useless).
        nested = any(isinstance(a, DTensor) for a in args) or any(
            isinstance(v, DTensor) for v in kwargs.values())
        # Step 1: inputs -> DTensor
        args, kwargs = boundary.redistribute_inputs(args, kwargs, as_dtensor=True)

        # Step 2: parameters stay DTensors; placement propagates via __torch_function__ dispatch
        outputs = original_forward(*args, **kwargs)

        # Step 3: [core validation] out_src -- native DTensor-propagated output vs declaration
        if spec.out_src is not None:
            _validate_out_src(outputs, spec, mesh_dim_names, module_name)

        # Step 4: redistribute to out_dst
        outputs = boundary.redistribute_outputs(outputs, as_dtensor_input=True)

        # Step 5: [defensive validation] out_dst -- terminal modules only
        if spec._is_terminal and spec.out_dst is not None:
            _validate_out_dst(outputs, spec, mesh_dim_names, module_name)

        # Step 6: return local (isomorphic to production boundary outputs) --
        # but under an outer DTensor-propagating boundary (D-14 nesting, 05
        # §13.4) keep the DTensor so the outer forward's dispatch chain is
        # unbroken; the outermost boundary exit unwraps.
        if nested:
            return outputs
        if isinstance(outputs, DTensor):
            outputs = outputs.to_local()
        elif isinstance(outputs, (tuple, list)):
            outputs = tuple(
                t.to_local() if isinstance(t, DTensor) else t for t in outputs
            )
        return outputs

    module.forward = validate_forward


def _out_placements_of(value, spec, mesh_dim_names, attr, out_name):
    return tuple(resolve_placements(spec.__dict__[attr][out_name], mesh_dim_names))


def _validate_out_src(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_src")


def _validate_out_dst(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_dst")


def _normalize_placements_ndim(placements, ndim):
    """Normalize negative dims like Shard(-1) against the tensor ndim (Shard(-1) == Shard(ndim-1))."""
    out = []
    for p in placements:
        if isinstance(p, Shard) and p.dim < 0:
            out.append(Shard(p.dim + ndim))
        else:
            out.append(p)
    return tuple(out)


def _validate_outputs(outputs, spec, mesh_dim_names, module_name, stage):
    """Placement validation for single/multi outputs (shared by out_src / out_dst).

    Multi outputs are mapped to tuple positions via spec.out_names (falling
    back to declaration key order); outputs that are not returned or are not
    DTensors are skipped. Declared and actual placements are
    negative-dim-normalized before comparison.
    """
    declared = getattr(spec, stage)
    if isinstance(outputs, (tuple, list)):
        out_names = getattr(spec, "out_names", None) or list(declared.keys())
        name_to_idx = {name: i for i, name in enumerate(out_names)}
        items = list(outputs)
    else:
        name_to_idx = {name: 0 for name in declared}
        items = [outputs]
    for out_name, expected_named in declared.items():
        idx = name_to_idx.get(out_name)
        if idx is None or idx >= len(items):
            continue
        tensor = items[idx]
        if not isinstance(tensor, DTensor):
            continue
        ndim = len(tensor.shape)
        expected = _normalize_placements_ndim(
            tuple(resolve_placements(expected_named, mesh_dim_names)), ndim)
        actual = _normalize_placements_ndim(tuple(tensor.placements), ndim)
        if expected != actual:
            suffix = f"[{out_name}]" if len(declared) > 1 else ""
            raise PlacementMismatchError(
                module_name, expected, actual, f"{stage}{suffix}"
            )


# ────────────────────────────────────────────────────────────────────────────
# Phase C: MoE EP local region (05 §4.4.3 + D-03')
# ────────────────────────────────────────────────────────────────────────────

def _build_local_compute_factory(factory, module, mesh, mesh_dim_names,
                                 expert_mesh, *, configured=None, source):
    """Build a ``@local_compute`` factory into the region compute_fn (apply-time).

    The factory is invoked ONCE with the framework context filled BY NAME per
    its declared context (``meta.context``): the mesh family ``mesh`` /
    ``tp_mesh`` / ``cp_mesh`` / ``ep_mesh`` (mandatory declarations; None when
    the axis is inactive), plus the optional anchor ``module``. Context keys
    are RESERVED — a user-configured same-name key fails fast
    (fill_context_kwargs); every fill is logged at INFO. Behavior choices
    (routing, layouts, ...) are written INTO the factory — config keys carry
    data only, never functions.
    The returned compute fn is validated against the module's forward
    signature (validate_local_compute_signature: params must match the
    original forward) and bound to *module* by the caller.
    """
    meta = require_injection_meta(factory, LOCAL_COMPUTE, source=source)
    context = {
        "module": module,
        "mesh": mesh,
        "tp_mesh": _get_tp_submesh(mesh, mesh_dim_names),
        "cp_mesh": _get_cp_submesh(mesh, mesh_dim_names),
        "ep_mesh": expert_mesh,
    }
    build_kwargs = fill_context_kwargs(
        meta, context, configured or {}, source=source)
    # {**configured, **context}：与 Target.build 的绑定次序一致（上下文键是
    # 保留名，fill_context_kwargs 已拒绝 configured 里的同名键）
    compute_fn = factory(**{**(configured or {}), **build_kwargs})
    if not callable(compute_fn):
        raise TypeError(
            f"{source} returned {type(compute_fn).__name__}, not callable — "
            "the @local_compute factory must return the region compute fn "
            "fn(module, *local_args) (e.g. hyper_models.components.distributed."
            "ep_compute.hf_native_ep_compute_fn)")
    validate_local_compute_signature(
        compute_fn, module.forward,
        owner=f"{source!r} on {type(module).__name__}")
    return compute_fn


def _resolve_local_compute_fn(module, spec, mesh, mesh_dim_names,
                              expert_mesh):
    """Resolve the compute_fn of the local region (**single resolution chain**, 05 §4.4.3).

    Whether a module takes the local-region skeleton is derived by this chain
    (a non-None return means it does) -- the gate is not a stored bool but the
    resolution result. Since the explicit-injection rework the built-in EP
    auto-injection link is REMOVED; the remaining sources:
    1. spec.local_compute_fn: a user-defined FACTORY — single form since
       2026-08-10 (the direct compute-fn form was retired: every injection
       fn is invoked once at apply time with the mesh family filled by the
       framework, use-it-or-not being the user's choice — same discipline as
       @inner_wrapper). Accepted shapes: a ``@local_compute``-decorated
       factory callable (programmatic direct pass), or a Target wrapping one
       (config keys / YAML ``_target_`` reference); both are built at apply
       time by _build_local_compute_factory (e.g. the shipped reference
       ep_compute.hf_native_ep_compute_fn) — undecorated functions fail fast
       (injection discipline); the returned compute fn's params are
       validated against the module's forward (params must match the
       original function). Declaring it REQUIRES an explicit
       ``region_dispatch`` (no default — True: dispatchable pure-ops
       injection, validate dispatches through it and truly validates
       out_src; False: comm/custom-kernel injection, the skeleton runs it
       as a black box on local tensors);
    2. spec.region_dispatch=False (no injection): the module's own forward
       cannot dispatch (it IS the data-dependent logic, e.g. an EP-aware
       in-house MoE with the a2a already inside forward) — the skeleton
       runs it on local tensors;
    3. none of the above -> None (ordinary module; takes the
       validate/production path — and an EP-sharded boundary hitting this
       was already failed fast by _preflight_compute_injection).
    """
    custom = getattr(spec, "local_compute_fn", None)
    if custom is not None:
        _require_region_dispatch(spec, source="spec.local_compute_fn")
        if _is_delayed_target(custom):
            _check_target_config_keys(custom, "local_compute_fn")
            factory = getattr(custom, "_target_", None)
            source = (f"local_compute_fn factory "
                      f"{getattr(custom, '_target_path', custom)}")
            configured = getattr(custom, "_kwargs", {})
        else:
            factory = custom
            source = "spec.local_compute_fn"
            configured = None
        compute_fn = _build_local_compute_factory(
            factory, module, mesh, mesh_dim_names, expert_mesh,
            configured=configured, source=source)
        return functools.partial(compute_fn, module)
    if getattr(spec, "region_dispatch", None) is False:
        return module.forward
    return None


def _wrap_local_region_forward(module, boundary, spec, mesh, mesh_dim_names,
                               *, validate_mode=False, compute_fn=None,
                               exclude_subtrees=()):
    """Generic local-region forward wrapper (D-03', formerly the _wrap_moe_forward skeleton).

    Structure: boundary entry -> local region -> re-wrap per the declared
    out_src -> boundary exit. Applies to any module containing data-dependent
    logic that DTensor dispatch cannot express (e.g. MoE all-to-all) --
    injected by _apply_phase_c when _resolve_local_compute_fn resolves to
    non-None (derived gate, 05 §4.4.3).

    production: parameters were permanently unpacked at build time and inputs
    are local (boundary passthrough); validate: inputs are DTensors ->
    to_local -> temporarily unwrap parameters -> local computation -> re-wrap
    the output per the declared out_src via from_local (for data-dependent
    modules out_src is declarative validation -- the data dependence of
    all-to-all makes the placement underivable; this is an inherent
    limitation). Both modes share the same wrapper code (local_region
    tolerant passthrough semantics).

    compute_fn: the function actually executed inside the region; defaults to
    the module's own forward. Resolved uniformly by
    _resolve_local_compute_fn (user local_compute_fn / region_dispatch=False
    gate), independent of the original forward.

    spec.region_dispatch=True (dispatchable pure-ops injection): validate
    feeds the DTensors straight into compute_fn — strategy propagation runs
    THROUGH the injected fn and out_src is TRULY validated (propagated vs
    declared); production is unchanged (always local passthrough).
    """
    original_forward = module.forward
    if compute_fn is None:
        compute_fn = original_forward

    out_src_placements = None
    if spec.out_src:
        _out_src_named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_out_src_named, mesh_dim_names))
    dispatch_through = bool(getattr(spec, "region_dispatch", None))

    @functools.wraps(original_forward)
    def local_region_forward(*args, **kwargs):
        # Step 1: PrecompiledBoundary entry (e.g. TP all-gather; identity passthrough)
        args, kwargs = boundary.redistribute_inputs(
            args, kwargs, as_dtensor=validate_mode)

        # Step 2: local region -- the data-dependent computation (e.g. EP
        # dispatch/combine) executes on local tensors; region_dispatch=True
        # instead dispatches THROUGH the injected fn (pure standard ops) so
        # validate's strategy propagation covers it
        if validate_mode and dispatch_through:
            try:
                output = compute_fn(*args, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    f"{exc}\n[region_dispatch=True] validate 穿透注入函数时"
                    " dispatch 失败——注入物含不可 dispatch 的通信原语/自定义"
                    " kernel？请改声明 region_dispatch=False（骨架黑盒托管）"
                ) from exc
            # 真校验：传播结果 vs out_src 声明（声明错即 fail-fast）
            if spec.out_src is not None:
                _validate_out_src(output, spec, mesh_dim_names,
                                  type(module).__name__)
        elif validate_mode:
            local_args = tuple(
                a.to_local() if isinstance(a, DTensor) else a for a in args)
            local_kwargs = {
                k: (v.to_local() if isinstance(v, DTensor) else v)
                for k, v in kwargs.items()
            }
            with _temp_local_params(module, exclude=exclude_subtrees):
                output = compute_fn(*local_args, **local_kwargs)
        else:
            output = compute_fn(*args, **kwargs)

        # Step 3: local -> DTensor (re-wrap per the declared out_src, restoring
        # the DTensor metadata broken by all-to-all; under production the
        # boundary exit needs the same contract)
        if out_src_placements is not None and not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, out_src_placements)

        # Step 4: PrecompiledBoundary exit (e.g. TP reduce-scatter)
        output = boundary.redistribute_outputs(
            output, as_dtensor_input=validate_mode)
        # The final boundary exit is always local (when out_plan is empty, the
        # from_local wrap from Step 3 must also be unwrapped here)
        if isinstance(output, DTensor):
            output = output.to_local()
        return output

    module.forward = local_region_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase C: CP inner attention wrapper (05 §4.4.2 + D-01'' + D-04)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_inner_target(module, spec=None):
    """Resolve the inner-wrap target -- pure location resolution.

    ``inner_target`` is MANDATORY when ``inner_wrapper`` is declared (the
    pairing is enforced in _resolve_inner_wrapper before this is called):
    "self" means the boundary module itself, otherwise resolved by attribute
    name -- fail-fast if the attribute does not exist or has no forward (a
    typo must not silently degrade). The attention-domain auto-location
    heuristic (inner_attention/attn/attention attributes, class-name
    matching, q/k/v_proj structural fallback) was REMOVED (2026-08-10): the
    inner-wrap mechanism is generic (any module, any submodule), and a
    silently located target is a silent wrong-target hazard.
    """
    explicit = getattr(spec, "inner_target", None) if spec is not None else None
    if explicit is None:
        raise ValueError(
            "inner_target 未声明——声明 inner_wrapper 时必须成对显式声明 "
            "inner_target（\"self\" 或子模块属性名；自动定位启发式已删除）")
    if explicit == "self":
        return module
    inner = getattr(module, explicit, None)
    if inner is not None and hasattr(inner, "forward"):
        return inner
    raise ValueError(
        f"spec.inner_target={explicit!r} did not match anything on "
        f"{type(module).__name__} (attribute missing or has no forward) "
        f"-- check the spelling in plan_overrides")


def _apply_custom_inner_wrapper(custom_fn, context):
    """Apply a user-defined inner_wrapper (@inner_wrapper callable).

    Contract: custom_fn must be decorated ``@inner_wrapper``; it is invoked
    with its declared context (anchor target_module + the mandatory mesh
    family, filled by name) and replaces target.forward in place. The
    replacement forward runs in a LOCAL-TENSOR world — the dual-mode
    adapter installed by _wrap_inner_attention owns all DTensor
    conversion (to_local / _temp_local_params / from_local rewrap per the
    declared placements), so custom wrappers never touch DTensor.
    """
    meta = require_injection_meta(
        custom_fn, INNER_WRAPPER, source="spec.inner_wrapper")
    custom_fn(**{k: context[k] for k in sorted(meta.context)})


def _install_inner_adapter(target, user_fwd, boundary_module, spec, mesh,
                           mesh_dim_names, wrapper_name):
    """统一双模适配器：安装期解析重包规则，运行期零决策（05 §4.4.2 + D-01''）。

    用户 wrapper 的替换 forward 只面向 local 张量。适配器负责：
    - validate（任一入参是 DTensor）：所有 DTensor 入参 to_local（非张量透
      传）+ ``_temp_local_params(target)`` 临时解包参数 → 调用用户
      forward → 按声明重包回 DTensor（传播链接回，边界校验继续）；
    - production（无 DTensor 入参）：直通，零转换开销。

    重包 placements 来源（框架零推导零猜测，全部显式声明）：
    - 情形 A（target 是边界模块自身）：边界 ``spec.out_src`` 声明（多输出
      按 ``out_names``/声明键序逐位置）；
    - 情形 B（inner 子模块）：``spec.inner_out_src`` 显式声明——哨兵
      ``"first_input"``（输出布局 == 首个 DTensor 入参的运行时布局，
      layout-preserving wrapper 用，仅单输出合法）或 NamedPlacement /
      {name: NamedPlacement}（多输出按声明键序对 tuple 位置）；
      未声明 → 安装时 fail-fast。
    """
    is_self = target is boundary_module
    first_input_rule = False
    out_placements = None
    if is_self:
        if spec is not None and spec.out_src:
            out_names = list(spec.out_names or spec.out_src.keys())
            missing = [n for n in out_names if n not in spec.out_src]
            if missing:
                raise ValueError(
                    f"inner_wrapper {wrapper_name!r}: out_names {missing} 在 "
                    "out_src 中无声明——多输出契约必须逐名声明")
            out_placements = [
                tuple(resolve_placements(spec.out_src[n], mesh_dim_names))
                for n in out_names]
    else:
        declared = getattr(spec, "inner_out_src", None) if spec is not None else None
        if declared is None:
            raise ValueError(
                f"inner_wrapper {wrapper_name!r} 作用于 "
                f"{type(boundary_module).__name__} 的 inner 子模块，但未声明 "
                "inner_out_src——框架对 inner 输出布局零推导零猜测。请二选一："
                "① wrapper 是 layout-preserving 的（输出布局 == 首个输入布"
                "局，attention 类）：inner_out_src: \"first_input\"；"
                "② 显式声明 placement：inner_out_src: {cp: \"shard(2)\"}"
                "（多输出用 {name: {axis: placement}}）；③ 或改用 "
                "inner_target=\"self\" 以边界 out_src 契约重包")
        if isinstance(declared, str):
            if declared != "first_input":
                raise ValueError(
                    f"inner_out_src 的字符串值只接受哨兵 'first_input'，"
                    f"got {declared!r}")
            first_input_rule = True
        elif all(isinstance(v, Placement) for v in declared.values()):
            out_placements = [tuple(resolve_placements(declared, mesh_dim_names))]
        else:
            out_placements = [
                tuple(resolve_placements(named, mesh_dim_names))
                for named in declared.values()]

    @functools.wraps(user_fwd)
    def adapted(*args, **kwargs):
        src = None
        for a in args:
            if isinstance(a, DTensor):
                src = a
                break
        if src is None:
            for v in kwargs.values():
                if isinstance(v, DTensor):
                    src = v
                    break
        if src is None:
            return user_fwd(*args, **kwargs)          # production：直通
        if getattr(spec, "region_dispatch", None):
            # validate 穿透（region_dispatch=True）：注入物是纯标准算子——
            # DTensor 直入，dispatch 传播穿透 inner 区域；声明的重包规则
            # 升级为真校验基准（传播结果 vs 声明，不符即 fail-fast）
            try:
                out = user_fwd(*args, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    f"{exc}\n[region_dispatch=True] validate 穿透 inner_wrapper "
                    f"{wrapper_name!r} 时 dispatch 失败——注入物含不可 "
                    "dispatch 的通信原语/自定义 kernel？请改声明 "
                    "region_dispatch=False（适配器黑盒托管）") from exc
            expected = ([tuple(src.placements)] if first_input_rule
                        else out_placements)
            if expected is not None:
                outs = (list(out) if isinstance(out, (tuple, list))
                        else [out])
                if len(outs) != len(expected):
                    raise RuntimeError(
                        f"inner_wrapper {wrapper_name!r}: 输出数量 "
                        f"{len(outs)} 与声明的 {len(expected)} 个 placement "
                        "不符——多输出契约必须逐名声明且数量一致")
                for t, exp in zip(outs, expected):
                    if not isinstance(t, DTensor):
                        raise RuntimeError(
                            f"inner_wrapper {wrapper_name!r} "
                            f"[region_dispatch=True]: 穿透传播的输出不是 "
                            f"DTensor（{type(t).__name__}）——注入物疑似脱离 "
                            "dispatch 链，无法完成真校验")
                    if tuple(t.placements) != tuple(exp):
                        raise PlacementMismatchError(
                            f"{type(boundary_module).__name__} "
                            f"(inner_wrapper {wrapper_name!r})",
                            tuple(exp), tuple(t.placements), "inner_out_src")
            return out
        # validate：统一转 local（参数临时解包，退出恢复）
        local_args = tuple(
            a.to_local() if isinstance(a, DTensor) else a for a in args)
        local_kwargs = {k: (v.to_local() if isinstance(v, DTensor) else v)
                        for k, v in kwargs.items()}
        with _temp_local_params(target):
            out = user_fwd(*local_args, **local_kwargs)

        def _rw(t, placements, out_mesh):
            if not isinstance(t, torch.Tensor) or isinstance(t, DTensor):
                return t                               # 非张量透传 / 幂等
            return DTensor.from_local(t, out_mesh, placements)

        if first_input_rule:
            if isinstance(out, (tuple, list)):
                raise RuntimeError(
                    f"inner_wrapper {wrapper_name!r}: inner_out_src="
                    "'first_input' 仅支持单输出——多输出请显式声明 "
                    "inner_out_src 的 {name: {axis: placement}} 形态")
            return _rw(out, tuple(src.placements), src.device_mesh)
        if out_placements is None:
            return out                   # 情形 A 且无 out_src 声明：不重包
        if isinstance(out, (tuple, list)):
            if len(out) != len(out_placements):
                raise RuntimeError(
                    f"inner_wrapper {wrapper_name!r}: 替换 forward 返回了 "
                    f"{len(out)} 个输出，与声明的 {len(out_placements)} 个 "
                    "placement 不符——多输出契约必须逐名声明且数量一致")
            return tuple(_rw(t, p, mesh) for t, p in zip(out, out_placements))
        return _rw(out, out_placements[0], mesh)

    target.forward = adapted


def _resolve_inner_wrapper(module, spec, cp_mesh, mesh, mesh_dim_names,
                           tp_mesh=None, ep_mesh=None):
    """Resolve the inner-wrap scheme (**pure function, no side effects**) -- where the resolution chain converges.

    Returns (name, target, apply_fn) or None (None = no inner-wrap for this
    module; the gate is derived from exactly this). Since the
    explicit-injection rework there is NO heuristic dispatch: the wrapper
    must be declared explicitly, and every wrapper callable must be
    decorated ``@inner_wrapper`` (injection discipline: the anchor
    target_module and the mesh family mesh/tp_mesh/cp_mesh/ep_mesh are
    MANDATORY context, ALL framework-filled — None for inactive axes;
    spec is optional context (the declared I/O contract — the HF-path
    rewrap needs out_src, which is not derivable from runtime tensors);
    context keys are reserved — configuring them fails fast; undecorated
    callables/Targets fail fast).
    The mechanism itself is NOT CP-gated (declaration == application);
    only the four shipped CP wrappers require an active cp axis
    (self-guard fail-fast when the framework-filled cp_mesh is None).
    Chain:
    1. spec.inner_wrapper is a Target -> the target callable must be
       @inner_wrapper decorated; built at apply time with its declared
       context (every fill logged — nothing silent); a None return means
       the forward was replaced in place, a callable return (also
       @inner_wrapper decorated) is applied as a custom wrapper;
    2. spec.inner_wrapper is Callable -> fully custom ("custom"; must be
       @inner_wrapper decorated);
    3. spec.inner_wrapper is str -> INNER_WRAPPER_REGISTRY lookup (unknown name
       fail-fast; the registered fn must be @inner_wrapper decorated);
    4. spec.inner_target without inner_wrapper -> fail-fast (location alone
       cannot pick a scheme; no heuristic since the rework);
    5. inner_wrapper without inner_target -> fail-fast (the attention-domain
       auto-location heuristic was REMOVED 2026-08-10: the mechanism is
       generic, a silently located target is a silent wrong-target hazard —
       the two fields must be declared as an explicit pair);
    6. none of the above -> None.
    """
    custom = getattr(spec, "inner_wrapper", None) if spec is not None else None
    inner_target = getattr(spec, "inner_target", None) if spec is not None else None
    if custom is None:
        if inner_target is not None:
            raise ValueError(
                f"spec.inner_target={inner_target!r} 只是定位（replace whom）——"
                "改造后不再启发式选择包装方案，必须同时显式声明 "
                f"inner_wrapper：注册表名 {sorted(INNER_WRAPPER_REGISTRY)}、"
                "@inner_wrapper 装饰的 callable，或指向仓内参考实现的 "
                "Target（hyper_models.components.distributed.cp_wrappers.*）")
        return None

    _require_region_dispatch(spec, source="spec.inner_wrapper")
    if inner_target is None:
        raise ValueError(
            f"声明了 inner_wrapper={custom!r} 但未显式声明 inner_target——"
            "attention 域自动定位启发式已删除（inner-wrap 是通用机制，静默"
            "定位有包错目标风险）。两字段必须成对显式声明：包装边界模块自身 "
            "→ inner_target=\"self\"；包装子模块 → inner_target=\"<属性名>\"")
    target = _resolve_inner_target(module, spec)
    context = {
        "target_module": target,
        "mesh": mesh,
        "tp_mesh": tp_mesh,
        "cp_mesh": cp_mesh,
        "ep_mesh": ep_mesh,
    }
    if _is_delayed_target(custom):
        fn = getattr(custom, "_target_", None)
        source = (f"inner_wrapper Target "
                  f"{getattr(custom, '_target_path', custom)}")
        meta = require_injection_meta(fn, INNER_WRAPPER, source=source)
        _check_target_config_keys(custom, "inner_wrapper")

        def _apply_target():
            build_kwargs = fill_context_kwargs(
                meta, context, getattr(custom, "_kwargs", {}), source=source)
            result = custom.build(**build_kwargs)
            if result is None:
                return   # in-place forward replacement (registry-style fn)
            if callable(result):
                _apply_custom_inner_wrapper(result, context)
                return
            raise TypeError(
                f"inner_wrapper Target "
                f"{getattr(custom, '_target_path', custom)!r} returned "
                f"{type(result).__name__} — expected None (in-place forward "
                "replacement) or an @inner_wrapper decorated callable")

        name = getattr(custom, "_target_path", None) or "custom"
        return (name, target, _apply_target)

    if callable(custom):
        require_injection_meta(
            custom, INNER_WRAPPER, source="spec.inner_wrapper")
        return ("custom", target,
                lambda: _apply_custom_inner_wrapper(custom, context))

    if isinstance(custom, str):
        fn = INNER_WRAPPER_REGISTRY.get(custom)
        if fn is None:
            raise ValueError(
                f"inner_wrapper={custom!r} is not registered in "
                f"INNER_WRAPPER_REGISTRY (available: {sorted(INNER_WRAPPER_REGISTRY)})"
                f" -- check the spelling, or first register "
                f"INNER_WRAPPER_REGISTRY[{custom!r}] = your_fn")
        meta = require_injection_meta(
            fn, INNER_WRAPPER, source=f"INNER_WRAPPER_REGISTRY[{custom!r}]")
        context["target_module"] = target
        return (custom, target,
                lambda: fn(**{k: context[k] for k in sorted(meta.context)}))

    raise TypeError(
        f"inner_wrapper must be a registry name (str), an @inner_wrapper "
        f"decorated callable, or a Target — got {type(custom).__name__}")


def _wrap_inner_attention(module, cp_mesh, *, spec=None, mesh=None,
                          mesh_dim_names=(), tp_mesh=None, ep_mesh=None):
    """Inject an inner forward wrapper (one-shot replacement at apply time, 05 §4.4.2).

    General "weave into / replace the inner forward" mechanism — **not gated
    on CP**: whenever ``spec.inner_wrapper`` is declared the wrapper is
    applied (declaration == application; the resolution chain is the derived
    gate). The shipped CP wrappers (INNER_WRAPPER_REGISTRY) are its first-class
    built-in use case and still require an active cp axis (fail-fast in the
    resolution chain otherwise); custom callables/Targets receive
    ``cp_mesh=None`` when no cp axis exists and own their semantics.

    Resolution (_resolve_inner_wrapper, a pure function of the chain) is
    separated from application; when resolution returns None (no explicit
    declaration), this returns None and injects nothing. Returns the resolved
    wrapper name (or None).

    D-01'': production and validate inject **the same** wrapper, so the
    in-region computation is instruction-for-instruction identical
    (kernel-level equivalence).
    D-04: when is_causal and CP is active, replace it with an offset-aware
    explicit mask.
    Nothing is located silently: after injection an INFO log records the
    target/wrapper/source, and spec._resolved_inner_wrapper +
    spec._resolved_inner_target are written back for plan introspection.
    """
    resolved = _resolve_inner_wrapper(
        module, spec, cp_mesh, mesh, mesh_dim_names,
        tp_mesh=tp_mesh, ep_mesh=ep_mesh)
    if resolved is None:
        return None
    name, target, apply_fn = resolved
    orig_forward = target.forward
    apply_fn()
    # 检测"真的发生了替换"：绑定方法每次取属性都是新对象，`is` 比较恒为
    # 真——必须比较底层函数对象（__func__），否则纯探针 wrapper（不替换
    # forward）也会被误装适配器、被强求 inner_out_src 声明
    _new_forward = target.forward
    _replaced = (getattr(_new_forward, "__func__", _new_forward)
                 is not getattr(orig_forward, "__func__", orig_forward))
    if _replaced:
        # 原则 1：替换后的 forward 必须能接收原 forward 的全部入参
        validate_wrapped_forward(
            orig_forward, _new_forward,
            owner=f"inner_wrapper {name!r} on {type(module).__name__}")
        # 统一安装双模适配器：用户 wrapper 只面向 local 张量，DTensor
        # 转换与声明式重包由适配器托管（local_map 语义；validate 对
        # inner 区域跳过传播校验，安全网在边界层）
        _install_inner_adapter(
            target, _new_forward, module, spec, mesh, mesh_dim_names, name)
    target_name = _inner_target_name(module, target)
    if spec is not None:
        spec._resolved_inner_wrapper = name
        spec._resolved_inner_target = target_name
    if name == "custom":
        source = "custom callable"
    elif spec is not None and isinstance(
            getattr(spec, "inner_wrapper", None), str):
        source = "explicitly specified (registry)"
    else:
        source = "explicitly specified (Target)"
    logger.info("inner-wrap: %s target=%s <- wrapper %r (%s)",
                type(module).__name__, target_name, name, source)
    return name


def _inner_target_name(attn_module, target) -> str:
    """Readable name of the located inner-wrap target: child attribute name,
    or "self" when the boundary module itself is the target."""
    if target is attn_module:
        return "self"
    for child_name, child in attn_module.named_children():
        if child is target:
            return child_name
    return type(target).__name__


# ────────────────────────────────────────────────────────────────────────────
# Phase C: D-02 vocab-parallel embedding wrapper
# ────────────────────────────────────────────────────────────────────────────

def _is_vocab_parallel_embed(module, spec, tp_mesh) -> bool:
    """production embed boundary check: nn.Embedding + weight Shard(0) on TP + TP>1."""
    if tp_mesh is None or tp_mesh.size() <= 1:
        return False
    if not isinstance(module, nn.Embedding):
        return False
    weight_named = spec.params.get("weight", {})
    return weight_named.get("tp") == Shard(0)


def _wrap_vocab_parallel_embedding(module, tp_mesh):
    """D-02: Megatron-style masked embedding (injected at the production embed boundary).

    The vocab-range mask logic of DTensor dispatch is lost after the
    parameter unwrap -- HF native F.embedding would index out of range when
    given global token ids. The wrapper: tokens outside the local vocab
    interval [lo, hi) are zeroed and indices are shifted by the offset, so the
    output is naturally a Partial contribution and the boundary exit's
    Partial->Shard(1) reduction is unchanged.
    """
    original_forward = module.forward
    v_local = module.weight.shape[0]
    lo = tp_mesh.get_local_rank() * v_local
    hi = lo + v_local

    @functools.wraps(original_forward)
    def masked_embedding_forward(input_ids, *args, **kwargs):
        mask = (input_ids >= lo) & (input_ids < hi)
        local_ids = torch.where(mask, input_ids - lo, torch.zeros_like(input_ids))
        out = original_forward(local_ids, *args, **kwargs)
        return out * mask.unsqueeze(-1).to(out.dtype)

    module.forward = masked_embedding_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase D: tied weights
# ────────────────────────────────────────────────────────────────────────────

def detect_tied_weights(model):
    """Detect tied-weight pairs (embed_tokens.weight <-> lm_head.weight).

    In PP scenarios cross-stage pairs cannot be detected; the user must
    explicitly declare plan.tied_pairs.
    """
    tied = []
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        embed_fqn = lm_head_fqn = None
        # remove_duplicate=False: under the default dedup of named_parameters
        # a tied parameter appears only once; duplicates must be explicitly
        # retained to discover the FQNs of both ends.
        for name, _ in model.named_parameters(remove_duplicate=False):
            if name.endswith("embed_tokens.weight"):
                embed_fqn = name
            elif name.endswith("lm_head.weight"):
                lm_head_fqn = name
        if embed_fqn and lm_head_fqn:
            tied.append((embed_fqn, lm_head_fqn))
    return tied


def _broadcast_tied_param(model, tied_pair, mesh):
    """A tied-weight pair shares storage within this rank (end A's storage is authoritative; end B shares it).

    Cross-rank broadcast would be **wrong**: a tied pair (embed/lm_head) is
    usually Shard(0)-sharded on both ends, and each rank's local shard carries
    a different vocab interval -- broadcasting rank0's shard to rank1 would
    corrupt rank1's sharding. Tied semantics require that **within the same
    rank** the two ends are the same physical parameter (shared gradients),
    not cross-rank consistency (sharding is naturally consistent: same global
    source, same placement).
    """
    fqn_a, fqn_b = tied_pair
    try:
        param_a = _get_attr_by_path(model, fqn_a)
        param_b = _get_attr_by_path(model, fqn_b)
    except AttributeError:
        return
    if param_a is None or param_b is None:
        return
    tensor_a = param_a.to_local() if isinstance(param_a, DTensor) else param_a.data
    # B shares storage with A (a tied weight is the same physical parameter)
    if isinstance(param_b, DTensor):
        param_b._local_tensor = tensor_a
    else:
        param_b.data = tensor_a


def _replicate_tied_weights(model, mesh, tied_pairs=None):
    """Phase D: replicate tied weights across ranks."""
    for tied_pair in (tied_pairs if tied_pairs is not None
                      else detect_tied_weights(model)):
        _broadcast_tied_param(model, tied_pair, mesh)

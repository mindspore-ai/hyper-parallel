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
"""sharding_config: data model for the dual-mode DTensor parallel strategy (05 §3.1/§3.2/§3.5 canonical).

Contains:
- ``MeshAxisName``: mesh dimension name enum (canonical definition, imported and reused by later docs such as 06);
- ``NamedPlacement``: alias for ``dict[MeshAxisName, Placement]``;
- ``ShardingPlan`` / ``ModuleShardingSpec``: model-level plan and per-module I/O contract;
- ``ShardingTemplate`` / ``TEMPLATES``: semantic role → placement templates (TP+CP+EP, three dims);
- ``PlacementMismatchError``: error for placement declarations inconsistent with DTensor propagation;
- ``resolve_placements`` / ``_multi_dim`` / ``_normalize_out_fields``: placement utilities.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

logger = logging.getLogger(__name__)


class MeshAxisName(str, Enum):
    """Canonical enum of mesh dimension names (a str enum, directly comparable to plain strings like "tp" and usable as a dict key)."""
    TP = "tp"
    CP = "cp"
    EP = "ep"
    PP = "pp"
    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_CP = "dp_cp"
    EP_SHARD = "ep_shard"


# Shorthand aliases for the {TP: ..., CP: ..., EP: ...} literals in templates and examples.
# str-enum keys interoperate with plain string keys like "tp" (hash/eq are identical).
TP = MeshAxisName.TP
CP = MeshAxisName.CP
EP = MeshAxisName.EP
DP = MeshAxisName.DP

# NamedPlacement = {MeshAxisName: Placement}.
# The key is a mesh dimension name; the N in Shard(N) of a value is a tensor dimension index (05 §3.2.1).
NamedPlacement = Dict[MeshAxisName, Placement]


class PlacementMismatchError(ValueError):
    """DTensor propagation result is inconsistent with the ModuleShardingSpec declaration (05 §5.3)."""

    def __init__(self, module_name: str, expected, actual, stage: str):
        self.module_name = module_name
        self.expected = expected
        self.actual = actual
        self.stage = stage
        super().__init__(
            f"[{module_name}] {stage} placement mismatch:\n"
            f"  Expected (from ShardingConfig.{stage}): {expected}\n"
            f"  Actual   (from DTensor propagation):   {actual}\n"
            f"  → Check the ShardingConfig for this module."
        )


@dataclass
class ModuleShardingSpec:
    """Complete DTensor contract for a single module (05 §3.2).

    The four placement fields form the complete I/O contract — the runtime does no
    inference and executes exactly as declared:

      in_src:  placement of the input when it arrives at the module boundary
               (from the output of an upstream module or the dataloader)
      in_dst:  placement required by the module's internal computation
               (a mismatch triggers communication)
      out_src: placement naturally produced by the module's internal computation
               (used by validate mode)
      out_dst: placement expected by downstream modules (a mismatch triggers
               communication)

    plan_overrides input side only: the contract fields (params/in_src/in_dst/
    out_src/out_dst/out_names) follow one rule — **"不写继承，写了照办"**
    (2026-08-05): ``None`` (unset) or the sentinel ``"auto"`` means
    inherit-from-template (merge only); an explicit value IS the final value,
    including the empty dict ``{}`` (explicit "no sharding / no contract"
    — the ViT ``params={}`` pattern); the sentinel ``"none"`` is a readable
    alias for clearing (``{}`` for params/in_*, ``None`` for out_*). The
    planner resolves sentinels and normalizes ``None`` at merge/insert time;
    a spec inside a finished ShardingPlan always carries concrete dicts
    (params/in_src/in_dst) or Optional dicts (out_*).
    """
    # ── Parameter sharding: submodule path → NamedPlacement ──
    # Default None = "not declared" (merge: inherit); an explicit {} means
    # "this boundary shards nothing" (I/O-stitch-only citizen).
    params: Optional[Dict[str, NamedPlacement]] = None

    # ── Input contract ──
    in_src: Optional[Dict[str, NamedPlacement]] = None
    in_dst: Optional[Dict[str, NamedPlacement]] = None

    # ── Output contract ──
    # out_src=None: no src validation; out_dst=None: output needs no redistribution.
    # Single-output modules use {"output": NamedPlacement}; the scalar shorthand
    # {TP: ...} is wrapped into {"output": ...} during the normalization phase
    # (_normalize_out_fields).
    out_src: Optional[Dict[str, NamedPlacement]] = None
    out_dst: Optional[Dict[str, NamedPlacement]] = None
    # out_names: output-name ordering for multi-output modules (returning a tuple),
    # used to map the keys of out_src/out_dst to tuple positions
    # (RedistOp.arg_index). Defaults to the key order of out_src.
    out_names: Optional[List[str]] = None

    # ── Boundary flag ──
    is_boundary: bool = True

    # ── Structural flags (user-configurable) ──
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ The user extension-point interfaces: region_dispatch /           │
    # │ local_compute_fn / inner_target / inner_wrapper                  │
    # │                                                                 │
    # │ One axiom governs every boundary: 区域计算默认可 dispatch 穿透    │
    # │ （validate 下 DTensor 直入、策略传播、out_src 真校验）；不能       │
    # │ dispatch 的，显式声明 region_dispatch=False（骨架黑盒：           │
    # │ to_local → local 执行 → 声明式重包）。                          │
    # │                                                                 │
    # │ [local-region family] (module level: skeleton unchanged,         │
    # │   content swapped)                                               │
    # │   region_dispatch / local_compute_fn                             │
    # │   The skeleton = _wrap_local_region_forward: boundary            │
    # │   entry/exit stitching + local compute + validate dual-mode      │
    # │   fault tolerance (to_local/_temp_local_params/from_local        │
    # │   re-wrapping), shared by both modes. Whether a module runs      │
    # │   through the skeleton is **derived from a single resolution     │
    # │   chain** (not a stored bool) by _resolve_local_compute_fn:      │
    # │     chain link 1  local_compute_fn (user-defined computation:    │
    # │                   callable, or a factory Target built at apply   │
    # │                   time — e.g. the shipped reference              │
    # │                   ep_compute.hf_native_ep_compute_fn)            │
    # │     chain link 2  region_dispatch=False (the module's own        │
    # │                   forward can't dispatch — it IS the compute)    │
    # │     none present → None (skeleton not used)                      │
    # │                                                                 │
    # │   ★ EXPLICIT INJECTION (since the rework): the shipped EP        │
    # │   compute is NEVER auto-injected. The planner only shards the    │
    # │   expert params ({EP: Shard(0)} + _ep_size/_ep_stack metadata);  │
    # │   the compute must be declared via plan_overrides / direct       │
    # │   spec assignment, WITH an explicit region_dispatch (no          │
    # │   default for injections — the tutorial teaches False for        │
    # │   CP/EP-style comm-carrying injections and why). An EP-sharded   │
    # │   boundary whose chain resolves to None fails fast at apply      │
    # │   time (_preflight_compute_injection). HF-native MoE layouts     │
    # │   (per-expert / batched) get the template's region_dispatch      │
    # │   CLEARED by the planner (their forward is not EP-aware);        │
    # │   custom-named pre-stacked modules (w1/w2/w3) keep False         │
    # │   (EP-aware by construction).                                    │
    # │                                                                 │
    # │ [inner-wrap family] (submodule level: content and wrapping both  │
    # │   replaced)                                                      │
    # │   inner_target / inner_wrapper                                   │
    # │   The mechanism is a generic "locate an inner submodule +        │
    # │   replace its forward": inner_target answers "replace whom",     │
    # │   inner_wrapper answers "replace with what". It is NOT CP-gated  │
    # │   (declaration == application); the shipped reference domain is  │
    # │   CP (K/V all-gather), and the four INNER_WRAPPER_REGISTRY       │
    # │   entries still require an active cp axis (fail-fast otherwise)  │
    # │   — custom wrappers receive cp_mesh=None and own their           │
    # │   semantics.                                                     │
    # │                                                                 │
    # │   ★ EXPLICIT INJECTION (since the rework): NO heuristic          │
    # │   dispatch. inner_wrapper must be declared explicitly: a         │
    # │   INNER_WRAPPER_REGISTRY name ("sdpa_qkv"/"sdpa_hf"/"flex_qkv"/  │
    # │   "flex_hf"), a callable fn(target_module, cp_mesh), or a        │
    # │   Target pointing at a wrapper fn (e.g.                          │
    # │   cp_wrappers.sdpa_hf_cp_wrapper). _needs_cp_attn is now pure    │
    # │   METADATA (template recognition) — it no longer triggers any    │
    # │   injection; an attention boundary under an active cp mesh       │
    # │   without inner_wrapper fails fast at apply time.                │
    # │                                                                 │
    # │ The two families are orthogonal and composable: the same module  │
    # │ may declare both inner_* (wrapping of an inner attention) and    │
    # │ local_* (module-level skeleton).                                 │
    # └─────────────────────────────────────────────────────────────────┘
    #
    # region_dispatch: **区域计算（无论来源：模块自身 forward 或注入函数）的
    #   validate 执行模式声明**——全框架一条公理：区域计算默认可 dispatch
    #   穿透，不能的显式声明 False。
    #   - None（无注入）：普通边界——validate 下 DTensor 直入原 forward，
    #     dispatch 穿透 + out_src 真校验（公理缺省，非本字段的"默认值"）；
    #   - False（无注入）：模块自身 forward 不可 dispatch（数据依赖逻辑在
    #     forward 内，如自研 EP-aware MoE 的 a2a）→ 走 local-region 骨架，
    #     compute = 模块自身 forward（模板识别出此类模块时由 planner 推导
    #     填入，日志可见）；
    #   - 有注入（local_compute_fn / inner_wrapper 非 None）：**必须显式
    #     声明**（无默认）——True = 注入物纯标准算子可 dispatch（validate
    #     穿透 + out_src/inner_out_src 真校验）；False = 注入物含通信/
    #     自定义 kernel（validate 黑盒 local 执行 + 声明式重包）。缺失 →
    #     apply 时 fail-fast；
    #   - True（无注入）：冗余声明 → fail-fast（普通边界天然穿透）。
    #   production 不受本字段影响（区域内恒 local 直通）。
    region_dispatch: Optional[bool] = None

    # ── inner-wrap custom entry points (user-configurable, 05 §4.4.2/§8.6) ──
    # inner_target: **pure location** — names the attribute of the inner
    #   submodule whose forward is to be replaced ("self" means the module
    #   itself). Automatic locating (_resolve_inner_target) fails fast, in which
    #   case the user must specify it via plan_overrides/injections. Declaring
    #   inner_target WITHOUT inner_wrapper is an error since the rework
    #   (location alone cannot pick a scheme — no heuristic dispatch).
    # inner_wrapper: **pure behavior** — selects which scheme wraps the target.
    #   NOT CP-gated: declaration == application, whatever the parallel mode.
    #   - str: a name in the INNER_WRAPPER_REGISTRY registry ("sdpa_qkv"/"sdpa_hf"/
    #     "flex_qkv"/"flex_hf", or a user-registered name); explicitly pins a
    #     built-in scheme; an unknown name fails fast; the four SHIPPED names
    #     are CP schemes and require an active cp axis (fail-fast otherwise);
    #   - Callable: a fully custom wrapper, which MUST be decorated
    #     ``@inner_wrapper`` (injection discipline, see injection.py —
    #     undecorated callables fail fast). Context contract:
    #     fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh) — the anchor plus
    #     the mesh family are MANDATORY context params, ALL filled by the
    #     framework at apply time (None for inactive axes; the user just uses
    #     them); ``spec`` is the only optional context. The wrapper replaces
    #     target.forward in place (use cp_utils.flex_cp_allgather for K/V
    #     all-gather). The replaced forward must accept the original
    #     forward's params (validated at apply time). region_dispatch 必须
    #     显式声明（见上）：注入物只面向 local 张量（False，适配器托管
    #     DTensor 转换）或纯标准算子可 dispatch（True，validate 穿透
    #     真校验）；
    #   - Target (hyper_models.trainer.config.Target): a delayed wrapper
    #     reference resolved from YAML; the referenced fn must likewise be
    #     @inner_wrapper decorated (the shipped built-ins
    #     cp_wrappers.sdpa_hf_cp_wrapper etc. satisfy this contract
    #     directly). A None return = in-place replacement done; a callable
    #     return (also decorated) = applied as a custom wrapper.
    inner_target: Optional[str] = None
    inner_wrapper: Optional[Union[str, Callable]] = None   # or Target
    # inner_out_src: inner-wrap 情形 B（inner_target 指向子模块而非 self）的
    #   输出 placement **显式声明**——框架对 inner 输出布局零推导零猜测：
    #   - 哨兵 "first_input"：layout-preserving 声明（输出布局 == 首个
    #     DTensor 入参的运行时布局；attention 类 wrapper 用，多输出非法）；
    #   - NamedPlacement（{axis: Placement}）：单输出显式声明；
    #   - {name: NamedPlacement}：多输出逐名声明（tuple 位置按声明键序）。
    #   情形 B 未声明 → apply 时 fail-fast（情形 A target=self 用边界
    #   out_src，无需本字段）。validate 对 inner 区域不做传播校验：声明错
    #   误由重包后的全局形状一致性 / 边界 out_src 校验 / 数值对拍兜底。
    inner_out_src: Optional[Union[str, Dict]] = None

    # ── local-region custom computation (user-configurable, 05 §4.4.3/§8.6) ──
    # local_compute_fn: the custom compute FACTORY of the local region
    # (chain link 1; single form since 2026-08-10 — the direct compute-fn
    # form was retired, mesh family is always filled, use-it-or-not):
    #   - Callable: a ``@local_compute``-decorated factory
    #     fn(mesh, tp_mesh, cp_mesh, ep_mesh, [module], <config keys...>)
    #     -> compute_fn (programmatic direct pass; injection discipline:
    #     mesh family mandatory, no defaults, no *args/**kwargs);
    #   - Target: the same factory as a delayed reference resolved from
    #     YAML (config keys bound by name; typo fail-fast).
    #   The factory is built ONCE at apply time with the framework context
    #   filled by name (mesh / tp_mesh / cp_mesh / ep_mesh mandatory —
    #   ep_mesh is the same object the expert params were sharded on;
    #   module optional anchor) and must RETURN the region compute fn
    #   fn(module, *local_args) -> Tensor, whose params MUST match the
    #   module's forward params (validated at apply time) — executed on
    #   local tensors inside the _wrap_local_region_forward skeleton (the
    #   skeleton handles boundary entry/exit stitching + validate dual-mode
    #   to_local/_temp_local_params/from_local re-wrapping). Config keys
    #   carry DATA only (function-typed config is rejected by discipline —
    #   custom behavior = your own injected function, e.g. a MoE factory
    #   with its router written inline).
    # Suitable for custom modules that want to reuse the skeleton with their
    # own data-dependent logic: typically a custom MoE (router not in
    # MOE_ROUTER_ADAPTERS / expert layout not using HF-standard naming /
    # hooked to a DeepEP fused dispatcher).
    # Priority: local_compute_fn > region_dispatch=False gate (the module's own
    #   forward) — a single resolution chain (_resolve_local_compute_fn);
    #   declaring it takes effect immediately: **there is no need — and it is
    #   wrong — to also set region_dispatch=False** unless the injected fn is
    #   itself non-dispatchable; the skeleton gate is derived from the
    #   resolution chain (non-None means the skeleton is used).
    # Dual-mode convention: with region_dispatch=False the inputs are always
    #   local tensors and the return value is a local tensor; validate's
    #   DTensor unwrap/re-wrapping is done by the skeleton, so the compute_fn
    #   need not be aware of the mode. With region_dispatch=True the compute
    #   fn must be pure dispatchable ops (validate feeds DTensors straight
    #   through and truly validates out_src).
    local_compute_fn: Optional[Callable] = None   # or factory Target

    # ── Internal flags (set automatically by ShardingPlanner / applier) ──
    _is_terminal: bool = False    # marked automatically during chained propagation
    # _needs_cp_attn: template recognition METADATA (attention module: the
    # inner attention needs a CP-aware forward under an active cp mesh). Since
    # the explicit-injection rework it does NOT trigger any injection — it is
    # used only by the apply-time preflight (an attention boundary under cp>1
    # without inner_wrapper fails fast) and for introspection.
    _needs_cp_attn: bool = False
    # _resolved_inner_wrapper: written back by the applier after resolution (for
    # introspection) — the inner wrapper actually injected:
    # "sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"/"custom"/<target_path>/None.
    _resolved_inner_wrapper: Optional[str] = None
    # _resolved_inner_target: written back by the applier after resolution —
    # the inner-wrap target actually located: the child attribute name, or
    # "self" when the boundary module itself is wrapped. Nothing is located
    # silently: the resolved target is always visible here and in the INFO log.
    _resolved_inner_target: Optional[str] = None
    # D-09 (05 §6.4.7): EP pass-through for HF-native MoE. A non-empty _ep_stack
    # means per-expert parameters must be pre-stacked into [E, ...] in Phase A.
    # Since the explicit-injection rework the compute side is NOT auto-injected:
    # declare local_compute_fn explicitly (e.g. Target →
    # ep_compute.hf_native_ep_compute_fn).
    _ep_stack: Dict[str, List[str]] = field(default_factory=dict)
    # D-10 (05 §6.4.8): TP-extend-EP parameter-sharding marker. When >0 this is
    # the extended EP group size (= ep_size; the a2a communication domain
    # includes TP ranks); the MoE uses an SP-in identity boundary + a derived
    # expert mesh (edp, ep); expert weights are only Shard(0) along the expert
    # dim. It drives ONLY the parameter sharding/expert-mesh derivation — the
    # compute must be injected explicitly (apply-time preflight fails fast
    # otherwise).
    _ep_size: int = 0


@dataclass
class ShardingPlan:
    """Complete sharding plan for a model (05 §3.1)."""
    # {module_fqn: ModuleShardingSpec} — contains only modules with is_boundary=True
    modules: Dict[str, ModuleShardingSpec] = field(default_factory=dict)

    # Global switches
    sequence_parallel: bool = True
    loss_parallel: bool = False

    # Special parameter handlers: {module_fqn.param_name: handler_name}
    special_handlers: Dict[str, str] = field(default_factory=dict)

    # Mesh dimension names (consistent with DeviceMesh.mesh_dim_names)
    mesh_dim_names: Tuple[str, ...] = ()

    # Tied-weight pairs: [(fqn_a, fqn_b)], parameters sharing storage
    # (embed_tokens <-> lm_head).
    tied_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def explain(self, fqn: Optional[str] = None) -> str:
        """Human-readable introspection report of this plan (usability tool).

        Per boundary: the parameter sharding table (param → placement), the
        compiled boundary communication plan (in/out RedistOps — which
        tensor, from which layout to which, which collective), the injection
        declarations and their resolution result, and any special-handler
        entries. All of this information already lives in the plan/specs —
        this is purely a formatting outlet. The intended learning path:
        read your own model's actual sharding first, then reverse-engineer
        the concepts — instead of building the layout mental model from
        docs up front.

        Args:
            fqn: optional exact boundary FQN — report just that boundary;
                None reports all boundaries.
        """
        # Lazy: sharding_config must not import precompiled_boundary at
        # module level (precompiled_boundary already imports this module).
        from hyper_models.components.distributed.precompiled_boundary import (
            PrecompiledBoundary,
        )

        def fmt_named(named: Optional[NamedPlacement]) -> str:
            if not named:
                return "{}"
            items = [
                (getattr(axis, "value", axis), p) for axis, p in named.items()
                # 只显示本 plan 拓扑轴（spec 可能携带模板的全轴声明，
                # resolve 时才按 mesh_dim_names 过滤——报告提前过滤避免误导）
                if not self.mesh_dim_names
                or getattr(axis, "value", axis) in self.mesh_dim_names
            ]
            return "{" + ", ".join(f"{a}: {p!r}" for a, p in items) + "}"

        def fmt_callable(obj) -> str:
            if obj is None:
                return "-"
            path = getattr(obj, "_target_path", None)   # Target 实例
            if path:
                return path
            if isinstance(obj, str):
                return obj
            return getattr(obj, "__qualname__", repr(obj))

        lines = [
            "=== ShardingPlan 内省报告 ===",
            f"mesh_dim_names={self.mesh_dim_names}  "
            f"sequence_parallel={self.sequence_parallel}  "
            f"loss_parallel={self.loss_parallel}",
            f"边界数: {len(self.modules)}  tied_pairs: "
            + (", ".join(f"{a}↔{b}" for a, b in self.tied_pairs) or "无"),
        ]
        if fqn is not None and fqn not in self.modules:
            lines.append(f"\n[!] {fqn!r} 不是 plan 中的边界（现有边界见上）")
            return "\n".join(lines)
        selected = (
            {fqn: self.modules[fqn]} if fqn is not None else self.modules)

        for name, spec in selected.items():
            lines.append(f"\n[{name}]")
            # ── 参数切分表 ──
            if spec.params:
                lines.append("  参数切分:")
                for pname, named in spec.params.items():
                    lines.append(f"    {pname}: {fmt_named(named)}")
            else:
                lines.append("  参数切分: 无（{} = 本边界不切参数，仅 I/O 缝合）")
            # ── 边界通信计划（编译结果，mesh=None 仅取 RedistOp 描述） ──
            boundary = PrecompiledBoundary(spec, None, self.mesh_dim_names)
            if boundary.in_plan:
                lines.append("  输入通信计划 (in_src → in_dst):")
                for op in boundary.in_plan:
                    tag = "直通" if op.collective_type == "identity" \
                        else op.collective_type
                    lines.append(
                        f"    {op.arg_name}: "
                        f"{tuple(map(repr, op.src_placements))}"
                        f" → {tuple(map(repr, op.dst_placements))}  [{tag}]")
            if boundary.out_plan:
                lines.append("  输出通信计划 (out_src → out_dst):")
                for op in boundary.out_plan:
                    lines.append(
                        f"    {op.arg_name}(tuple[{op.arg_index}]): "
                        f"{tuple(map(repr, op.src_placements))}"
                        f" → {tuple(map(repr, op.dst_placements))}"
                        f"  [{op.collective_type}]")
            if not boundary.in_plan and not boundary.out_plan:
                lines.append("  边界通信: 无")
            # ── 注入声明与解析 ──
            injection = []
            if spec.local_compute_fn is not None:
                injection.append(
                    f"local_compute_fn={fmt_callable(spec.local_compute_fn)}")
            if spec.inner_wrapper is not None:
                injection.append(
                    f"inner_wrapper={fmt_callable(spec.inner_wrapper)}"
                    f"(target={spec.inner_target or 'auto'})")
            if spec.inner_out_src is not None:
                injection.append(f"inner_out_src={spec.inner_out_src}")
            if spec.region_dispatch is not None:
                meaning = ("黑盒托管（区域内跳过传播校验，声明式重包）"
                           if spec.region_dispatch is False
                           else "dispatch 穿透（validate 真校验已启用）")
                injection.append(
                    f"region_dispatch={spec.region_dispatch} → {meaning}")
            if injection:
                lines.append("  注入: " + "; ".join(injection))
            else:
                lines.append("  注入: 无（普通边界，validate 下 dispatch 穿透）")
            # ── 特殊参数处理 ──
            handlers = {k: v for k, v in self.special_handlers.items()
                        if k.startswith(name + ".")}
            for key, handler in handlers.items():
                lines.append(f"  特殊处理: {key[len(name) + 1:]} → {handler}")
        return "\n".join(lines)


@dataclass
class ShardingTemplate:
    """Semantic role → placement template (05 §3.5).

    Each I/O field declares placements for all active mesh dimensions
    (TP+CP+EP); the ShardingPlanner filters out dimensions not enabled by the
    actual mesh_dim_names (resolve_placements looks up keys by mesh_dim_names,
    so extra keys are naturally dropped).

    Note: sp_out_src / nosp_out_src etc. are scalar NamedPlacement shorthands
    (single-output modules); they are wrapped into {"output": ...} during
    normalization in _build_spec_from_template.
    """
    # Parameter sharding rules
    colwise_placement: Placement = field(default_factory=lambda: Shard(0))
    rowwise_placement: Placement = field(default_factory=lambda: Shard(1))
    norm_placement: Placement = field(default_factory=Replicate)
    moe_expert_placement: Placement = field(default_factory=lambda: Shard(0))

    # SP-mode I/O (full TP+CP+EP, three dims)
    sp_in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    sp_in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)
    sp_out_src: Optional[NamedPlacement] = None
    sp_out_dst: Optional[NamedPlacement] = None

    # non-SP-mode I/O
    nosp_in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    nosp_in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)
    nosp_out_src: Optional[NamedPlacement] = None
    nosp_out_dst: Optional[NamedPlacement] = None

    # Special flags
    # region_dispatch: template-level declaration that the matched module's own
    # forward CANNOT dispatch (data-dependent logic inside, e.g. an EP-aware
    # custom MoE's a2a) — the planner copies False into the spec (derived
    # metadata, logged); None = ordinary dispatchable module.
    region_dispatch: Optional[bool] = None
    needs_cp_attn: bool = False   # CP: inner attention needs a CP-aware forward


def _multi_dim(tp=None, cp=None, ep=None) -> NamedPlacement:
    """Build multi-dim placement dict, filtering out None dims."""
    result = {}
    if tp is not None:
        result[TP] = tp
    if cp is not None:
        result[CP] = cp
    if ep is not None:
        result[EP] = ep
    return result


def resolve_placements(
    named: NamedPlacement,
    mesh_dim_names: Tuple[str, ...],
) -> List[Placement]:
    """Arrange placements in mesh_dim_names order, fill missing axes with Replicate()."""
    return [named.get(axis, Replicate()) for axis in mesh_dim_names]


_PLACEMENT_DSL_DOC = (
    '"replicate" / "partial" / "shard(N)"（N 为张量维下标，如 "shard(0)"）')


def parse_placement(text, *, path: str = "placement") -> Placement:
    """Parse the YAML placement DSL string into a Placement object.

    Grammar (closed set): ``replicate`` / ``partial`` / ``shard(N)`` — case
    insensitive, whitespace tolerant. Anything else fails fast listing the
    grammar (typo self-discovery, same philosophy as
    ``_check_target_config_keys``).
    """
    if not isinstance(text, str):
        raise ValueError(
            f"{path}: placement 必须是字符串（{_PLACEMENT_DSL_DOC}），"
            f"got {type(text).__name__} {text!r}")
    normalized = re.sub(r"\s+", "", text).lower()
    if normalized == "replicate":
        return Replicate()
    if normalized == "partial":
        return Partial()
    match = re.fullmatch(r"shard\((\d+)\)", normalized)
    if match:
        return Shard(int(match.group(1)))
    raise ValueError(
        f"{path}: 无法解析 placement {text!r} —— 合法文法：{_PLACEMENT_DSL_DOC}")


def parse_named_placement(raw, *, path: str = "named_placement") -> NamedPlacement:
    """Parse the YAML form ``{axis: placement_str}`` into a NamedPlacement.

    Axis names stay plain strings (str-enum interop; custom mesh dims such as
    "encoder_dp" are legal) — validity against the actual mesh is checked at
    plan-merge time (ShardingPlanner._validate_override_axes), because the
    mesh does not exist at config time.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{path}: 期望非空映射 {{轴名: placement 字符串}}，got {raw!r}")
    named: NamedPlacement = {}
    for axis, placement_text in raw.items():
        if not isinstance(axis, str) or not axis:
            raise ValueError(
                f"{path}: 轴名必须是非空字符串，got {axis!r}")
        named[axis] = parse_placement(
            placement_text, path=f"{path}.{axis}")
    return named


def _normalize_out_fields(spec: ModuleShardingSpec) -> ModuleShardingSpec:
    """Normalize the scalar shorthand {TP: ...} into {'output': {TP: ...}} (05 §3.5).

    Detection heuristic: if val is a non-None dict and any of its values is not a
    dict, it is judged to be a scalar NamedPlacement shorthand. Idempotent — a
    second call on an already-normalized dict contract changes nothing.
    """
    for attr in ("out_src", "out_dst"):
        val = getattr(spec, attr, None)
        if val and not all(isinstance(v, dict) for v in val.values()):
            setattr(spec, attr, {"output": dict(val)})
    return spec


def _hid(tp_p, cp_p, ep_p=None) -> Dict[str, NamedPlacement]:
    """Shorthand for the single-input hidden_states contract."""
    return {"hidden_states": _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())}


def _out(tp_p, cp_p, ep_p=None) -> NamedPlacement:
    """Shorthand for the single-output (scalar shorthand) contract."""
    return _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())


# ── TEMPLATES: complete templates for the 7 semantic roles (05 §3.5, declared over TP+CP+EP) ──
# CP-dim rule: parameters are always Replicate (CP does not shard parameters);
# activations are Shard(1) (sequence dim) or Replicate.
# EP-dim rule: non-MoE modules Replicate; MoE experts Shard(0).
TEMPLATES: Dict[str, ShardingTemplate] = {
    # ── Attention (q/k/v Colwise + o Rowwise) ──
    # The CP dim keeps Shard(1) in in_dst: the K/V all-gather is done by the
    # inner attention wrapper inside SDPA/FlexAttention (needs_cp_attn=True),
    # not at the boundary layer.
    "attention": ShardingTemplate(
        colwise_placement=Shard(0),          # q/k/v: [H/tp, H]
        rowwise_placement=Shard(1),          # o: [H, H/tp]
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Partial(), Shard(1)),     # local Q-chunk output → CP Shard(1)
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
        needs_cp_attn=True,
    ),

    # ── MLP (gate/up Colwise + down Rowwise) ──
    # The CP dim stays Shard(1) throughout (revision D-06): MLP is pointwise and
    # CP needs no communication; if in_dst had CP=Replicate, the full-sequence
    # reduce-scatter under TP×CP would produce a tp-major sequence layout
    # inconsistent with embed/attention (cp-major).
    "mlp": ShardingTemplate(
        colwise_placement=Shard(0),
        rowwise_placement=Shard(1),
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Partial(), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── Norm (RMSNorm/LayerNorm: weight replicated, zero communication) ──
    "norm": ShardingTemplate(
        norm_placement=Replicate(),
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Shard(1), Shard(1)),      # identity
        sp_out_src=_out(Shard(1), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),     # identity
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Replicate(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── Embedding (weight Shard(0) along the vocab dim, output Partial → SP+CP) ──
    "embed": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        sp_in_dst={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        sp_out_src=_out(Partial(), Replicate()),
        sp_out_dst=_out(Shard(1), Shard(1)),     # reduce-scatter → SP+CP
        nosp_in_src={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_in_dst={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── LM Head (weight Shard(0), output Shard(-1); out_dst is overridden according to loss_parallel) ──
    # The CP dim stays Shard(1) throughout (revision D-07): R8 — the CP dim is
    # always identity at the boundary layer (the CP sequence all-gather happens
    # only inside attention for K/V). lm_head computes logits/loss on the local
    # CP chunk (the standard Megatron CP practice), with no CP gather.
    "lm_head": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Shard(-1), Shard(1)),
        sp_out_dst=_out(Shard(-1), Shard(1)),   # loss_parallel=true default;
        # when loss_parallel=false, _build_spec_from_template overrides it to
        # {TP: Replicate, CP: Shard(1)}
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Shard(-1), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── MoE Gate (Router: weight replicated, output redistributes → EP) ──
    "moe_gate": ShardingTemplate(
        norm_placement=Replicate(),          # router weight/bias: replicated
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Replicate()),
        sp_out_src=_out(Replicate(), Replicate()),
        sp_out_dst=_out(Replicate(), Replicate(), Shard(0)),
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Replicate(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate(), Shard(0)),
    ),

    # ── MoE MLP (gate + routed experts + optional shared experts) ──
    # CP dim same as mlp (D-06): pointwise per-token, CP stays Shard(1) throughout.
    "moe_mlp": ShardingTemplate(
        colwise_placement=Shard(0),          # expert w1/w3: Colwise on TP
        rowwise_placement=Shard(1),          # expert w2: Rowwise on TP
        norm_placement=Replicate(),          # gate/norm: replicated
        moe_expert_placement=Shard(0),       # expert params: Shard(0) on EP
        sp_in_src={"x_BLD": _multi_dim(tp=Shard(1), cp=Shard(1), ep=Replicate())},
        sp_in_dst={"x_BLD": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        sp_out_src=_out(Partial(), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src={"x_BLD": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_in_dst={"x_BLD": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
        region_dispatch=False,           # MoE forward 自带 a2a，不可 dispatch
    ),
}

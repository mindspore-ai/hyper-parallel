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
"""sharding_applier: ShardingPlan 的运行时应用（05 §4 canonical）。

apply_sharding_plan: Phase 0 归一化 → A 参数分片 → B 特殊处理器 →
C 入口解包 + tp_grad_info → C forward 包装（production/validate/moe/cp/vocab_embed
五路）→ D tied weights。

双模式架构约束（05 §1.4）：production 零 DTensor dispatch（build-time unpack +
PrecompiledBoundary）；validate 与 production 的唯一差异是边界缝合方式——凡
DTensor dispatch 隐含数据相关逻辑的模块（embedding mask / attention K/V gather /
MoE all-to-all），两模式用同一份 local-region wrapper 显式重建（D-01''/D-02/D-03'）。
"""

import functools
import inspect
import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.components.distributed.cp_utils import (
    _cp_offset_causal_mask,
    flex_cp_allgather,
)
from hyper_parallel.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_parallel.components.distributed.sharding.apply import (
    _get_attr_by_path,
    _local_params_context,
    _resolve_module,
    _set_param_by_path,
)
from hyper_parallel.components.distributed.sharding_config import (
    PlacementMismatchError,
    _normalize_out_fields,
    resolve_placements,
)
from hyper_parallel.components.distributed.sharding_planner import SPECIAL_HANDLERS
from hyper_parallel.components.distributed.tp_grad import build_tp_grad_info

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# 主入口（05 §4.1）
# ────────────────────────────────────────────────────────────────────────────

def apply_sharding_plan(model, plan, mesh, *, validate_mode=False):
    """对任意 nn.Module（或 PP 多 part 列表）应用 ShardingPlan，启用双模式 DTensor。

    返回 (model, tp_grad_info)：
    - production：Phase C 入口一次性 `_local_params_context` 把 DTensor 参数永久
      解包为 plain local tensor，并构造 tp_grad_info 供 fully_shard 使用；
    - validate：不解包（参数保持 DTensor），tp_grad_info 为 None。
    """
    mesh_dim_names = plan.mesh_dim_names
    # 活跃子 mesh：planner 会剔除 size=1 轴（plan.mesh_dim_names），但传入的
    # mesh 可能仍含这些轴——placements 按 plan.mesh_dim_names 解析，维度数
    # 必须与 mesh 对齐，否则 distribute_tensor 会静默错轴分片。
    mesh = _get_active_mesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    models = model if isinstance(model, list) else [model]

    # ====== Phase 0: 归一化 out_src/out_dst 标量简写（幂等，覆盖用户注入路径） ======
    for spec in plan.modules.values():
        _normalize_out_fields(spec)

    # ====== Phase A: 参数分片 ======
    for part in models:
        for module_fqn, spec in plan.modules.items():
            module = _resolve_module(part, module_fqn)
            _shard_module_params(module, spec.params, mesh, mesh_dim_names)

    # ====== Phase B: 特殊处理器 ======
    for part in models:
        for param_ref, handler_name in plan.special_handlers.items():
            handler = SPECIAL_HANDLERS.get(handler_name)
            if handler is None:
                logger.warning("SPECIAL_HANDLERS 未注册 handler: %s", handler_name)
                continue
            module_fqn, param_name = param_ref.rsplit(".", 1)
            handler(_resolve_module(part, module_fqn), param_name, mesh)

    # ====== Phase C 入口: build 期一次性解包（production 专用） ======
    tp_grad_info = None
    if not validate_mode:
        tp_grad_records = {}
        for part in models:
            tp_grad_records.update(_local_params_context(part))
        if tp_grad_records and tp_mesh is not None:
            tp_grad_info = build_tp_grad_info(plan, tp_mesh)

    # ====== Phase C: 包装 forward ======
    for part in models:
        _apply_phase_c(part, plan, mesh, validate_mode)

    # ====== Phase D: tied weights ======
    tied_pairs = list(plan.tied_pairs) or detect_tied_weights(models[0])
    for part in models:
        _replicate_tied_weights(part, mesh, tied_pairs)

    return model, tp_grad_info


def _get_active_mesh(mesh, mesh_dim_names):
    """取与 plan.mesh_dim_names 对齐的活跃子 mesh（剔除 size=1 轴后的维度集合）。"""
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


# ────────────────────────────────────────────────────────────────────────────
# Phase A: 参数分片（05 §4.2）
# ────────────────────────────────────────────────────────────────────────────

def _shard_module_params(module, param_specs, mesh, mesh_dim_names):
    """distribute_tensor() 转换参数为 DTensor。

    - meta tensor → DTensor：_local_tensor 仍为 meta（零显存路径）；
    - real tensor → DTensor：物理切分，每 rank 持 local shard；
    - 已是 DTensor：placement 一致跳过，不一致抛 PlacementMismatchError。
    """
    for param_path, named in param_specs.items():
        param = _get_attr_by_path(module, param_path)
        placements = tuple(resolve_placements(named, mesh_dim_names))
        if not placements:
            continue  # 无活跃 DTensor 轴（全部 size 1）——无需分片

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
# Phase C: forward 包装（05 §4.4）
# ────────────────────────────────────────────────────────────────────────────

def _apply_phase_c(model, plan, mesh, validate_mode):
    """Phase C: 包装 forward（production/validate/moe/cp/vocab_embed 五路）。"""
    mesh_dim_names = plan.mesh_dim_names
    cp_mesh = _get_cp_submesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    for module_fqn, spec in plan.modules.items():
        if not spec.is_boundary:
            continue
        module = _resolve_module(model, module_fqn)
        boundary = PrecompiledBoundary(spec, mesh, mesh_dim_names)
        _bind_input_indices(boundary, module)

        # Step 1: CP inner attention wrapper（D-01''：production 与 validate
        # 注入同一个 all-gather wrapper，区域内计算逐指令一致）
        if (cp_mesh is not None and cp_mesh.size() > 1
                and getattr(spec, "_needs_cp_attn", False)):
            _wrap_cp_inner_attention(
                module, cp_mesh, spec=spec, mesh=mesh,
                mesh_dim_names=mesh_dim_names,
            )

        # Step 2: forward 包装
        if validate_mode:
            if spec._use_local_map:
                _wrap_moe_forward(module, boundary, spec, mesh, mesh_dim_names,
                                  validate_mode=True)
            else:
                _wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names)
        elif spec._use_local_map:
            _wrap_moe_forward(module, boundary, spec, mesh, mesh_dim_names,
                              validate_mode=False)
        else:
            # D-02: production vocab-parallel embedding masked wrapper
            if _is_vocab_parallel_embed(module, spec, tp_mesh):
                _wrap_vocab_parallel_embedding(module, tp_mesh)
            _wrap_production_forward(module, boundary)


def _bind_input_indices(boundary, module):
    """把 in_plan 的 arg_name 绑定到 forward 签名的 positional 下标。

    模块间调用多为 positional（layer 内 self.mlp(x)），RedistOp 的 kwargs
    按名查找会 miss——编译期绑定签名下标，运行时 _get_arg 先 kwargs 后 args。
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
    # 位置回退：单输入契约（in_plan 仅 1 个 op）绑定到首个 positional 参数——
    # 覆盖模板 key（如 "hidden_states"）与叶模块签名（nn.Linear.forward(input)）
    # 不同名的场景。
    if len(boundary.in_plan) == 1 and boundary.in_plan[0].arg_index is None:
        boundary.in_plan[0].arg_index = 0


def _wrap_production_forward(module, boundary):
    """生产模式：纯 local tensor 计算 + 预编译边界通信（05 §4.4.1）。

    _local_params_context 已在 Phase C 入口调用（参数永久 unpack）。
    """
    original_forward = module.forward

    @functools.wraps(original_forward)
    def production_forward(*args, **kwargs):
        args, kwargs = boundary.redistribute_inputs(args, kwargs)
        outputs = original_forward(*args, **kwargs)
        return boundary.redistribute_outputs(outputs)

    module.forward = production_forward


def _wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names):
    """校验模式：DTensor 全程传播 → 校验 out_src（核心）+ out_dst（仅末端）。

    自研 DTensor 前向-only：校验仅覆盖前向 placement 传播；backward 两模式
    同为 local autograd（05 §1.0），梯度等价由 testing/grad_equiv.py 保证。
    """
    original_forward = module.forward
    module_name = type(module).__name__

    @functools.wraps(original_forward)
    def validate_forward(*args, **kwargs):
        # Step 1: 输入 → DTensor
        args, kwargs = boundary.redistribute_inputs(args, kwargs, as_dtensor=True)

        # Step 2: 参数保持 DTensor，走 __torch_function__ dispatch 传播 placement
        outputs = original_forward(*args, **kwargs)

        # Step 3: 【核心校验】out_src — DTensor 传播原生输出 vs 声明
        if spec.out_src is not None:
            _validate_out_src(outputs, spec, mesh_dim_names, module_name)

        # Step 4: redistribute 到 out_dst
        outputs = boundary.redistribute_outputs(outputs, as_dtensor_input=True)

        # Step 5: 【防御性校验】out_dst — 仅末端模块
        if spec._is_terminal and spec.out_dst is not None:
            _validate_out_dst(outputs, spec, mesh_dim_names, module_name)

        # Step 6: 返回 local（与 production 边界输出同构）
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
    """Shard(-1) 等负维度按 tensor ndim 归一化（Shard(-1) == Shard(ndim-1)）。"""
    out = []
    for p in placements:
        if isinstance(p, Shard) and p.dim < 0:
            out.append(Shard(p.dim + ndim))
        else:
            out.append(p)
    return tuple(out)


def _validate_outputs(outputs, spec, mesh_dim_names, module_name, stage):
    """单/多输出的 placement 校验（out_src / out_dst 共用）。

    多输出按 spec.out_names（缺省按声明 key 顺序）映射到 tuple 位置；
    未返回/非 DTensor 的输出跳过。比较前对声明与实际 placement 做负维度归一化。
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
# Phase C: MoE EP local region（05 §4.4.3 + D-03'）
# ────────────────────────────────────────────────────────────────────────────

@contextmanager
def _temp_local_params(module):
    """validate 模式 local region 内临时解包 DTensor 参数（退出时恢复）。

    production 下参数已 build 期永久解包，无需本 context。validate 的 local
    region（MoE all-to-all / HF CP attention）内部在 local tensor 上计算，
    需要 local 参数；恢复后 DTensor 传播链不断。
    """
    saved = []
    for name, param in list(module.named_parameters(recurse=True)):
        if isinstance(param, DTensor):
            saved.append((name, param))
            _set_param_by_path(module, name, nn.Parameter(
                param.to_local(), requires_grad=param.requires_grad))
    try:
        yield
    finally:
        for name, param in saved:
            _set_param_by_path(module, name, param)


def _wrap_moe_forward(module, boundary, spec, mesh, mesh_dim_names,
                      *, validate_mode=False):
    """MoE forward wrapper（D-03'）：boundary 入口 → local region → boundary 出口。

    production：参数已 build 期永久 unpack，输入为 local（boundary 直通）；
    validate：输入为 DTensor → to_local → 临时解包参数 → local all-to-all →
    输出按声明 out_src from_local 重包装（out_src 对 MoE 为声明式校验——
    all-to-all 的数据相关性使 placement 无法派生，这是本质限制）。
    两模式共用同一份 wrapper 代码（local_region 容错透传语义）。
    """
    original_forward = module.forward

    out_src_placements = None
    if spec.out_src:
        _out_src_named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_out_src_named, mesh_dim_names))

    @functools.wraps(original_forward)
    def moe_forward(*args, **kwargs):
        # Step 1: PrecompiledBoundary 入口（TP all-gather）
        args, kwargs = boundary.redistribute_inputs(
            args, kwargs, as_dtensor=validate_mode)

        # Step 2: local region —— EP dispatch/combine 在 local tensor 上执行
        if validate_mode:
            local_args = tuple(
                a.to_local() if isinstance(a, DTensor) else a for a in args)
            local_kwargs = {
                k: (v.to_local() if isinstance(v, DTensor) else v)
                for k, v in kwargs.items()
            }
            with _temp_local_params(module):
                output = original_forward(*local_args, **local_kwargs)
        else:
            output = original_forward(*args, **kwargs)

        # Step 3: local → DTensor（按声明 out_src 重包装，恢复 all-to-all
        # 打断的 DTensor 元数据；production 下边界出口需要同一契约）
        if out_src_placements is not None and not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, out_src_placements)

        # Step 4: PrecompiledBoundary 出口（TP reduce-scatter）
        output = boundary.redistribute_outputs(
            output, as_dtensor_input=validate_mode)
        # 边界最终出口恒为 local（out_plan 为空时 Step 3 的 from_local
        # 包装也需要在此解包）
        if isinstance(output, DTensor):
            output = output.to_local()
        return output

    module.forward = moe_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase C: CP inner attention wrapper（05 §4.4.2 + D-01'' + D-04）
# ────────────────────────────────────────────────────────────────────────────

def _find_inner_attention(module):
    """定位 attention 模块内的 inner attention 子模块。

    1. 显式属性 inner_attention / attn / attention（NeMo/Megatron 风格）；
    2. HF 标准：类名含 "SdpaAttention" 或以 "Attention" 结尾——模块本身即 inner；
    3. 结构兜底：直接持有 q_proj/k_proj/v_proj。
    """
    for name in ("inner_attention", "attn", "attention"):
        inner = getattr(module, name, None)
        if inner is not None and hasattr(inner, "forward"):
            return inner
    cls_name = type(module).__name__
    if "SdpaAttention" in cls_name or cls_name.endswith("Attention"):
        return module
    if (hasattr(module, "q_proj") and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")):
        return module
    return None


def _attn_implementation(module):
    cfg = getattr(module, "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl is None and isinstance(cfg, dict):
        impl = cfg.get("attn_implementation")
    return impl


def _is_sdpa_attention(module) -> bool:
    impl = _attn_implementation(module)
    return (impl == "sdpa") or ("SdpaAttention" in type(module).__name__)


def _is_flex_attention(module) -> bool:
    impl = _attn_implementation(module)
    return (impl == "flex_attention") or ("FlexAttention" in type(module).__name__)


def _is_hf_style_attention(module) -> bool:
    """HF 风格（forward(hidden_states,...)，投影在 forward 内）→ 原语拦截路径。"""
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


def _wrap_cp_inner_attention(attn_module, cp_mesh, *, spec=None, mesh=None,
                             mesh_dim_names=()):
    """注入 CP-aware inner forward（编译期一次性替换，05 §4.4.2）。

    D-01''：production 与 validate 注入**同一个** all-gather wrapper——
    K/V all-gather + 本地 Q chunk SDPA，区域内计算逐指令一致（kernel 级等价）。
    wrapper 入口容错 DTensor/local（local_region 透传语义），validate 出口按
    声明重包装 DTensor。
    D-04：is_causal 且 q_len ≠ kv_len 时替换为 offset-aware 显式 mask。
    """
    inner_attn = _find_inner_attention(attn_module)
    if inner_attn is None:
        logger.warning("_wrap_cp_inner_attention: no inner attention found on %s",
                       type(attn_module).__name__)
        return

    if _is_hf_style_attention(inner_attn):
        # HF: forward(hidden_states,...) → 原语拦截，复用 HF 投影/RoPE
        if _is_flex_attention(inner_attn):
            _wrap_hf_flex_for_cp(inner_attn, cp_mesh, spec=spec, mesh=mesh,
                                 mesh_dim_names=mesh_dim_names)
        else:
            _wrap_hf_sdpa_for_cp(inner_attn, cp_mesh, spec=spec, mesh=mesh,
                                 mesh_dim_names=mesh_dim_names)
    else:
        # NeMo/Megatron: inner_attention.forward(q,k,v,...) → (q,k,v) wrapper
        if _is_flex_attention(inner_attn):
            _wrap_flex_attn_for_cp(inner_attn, cp_mesh)
        else:
            _wrap_sdpa_for_cp(inner_attn, cp_mesh)


def _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kwargs):
    """CP-aware SDPA：K/V all-gather + D-04 offset-aware causal mask。"""
    cp_dim = 2  # [B, N, S, H] 布局的序列维
    global_k, global_v = flex_cp_allgather(
        k.contiguous(), v.contiguous(), cp_dim, cp_mesh)
    if kwargs.get("is_causal") and q.shape[cp_dim] != global_k.shape[cp_dim]:
        # G4：is_causal 在 q_len ≠ kv_len 时右下对齐，rank>0 chunk 掩码错误
        # → 替换为按本 rank Q 全局偏移 lo 的显式下三角 mask。
        cp_rank = cp_mesh.get_local_rank()
        lo = cp_rank * q.shape[cp_dim]
        kwargs = dict(kwargs)
        kwargs.pop("is_causal")
        kwargs["attn_mask"] = _cp_offset_causal_mask(
            q.shape[cp_dim], global_k.shape[cp_dim], lo, q.device)
    return orig_sdpa(q, global_k, global_v, **kwargs)


def _wrap_sdpa_for_cp(inner_attn, cp_mesh):
    """NeMo/Megatron SDPA 路径：inner_attention.forward(q,k,v,...) → 显式 all-gather K/V。

    双模式共用：q/k/v 为 DTensor 时 unwrap（validate），输出按 q 的
    placements 重包装；local 输入透传（production）。
    """
    original_forward = inner_attn.forward

    @functools.wraps(original_forward)
    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)
        q_placements = tuple(q.placements) if was_dtensor else None
        mesh = q.device_mesh if was_dtensor else None
        ql, kl, vl = (t.to_local() if isinstance(t, DTensor) else t
                      for t in (q, k, v))
        out = _cp_sdpa_call(
            lambda *a, **kw: original_forward(*a, **kw),
            cp_mesh, ql, kl, vl, kwargs)
        if was_dtensor and isinstance(out, torch.Tensor):
            out = DTensor.from_local(out, mesh, q_placements)
        return out

    inner_attn.forward = cp_forward


def _wrap_flex_attn_for_cp(inner_attn, cp_mesh):
    """NeMo/Megatron FlexAttention 路径：显式 all-gather K/V（双模式共用）。"""
    original_forward = inner_attn.forward

    @functools.wraps(original_forward)
    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)
        q_placements = tuple(q.placements) if was_dtensor else None
        mesh = q.device_mesh if was_dtensor else None
        ql, kl, vl = (t.to_local() if isinstance(t, DTensor) else t
                      for t in (q, k, v))
        global_k, global_v = flex_cp_allgather(
            kl.contiguous(), vl.contiguous(), 2, cp_mesh)
        out = original_forward(ql, global_k, global_v, **kwargs)
        if was_dtensor and isinstance(out, torch.Tensor):
            out = DTensor.from_local(out, mesh, q_placements)
        return out

    inner_attn.forward = cp_forward


def _wrap_hf_sdpa_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                         mesh_dim_names=()):
    """HF 标准 SDPA 路径：forward(hidden_states,...) → 原语拦截（05 §4.4.2）。

    双模式共用（D-01''）：hidden_states 为 DTensor 时（validate）unwrap +
    临时解包模块参数，出口按 spec.out_src 声明重包装；local 输入透传
    （production）。原语拦截为临时全局函数替换（try/finally 还原），非线程
    安全；单进程 SPMD 训练下安全（与 TorchTitan CP 实现一致）。
    """
    original_forward = inner_attn.forward
    orig_sdpa = F.scaled_dot_product_attention

    out_src_placements = None
    if spec is not None and spec.out_src:
        _named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_named, mesh_dim_names))

    def cp_aware_sdpa(q, k, v, **kwargs):
        return _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kwargs)

    @functools.wraps(original_forward)
    def cp_forward(hidden_states, *args, **kwargs):
        was_dtensor = isinstance(hidden_states, DTensor)
        hs = hidden_states.to_local() if was_dtensor else hidden_states
        F.scaled_dot_product_attention = cp_aware_sdpa
        try:
            if was_dtensor:
                with _temp_local_params(inner_attn):
                    out = original_forward(hs, *args, **kwargs)
            else:
                out = original_forward(hs, *args, **kwargs)
        finally:
            F.scaled_dot_product_attention = orig_sdpa
        if (was_dtensor and out_src_placements is not None
                and not isinstance(out, DTensor) and isinstance(out, torch.Tensor)):
            out = DTensor.from_local(out, mesh, out_src_placements)
        return out

    inner_attn.forward = cp_forward


def _wrap_hf_flex_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                         mesh_dim_names=()):
    """HF 标准 FlexAttention 路径：拦截 flex_attention（结构同 SDPA 路径）。"""
    original_forward = inner_attn.forward
    from torch.nn.attention.flex_attention import flex_attention as _orig_flex

    out_src_placements = None
    if spec is not None and spec.out_src:
        _named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_named, mesh_dim_names))

    def cp_aware_flex(q, k, v, **kwargs):
        global_k, global_v = flex_cp_allgather(
            k.contiguous(), v.contiguous(), 2, cp_mesh)
        return _orig_flex(q, global_k, global_v, **kwargs)

    @functools.wraps(original_forward)
    def cp_forward(hidden_states, *args, **kwargs):
        import torch.nn.attention.flex_attention as _flex_mod
        was_dtensor = isinstance(hidden_states, DTensor)
        hs = hidden_states.to_local() if was_dtensor else hidden_states
        _flex_mod.flex_attention = cp_aware_flex
        try:
            if was_dtensor:
                with _temp_local_params(inner_attn):
                    out = original_forward(hs, *args, **kwargs)
            else:
                out = original_forward(hs, *args, **kwargs)
        finally:
            _flex_mod.flex_attention = _orig_flex
        if (was_dtensor and out_src_placements is not None
                and not isinstance(out, DTensor) and isinstance(out, torch.Tensor)):
            out = DTensor.from_local(out, mesh, out_src_placements)
        return out

    inner_attn.forward = cp_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase C: D-02 vocab-parallel embedding wrapper
# ────────────────────────────────────────────────────────────────────────────

def _is_vocab_parallel_embed(module, spec, tp_mesh) -> bool:
    """production embed 边界判定：nn.Embedding + weight 在 TP 上 Shard(0) + TP>1。"""
    if tp_mesh is None or tp_mesh.size() <= 1:
        return False
    if not isinstance(module, nn.Embedding):
        return False
    weight_named = spec.params.get("weight", {})
    return weight_named.get("tp") == Shard(0)


def _wrap_vocab_parallel_embedding(module, tp_mesh):
    """D-02：Megatron 风格 masked embedding（production embed 边界注入）。

    DTensor dispatch 的 vocab 范围 mask 逻辑在参数解包后丢失——HF 原生
    F.embedding 收到全局 token id 会索引越界。wrapper：本地 vocab 区间
    [lo, hi) 外的 token 置 0、索引减去偏移，输出即天然 Partial 贡献，
    boundary 出口 Partial→Shard(1) 归约不变。
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
    """检测 tied-weight 对（embed_tokens.weight <-> lm_head.weight）。

    PP 场景跨 stage 检测不到，需用户显式声明 plan.tied_pairs。
    """
    tied = []
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        embed_fqn = lm_head_fqn = None
        # remove_duplicate=False：tied 参数在 named_parameters 默认去重下
        # 只出现一次，必须显式保留重复项才能发现两端 FQN。
        for name, _ in model.named_parameters(remove_duplicate=False):
            if name.endswith("embed_tokens.weight"):
                embed_fqn = name
            elif name.endswith("lm_head.weight"):
                lm_head_fqn = name
        if embed_fqn and lm_head_fqn:
            tied.append((embed_fqn, lm_head_fqn))
    return tied


def _broadcast_tied_param(model, tied_pair, mesh):
    """tied-weight 对本 rank 内共享存储（A 端存储为准，B 端共享）。

    跨 rank 广播是**错误**的：tied 对（embed/lm_head）通常同为 Shard(0)
    分片，各 rank 的 local shard 承载不同 vocab 区间——把 rank0 的 shard
    广播给 rank1 会破坏 rank1 的分片。tied 语义要求的是**同一 rank 内**
    两端是同一物理参数（梯度共享），而非跨 rank 一致（分片天然一致：
    同一 global 来源、同一 placement）。
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
    # B 与 A 共享存储（tied weight 同一物理参数）
    if isinstance(param_b, DTensor):
        param_b._local_tensor = tensor_a
    else:
        param_b.data = tensor_a


def _replicate_tied_weights(model, mesh, tied_pairs=None):
    """Phase D：tied weights 跨 rank replicate。"""
    for tied_pair in (tied_pairs if tied_pairs is not None
                      else detect_tied_weights(model)):
        _broadcast_tied_param(model, tied_pair, mesh)

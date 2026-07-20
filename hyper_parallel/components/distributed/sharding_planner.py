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
"""sharding_planner: ShardingPlanner 6-phase 推导管线（05 §3.6 canonical）。

Phase 1  参数角色分类（ParameterClassifier + ARCH_OVERRIDES）
Phase 2  通信边界分组（两趟：先按直属模块分组，再深度优先向上合并——
         修正 05 §3.6.6 伪代码"单参数 group 推断"会把 q_proj 叶模块误判为
         mlp 边界的缺陷）
Phase 3  语义角色推断（FQN 显式模式 > 结构守卫 > 参数角色组合）
Phase 4  模板查表生成 spec（_build_spec_from_template）
Phase 4.5 用户 plan_overrides 合并（_merge_plan_overrides，05 §3.6.7）
Phase 5  链式传播校验（填充缺省 in_src + 校验相邻契约 + _is_terminal 标记）
Phase 6  特殊参数处理器收集（SPECIAL_HANDLERS）

注册表：
- ``ARCH_OVERRIDES``: {arch_name: [(pattern | [patterns], ParamRole)]}
- ``SPECIAL_HANDLERS``: {handler_name: callable(module, param_name, mesh)}
"""

import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple

from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
    _match_any,
)
from hyper_parallel.components.distributed.sharding_config import (
    EP,
    TP,
    ModuleShardingSpec,
    NamedPlacement,
    PlacementMismatchError,
    ShardingPlan,
    ShardingTemplate,
    TEMPLATES,
    _multi_dim,
    _normalize_out_fields,
    resolve_placements,
)

logger = logging.getLogger(__name__)

# {arch_name: [(pattern | [patterns], ParamRole)]} —— 架构级命名覆盖（方式 B）。
# pattern 为小写子串（或子串列表），命中即强制为该角色。
ARCH_OVERRIDES: Dict[str, list] = {
    "llama": [],
    "qwen2": [],
    "qwen3": [],
    "mixtral": [],
}


def _shard_gated_delta(module, param_name, mesh):
    """gated_delta 模块自定义 TP 分片骨架（SSM/Mamba 类模块，05 §6.4.6）。

    按 SSM head 结构切分而非标准 colwise/rowwise。骨架实现：结构识别与
    标准 Shard(0) 回退；head 对齐的精细切分留待具体模型接入时补全。
    """
    import torch.nn as nn

    param = getattr(module, param_name, None)
    if param is None:
        return
    sharded = distribute_tensor(param.data, mesh, [Shard(0)])
    module.register_parameter(param_name, nn.Parameter(sharded))


# {handler_name: callable(module, param_name, mesh)} —— Phase B 特殊参数处理器。
SPECIAL_HANDLERS: Dict[str, Callable] = {
    "gated_delta_tp_shard": _shard_gated_delta,
}

# planner 侧 pattern → handler_name 映射（fqn 子串小写匹配）。
_SPECIAL_HANDLER_PATTERNS: Dict[str, str] = {
    "gated_delta": "gated_delta_tp_shard",
    "a_log": "gated_delta_tp_shard",
    "dt_bias": "gated_delta_tp_shard",
}

# 叶子投影/容器段名守卫：这些段名自身不是边界容器，推断时返回 unknown 继续向上。
_LEAF_SEGMENT_GUARD = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv_proj", "fused_qkv", "gate_up_proj", "query_key_value",
    "experts", "shared_experts", "gate", "linear", "proj",
    "fc1", "fc2", "w1", "w2", "w3", "w13", "dense", "dense_h_to_4h", "dense_4h_to_h",
})

_ATTN_PATTERNS = ("attn", "attention")
_MLP_PATTERNS = ("mlp", "ffn", "feed_forward")
_MOE_CONTAINER_PATTERNS = ("mlp", "moe", "moe_block", "moe_layer")


def _last_segment(fqn: str) -> str:
    return fqn.rsplit(".", 1)[-1].lower() if fqn else ""


def _infer_colwise_vs_rowwise(param_path: str, template: ShardingTemplate):
    """按参数名后缀推断 TP placement：w2/down → rowwise，其余 → colwise。"""
    name = param_path.lower()
    if any(k in name for k in ("w2", "down_proj", "down.")):
        return template.rowwise_placement
    return template.colwise_placement


def _moe_expert_tp_placement(param_path: str, ndim: int,
                             template: ShardingTemplate):
    """MOE_EXPERT 的 TP placement（修订 D-08，按参数 ndim 感知）。

    expert 权重为 batched 3D 布局 [E, H_out, H_in]（ndim>=3）时，tensor dim 0
    是 expert 维（归 EP Shard(0)），TP 的 colwise/rowwise 须作用在 +1 维：
    colwise（切 H_out）→ Shard(1)；rowwise（切 contraction 维 H_in）→ Shard(2)。
    per-expert 2D 布局（experts.N.w1 [H_out, H_in]）沿用标准 Shard(0)/Shard(1)
    ——但此时 EP Shard(0) 会切 H_out，语义不成立：EP 应按"每 rank 持有 expert
    子集"实现（module 级），需 ARCH_OVERRIDES/SpecialHandler，不在模板覆盖范围。
    """
    name = param_path.lower()
    is_rowwise = any(k in name for k in ("w2", "down_proj", "down."))
    if ndim >= 3:
        return Shard(2) if is_rowwise else Shard(1)
    return template.rowwise_placement if is_rowwise else template.colwise_placement


class ShardingPlanner:
    """从任意 HF 风格模型自动推导 ShardingPlan（05 §3.6.6）。

    ``plan_overrides``: {module_fqn: ModuleShardingSpec} —— 用户手写 spec，
    在 Phase 5 链式传播之前整体替换/插入（05 §3.6.7）。覆盖 spec 仍参与
    相邻契约校验与 terminal 标记，比 plan() 返回后再打补丁安全。
    """

    def __init__(
        self,
        plan_overrides: Optional[Dict[str, ModuleShardingSpec]] = None,
    ):
        self._classifier = ParameterClassifier(arch_overrides=ARCH_OVERRIDES)
        self._templates = TEMPLATES
        self._special_handler_patterns = dict(_SPECIAL_HANDLER_PATTERNS)
        self._plan_overrides = dict(plan_overrides or {})

    # ── 主入口 ──────────────────────────────────────────────────────────

    def plan(
        self,
        model,
        mesh,
        *,
        tp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        sequence_parallel: bool = True,
        loss_parallel: bool = False,
    ) -> ShardingPlan:
        arch = self._get_architecture(model)
        mesh_dim_names = self._build_mesh_dim_names(mesh, tp_size, cp_size, ep_size)

        # Phase 1: 参数角色分类
        param_roles = self._classify_all_params(model, arch)

        # Phase 2: 通信边界分组
        boundary_groups = self._group_by_boundary(param_roles)

        # Phase 3+4: 语义推断 + 模板填充 I/O
        param_ndims = {name: p.ndim for name, p in model.named_parameters()}
        plan = ShardingPlan(
            mesh_dim_names=mesh_dim_names,
            sequence_parallel=sequence_parallel,
            loss_parallel=loss_parallel,
        )
        inferred_templates: Dict[str, ShardingTemplate] = {}
        for boundary_fqn, group in boundary_groups.items():
            boundary_type = self._infer_boundary_type(boundary_fqn, group)
            template = self._templates.get(boundary_type)
            if template is None:
                logger.warning(
                    "No template for boundary_type=%s at %s", boundary_type, boundary_fqn
                )
                continue
            spec = self._build_spec_from_template(
                boundary_fqn, group, template,
                sequence_parallel, loss_parallel, mesh_dim_names,
                param_ndims=param_ndims,
            )
            if spec is not None:
                plan.modules[boundary_fqn] = spec
                inferred_templates[boundary_fqn] = template

        # Phase 4.5: 用户 plan_overrides 合并（05 §3.6.7，须在 Phase 5 之前——
        # 覆盖 spec 仍要参与链式契约校验与 terminal 标记）
        self._merge_plan_overrides(plan, model, inferred_templates)

        # Phase 5: 链式传播校验
        plan = self._chain_propagate_and_validate(plan, model)

        # Phase 6: 特殊参数处理
        plan.special_handlers = self._collect_special_handlers(param_roles)

        # tied-weight 检测（embed <-> lm_head 共享存储）
        plan.tied_pairs = self._detect_tied_pairs(model)

        return plan

    # ── 架构检测 ────────────────────────────────────────────────────────

    def _get_architecture(self, model) -> str:
        """检测 canonical 架构名：config.architectures[0] > config.model_type > 类名，
        小写化并剥离 ForCausalLM 等后缀。"""
        cfg = getattr(model, "config", None)
        arch_str = None
        archs = getattr(cfg, "architectures", None)
        if archs:
            arch_str = archs[0]
        if not arch_str:
            arch_str = getattr(cfg, "model_type", None)
        if not arch_str:
            arch_str = type(model).__name__

        s = arch_str.lower()
        for suffix in ("forcausallm", "forconditionalgeneration",
                       "forsequenceclassification", "forimagetexttotext"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        return s

    def _build_mesh_dim_names(
        self, mesh, tp_size: int, cp_size: int, ep_size: int,
    ) -> Tuple[str, ...]:
        """以 mesh.mesh_dim_names 为权威顺序过滤 tp/cp/ep；未声明时按 (tp,cp,ep)
        回退；size=1 轴剔除。"""
        mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        dtensor_axes = ("tp", "cp", "ep")
        active = {ax for ax, sz in (("tp", tp_size), ("cp", cp_size), ("ep", ep_size))
                  if sz and sz > 1}
        if mesh_names:
            return tuple(n for n in mesh_names if n in dtensor_axes and n in active)
        return tuple(ax for ax in dtensor_axes if ax in active)

    # ── Phase 1 ─────────────────────────────────────────────────────────

    def _classify_all_params(self, model, arch: str) -> Dict[str, ParamRole]:
        return self._classifier.classify(model, arch)

    # ── Phase 2 ─────────────────────────────────────────────────────────

    def _group_by_boundary(
        self, param_roles: Dict[str, ParamRole],
    ) -> Dict[str, List[Tuple[str, ParamRole]]]:
        """两趟分组（修正 05 §3.6.6 伪代码的单参数 group 缺陷）：

        趟 1：按直属模块 FQN 分组（去掉 leaf 参数名）。
        趟 2：工作队列深度优先——组内角色齐全时做边界推断；unknown 则把整组
              参数向上合并到父模块并入队（父模块更浅、必然后处理；兄弟模块的
              参数先合并齐备再推断，避免 q_proj 单独被误判）。回溯到根仍
              unknown 归入参数所在模块（后续无模板命中 → warning 跳过）。
        """
        # 趟 1
        own: Dict[str, List[Tuple[str, ParamRole]]] = {}
        for fqn, role in param_roles.items():
            module_fqn = ".".join(fqn.split(".")[:-1])
            own.setdefault(module_fqn, []).append((fqn, role))

        # 趟 2
        merged: Dict[str, List[Tuple[str, ParamRole]]] = {
            mfqn: list(params) for mfqn, params in own.items()
        }
        pending = sorted(merged.keys(), key=lambda f: f.count("."), reverse=True)
        consumed: set = set()
        groups: Dict[str, List[Tuple[str, ParamRole]]] = {}
        i = 0
        while i < len(pending):
            mfqn = pending[i]
            i += 1
            if mfqn in consumed:
                continue
            params = merged.get(mfqn, [])
            if self._infer_boundary_type(mfqn, params) != "unknown":
                groups[mfqn] = params
            else:
                parent = mfqn.rsplit(".", 1)[0] if "." in mfqn else ""
                if parent:
                    if parent not in merged:
                        merged[parent] = []
                        pending.append(parent)  # 父模块更浅，尾部入队即可
                    merged[parent].extend(params)
                else:
                    # 回溯到根仍 unknown：归入参数所在模块（后续无模板 → 跳过）
                    origin = ".".join(params[0][0].split(".")[:-1]) if params else mfqn
                    groups.setdefault(origin, params)
            consumed.add(mfqn)
        return groups

    # ── Phase 3 ─────────────────────────────────────────────────────────

    def _infer_boundary_type(self, fqn: str, group: List[Tuple[str, ParamRole]]) -> str:
        """从模块 FQN + 组内参数角色识别语义角色。

        优先级：显式 FQN 模式 > 叶子段守卫 > MoE 角色 > 参数角色组合 > 默认。
        """
        fqn_lower = fqn.lower()
        seg = _last_segment(fqn)

        # 1. 显式规则（最高优先级，叶模块即边界）
        if _match_any(fqn_lower, ["embed_tokens", "wte", ".embed.", "tok_embeddings",
                                  "embed_in", "word_embeddings"]):
            return "embed"
        if _match_any(fqn_lower, ["lm_head", "embed_out", "output_layer"]):
            return "lm_head"
        if _match_any(fqn_lower, ["norm", "layernorm", "rmsnorm", "ln_"]):
            return "norm"
        if _match_any(seg, ["router"]):
            return "moe_gate"

        # 2. 叶子段守卫：投影/expert 叶模块自身不是边界容器
        if seg in _LEAF_SEGMENT_GUARD:
            return "unknown"

        # 3. MoE 角色：含 MOE_* 角色的组向上聚合到 moe 容器边界
        roles = {r for _, r in group}
        moe_roles = {ParamRole.MOE_EXPERT, ParamRole.SHARED_EXPERT, ParamRole.MOE_GATE}
        if roles & moe_roles:
            if _match_any(fqn_lower, list(_MOE_CONTAINER_PATTERNS)):
                return "moe_mlp"
            return "unknown"

        # 4. 参数角色组合
        has_colwise = any(r in (ParamRole.COLWISE, ParamRole.FUSED_QKV,
                                ParamRole.FUSED_GATE_UP) for _, r in group)
        has_rowwise = any(r == ParamRole.ROWWISE for _, r in group)
        if has_colwise and has_rowwise:
            if _match_any(fqn_lower, list(_ATTN_PATTERNS)):
                return "attention"
            if _match_any(fqn_lower, list(_MLP_PATTERNS)):
                return "mlp"
            return "attention"  # 默认 attention（更保守的 SP 通信）
        if has_colwise and not has_rowwise:
            if _match_any(fqn_lower, list(_MLP_PATTERNS)):
                return "mlp"
            return "unknown"

        return "unknown"

    # ── Phase 4 ─────────────────────────────────────────────────────────

    def _build_spec_from_template(
        self, boundary_fqn: str, group: List[Tuple[str, ParamRole]],
        template: ShardingTemplate, sequence_parallel: bool, loss_parallel: bool,
        mesh_dim_names: Tuple[str, ...], param_ndims: Optional[Dict[str, int]] = None,
    ) -> Optional[ModuleShardingSpec]:
        """Template + ParamRole → ModuleShardingSpec（05 §3.5 Template Mapping）。"""
        has_tp = "tp" in mesh_dim_names
        has_ep = "ep" in mesh_dim_names
        spec = ModuleShardingSpec()

        # Step 1: 按 ParamRole 填充 spec.params
        for param_fqn, role in group:
            param_path = param_fqn[len(boundary_fqn) + 1:]
            ndim = (param_ndims or {}).get(param_fqn, 2)
            placement = self._placement_for_role(param_path, role, template,
                                                 has_tp, has_ep, ndim=ndim)
            if placement is not None:
                spec.params[param_path] = placement

        # Step 2: 按 SP 开关选择 I/O 契约（深拷贝，避免链式传播改脏共享模板）
        if sequence_parallel:
            spec.in_src = copy.deepcopy(template.sp_in_src)
            spec.in_dst = copy.deepcopy(template.sp_in_dst)
            spec.out_src = copy.deepcopy(template.sp_out_src)
            spec.out_dst = copy.deepcopy(template.sp_out_dst)
        else:
            spec.in_src = copy.deepcopy(template.nosp_in_src)
            spec.in_dst = copy.deepcopy(template.nosp_in_dst)
            spec.out_src = copy.deepcopy(template.nosp_out_src)
            spec.out_dst = copy.deepcopy(template.nosp_out_dst)

        # Step 2.5: lm_head 的 out_dst 取决于 loss_parallel（运行时决策）。
        # CP 维恒 Shard(1)（D-07/R8）：CP 下在本地 chunk 上算 loss，不做 gather。
        if template is self._templates.get("lm_head"):
            spec.out_dst = _multi_dim(
                tp=Shard(-1) if loss_parallel else Replicate(),
                cp=Shard(1), ep=Replicate(),
            )

        # Step 2.6: embed 的 CP 契约（修订 D-05）：CP 数据管道
        # （shard_batch_for_cp，05 §6.3.4）已把 input_ids 按 CP 切好——
        # in/out 的 CP 维为 Shard(1) 而非模板默认的 Replicate，否则 boundary
        # 会把已切分的 chunk 再 scatter 一次（序列被切两次）。
        has_cp = "cp" in mesh_dim_names
        if template is self._templates.get("embed") and has_cp and sequence_parallel:
            spec.in_src = {"input": _multi_dim(tp=Replicate(), cp=Shard(1),
                                               ep=Replicate())}
            spec.in_dst = {"input": _multi_dim(tp=Replicate(), cp=Shard(1),
                                               ep=Replicate())}
            spec.out_src = _multi_dim(tp=Partial(), cp=Shard(1), ep=Replicate())

        # Step 3: 特殊标记
        spec._use_local_map = template.use_local_map
        if template.needs_cp_attn:
            spec._needs_cp_attn = True

        # Step 4: 归一化 out_src/out_dst 标量简写
        return _normalize_out_fields(spec)

    @staticmethod
    def _placement_for_role(
        param_path: str, role: ParamRole, template: ShardingTemplate,
        has_tp: bool, has_ep: bool, ndim: int = 2,
    ) -> Optional[NamedPlacement]:
        """13 角色 → placement 映射（05 §3.5 映射表 + D-08 ndim 感知）。"""
        if role in (ParamRole.COLWISE, ParamRole.EMBED, ParamRole.LM_HEAD,
                    ParamRole.FUSED_QKV, ParamRole.FUSED_GATE_UP):
            return _multi_dim(tp=template.colwise_placement if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role == ParamRole.ROWWISE:
            return _multi_dim(tp=template.rowwise_placement if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role in (ParamRole.NORM, ParamRole.MOE_GATE):
            return _multi_dim(tp=template.norm_placement if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role == ParamRole.MOE_EXPERT:
            # 05 §3.5 NOTE：has_tp=False 时显式 Replicate（而非省略 TP 键）。
            # D-08：3D expert 权重 [E, H_out, H_in] 的 TP 维按 ndim 平移。
            tp_p = (_moe_expert_tp_placement(param_path, ndim, template)
                    if has_tp else Replicate())
            return _multi_dim(tp=tp_p, cp=Replicate(),
                              ep=template.moe_expert_placement if has_ep else None)
        if role == ParamRole.SHARED_EXPERT:
            # EP 维全复制；TP 按 w1/w3(colwise)/w2(rowwise)
            tp_p = _infer_colwise_vs_rowwise(param_path, template)
            return _multi_dim(tp=tp_p if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role == ParamRole.BIAS:
            return _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
        # SPECIAL → Phase 6；SKIP → 不分片
        return None

    # ── Phase 4.5: 用户 spec 覆盖（05 §3.6.7） ───────────────────────────

    def _merge_plan_overrides(
        self, plan: ShardingPlan, model,
        inferred_templates: Dict[str, ShardingTemplate],
    ) -> None:
        """合并用户手写 spec（plan_overrides），在 Phase 5 之前执行。

        语义：
        - fqn 已命中 planner 生成的 spec → 整体替换（用户 spec 为权威）；
        - fqn 未命中（planner 漏识别/无模板/无参数模块）→ 插入；
        - 结构标记 ``_use_local_map`` / ``_needs_cp_attn`` 从推断模板补齐
          （它们是模块结构属性而非 I/O 契约：MoE all-to-all 与 CP K/V
          all-gather 缺失会导致数值错误，因此模板推断为 True 时强制置位，
          用户 spec 无需也不应负责）；
        - ``out_src``/``out_dst`` 标量简写在此归一化；
        - ``_is_terminal`` 由 Phase 5 统一标记，用户预设值会被覆盖；
        - 深拷贝用户 spec——plan() 可重复调用，chain 传播会就地改 in_src，
          不能污染调用方持有的对象。
        """
        if not self._plan_overrides:
            return
        module_names = {name for name, _ in model.named_modules()}
        for fqn, user_spec in self._plan_overrides.items():
            if not isinstance(user_spec, ModuleShardingSpec):
                raise TypeError(
                    f"plan_overrides[{fqn!r}] 必须是 ModuleShardingSpec，"
                    f"得到 {type(user_spec).__name__}"
                )
            if fqn not in module_names:
                raise ValueError(
                    f"plan_overrides 的 FQN 未在模型 named_modules 中命中: {fqn!r}"
                    f"（检查拼写；PP 场景请对单 part 模型分别 plan）"
                )
            spec = copy.deepcopy(user_spec)
            template = inferred_templates.get(fqn)
            if template is not None:
                if template.use_local_map:
                    spec._use_local_map = True
                if template.needs_cp_attn:
                    spec._needs_cp_attn = True
            _normalize_out_fields(spec)
            action = "替换" if fqn in plan.modules else "插入"
            logger.info("plan_overrides: %s模块 %s 的 spec", action, fqn)
            plan.modules[fqn] = spec

    # ── Phase 5 ─────────────────────────────────────────────────────────

    def _chain_propagate_and_validate(self, plan: ShardingPlan, model) -> ShardingPlan:
        """链式传播：填充缺省 in_src + 校验相邻模块契约一致性。

        匹配规则（对 05 §3.6.5 的修订——模板 in_src key 与上游 out_dst key
        可能不同名，如 attention out "output" vs moe_mlp in "x_BLD"）：
        - 双方都恰好 1 个 entry 时按"唯一 arg"配对（名字无关）；
        - 否则按 key 名配对；
        - next.in_src 整体为空时，用上游唯一 out_dst 值填充其 in_dst 声明的 key。
        """
        sorted_fqns = self._topological_sort_by_forward_order(
            list(plan.modules.keys()), model
        )

        non_terminal: set = set()
        for i in range(len(sorted_fqns) - 1):
            curr_fqn, next_fqn = sorted_fqns[i], sorted_fqns[i + 1]
            curr_spec = plan.modules[curr_fqn]
            next_spec = plan.modules[next_fqn]
            if curr_spec.out_dst is None:
                continue

            pairs = self._pair_contracts(curr_spec.out_dst, next_spec)
            for out_key, in_key in pairs:
                out_placement = curr_spec.out_dst[out_key]
                if in_key is None:
                    continue
                non_terminal.add(curr_fqn)  # out_dst 被下游引用
                declared = next_spec.in_src.get(in_key)
                if not declared:
                    # 场景 1：填充缺省
                    next_spec.in_src[in_key] = out_placement
                    continue
                # 场景 3：校验一致性
                next_in = tuple(resolve_placements(declared, plan.mesh_dim_names))
                curr_out = tuple(resolve_placements(out_placement, plan.mesh_dim_names))
                if next_in != curr_out:
                    raise PlacementMismatchError(
                        f"{curr_fqn} → {next_fqn}", curr_out, next_in, "chain"
                    )

        # _is_terminal 标记：out_dst 未被任何下游 in_src 引用 → terminal
        # （按链式相邻关系判定——不做跨模块 placement 值相等匹配，避免
        # lm_head 的 Replicate out_dst 被 embed 的 Replicate in_src 误引用。）
        for fqn, spec in plan.modules.items():
            spec._is_terminal = fqn not in non_terminal
        return plan

    @staticmethod
    def _pair_contracts(out_dst: Dict[str, NamedPlacement],
                        next_spec: ModuleShardingSpec):
        """产出 (out_key, in_key|None) 配对：单 entry 名字无关配对，否则按名配对。"""
        in_keys = list(next_spec.in_src.keys()) or list(next_spec.in_dst.keys())
        if len(out_dst) == 1 and len(in_keys) <= 1:
            out_key = next(iter(out_dst))
            in_key = in_keys[0] if in_keys else None
            return [(out_key, in_key)]
        pairs = []
        for out_key in out_dst:
            pairs.append((out_key, out_key if out_key in next_spec.in_src else None))
        return pairs

    def _topological_sort_by_forward_order(self, fqns: List[str], model) -> List[str]:
        """按 named_modules 注册顺序排序；未命中 FQN 追加到末尾并 warning。"""
        fqn_set = set(fqns)
        ordered: List[str] = []
        seen: set = set()
        for name, _module in model.named_modules():
            if name in fqn_set and name not in seen:
                ordered.append(name)
                seen.add(name)
        missing = fqn_set - seen
        if missing:
            logger.warning(
                "_topological_sort_by_forward_order: %d FQN 未在 named_modules "
                "中命中，追加到末尾: %s", len(missing), sorted(missing)[:5],
            )
            ordered.extend(sorted(missing))
        return ordered

    # ── Phase 6 ─────────────────────────────────────────────────────────

    def _collect_special_handlers(
        self, param_roles: Dict[str, ParamRole],
    ) -> Dict[str, str]:
        """SPECIAL 角色参数 → handler 名（未注册模式归 "default"）。"""
        result: Dict[str, str] = {}
        for fqn, role in param_roles.items():
            if role != ParamRole.SPECIAL:
                continue
            handler_name = "default"
            for pattern, hname in self._special_handler_patterns.items():
                if _match_any(fqn.lower(), [pattern.lower()]):
                    handler_name = hname
                    break
            result[fqn] = handler_name
        return result

    # ── tied weights ────────────────────────────────────────────────────

    @staticmethod
    def _detect_tied_pairs(model) -> List[Tuple[str, str]]:
        """检测 embed_tokens.weight <-> lm_head.weight 的 tied 对。

        HF tie_word_embeddings 时两端共享存储；PP 场景跨 stage 检测不到，
        需用户显式声明 plan.tied_pairs（05 detect_tied_weights 注释）。
        """
        if not getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            return []
        embed_fqn = lm_head_fqn = None
        # remove_duplicate=False：tied 参数在 named_parameters 默认去重下只出现一次。
        for name, _ in model.named_parameters(remove_duplicate=False):
            if name.endswith("embed_tokens.weight"):
                embed_fqn = name
            elif name.endswith("lm_head.weight"):
                lm_head_fqn = name
        if embed_fqn and lm_head_fqn:
            return [(embed_fqn, lm_head_fqn)]
        return []


def validate_model_compatibility(
    model, *, tp_size: int = 1, cp_size: int = 1, ep_size: int = 1,
    seq_len: Optional[int] = None,
) -> None:
    """模型侧兼容性校验（05 §6.5；与 06 的拓扑校验分工——这里只看模型 config）。"""
    config = getattr(model, "config", None)
    if config is None:
        return

    if tp_size > 1:
        heads = getattr(config, "num_attention_heads", None)
        if heads is not None and heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({heads}) must be divisible by TP ({tp_size})"
            )
        kv_heads = getattr(config, "num_key_value_heads", None)
        if kv_heads is not None and kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads ({kv_heads}) must be divisible by TP ({tp_size})"
            )
        moe_inter = getattr(config, "moe_intermediate_size", None)
        if ep_size > 1 and moe_inter is not None and moe_inter % tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size ({moe_inter}) must be divisible by TP ({tp_size})"
            )

    if cp_size > 1 and seq_len is not None and seq_len % (cp_size * 2) != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by 2*cp ({2 * cp_size})"
        )

    if ep_size > 1:
        num_experts = (getattr(config, "num_experts", None)
                       or getattr(config, "n_routed_experts", None) or 0)
        if num_experts <= 0:
            raise ValueError("EP>1 requires MoE model (num_experts > 0)")
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by EP ({ep_size})"
            )

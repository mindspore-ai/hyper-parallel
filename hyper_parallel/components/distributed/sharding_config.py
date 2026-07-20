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
"""sharding_config: 双模式 DTensor 并行策略的数据模型（05 §3.1/§3.2/§3.5 canonical）。

包含：
- ``MeshAxisName``：mesh 维度名枚举（canonical 定义，06 等后续文档 import 复用）；
- ``NamedPlacement``：``dict[MeshAxisName, Placement]`` 别名；
- ``ShardingPlan`` / ``ModuleShardingSpec``：模型级计划与单模块 I/O 契约；
- ``ShardingTemplate`` / ``TEMPLATES``：语义角色 → placement 模板（TP+CP+EP 三维）；
- ``PlacementMismatchError``：placement 声明与 DTensor 传播不一致的错误；
- ``resolve_placements`` / ``_multi_dim`` / ``_normalize_out_fields``：placement 工具。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

logger = logging.getLogger(__name__)


class MeshAxisName(str, Enum):
    """mesh 维度名的 canonical 枚举（str 枚举，可直接与 "tp" 等字符串比较/做 dict key）。"""
    TP = "tp"
    CP = "cp"
    EP = "ep"
    PP = "pp"
    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_SHARD_CP = "dp_shard_cp"
    DP_CP = "dp_cp"
    EP_SHARD = "ep_shard"


# 简写别名：模板与示例中的 {TP: ..., CP: ..., EP: ...} 字面量。
# str 枚举 key 与 "tp" 等 plain string key 互通（hash/eq 一致）。
TP = MeshAxisName.TP
CP = MeshAxisName.CP
EP = MeshAxisName.EP

# NamedPlacement = {MeshAxisName: Placement}。
# Key 是 mesh 维度名；value 中 Shard(N) 的 N 是 tensor 维度索引（05 §3.2.1）。
NamedPlacement = Dict[MeshAxisName, Placement]


class PlacementMismatchError(ValueError):
    """DTensor 传播结果与 ModuleShardingSpec 声明不一致（05 §5.3）。"""

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
    """单个模块的完整 DTensor 契约（05 §3.2）。

    四个 placement 字段构成完整 I/O 契约——运行时不做推断，直接按声明执行：

      in_src:  输入到达模块边界时的 placement（从上游模块的输出或 dataloader 来）
      in_dst:  模块内部计算需要的 placement（不等则触发通信）
      out_src: 模块内部计算自然产生的 placement（校验模式使用）
      out_dst: 下游模块期望的 placement（不等则触发通信）
    """
    # ── 参数分片：子模块路径 → NamedPlacement ──
    params: Dict[str, NamedPlacement] = field(default_factory=dict)

    # ── 输入契约 ──
    in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)

    # ── 输出契约 ──
    # out_src=None: 不做 src 校验；out_dst=None: 输出不需要 redistribution。
    # 单输出模块使用 {"output": NamedPlacement}；标量简写 {TP: ...} 会在
    # 归一化阶段（_normalize_out_fields）包装为 {"output": ...}。
    out_src: Optional[Dict[str, NamedPlacement]] = None
    out_dst: Optional[Dict[str, NamedPlacement]] = None
    # out_names: 多输出模块（返回 tuple）的输出名顺序，用于把 out_src/out_dst
    # 的 key 映射到 tuple 位置（RedistOp.arg_index）。缺省按 out_src 的 key 顺序。
    out_names: Optional[List[str]] = None

    # ── 边界标记 ──
    is_boundary: bool = True

    # ── 内部标记（由 ShardingPlanner 自动设置） ──
    _is_terminal: bool = False    # 链式传播时自动标记
    _use_local_map: bool = False  # MoE 模块: forward 内部需要 DTensor→local→DTensor
    _needs_cp_attn: bool = False  # attention 模块: inner attention 需要 CP-aware forward 替换


@dataclass
class ShardingPlan:
    """一个模型的完整分片计划（05 §3.1）。"""
    # {module_fqn: ModuleShardingSpec} — 只包含 is_boundary=True 的模块
    modules: Dict[str, ModuleShardingSpec] = field(default_factory=dict)

    # 全局开关
    sequence_parallel: bool = True
    loss_parallel: bool = False

    # 特殊参数处理器: {module_fqn.param_name: handler_name}
    special_handlers: Dict[str, str] = field(default_factory=dict)

    # mesh 维度名（与 DeviceMesh.mesh_dim_names 一致）
    mesh_dim_names: Tuple[str, ...] = ()

    # tied-weight 对：[(fqn_a, fqn_b)]，共享存储的参数（embed_tokens <-> lm_head）。
    tied_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class ShardingTemplate:
    """语义角色 → placement 模板（05 §3.5）。

    每个 I/O 字段声明全部活跃 mesh 维度（TP+CP+EP）的 placement；
    ShardingPlanner 依据实际 mesh_dim_names 过滤未启用的维度
    （resolve_placements 按 mesh_dim_names 取键，多余键自然丢弃）。

    注意：sp_out_src / nosp_out_src 等为标量 NamedPlacement 简写（单输出模块），
    在 _build_spec_from_template 归一化时包装为 {"output": ...}。
    """
    # 参数分片规则
    colwise_placement: Placement = field(default_factory=lambda: Shard(0))
    rowwise_placement: Placement = field(default_factory=lambda: Shard(1))
    norm_placement: Placement = field(default_factory=Replicate)
    moe_expert_placement: Placement = field(default_factory=lambda: Shard(0))

    # SP 模式 I/O（完整 TP+CP+EP 三维）
    sp_in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    sp_in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)
    sp_out_src: Optional[NamedPlacement] = None
    sp_out_dst: Optional[NamedPlacement] = None

    # non-SP 模式 I/O
    nosp_in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    nosp_in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)
    nosp_out_src: Optional[NamedPlacement] = None
    nosp_out_dst: Optional[NamedPlacement] = None

    # 特殊标记
    use_local_map: bool = False   # MoE EP: forward 需要 local region
    needs_cp_attn: bool = False   # CP: inner attention 需要 CP-aware forward


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


def _normalize_out_fields(spec: ModuleShardingSpec) -> ModuleShardingSpec:
    """标量简写 {TP: ...} 归一化为 {'output': {TP: ...}}（05 §3.5）。

    检测启发式：若 val 是非 None dict 且任意 value 不是 dict，则判定为标量
    NamedPlacement 简写。幂等——已是 dict 契约的二次调用不变。
    """
    for attr in ("out_src", "out_dst"):
        val = getattr(spec, attr, None)
        if val and not all(isinstance(v, dict) for v in val.values()):
            setattr(spec, attr, {"output": dict(val)})
    return spec


def _hid(tp_p, cp_p, ep_p=None) -> Dict[str, NamedPlacement]:
    """hidden_states 单输入契约简写。"""
    return {"hidden_states": _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())}


def _out(tp_p, cp_p, ep_p=None) -> NamedPlacement:
    """单输出（标量简写）契约简写。"""
    return _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())


# ── TEMPLATES：7 个语义角色的完整模板（05 §3.5，TP+CP+EP 三维声明） ──
# CP 维规则：参数恒 Replicate（CP 不切参数）；激活 Shard(1)（序列维）或 Replicate。
# EP 维规则：非 MoE 模块 Replicate；MoE experts Shard(0)。
TEMPLATES: Dict[str, ShardingTemplate] = {
    # ── Attention（q/k/v Colwise + o Rowwise） ──
    # CP 维 in_dst 保持 Shard(1)：K/V all-gather 由 inner attention wrapper 在
    # SDPA/FlexAttention 内部完成（needs_cp_attn=True），不在 boundary 层。
    "attention": ShardingTemplate(
        colwise_placement=Shard(0),          # q/k/v: [H/tp, H]
        rowwise_placement=Shard(1),          # o: [H, H/tp]
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Partial(), Shard(1)),     # 本地 Q 段输出 → CP Shard(1)
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
        needs_cp_attn=True,
    ),

    # ── MLP（gate/up Colwise + down Rowwise） ──
    # CP 维全程 Shard(1)（修订 D-06）：MLP 是 pointwise，CP 无需通信；
    # 若 in_dst CP=Replicate，TP×CP 下全序列 reduce-scatter 会产生与
    # embed/attention（cp-major）不一致的 tp-major 序列布局。
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

    # ── Norm（RMSNorm/LayerNorm：weight 全复制，零通信） ──
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

    # ── Embedding（weight Shard(0) 沿词表，输出 Partial → SP+CP） ──
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

    # ── LM Head（weight Shard(0)，输出 Shard(-1)；out_dst 视 loss_parallel 覆盖） ──
    # CP 维全程 Shard(1)（修订 D-07）：R8——boundary 层 CP 维恒 identity
    # （CP 序列 all-gather 仅发生在 attention 内部 K/V）。lm_head 在本地
    # CP chunk 上计算 logits/loss（Megatron CP 标准做法），不做 CP gather。
    "lm_head": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Shard(-1), Shard(1)),
        sp_out_dst=_out(Shard(-1), Shard(1)),   # loss_parallel=true 默认；
        # loss_parallel=false 时由 _build_spec_from_template 覆盖为 {TP: Replicate, CP: Shard(1)}
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Shard(-1), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── MoE Gate（Router：weight 全复制，出口 redistribute → EP） ──
    "moe_gate": ShardingTemplate(
        norm_placement=Replicate(),          # router weight/bias: 全复制
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Replicate()),
        sp_out_src=_out(Replicate(), Replicate()),
        sp_out_dst=_out(Replicate(), Replicate(), Shard(0)),
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Replicate(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate(), Shard(0)),
    ),

    # ── MoE MLP（gate + routed experts + optional shared experts） ──
    # CP 维同 mlp（D-06）：pointwise per-token，CP 全程 Shard(1)。
    "moe_mlp": ShardingTemplate(
        colwise_placement=Shard(0),          # expert w1/w3: Colwise on TP
        rowwise_placement=Shard(1),          # expert w2: Rowwise on TP
        norm_placement=Replicate(),          # gate/norm: 全复制
        moe_expert_placement=Shard(0),       # expert params: Shard(0) on EP
        sp_in_src={"x_BLD": _multi_dim(tp=Shard(1), cp=Shard(1), ep=Replicate())},
        sp_in_dst={"x_BLD": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        sp_out_src=_out(Partial(), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src={"x_BLD": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_in_dst={"x_BLD": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
        use_local_map=True,
    ),
}

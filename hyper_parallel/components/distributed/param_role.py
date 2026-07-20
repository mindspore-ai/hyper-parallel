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
"""param_role: ParamRole 枚举 + ParameterClassifier（05 §3.6 Phase 1）。

ParamRole 是命名规则与 ShardingTemplate 之间的桥梁：
- Phase 1 把 named_parameters() 按命名规则分类为 ParamRole；
- Phase 2 按 ParamRole 聚合通信边界；
- Phase 4 由 Template 按 ParamRole 填充 spec.params 的 placement。

ParamRole 不决定 I/O 契约——那是 Template 的语义角色（attention/mlp/...）决定的。
"""

import logging
from enum import Enum, auto
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ParamRole(Enum):
    """参数语义角色（13 个枚举值，05 §3.6）。"""
    COLWISE = auto()        # 列切线性层: q/k/v/gate/up proj → Shard(0)
    ROWWISE = auto()        # 行切线性层: o/down proj → Shard(1)
    NORM = auto()           # RMSNorm/LayerNorm weight → Replicate
    EMBED = auto()          # embedding weight → Shard(0)（词表维）
    LM_HEAD = auto()        # lm_head weight → Shard(0)（词表维）
    MOE_GATE = auto()       # MoE router/gate → Replicate
    MOE_EXPERT = auto()     # MoE routed expert → EP Shard(0) + TP colwise/rowwise
    SHARED_EXPERT = auto()  # MoE shared expert → EP Replicate + TP colwise/rowwise
    FUSED_QKV = auto()      # 融合 QKV → Shard(0)（后续 SpecialHandler 可调整）
    FUSED_GATE_UP = auto()  # 融合 gate/up → Shard(0)
    BIAS = auto()           # bias → 恒 Replicate
    SPECIAL = auto()        # 特殊参数（gated_delta 等）→ Phase 6 SpecialHandler
    SKIP = auto()           # 冻结/不分片 → 不进入 spec.params


def _match_any(name: str, patterns: List[str]) -> bool:
    """子串匹配：name 中包含任一 pattern。"""
    return any(p in name for p in patterns)


def _build_default_rules() -> List[Tuple[List[str], ParamRole]]:
    """默认命名规则：list[(patterns, ParamRole)]，按顺序首匹配。

    排序原则：更具体的规则在前（shared_experts 先于 experts；moe gate 的
    带圆点模式先于裸 "gate" 词；bias 先于 colwise/rowwise，否则 q_proj.bias
    会被 colwise 截获）。"ln"/"norm" 类模式不会误伤 "linear"/"kernel"
    （子串不含）。
    """
    return [
        (["embed_tokens.weight", "wte.weight", "tok_embeddings.weight",
          "embed_in.weight", "word_embeddings.weight"], ParamRole.EMBED),
        (["lm_head.weight", "embed_out.weight", "output_layer.weight"], ParamRole.LM_HEAD),
        (["shared_expert"], ParamRole.SHARED_EXPERT),
        (["experts"], ParamRole.MOE_EXPERT),
        ([".mlp.gate.", ".router.", "moe_gate", "mlp.router"], ParamRole.MOE_GATE),
        (["fused_qkv", "qkv_proj", "query_key_value"], ParamRole.FUSED_QKV),
        (["gate_up_proj", "fused_gate_up", ".w13."], ParamRole.FUSED_GATE_UP),
        (["a_log", "dt_bias", "gated_delta"], ParamRole.SPECIAL),
        (["norm", "layernorm", "rmsnorm", "ln_"], ParamRole.NORM),
        ([".bias"], ParamRole.BIAS),
        (["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", ".w1.", ".w3."],
         ParamRole.COLWISE),
        (["o_proj", "down_proj", ".w2."], ParamRole.ROWWISE),
    ]


class ParameterClassifier:
    """按命名规则 + 架构覆盖把 named_parameters 分类为 ParamRole（05 §3.6.6）。

    规则来源（优先级递减）：
      1. ``arch_overrides[arch]`` —— 显式 (pattern | [patterns], ParamRole) 覆盖；
      2. 默认命名规则（首匹配）；
      3. 未命中 → ``ParamRole.SKIP``。
    """

    def __init__(self, name_rules=None, arch_overrides=None):
        self._name_rules = (
            name_rules if name_rules is not None else _build_default_rules()
        )
        self._arch_overrides = arch_overrides if arch_overrides is not None else {}

    def classify(self, model, arch: str = "") -> Dict[str, ParamRole]:
        """遍历所有命名参数，返回 {param_fqn: ParamRole}。"""
        roles: Dict[str, ParamRole] = {}
        overrides = self._arch_overrides.get(arch, [])
        for name, _ in model.named_parameters():
            roles[name] = self.classify_param(name, overrides)
        return roles

    def classify_param(self, name: str, overrides=None) -> ParamRole:
        """单参数分类（overrides 缺省时不应用架构覆盖）。"""
        name_lower = name.lower()
        # 1. 架构显式覆盖（精确 FQN / 子串 / list-of-patterns 三种写法）
        for pattern, forced_role in (overrides or []):
            patterns = [pattern] if isinstance(pattern, str) else list(pattern)
            if _match_any(name_lower, [p.lower() for p in patterns]):
                return forced_role
        # 2. 默认命名规则（首匹配）
        for patterns, default_role in self._name_rules:
            if _match_any(name_lower, patterns):
                return default_role
        # 3. 兜底
        return ParamRole.SKIP

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
"""param_role: ParamRole enum + ParameterClassifier (05 §3.6 Phase 1).

ParamRole is the bridge between naming rules and ShardingTemplate:
- Phase 1 classifies named_parameters() into ParamRole by naming rules;
- Phase 2 aggregates communication boundaries by ParamRole;
- Phase 4 fills the placements of spec.params from the Template by ParamRole.

ParamRole does not decide the I/O contract — that is decided by the Template's
semantic role (attention/mlp/...).
"""

import logging
from enum import Enum, auto
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ParamRole(Enum):
    """Semantic roles of parameters (14 enum values, 05 §3.6)."""
    COLWISE = auto()        # column-sharded linear layers: q/k/v/gate/up proj → Shard(0)
    ROWWISE = auto()        # row-sharded linear layers: o/down proj → Shard(1)
    NORM = auto()           # RMSNorm/LayerNorm weight → Replicate
    EMBED = auto()          # embedding weight → Shard(0) (vocab dim)
    LM_HEAD = auto()        # lm_head weight → Shard(0) (vocab dim)
    MOE_GATE = auto()       # MoE router/gate → Replicate
    MOE_EXPERT = auto()     # MoE routed expert → EP Shard(0) + TP colwise/rowwise
    SHARED_EXPERT = auto()  # MoE shared expert → EP Replicate + TP colwise/rowwise
    FUSED_QKV = auto()      # fused QKV → Shard(0) (a later SpecialHandler may adjust)
    FUSED_GATE_UP = auto()  # fused gate/up → Shard(0)
    BIAS = auto()           # bias → always Replicate
    REPLICATED = auto()     # linear-layer weights forced to be replicated
                            # (e.g. MLA q_a/kv_a down projections)
                            # → Replicate on all dims; assigned only explicitly via
                            # ARCH_OVERRIDES, never produced by the default naming rules
    SPECIAL = auto()        # special parameters (gated_delta etc.) → Phase 6 SpecialHandler
    SKIP = auto()           # frozen / not sharded → excluded from spec.params


def _match_any(name: str, patterns: List[str]) -> bool:
    """Substring matching: name contains any of the patterns."""
    return any(p in name for p in patterns)


def _build_default_rules() -> List[Tuple[List[str], ParamRole]]:
    """Default naming rules: list[(patterns, ParamRole)], first match wins.

    Ordering principle: more specific rules come first (shared_experts before
    experts; dotted patterns of the MoE gate before the bare "gate" word; bias
    before colwise/rowwise, otherwise q_proj.bias would be captured by
    colwise). The "ln"/"norm"-style patterns do not misfire on
    "linear"/"kernel" (neither contains them as a substring).
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
    """Classifies named_parameters into ParamRole by naming rules + arch overrides (05 §3.6.6).

    Rule sources (in decreasing priority):
      1. ``arch_overrides[arch]`` — explicit (pattern | [patterns], ParamRole) overrides;
      2. default naming rules (first match);
      3. no match → ``ParamRole.SKIP``.
    """

    def __init__(self, name_rules=None, arch_overrides=None):
        self._name_rules = (
            name_rules if name_rules is not None else _build_default_rules()
        )
        self._arch_overrides = arch_overrides if arch_overrides is not None else {}

    def classify(self, model, arch: str = "") -> Dict[str, ParamRole]:
        """Iterate over all named parameters and return {param_fqn: ParamRole}."""
        roles: Dict[str, ParamRole] = {}
        overrides = self._arch_overrides.get(arch, [])
        for name, _ in model.named_parameters():
            roles[name] = self.classify_param(name, overrides)
        return roles

    def classify_param(self, name: str, overrides=None) -> ParamRole:
        """Classify a single parameter (arch overrides are not applied when overrides is omitted)."""
        name_lower = name.lower()
        # 1. Explicit arch overrides (three forms: exact FQN / substring /
        #    list-of-patterns)
        for pattern, forced_role in (overrides or []):
            patterns = [pattern] if isinstance(pattern, str) else list(pattern)
            if _match_any(name_lower, [p.lower() for p in patterns]):
                return forced_role
        # 2. Default naming rules (first match)
        for patterns, default_role in self._name_rules:
            if _match_any(name_lower, patterns):
                return default_role
        # 3. Fallback
        return ParamRole.SKIP

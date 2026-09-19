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
from typing import Any, Dict, List, Optional, Tuple

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
    BIAS = auto()           # unmatched bias → Replicate; Linear bias follows its weight role
    REPLICATED = auto()     # linear-layer weights forced to be replicated
                            # (e.g. MLA q_a/kv_a down projections)
                            # → Replicate on all dims; assigned only explicitly via
                            # the family's sharding_rules (ModelAdapterSpec),
                            # never produced by the default naming rules
    SPECIAL = auto()        # special parameters (gated_delta etc.) → Phase 6 SpecialHandler
    SKIP = auto()           # frozen / not sharded → excluded from spec.params


def _match_any(name: str, patterns: List[str]) -> bool:
    """Substring matching: name contains any of the patterns."""
    return any(p in name for p in patterns)


# Match modes for default naming rules (accuracy fix F1 — segment-aware
# matching; the accuracy_problem.md 10.1 misclassification came from matching
# "shared_expert" as a raw substring of the whole FQN, which also hits
# "shared_expert_gate").
SEGMENT_EXACT = "segment_exact"          # a full '.'-segment must equal the pattern
SEGMENT_SUBSTRING = "segment_substring"  # pattern is a substring OF ONE segment


def _match_pattern(name_lower: str, pattern: str, mode: str) -> bool:
    """Match one pattern against a lower-cased parameter FQN.

    Dotted patterns (e.g. ".mlp.gate.", "embed_tokens.weight", ".bias")
    always keep the legacy dotted-path substring semantics. Dot-less patterns
    are matched per segment according to the rule's declared mode:

    - SEGMENT_EXACT: a whole segment must equal the pattern — for container
      names ("shared_expert"/"shared_experts") where a longer sibling name
      ("shared_expert_gate") must NOT match;
    - SEGMENT_SUBSTRING: the pattern may be a fragment of one segment — for
      leaf-name fragments ("norm" hits "input_layernorm"; "q_proj" hits
      "q_proj"), while never spanning a '.' boundary.
    """
    if "." in pattern:
        return pattern in name_lower
    segments = name_lower.split(".")
    if mode == SEGMENT_EXACT:
        return any(seg == pattern for seg in segments)
    return any(pattern in seg for seg in segments)


def _match_rule(name_lower: str, patterns: List[str], mode: str) -> bool:
    """Return True when any pattern of one rule matches the lower-cased FQN."""
    return any(_match_pattern(name_lower, p, mode) for p in patterns)


def _build_default_rules() -> List[Tuple[List[str], ParamRole, str]]:
    """Default naming rules: list[(patterns, ParamRole, match_mode)],
    first match wins.

    Ordering principle: more specific rules come first (shared_expert(s)
    before experts; dotted patterns of the MoE gate before the bare "gate"
    word). Colwise Linear-name rules intentionally precede the generic bias
    fallback so their bias follows the output-channel shard. The fallback
    precedes rowwise rules: rowwise bias is a full output vector and remains
    replicated; adding it exactly once after TP reduction is an execution-
    layer concern. The "ln"/"norm"-style patterns do not misfire on
    "linear"/"kernel" (neither contains them within one segment).

    Match modes (F1): MoE container names use SEGMENT_EXACT so that e.g.
    "shared_expert_gate" is NOT classified SHARED_EXPERT (it is a scalar
    gate Linear, not the shared expert body — see accuracy_fix_plan.md §2);
    leaf-name fragments use SEGMENT_SUBSTRING; dotted patterns keep the
    legacy path-substring semantics regardless of the declared mode.
    """
    return [
        (["embed_tokens.weight", "wte.weight", "tok_embeddings.weight",
          "embed_in.weight", "word_embeddings.weight"], ParamRole.EMBED, SEGMENT_SUBSTRING),
        (["lm_head.weight", "embed_out.weight", "output_layer.weight"], ParamRole.LM_HEAD, SEGMENT_SUBSTRING),
        (["shared_expert", "shared_experts"], ParamRole.SHARED_EXPERT, SEGMENT_EXACT),
        (["experts"], ParamRole.MOE_EXPERT, SEGMENT_SUBSTRING),
        ([".mlp.gate.", ".router.", "moe_gate", "mlp.router"], ParamRole.MOE_GATE, SEGMENT_SUBSTRING),
        (["fused_qkv", "linear_qkv", "qkv_proj", "query_key_value"],
         ParamRole.FUSED_QKV, SEGMENT_SUBSTRING),
        (["gate_up_proj", "fused_gate_up", ".w13."], ParamRole.FUSED_GATE_UP, SEGMENT_SUBSTRING),
        (["a_log", "dt_bias", "gated_delta"], ParamRole.SPECIAL, SEGMENT_SUBSTRING),
        (["norm", "layernorm", "rmsnorm", "ln_"], ParamRole.NORM, SEGMENT_SUBSTRING),
        (["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", ".w1.", ".w3."],
         ParamRole.COLWISE, SEGMENT_SUBSTRING),
        ([".bias"], ParamRole.BIAS, SEGMENT_SUBSTRING),
        (["o_proj", "down_proj", ".w2."], ParamRole.ROWWISE, SEGMENT_SUBSTRING),
    ]


class ParameterClassifier:
    """Classifies named_parameters into ParamRole by naming rules + arch overrides (05 §3.6.6).

    Rule sources (in decreasing priority):
      1. ``arch_overrides[arch]`` — explicit (pattern | [patterns], ParamRole)
         overrides (legacy full-name substring semantics — they are explicit
         user intent);
      2. default naming rules (first match; each rule declares its match mode,
         see ``_build_default_rules``);
      3. no match → ``ParamRole.SKIP``.

    User-supplied ``name_rules`` may use either the current 3-tuple form
    ``(patterns, role, mode)`` or the legacy 2-tuple form ``(patterns, role)``
    (treated as full-name substring, the pre-F1 semantics).
    """

    def __init__(
        self,
        name_rules: Optional[List[tuple]] = None,
        arch_overrides: Optional[Dict[str, list]] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            name_rules: Custom naming rules; falls back to
                ``_build_default_rules()`` when None.
            arch_overrides: ``{arch: [(pattern | [patterns], ParamRole)]}``
                per-architecture overrides; empty when None.
        """
        self._name_rules = (
            name_rules if name_rules is not None else _build_default_rules()
        )
        self._arch_overrides = arch_overrides if arch_overrides is not None else {}

    def classify(self, model: Any, arch: str = "") -> Dict[str, ParamRole]:
        """Iterate over all named parameters and return {param_fqn: ParamRole}.

        Args:
            model: The model whose ``named_parameters()`` are classified.
            arch: Architecture key selecting the ``arch_overrides`` entry.

        Returns:
            Mapping of parameter FQN to its classified ParamRole.
        """
        roles: Dict[str, ParamRole] = {}
        overrides = self._arch_overrides.get(arch, [])
        for name, _ in model.named_parameters():
            roles[name] = self.classify_param(name, overrides)
        return roles

    def classify_param(
        self,
        name: str,
        overrides: Optional[List[Tuple[Any, ParamRole]]] = None,
    ) -> ParamRole:
        """Classify a single parameter (arch overrides are not applied when overrides is omitted).

        Args:
            name: The parameter FQN to classify.
            overrides: Explicit ``(pattern | [patterns], ParamRole)`` override
                rules checked before the default naming rules.

        Returns:
            The matched ParamRole; ``ParamRole.SKIP`` when nothing matches.
        """
        name_lower = name.lower()
        # 1. Explicit arch overrides (three forms: exact FQN / substring /
        #    list-of-patterns)
        for pattern, forced_role in (overrides or []):
            patterns = [pattern] if isinstance(pattern, str) else list(pattern)
            if _match_any(name_lower, [p.lower() for p in patterns]):
                return forced_role
        # 2. Default naming rules (first match)
        for rule in self._name_rules:
            if len(rule) == 3:
                patterns, default_role, mode = rule
            else:  # legacy 2-tuple: full-name substring
                patterns, default_role = rule
                mode = None
            if mode is None:
                if _match_any(name_lower, patterns):
                    return default_role
            elif _match_rule(name_lower, patterns, mode):
                return default_role
        # 3. Fallback
        return ParamRole.SKIP

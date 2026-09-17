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
"""registration: architecture ID / capability contract for DeepSeek-V3.

Registers the family's ``ModelAdapterSpec`` with the shared
``models/registry.py`` (adjust doc §7.2). The MLA naming-rule overrides
(sharding_rules) moved here from the planner's transitional
``ARCH_OVERRIDES`` table — the generic planner no longer carries
per-family knowledge.

DeepSeek-V2 shares the same MLA structure, so its spec is registered here
too with the same sharding rules (registry ``_FAMILY_DIR_ALIASES`` routes
the ``deepseek_v2`` lookup to this module).

The providers stay lazy so registry discovery keeps working on CPU-only
checkouts; this module itself never imports Trainer/Data or torch.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return the family's replacement-factory module (lazy provider)."""
    from hyper_parallel.models.deepseek_v3.adapter import (  # pylint: disable=C0415
        replacements,
    )
    return replacements


def _load_sharding_rules():
    """Return the MLA naming-rule overrides (lazy provider).

    DeepSeek MLA (deepseek_v2/v3 share the same structure): the q_a/kv_a
    down-projections are forced to replicated (the LoRA rank dim is not
    sharded); the q_b/kv_b up-projections are colwise along the head dim —
    isomorphic to the standard attention template (the o_proj rowwise
    contract over the head dim is unchanged).
    """
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )
    return [
        (["q_a_proj", "kv_a_proj_with_mqa"], ParamRole.REPLICATED),
        (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
    ]


DEEPSEEK_V3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV3ForCausalLM",
    model_type="deepseek_v3",
    replacements=_load_replacements,
    sharding_rules=_load_sharding_rules,
)
register_model_adapter(DEEPSEEK_V3_ADAPTER_SPEC)

# DeepSeek-V2 is structurally isomorphic for sharding purposes (same MLA
# naming); it gets its own spec identity but shares the rules provider.
DEEPSEEK_V2_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV2ForCausalLM",
    model_type="deepseek_v2",
    sharding_rules=_load_sharding_rules,
)
register_model_adapter(DEEPSEEK_V2_ADAPTER_SPEC)

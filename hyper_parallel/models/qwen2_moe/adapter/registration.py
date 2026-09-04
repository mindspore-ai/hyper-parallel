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
"""registration: architecture ID / capability contract for Qwen2-MoE.

Registers the family's ``ModelAdapterSpec`` with the shared
``models/registry.py`` (adjust doc §7.2). Only ``sharding_rules`` is
declared: the shared_expert_gate naming override (accuracy fix F2,
accuracy_fix_plan.md §2) moved here from the planner's transitional
``ARCH_OVERRIDES`` table. The family otherwise uses HF-native modeling
code with the generic templates.

The provider stays lazy so registry discovery keeps working on CPU-only
checkouts; this module itself never imports Trainer/Data or torch.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_sharding_rules():
    """Return the shared_expert_gate naming override (lazy provider).

    shared_expert_gate is a scalar-gate Linear(H, 1) computed per token —
    "the parameter must be replicated" ≠ "the module has router semantics",
    so it is forced to REPLICATED (never MOE_GATE, which would anchor a
    spurious router boundary; never SHARED_EXPERT, which would shard its
    single row — see accuracy_problem.md 10.1).
    """
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )
    return [
        (["shared_expert_gate"], ParamRole.REPLICATED),
    ]


QWEN2_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen2MoeForCausalLM",
    model_type="qwen2_moe",
    sharding_rules=_load_sharding_rules,
)
register_model_adapter(QWEN2_MOE_ADAPTER_SPEC)

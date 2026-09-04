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
"""adapter_spec: ModelAdapterSpec — the shared model-adapter data contract.

One ``ModelAdapterSpec`` per model family declares the architecture
identity and the family's adapter providers (structure replacements,
attention contract, checkpoint mapping, model-specific TP/CP/EP rules).
The generic builders receive the spec through ``models/registry.py`` —
they never branch on model class names themselves (05 §15.9 step 1,
adjust doc §4/§7.2). This module holds only the data contract, never
model-class-name branches.

Provider fields stay ``None`` until the family's adapter modules land
(Qwen3-MoE: replacements/attention in M2, distributed rules in M3).
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ModelAdapterSpec:
    """Architecture identity + adapter providers for one model family.

    Attributes:
        architecture: HF ``config.architectures[0]`` (e.g.
            ``"Qwen3MoeForCausalLM"``); native (non-HF) models register
            their own model class name here.
        model_type: HF ``config.model_type`` (e.g. ``"qwen3_moe"``) — the
            registry lookup key.
        replacements: provider returning the family's module-replacement
            declarations (pointing at the generic high-performance
            ``modules`` entries — never re-implementing kernels).
        attention: provider returning the family's attention contract
            (parameter names, mask/cache/forward adaptation).
        checkpoint: provider returning family-specific checkpoint
            key/layout mappings, when the generic mapping is insufficient.
        context_parallel: provider returning the family's CP wrappers.
        expert_parallel: provider returning the family's EP compute
            factories.
        sharding_rules: provider returning the family's planner naming-rule
            overrides — ``[(pattern | [patterns], ParamRole), ...]`` checked
            before the default naming rules in Phase 1 (e.g. DeepSeek MLA's
            replicated down-projections). Lives here so the generic planner
            never carries per-family knowledge.
    """

    architecture: str
    model_type: str
    replacements: Optional[Callable[..., Any]] = None
    attention: Optional[Callable[..., Any]] = None
    checkpoint: Optional[Callable[..., Any]] = None
    context_parallel: Optional[Callable[..., Any]] = None
    expert_parallel: Optional[Callable[..., Any]] = None
    sharding_rules: Optional[Callable[..., Any]] = None

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
# pylint: disable=missing-apache-license-header
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class RecomputePolicy:
    """Model-owned activation-checkpoint regions used by normal training.

    One declaration covers both region shapes a model may need:

    * ``region_patterns`` names the modules that become checkpoint regions. The
      outermost match inside every selected layer is used, so a pattern set of
      leaf submodules keeps one region per submodule while a layer pattern wraps
      the whole layer and saves a single copy of the layer input.
    * ``exclude_patterns`` names descendants that must stay outside the
      recompute region, for a submodule whose forward may not run a second time
      at all. A cross-layer state producer or consumer does not need an
      exclusion: a model may instead guard its state writes with
      ``is_recomputing()`` and pin the discrete results a recompute has to
      reuse, as DeepSeek-V4.1 does for its indexer selection.

    A module may not be a region root and an excluded descendant at the same
    time, and an excluded module may not contain a region root. Layer coverage
    is selected independently through the Trainer's activation-checkpoint
    config.

    ``on_applied`` lets a model enable runtime behaviour that depends on the
    region shape finally installed, for example pinning the discrete selection
    of every layer that will be recomputed.

    ``exclude_predicate`` complements the patterns with an attribute rule that a
    path cannot express, such as "keep the attention of every role that does not
    recompute out of the region". It receives ``(module_fqn, module)`` and is
    evaluated together with ``exclude_patterns``.
    """

    region_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...] = ()
    exclude_predicate: Optional[Callable[[str, Any], bool]] = None
    on_applied: Optional[Callable[[Any], None]] = None


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
        materialization: provider accepting the constructed model and registering
            deterministic derived buffers or state hooks before sharding and
            ``to_empty()``. This supports unmodified native HF model classes.
        context_parallel: provider returning the family's CP wrappers.
        expert_parallel: provider returning the family's EP compute
            factories.
        sharding_rules: provider returning the family's planner naming-rule
            overrides — ``[(pattern | [patterns], ParamRole), ...]`` checked
            before the default naming rules in Phase 1 (e.g. DeepSeek MLA's
            replicated down-projections). Lives here so the generic planner
            never carries per-family knowledge.
        fsdp_wrap_modules: provider accepting the built model and returning
            exact module FQNs that form additional FSDP child units. Use this
            for non-decoder execution branches such as a vision tower or
            multimodal projector; the generic HF decoder discovery remains the
            default when this provider is absent.
        fsdp_excluded_subtrees: provider returning exact module FQNs whose
            subtrees must not be interpreted as HF decoder containers. The
            adapter may still declare nested FSDP units inside those branches.
        fsdp_execution_order: provider accepting the built model and every
            selected child-unit FQN, then returning their first-forward
            execution order. This lets conditional multimodal models override
            module-registration order for FSDP communication prefetching.
        recompute: provider returning normal-training activation-checkpoint
            policy. This must not depend on validation-only imports.
        validation: lazy provider returning the family's ``ModelValidationSpec``.
            Normal training never calls it; authoritative repositories and
            validation-only dependencies therefore stay outside import paths.
        loss: provider returning model-family output-loss adapters that must
            intercept the model before a full terminal output is materialized.
        encoder_parallel: lazy provider returning construction-time visual
            replacement/isolation rules and a factory for a model-bound
            Encoder-DP/Encoder-CP bridge.
    """

    architecture: str
    model_type: str
    replacements: Optional[Callable[..., Any]] = None
    attention: Optional[Callable[..., Any]] = None
    checkpoint: Optional[Callable[..., Any]] = None
    materialization: Optional[Callable[..., Any]] = None
    external_state: Optional[Callable[..., Any]] = None
    context_parallel: Optional[Callable[..., Any]] = None
    expert_parallel: Optional[Callable[..., Any]] = None
    sharding_rules: Optional[Callable[..., Any]] = None
    fsdp_wrap_modules: Optional[Callable[..., Any]] = None
    fsdp_excluded_subtrees: Optional[Callable[..., Any]] = None
    fsdp_execution_order: Optional[Callable[..., Any]] = None
    recompute: Optional[Callable[..., RecomputePolicy]] = None
    validation: Optional[Callable[..., Any]] = None
    loss: Optional[Callable[..., Any]] = None
    encoder_parallel: Optional[Callable[..., Any]] = None

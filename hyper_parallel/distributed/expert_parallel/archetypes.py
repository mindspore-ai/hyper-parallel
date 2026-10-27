# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Framework structure->archetype table for inline EP/codegen declarations.

The MoE dispatch is selected by *module structure* (never by model identity).
Every ep-strategy fact the generated artifact needs -- the strategy ``kind``,
the ``strip_boundary_subpatterns``, the ``moe_ep_forward`` structural keys --
is owned here, keyed by the ``detect_moe_structure`` fingerprint. The codegen
framework projects from this table instead of hand-writing family-named
literals; the manager reuses the same single source for the compute-factory
``_target_``.

The table is a pure *capability* matrix: it declares what the framework can
orchestrate for a structure, never which class carries it. The concrete block
class (``DeepseekV3MoE``, ``DeepseekV32MoE``, any custom name) is a model
fact the codegen pipeline derives per artifact from the frozen boundary map,
so a new model with a known structure needs no registration here.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Structural fingerprint fields, mirroring ``MoeStructure``.
FINGERPRINT_ROUTER = 0
FINGERPRINT_SHARED = 1
FINGERPRINT_STORAGE = 2


@dataclass(frozen=True)
class MoeArchetype:
    """Inline EP capabilities that structure detection proves for a MoE.

    Deliberately carries no target class: the concrete MoE block class is a
    per-model fact (any custom model may name it anything) and is derived by
    the codegen pipeline from the frozen boundary map at materialization
    time. This table only answers "what can the framework orchestrate for
    this structural fingerprint".
    """

    #: Structure-keyed strategy ``kind`` (not a model-family dispatch literal).
    kind: str
    strip_boundary_subpatterns: tuple[str, ...]
    #: Runtime router-adapter key the generic ``moe_ep_forward`` resolves
    #: through ``MOE_ROUTER_ADAPTERS``. This is the *merge contract* key and
    #: may differ from the detection fingerprint's router field (for example a
    #: grouped-sigmoid family whose gate module is detected as a top-k router
    #: module still merges with the ``sigmoid_group`` adapter).
    router_kind: str
    shared: str
    #: The importable compute-factory ``_target_`` the YAML/native path anchors.
    compute_factory: str


_MOE_ARCHETYPE_BY_STRUCTURE = {
    ("topk_router_module", "none", "batched_parameters"): MoeArchetype(
        kind="moe_ep_routed",
        strip_boundary_subpatterns=(),
        router_kind="topk_router_module",
        shared="none",
        compute_factory=(
            "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
            "qwen3moe_ep_compute_fn"
        ),
    ),
    ("topk_router_module", "additive", "batched_parameters"): MoeArchetype(
        kind="moe_ep_routed",
        strip_boundary_subpatterns=("experts.*", "shared_experts"),
        router_kind="sigmoid_group",
        shared="additive",
        compute_factory=(
            "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn"
        ),
    ),
}


def moe_archetype(
    router: str,
    shared_experts: str,
    expert_storage: str,
    *,
    default: MoeArchetype | None = None,
) -> MoeArchetype | None:
    """Look up the EP archetype for a MoE structural fingerprint."""
    return _MOE_ARCHETYPE_BY_STRUCTURE.get(
        (router, shared_experts, expert_storage), default
    )


def moe_archetypes() -> dict[tuple[str, str, str], MoeArchetype]:
    """Return the framework structure->archetype table (read-only by convention)."""
    return dict(_MOE_ARCHETYPE_BY_STRUCTURE)


def ep_compute_factory_for(router: str, shared_experts: str, expert_storage: str) -> str | None:
    """Return the importable compute-factory ``_target_`` for a fingerprint."""
    archetype = moe_archetype(router, shared_experts, expert_storage)
    return archetype.compute_factory if archetype is not None else None


def supported_fingerprints() -> tuple[tuple[str, str, str], ...]:
    """The structure fingerprints the framework has EP capabilities for."""
    return tuple(sorted(_MOE_ARCHETYPE_BY_STRUCTURE))


def format_supported_fingerprints() -> str:
    """Human-readable support matrix for fail-fast diagnostics."""
    return "; ".join(
        f"(router={router}, shared={shared}, storage={storage})"
        for router, shared, storage in supported_fingerprints()
    )


__all__ = [
    "MoeArchetype",
    "ep_compute_factory_for",
    "format_supported_fingerprints",
    "moe_archetype",
    "moe_archetypes",
    "supported_fingerprints",
]

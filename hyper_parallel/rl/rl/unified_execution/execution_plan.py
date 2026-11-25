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
"""JSON-only contracts for shared row projections and role-specific EP."""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase

from rl.unified_execution.models import get_model_family, model_families

EXPERTS_TARGET = "hyper_parallel.components.modules.GroupedExperts"
ROLLOUT_EXPERTS_TARGET = "rl.unified_execution.vllm.moe.grouped_experts.build_rollout_experts"
PIK_GQA_TARGET = "rl.unified_execution.modules.grouped_query_attention.build_grouped_query_attention"
PIK_LINEAR_TARGET = "rl.unified_execution.modules.row_parallel_linear.build_pik_linear"


def pik_row_matches(family: str) -> tuple[str, ...]:
    """Keep expert-internal projections outside the ordinary-row contract."""
    return get_model_family(family).row_projections


def _target(entry: dict) -> str | None:
    implementation = entry.get("replace", {}).get("implementation", {})
    return implementation.get("_target_") if isinstance(implementation, dict) else implementation


def _effective(plan: list[dict], name: str, sp: bool) -> dict:
    """Resolve matching method and layout rules in declaration order."""
    effective = {}
    for entry in plan:
        if entry.get("replace", {}).get("kind") == "module":
            continue
        condition = entry.get("when")
        if condition is not None and not (condition == "sequence_parallel" and sp):
            continue
        match = entry["match"]
        patterns = [match] if isinstance(match, str) else match
        if any(fnmatchcase(name, pattern) for pattern in patterns):
            effective.update(entry)
    return effective


def _validate_layouts(plan: list[dict], matches: tuple[str, ...]) -> None:
    """Require identity row outputs and matching parent boundaries."""
    for sp in (False, True):
        output = "shard(1)" if sp else "replicate"
        for match in matches:
            name = match.replace("*", "model.layers.0")
            if name.endswith(".self_attn.o_proj"):
                continue  # HP validates the composed GQA layout during planning.
            parent = _effective(plan, name.rsplit(".", 1)[0], sp)
            if parent.get("region_dispatch") is not False:
                raise ValueError("PIK parents require region_dispatch=false")
            for field in ("out_src", "out_dst"):
                if parent.get(field, {}).get("output", {}).get("tp") != output:
                    raise ValueError("PIK parents must preserve the already-reduced output layout")


def _pik_entry_options(entry: dict, description, attention: bool) -> dict:
    """Validate one unconditional replacement and its fixed leaf arithmetic."""
    action = entry["replace"]
    implementation = action["implementation"]
    if (entry.get("when") is not None or action.get("kind") != "module"
            or action.get("source_type") != (description.execution_targets["attention"]
                                            if attention else "torch.nn.Linear")
            or (not attention and action.get("exact_type") is not True)):
        raise ValueError("PIK requires unconditional exact Linear module replacement")
    if not isinstance(implementation, dict) or set(implementation) != {
            "_target_", "num_leaves", "leaf_layout", "fill_uninitialized_memory"}:
        raise ValueError("PIK must explicitly fix num_leaves, leaf_layout and memory filling")
    count = implementation["num_leaves"]
    if count != "auto" and (isinstance(count, bool) or not isinstance(count, int)
                            or count < 1 or count & (count - 1)):
        raise ValueError("PIK num_leaves must be auto or a positive power of two")
    if implementation["leaf_layout"] not in ("strided", "packed"):
        raise ValueError("PIK leaf_layout must be strided or packed")
    if not isinstance(implementation["fill_uninitialized_memory"], bool):
        raise ValueError("PIK fill_uninitialized_memory must be bool")
    return {key: value for key, value in implementation.items() if key != "_target_"}


def resolve_pik_leaves(plan: list[dict], training: Mapping, inference: Mapping) -> None:
    """Freeze automatic leaves on a private plan before either model is built.

    Each TP rank must own a complete subtree. With power-of-two TP degrees,
    max(train TP, rollout TP) is the smallest common global leaf count. An
    explicit larger count remains useful when comparing several run topologies.
    """
    for entry in plan:
        if _target(entry) not in (PIK_LINEAR_TARGET, PIK_GQA_TARGET):
            continue
        degrees = (training["tp"], inference["tp"])
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1
               or size & (size - 1) for size in degrees):
            raise ValueError("PIK training and inference TP must be powers of two (including 1)")
        implementation = entry["replace"]["implementation"]
        if implementation.get("num_leaves") == "auto":
            implementation["num_leaves"] = max(degrees)


def pik_row_options(plan: list[dict], family: str | None = None) -> dict | None:
    """Validate unconditional arithmetic installation and topology-specific boundaries."""
    selected = [entry for entry in plan if _target(entry) in (PIK_LINEAR_TARGET, PIK_GQA_TARGET)]
    if not selected:
        return None
    sources = {entry.get("replace", {}).get("source_type") for entry in selected
               if _target(entry) == PIK_GQA_TARGET}
    if family is None:
        families = [item.model_type for item in model_families() if item.execution_targets["attention"] in sources]
        if len(families) != 1:
            raise ValueError("PIK requires one unambiguous model attention source")
        family = families[0]
    description = get_model_family(family)
    matches = pik_row_matches(family)
    covered, options = [], []
    for entry in selected:
        patterns = entry["match"] if isinstance(entry["match"], list) else [entry["match"]]
        attention = _target(entry) == PIK_GQA_TARGET
        covered.extend(pattern + "." + description.attention_output if attention else pattern for pattern in patterns)
        options.append(_pik_entry_options(entry, description, attention))
    if sorted(covered) != sorted(matches) or any(option != options[0] for option in options):
        raise ValueError(f"PIK requires exactly {matches} with the same arithmetic; expert PIK is not integrated")
    _validate_layouts(plan, matches)
    return options[0]


def validate_pik_contract(plan: list[dict], policy: dict | None, family: str,
                          training: dict, inference: dict) -> None:
    """Reject incompatible topology or numerical policy before model construction."""
    options = pik_row_options(plan, family)
    if options is None:
        return
    if policy is None or policy["fill_uninitialized_memory"] != options["fill_uninitialized_memory"]:
        raise ValueError("PIK requires the same explicit Level 3 policy on both sides")
    for topology in (training, inference):
        if topology["pp"] != 1:
            raise ValueError("PIK requires PP=1")
        if options["num_leaves"] % topology["tp"]:
            raise ValueError("Both training and inference TP must divide the fixed PIK leaf count")


def validate_cp_plan(plan: list[dict], training: dict, backend: str, policy: dict | None) -> None:
    """Require the numerical recipe used by the HP CP GQA member."""
    if training["cp"] == 1:
        return
    if backend != "flash_attention_npu_v4" or policy is None:
        raise ValueError("Training CP requires Level 3 V4")
    if pik_row_options(plan) is None:
        raise ValueError("Training CP consistency requires the shared PIK attention recipe")


def validate_pcp_plan(plan: list[dict], inference: dict, backend: str, policy: dict | None) -> None:
    """Keep prefill CP on the shared V4/PIK arithmetic without a training CP wrapper."""
    if inference["cp"] == 1:
        return
    if backend != "flash_attention_npu_v4" or policy is None or pik_row_options(plan) is None:
        raise ValueError("Inference PCP requires Level 3 V4 and the shared PIK recipe")


def has_unified_ep(plan: list[dict]) -> bool:
    """Recognize the HP expert family selected for the complete decoder."""
    return any(entry.get("match") in ("*.mlp.experts", "model.layers.*.mlp.experts")
               and entry.get("when") is None and entry["replace"].get("kind") == "module"
               and _target(entry) == EXPERTS_TARGET for entry in plan)

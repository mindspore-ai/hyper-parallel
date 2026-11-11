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
"""JSON-only contracts for shared row projections and fixed-capacity EP."""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase

from rl.unified_execution.models import get_model_family, model_families


FIXED_EP_TARGET = "rl.unified_execution.components.moe.fixed_capacity_ep_compute_fn"
PIK_ROW_TARGET = "rl.unified_execution.components.row_parallel.pik_row_compute_fn"
PIK_ATTENTION_TARGET = "rl.unified_execution.components.row_parallel.replace_attention_with_pik"
PIK_MODULE_TARGET = "rl.unified_execution.components.row_parallel.replace_pik_linear"


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
        if isinstance(match, str) and fnmatchcase(name, match):
            effective.update(entry)
    return effective


def _validate_layouts(plan: list[dict], matches: tuple[str, ...]) -> None:
    """Require identity row outputs and matching parent boundaries."""
    for sp in (False, True):
        output = "shard(1)" if sp else "replicate"
        for match in matches:
            name = match.replace("*", "model.layers.0")
            child = _effective(plan, name + ".forward", sp)
            expected = {"_target_": PIK_ROW_TARGET, "sequence_parallel": sp}
            if (child.get("replace", {}).get("implementation") != expected
                    or child.get("params") != {} or child.get("region_dispatch", False) is not False):
                raise ValueError("PIK requires explicit shared row method binding for SP and non-SP")
            for field in ("in_src", "in_dst"):
                if child.get(field) != {"input": {"tp": "shard(2)"}}:
                    raise ValueError("Qwen3 PIK input must preserve HP's [batch,sequence,K] shard")
            for field in ("out_src", "out_dst"):
                if child.get(field) != {"output": {"tp": output}}:
                    raise ValueError("PIK child output must be an identity layout transition")
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
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count & (count - 1):
        raise ValueError("PIK num_leaves must be a positive power of two")
    if implementation["leaf_layout"] not in ("strided", "packed"):
        raise ValueError("PIK leaf_layout must be strided or packed")
    if not isinstance(implementation["fill_uninitialized_memory"], bool):
        raise ValueError("PIK fill_uninitialized_memory must be bool")
    return {key: value for key, value in implementation.items() if key != "_target_"}


def pik_row_options(plan: list[dict], family: str | None = None) -> dict | None:
    """Validate unconditional arithmetic installation and topology-specific boundaries."""
    selected = [entry for entry in plan if _target(entry) in (PIK_MODULE_TARGET, PIK_ATTENTION_TARGET)]
    if not selected:
        if any(_target(entry) == PIK_ROW_TARGET for entry in plan):
            raise ValueError("PIK method binding requires the shared module replacement, including TP1")
        return None
    sources = {entry.get("replace", {}).get("source_type") for entry in selected
               if _target(entry) == PIK_ATTENTION_TARGET}
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
        attention = _target(entry) == PIK_ATTENTION_TARGET
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
        if topology["cp"] != 1 or topology["pp"] != 1:
            raise ValueError("PIK model integration currently requires CP=PP=1")
        if options["num_leaves"] % topology["tp"]:
            raise ValueError("Both training and inference TP must divide the fixed PIK leaf count")


def fixed_capacity_ep_capacity(plan: list[dict]) -> int | None:
    """Return the explicit all-layer EP capacity, or None for the original recipe."""
    entries = [entry for entry in plan if entry.get("match") == "*.mlp.forward" and entry.get("when") == "ep"]
    if not entries:
        return None
    implementation = entries[-1].get("replace", {}).get("implementation", {})
    if not isinstance(implementation, Mapping) or implementation.get("_target_") != FIXED_EP_TARGET:
        return None
    if len(entries) != 1 or entries[-1]["replace"].get("kind") != "method":
        raise ValueError("Graph EP requires one unambiguous shared all-layer method recipe")
    if set(implementation) - {"_target_", "capacity"}:
        raise ValueError("Graph EP supports only the capacity recipe option")
    capacity = implementation.get("capacity", 256)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError("Graph EP capacity must be a positive integer")
    return capacity

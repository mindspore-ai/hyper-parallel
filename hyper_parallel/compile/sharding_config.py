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
"""
Sharding Configuration - Graph-mode FSDP module configuration.

Declares which modules the graph-mode ``FSDPPass`` should shard. ``FSDPPass``
itself owns all the actual sharding logic (all_gather on parameter
placeholders, reduce_scatter on gradient outputs, in-place live-model
sharding), so this is purely a *which modules* lookup.

Note:
    ``FSDPModuleConfig`` previously carried ``reshard_after_forward``,
    ``use_cpu_offload`` and ``wrap_separately`` fields. None of them were
    read by ``FSDPPass`` — graph-mode owns reshard as a future pass, CPU
    offload lives elsewhere, and per-module separate-wrapping is implicit
    (every marked module is sharded individually). They are removed as dead
    surface; when a real reshard/offload pass lands it can re-add fields
    with a real consumer.
"""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).parent / "examples"

__all__ = [
    "PassPlan",
    "FSDPModuleConfig",
    "create_sharding_plan_from_yaml",
    "create_simple_sharding_plan",
]


@dataclass
class FSDPModuleConfig:
    """Marker that a single module FQN is FSDP-wrapped.

    Attributes:
        module_fqn: Module fully qualified name (or wildcard pattern). The
            same value is also the dict key in ``PassPlan.fsdp_modules``
            / ``fsdp_patterns``; it is kept on the dataclass so iterating
            ``plan.fsdp_modules.values()`` stays self-describing.
    """

    module_fqn: str


@dataclass
class PassPlan:
    """Declare which modules ``FSDPPass`` should shard.

    Two registries (exact FQN match + wildcard patterns) are checked in
    order: exact wins first, then patterns in insertion order (first match
    wins when patterns overlap).

    Pipeline-parallel stage assignment lives in
    ``pp_module_fqns_per_stage``: a list whose i-th entry is the list of
    module FQNs (exact, no wildcards) assigned to stage ``i``. When it is
    ``None`` the ``PpPass`` falls back to an automatic even split of the
    model's layer-like children (see ``pp_pass._auto_stage_split``).

    Example:
        plan = PassPlan()
        plan.fsdp_wrap("tok_embeddings")
        plan.fsdp_wrap_pattern("layers.*")
        plan.pp_stage(0, ["tok_embeddings", "layers.0"])
        plan.pp_stage(1, ["layers.1", "norm", "lm_head"])
    """

    fsdp_modules: Dict[str, FSDPModuleConfig] = field(default_factory=dict)
    fsdp_patterns: Dict[str, FSDPModuleConfig] = field(default_factory=dict)
    pp_module_fqns_per_stage: Optional[List[List[str]]] = None

    def merge(self, other: "PassPlan") -> "PassPlan":
        """Return a new plan with both registries merged (other wins on key conflict).

        Args:
            other: Plan to merge in. Entries in ``other`` overwrite entries
                with the same FQN / pattern in ``self``; a PP stage plan on
                ``other`` replaces ``self``'s wholesale (per-stage merges
                are ambiguous and unsupported).

        Returns:
            A new ``PassPlan``; ``self`` and ``other`` are not mutated.
        """
        merged = PassPlan()
        merged.fsdp_modules = {**self.fsdp_modules, **other.fsdp_modules}
        merged.fsdp_patterns = {**self.fsdp_patterns, **other.fsdp_patterns}
        if other.pp_module_fqns_per_stage is not None:
            merged.pp_module_fqns_per_stage = other.pp_module_fqns_per_stage
        elif self.pp_module_fqns_per_stage is not None:
            merged.pp_module_fqns_per_stage = self.pp_module_fqns_per_stage
        return merged

    def fsdp_wrap(self, module_fqn: str) -> "PassPlan":
        """Mark a specific module for FSDP wrapping (exact match).

        Args:
            module_fqn: Module fully qualified name.

        Returns:
            ``self`` (chainable).

        Example:
            plan.fsdp_wrap("tok_embeddings")
        """
        self.fsdp_modules[module_fqn] = FSDPModuleConfig(module_fqn=module_fqn)
        return self

    def fsdp_wrap_pattern(self, pattern: str) -> "PassPlan":
        """Mark modules for FSDP wrapping (wildcard match).

        Args:
            pattern: Module FQN pattern (``fnmatch`` wildcards, e.g. ``*``,
                ``layers.*``).

        Returns:
            ``self`` (chainable).

        Example:
            plan.fsdp_wrap_pattern("layers.*")
        """
        self.fsdp_patterns[pattern] = FSDPModuleConfig(module_fqn=pattern)
        return self

    def is_fsdp_module(self, module_fqn: str) -> bool:
        """Check if a module should be wrapped with FSDP."""
        if module_fqn in self.fsdp_modules:
            return True

        for pattern in self.fsdp_patterns:
            if fnmatch.fnmatch(module_fqn, pattern):
                return True

        return False

    def get_fsdp_config(self, module_fqn: str) -> Optional[FSDPModuleConfig]:
        """Get FSDP configuration for a module, or ``None`` if not wrapped."""
        if module_fqn in self.fsdp_modules:
            return self.fsdp_modules[module_fqn]

        for pattern, config in self.fsdp_patterns.items():
            if fnmatch.fnmatch(module_fqn, pattern):
                return config

        return None

    def pp_stage(self, stage_idx: int, module_fqns: List[str]) -> "PassPlan":
        """Declare the module FQNs of one pipeline stage (exact match).

        Args:
            stage_idx: Zero-based stage index, ``stage_idx >= 0``. Stages
                may be declared in any order / sparsely; ``PpPass``
                validates completeness (every stage ``0..pp_degree-1``
                declared, no FQN assigned twice) before splitting.
            module_fqns: Module FQNs owned by this stage, in model order.
                Exact FQNs only — wildcards are rejected because a stage
                cut must be unambiguous.

        Returns:
            ``self`` (chainable).

        Raises:
            ValueError: If ``stage_idx`` is negative or a module FQN
                contains wildcard characters.

        Example:
            plan.pp_stage(0, ["tok_embeddings", "layers.0"])
            plan.pp_stage(1, ["layers.1", "norm", "lm_head"])
        """
        if stage_idx < 0:
            raise ValueError(f"stage_idx must be >= 0, got {stage_idx}")
        for fqn in module_fqns:
            if any(ch in fqn for ch in "*?["):
                raise ValueError(
                    f"pp_stage() takes exact module FQNs, got wildcard pattern "
                    f"'{fqn}' — a stage cut must be unambiguous"
                )
        if self.pp_module_fqns_per_stage is None:
            self.pp_module_fqns_per_stage = []
        while len(self.pp_module_fqns_per_stage) <= stage_idx:
            self.pp_module_fqns_per_stage.append([])
        self.pp_module_fqns_per_stage[stage_idx] = list(module_fqns)
        return self


def create_sharding_plan_from_yaml(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> PassPlan:
    """Create PassPlan from a YAML configuration file.

    Args:
        config_path: Path to YAML config file.
        model_name: Model name (looks up in ``examples/{model_name}/config.yaml``).

    Returns:
        PassPlan object.

    Raises:
        ValueError: When neither argument is given, ``model_name`` is empty
            or contains path separators, or the YAML is not a mapping.
        FileNotFoundError: When the resolved config file does not exist.

    Example:
        plan = create_sharding_plan_from_yaml(model_name="llama3")
        plan = create_sharding_plan_from_yaml(config_path="path/to/config.yaml")
    """
    if config_path is None and model_name is None:
        raise ValueError("Must provide either config_path or model_name")

    if config_path is None:
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if ".." in model_name or "/" in model_name or "\\" in model_name:
            raise ValueError(
                f"Invalid model_name '{model_name}': must not contain path "
                "separators or parent directory references"
            )
        config_path = DEFAULT_CONFIG_DIR / model_name / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"YAML config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise ValueError(
            f"YAML config must be a mapping (dict), "
            f"got {type(config).__name__}: {config_path}"
        )

    plan = PassPlan()

    fsdp_config = _yaml_section(config, "fsdp", config_path)
    _maybe_process_fsdp(plan, fsdp_config)

    pp_config = _yaml_section(config, "pp", config_path)
    if pp_config.get("stages"):
        _process_pp(plan, pp_config)

    return plan


def _yaml_section(config: dict, key: str, config_path: Path) -> dict:
    """Return YAML ``key`` as a mapping; empty/``None`` becomes ``{}``.

    ``or {}``: a YAML key present but empty (``fsdp:`` with only comments
    under it) parses to None, and ``dict.get(key, {})`` then returns
    None instead of the default.
    """
    section = config.get(key) or {}
    if not isinstance(section, dict):
        # A present-but-empty section parses to None and is normalized to {}
        # above, so anything landing here is a real scalar/sequence typo.
        raise ValueError(
            f"YAML '{key}' section must be a mapping (e.g. nested keys or an "
            f"empty section); got {type(section).__name__} in {config_path}"
        )
    return section


def _maybe_process_fsdp(plan: PassPlan, fsdp_config: dict) -> None:
    """Run the FSDP processor when the section is enabled.

    Enabled defaults to True when explicit modules/patterns are declared.
    """
    has_explicit = bool(fsdp_config.get("modules")) or bool(fsdp_config.get("patterns"))
    if fsdp_config.get("enabled", has_explicit):
        _process_fsdp(plan, fsdp_config)


def _process_fsdp(plan: PassPlan, fsdp_config: dict) -> None:
    """Process FSDP configuration (modules + patterns) into the plan."""
    for module_config in fsdp_config.get("modules", []):
        plan.fsdp_wrap(module_config["name"])

    for pattern_config in fsdp_config.get("patterns", []):
        plan.fsdp_wrap_pattern(pattern_config["pattern"])


def _process_pp(plan: PassPlan, pp_config: dict) -> None:
    """Process PP configuration (explicit per-stage FQN lists) into the plan.

    YAML shape::

        pp:
          stages:
            - [tok_embeddings, layers.0]
            - [layers.1, norm, lm_head]

    An entry may also be a mapping with ``stage`` / ``modules`` keys for
    readability::

        pp:
          stages:
            - stage: 0
              modules: [tok_embeddings, layers.0]
    """
    for idx, stage in enumerate(pp_config["stages"]):
        if isinstance(stage, dict):
            stage_idx = stage.get("stage", idx)
            module_fqns = list(stage.get("modules", []))
        else:
            stage_idx = idx
            module_fqns = list(stage)
        plan.pp_stage(stage_idx, module_fqns)


def create_simple_sharding_plan() -> PassPlan:
    """Create a plan that FSDP-wraps every module (``*`` pattern).

    Convenience for tests / quick demos.
    """
    plan = PassPlan()
    plan.fsdp_wrap_pattern("*")
    return plan

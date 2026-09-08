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
from typing import Dict, Optional

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

    Example:
        plan = PassPlan()
        plan.fsdp_wrap("tok_embeddings")
        plan.fsdp_wrap_pattern("layers.*")
    """

    fsdp_modules: Dict[str, FSDPModuleConfig] = field(default_factory=dict)
    fsdp_patterns: Dict[str, FSDPModuleConfig] = field(default_factory=dict)

    def merge(self, other: "PassPlan") -> "PassPlan":
        """Return a new plan with both registries merged (other wins on key conflict).

        Args:
            other: Plan to merge in. Entries in ``other`` overwrite entries
                with the same FQN / pattern in ``self``.

        Returns:
            A new ``PassPlan``; ``self`` and ``other`` are not mutated.
        """
        merged = PassPlan()
        merged.fsdp_modules = {**self.fsdp_modules, **other.fsdp_modules}
        merged.fsdp_patterns = {**self.fsdp_patterns, **other.fsdp_patterns}
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

    fsdp_config = config.get("fsdp", {})
    has_explicit_modules = bool(fsdp_config.get("modules"))
    has_explicit_patterns = bool(fsdp_config.get("patterns"))
    is_enabled = fsdp_config.get(
        "enabled", has_explicit_modules or has_explicit_patterns
    )

    if is_enabled:
        _process_fsdp(plan, fsdp_config)

    return plan


def _process_fsdp(plan: PassPlan, fsdp_config: dict) -> None:
    """Process FSDP configuration (modules + patterns) into the plan."""
    for module_config in fsdp_config.get("modules", []):
        plan.fsdp_wrap(module_config["name"])

    for pattern_config in fsdp_config.get("patterns", []):
        plan.fsdp_wrap_pattern(pattern_config["pattern"])


def create_simple_sharding_plan() -> PassPlan:
    """Create a plan that FSDP-wraps every module (``*`` pattern).

    Convenience for tests / quick demos.
    """
    plan = PassPlan()
    plan.fsdp_wrap_pattern("*")
    return plan

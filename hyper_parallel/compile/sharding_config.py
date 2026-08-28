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
Sharding Configuration - FSDP Module Configuration

Provides configuration for FSDP module wrapping.
"""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

# Default config directory (examples/)
DEFAULT_CONFIG_DIR = Path(__file__).parent / "examples"

__all__ = [
    "ShardingPlan",
    "FSDPModuleConfig",
    "create_sharding_plan_from_yaml",
    "create_simple_sharding_plan",
]


@dataclass
class FSDPModuleConfig:
    """
    FSDP Module Configuration

        Configures which modules should be wrapped with FSDP.

        Args:
            module_fqn: Module fully qualified name
            reshard_after_forward: Whether to reshard after forward
            use_cpu_offload: Whether to offload to CPU
            wrap_separately: Whether to wrap this module separately (not with children)
    """

    module_fqn: str
    reshard_after_forward: bool = True
    use_cpu_offload: bool = False
    wrap_separately: bool = False


@dataclass
class ShardingPlan:
    """
    Sharding Plan - Configure which modules to wrap with FSDP.

    Example:
        plan = ShardingPlan()
        plan.fsdp_wrap("tok_embeddings")
        plan.fsdp_wrap_pattern("layers.*")
    """

    fsdp_modules: Dict[str, FSDPModuleConfig] = field(default_factory=dict)
    fsdp_patterns: Dict[str, FSDPModuleConfig] = field(default_factory=dict)

    def merge(self, other: "ShardingPlan") -> "ShardingPlan":
        """Merge two sharding plans"""
        self.fsdp_modules.update(other.fsdp_modules)
        self.fsdp_patterns.update(other.fsdp_patterns)
        return self

    def fsdp_wrap(
        self,
        module_fqn: str,
        reshard_after_forward: bool = True,
        use_cpu_offload: bool = False,
        wrap_separately: bool = False,
    ) -> "ShardingPlan":
        """
        Mark a specific module for FSDP wrapping (exact match).

        Args:
            module_fqn: Module fully qualified name
            reshard_after_forward: Whether to reshard after forward
            use_cpu_offload: Whether to offload to CPU
            wrap_separately: Whether to wrap this module separately

        Example:
            plan.fsdp_wrap("tok_embeddings")  # Specific module
        """
        self.fsdp_modules[module_fqn] = FSDPModuleConfig(
            module_fqn=module_fqn,
            reshard_after_forward=reshard_after_forward,
            use_cpu_offload=use_cpu_offload,
            wrap_separately=wrap_separately,
        )
        return self

    def fsdp_wrap_pattern(
        self,
        pattern: str,
        reshard_after_forward: bool = True,
        use_cpu_offload: bool = False,
        wrap_separately: bool = False,
    ) -> "ShardingPlan":
        """
        Mark modules for FSDP wrapping (wildcard match).

        Args:
            pattern: Module FQN pattern (supports wildcards)
            reshard_after_forward: Whether to reshard after forward
            use_cpu_offload: Whether to offload to CPU
            wrap_separately: Whether to wrap this module separately

        Example:
            plan.fsdp_wrap_pattern("layers.*")  # All transformer blocks
        """
        self.fsdp_patterns[pattern] = FSDPModuleConfig(
            module_fqn=pattern,
            reshard_after_forward=reshard_after_forward,
            use_cpu_offload=use_cpu_offload,
            wrap_separately=wrap_separately,
        )
        return self

    def is_fsdp_module(self, module_fqn: str) -> bool:
        """Check if a module should be wrapped with FSDP"""
        if module_fqn in self.fsdp_modules:
            return True

        for pattern in self.fsdp_patterns:
            if fnmatch.fnmatch(module_fqn, pattern):
                return True

        return False

    def get_fsdp_config(self, module_fqn: str) -> Optional[FSDPModuleConfig]:
        """Get FSDP configuration for a module"""
        if module_fqn in self.fsdp_modules:
            return self.fsdp_modules[module_fqn]

        for pattern, config in self.fsdp_patterns.items():
            if fnmatch.fnmatch(module_fqn, pattern):
                return config

        return None


def create_sharding_plan_from_yaml(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ShardingPlan:
    """
    Create ShardingPlan from YAML configuration file.

    Args:
        config_path: Path to YAML config file
        model_name: Model name (looks up in examples/{model_name}/config.yaml)

    Returns:
        ShardingPlan object

    Example:
        # Use predefined config
        plan = create_sharding_plan_from_yaml(model_name="llama3")

        # Use custom config
        plan = create_sharding_plan_from_yaml(config_path="path/to/config.yaml")
    """
    if config_path is None and model_name is None:
        raise ValueError("Must provide either config_path or model_name")

    if config_path is None:
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if ".." in model_name or "/" in model_name or "\\" in model_name:
            raise ValueError(
                f"Invalid model_name '{model_name}': must not contain path separators or parent directory references"
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
            f"YAML config must be a mapping (dict), got {type(config).__name__}: {config_path}"
        )

    plan = ShardingPlan()

    # Process FSDP config if explicitly enabled OR if patterns/modules are present
    fsdp_config = config.get("fsdp", {})
    has_explicit_modules = bool(fsdp_config.get("modules"))
    has_explicit_patterns = bool(fsdp_config.get("patterns"))
    is_enabled = fsdp_config.get(
        "enabled", has_explicit_modules or has_explicit_patterns
    )

    if is_enabled:
        _process_fsdp(plan, fsdp_config)

    return plan


def _process_fsdp(plan: ShardingPlan, fsdp_config: dict) -> None:
    """Process FSDP configuration"""

    for module_config in fsdp_config.get("modules", []):
        name = module_config["name"]
        reshard = module_config.get("reshard_after_forward", True)
        cpu_offload = module_config.get("use_cpu_offload", False)

        plan.fsdp_wrap(
            name,
            reshard_after_forward=reshard,
            use_cpu_offload=cpu_offload,
        )

    for pattern_config in fsdp_config.get("patterns", []):
        pattern = pattern_config["pattern"]
        reshard = pattern_config.get("reshard_after_forward", True)
        cpu_offload = pattern_config.get("use_cpu_offload", False)

        plan.fsdp_wrap_pattern(
            pattern,
            reshard_after_forward=reshard,
            use_cpu_offload=cpu_offload,
        )


def create_simple_sharding_plan() -> ShardingPlan:
    """
    Create simple sharding plan (for testing)

    FSDP wraps all modules by default.
    """
    plan = ShardingPlan()
    plan.fsdp_wrap_pattern("*")  # FSDP wrap all modules
    return plan

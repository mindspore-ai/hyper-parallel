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
"""``torchrun`` entry point for the minimal Hyper-RL GRPO demo."""

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rl.trainer import SyncTrainer


_OPTIONAL_OVERRIDE_PATHS = frozenset(("rollout.vllm.visible_devices",))

def _parse_override_value(raw_value: str) -> Any:
    """Parse a CLI override value with YAML scalar/list semantics."""
    return yaml.safe_load(raw_value)


def _apply_override(config: dict[str, Any], override: str) -> None:
    """Apply one strict ``--a.b=value`` override to an existing key."""
    if not override.startswith("--") or "=" not in override:
        raise ValueError(f"Invalid override '{override}'; expected --section.field=value")
    dot_path, raw_value = override[2:].split("=", maxsplit=1)
    keys = dot_path.split(".")
    current: dict[str, Any] = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Unknown configuration override path: {dot_path}")
        current = child
    final_key = keys[-1]
    if final_key not in current and dot_path not in _OPTIONAL_OVERRIDE_PATHS:
        raise ValueError(f"Unknown configuration override path: {dot_path}")
    current[final_key] = _parse_override_value(raw_value)


def load_config(config_path: str, overrides: list[str]) -> dict[str, Any]:
    """Load YAML and apply strict dot-path command-line overrides.

    Args:
        config_path: YAML configuration path.
        overrides: ``--section.field=value`` items.

    Returns:
        Fully merged configuration dictionary.

    Raises:
        ValueError: If the file or an override is invalid.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    config = deepcopy(loaded)
    for override in overrides:
        _apply_override(config, override)
    return config


def main() -> None:
    """Parse configuration and execute the Hyper-RL trainer."""
    parser = argparse.ArgumentParser(description="Hyper-Parallel minimal GRPO trainer")
    parser.add_argument(
        "config",
        help="Path to Hyper-RL YAML configuration",
    )
    args, overrides = parser.parse_known_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(args.config, overrides)
    trainer = SyncTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

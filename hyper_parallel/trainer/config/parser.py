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
"""Training configuration entry point: a YAML file plus dotted CLI overrides."""

from __future__ import annotations

__all__ = [
    "parse_training_args",
]

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from hyper_parallel.trainer.config.resolver import (
    ConfigResolutionError,
    replace_override_path,
    resolve_config,
)
from hyper_parallel.trainer.config.trainer import TrainerConfig


def _parse_override_tokens(tokens: Sequence[str]) -> list[tuple[str, str]]:
    """Split ``--field=value`` CLI tokens into validated paths and raw values."""
    parsed = []
    for token in tokens:
        if not token.startswith("--") or token == "--":
            raise ConfigResolutionError(
                "CLI", f"expected a dotted override in '--field=value' form, got {token!r}"
            )
        option = token[2:]
        if "=" not in option:
            raise ConfigResolutionError(
                "CLI", f"expected a dotted override in '--field=value' form, got {token!r}"
            )
        path, raw_value = option.split("=", 1)
        if not path or any(not part for part in path.split(".")):
            raise ConfigResolutionError("CLI", f"invalid override path {path!r}")
        parsed.append((path, raw_value))
    return parsed


def _apply_typed_overrides(
    config: TrainerConfig,
    overrides: Sequence[str],
) -> TrainerConfig:
    """Parse CLI override values and apply them to the resolved configuration."""
    config_with_overrides = config
    for dot_path, raw_value in _parse_override_tokens(overrides):
        try:
            parsed_value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigResolutionError(
                f"CLI.{dot_path}", f"invalid YAML value {raw_value!r}: {exc}"
            ) from exc
        config_with_overrides = replace_override_path(
            config_with_overrides, dot_path.split("."), parsed_value, path=""
        )
    return config_with_overrides


def _load_training_config(
    config_file: str | Path,
    cli_overrides: Sequence[str] = (),
) -> TrainerConfig:
    """Load a YAML configuration and apply CLI overrides."""
    config_path = Path(config_file)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigResolutionError(config_path, f"could not read config file: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigResolutionError(config_path, f"invalid YAML: {exc}") from exc

    config = resolve_config(raw)
    return _apply_typed_overrides(config, cli_overrides)


def parse_training_args(argv: Sequence[str] | None = None) -> TrainerConfig:
    """Parse a YAML configuration path and CLI overrides into a ``TrainerConfig``.

    Args:
        argv: Argument tokens excluding the program name. Uses ``sys.argv[1:]``
            when ``None``. The configuration path precedes any
            ``--field=value`` overrides.

    Returns:
        The resolved configuration with CLI overrides applied.

    Raises:
        ConfigResolutionError: The file cannot be read, or the YAML
            configuration or CLI overrides are invalid.
        SystemExit: Help is requested or required command-line arguments
            are missing.

    Example:
        config = parse_training_args(["train.yaml", "--accelerator.tp_size=4"])
    """
    parser = argparse.ArgumentParser(description="HyperParallel training config")
    parser.add_argument("config_file", help="Path to the YAML training config")
    args, overrides = parser.parse_known_args(argv)
    return _load_training_config(args.config_file, overrides)

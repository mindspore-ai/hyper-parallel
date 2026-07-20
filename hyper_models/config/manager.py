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
"""YAML loading and typed dotted overrides for ``TrainerConfig``."""

from __future__ import annotations

import argparse
import difflib
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Sequence, get_type_hints

import yaml

from hyper_models.config.resolver import (
    ConfigResolutionError,
    coerce_value,
    resolve_root,
)
from hyper_models.trainer.config import TrainerConfig


def _parse_override_tokens(tokens: Sequence[str]) -> list[tuple[str, str]]:
    """Parse canonical ``--field=value`` override tokens."""

    parsed = []
    for token in tokens:
        if not token.startswith("--") or token == "--":
            raise ConfigResolutionError(
                f"CLI: expected a dotted override in '--field=value' form, got {token!r}"
            )
        option = token[2:]
        if "=" not in option:
            raise ConfigResolutionError(
                f"CLI: expected a dotted override in '--field=value' form, got {token!r}"
            )
        path, raw_value = option.split("=", 1)
        if not path or any(not part for part in path.split(".")):
            raise ConfigResolutionError(f"CLI: invalid override path {path!r}")
        parsed.append((path, raw_value))
    return parsed


def _replace_path(config: object, parts: list[str], value: object, *, path: str) -> object:
    """Return a dataclass copy with one typed dotted-path value replaced."""

    location = f"CLI.{path}" if path else "CLI"
    if not is_dataclass(config):
        raise ConfigResolutionError(
            f"{location}: value of type {type(config).__name__} has no configurable fields"
        )

    config_fields = {field.name: field for field in fields(config)}
    name = parts[0]
    if name not in config_fields:
        matches = difflib.get_close_matches(name, config_fields, n=1)
        suggestion = f"; did you mean {matches[0]!r}?" if matches else ""
        raise ConfigResolutionError(
            f"{location}: unknown field {name!r} on {type(config).__name__}{suggestion}"
        )

    full_path = f"{path}.{name}" if path else name
    if len(parts) == 1:
        annotation = get_type_hints(type(config))[name]
        normalized = coerce_value(value, annotation, path=f"CLI.{full_path}")
        return replace(config, **{name: normalized})

    child = getattr(config, name)
    if child is None:
        raise ConfigResolutionError(
            f"CLI.{full_path}: component was not selected by the YAML"
        )
    updated_child = _replace_path(child, parts[1:], value, path=full_path)
    return replace(config, **{name: updated_child})


def _apply_typed_overrides(
    config: TrainerConfig,
    overrides: Sequence[str],
) -> TrainerConfig:
    """Apply CLI overrides to fields on the concrete resolved config objects."""

    result = config
    for dot_path, raw_value in _parse_override_tokens(overrides):
        try:
            parsed_value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigResolutionError(
                f"CLI.{dot_path}: invalid YAML value {raw_value!r}: {exc}"
            ) from exc
        result = _replace_path(result, dot_path.split("."), parsed_value, path="")
    return result


def _load_training_config(
    config_file: str | Path,
    cli_overrides: Sequence[str] = (),
) -> TrainerConfig:
    """Load a YAML file, resolve its targets, and apply parsed overrides."""

    path = Path(config_file)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigResolutionError(f"{path}: could not read config file: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigResolutionError(f"{path}: invalid YAML: {exc}") from exc

    config = resolve_root(raw)
    return _apply_typed_overrides(config, cli_overrides)


def parse_training_args(argv: Sequence[str] | None = None) -> TrainerConfig:
    """Parse the training command line into ``TrainerConfig``.

    Expected command-line arguments::

        train.yaml --accelerator.tp_size=4 --optimizer.lr=0.0003

    Example:
        config = parse_training_args()

    Args:
        argv (Sequence[str] | None, optional): Explicit argument tokens for tests.
            By default, arguments are read from ``sys.argv``. Default: ``None``.
    """

    parser = argparse.ArgumentParser(description="HyperParallel training config")
    parser.add_argument("config_file", help="Path to the YAML training config")
    args, overrides = parser.parse_known_args(argv)
    return _load_training_config(args.config_file, overrides)


__all__ = [
    "parse_training_args",
]

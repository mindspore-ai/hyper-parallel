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
"""Propagate model-owned configuration down the module tree before parallel layout."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from torch import nn


def resolve_num_layers(configs: Iterable[Any], default: int | None = None) -> int | None:
    """Resolve global depth from nearest-owner configs, including text subconfigs.

    Args:
        configs: Model configurations ordered from the nearest owner to the root.
        default: Inherited global depth when no configuration declares one.

    Returns:
        The positive global layer count, or None when unavailable.

    Raises:
        ValueError: A declared or inherited layer count is not a positive integer.
    """
    for config in configs:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            text_config = get_text_config()
        elif isinstance(config, Mapping):
            text_config = config.get("text_config")
        else:
            text_config = getattr(config, "text_config", None)
        for candidate in (text_config, config):
            for name in ("num_hidden_layers", "num_layers"):
                value = candidate.get(name) if isinstance(candidate, Mapping) else getattr(candidate, name, None)
                if value is not None:
                    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                        raise ValueError(f"model {name} must be a positive integer, got {value!r}")
                    return value
    if default is not None and (not isinstance(default, int) or isinstance(default, bool) or default < 1):
        raise ValueError(f"context num_layers must be a positive integer, got {default!r}")
    return default


def context_with_config(context: Mapping[str, Any], config: Any) -> Mapping[str, Any]:
    """Inherit job context while applying the current owner's model configuration.

    Args:
        context: Parent context containing the inherited global layer count.
        config: Current owner's configuration, if present.

    Returns:
        A read-only context with nearest-owner config precedence.
    """
    result = dict(context)
    if config is not None:
        result["module_configs"] = (config, *(item for item in context.get("module_configs", ()) if item is not config))
    num_layers = resolve_num_layers(() if config is None else (config,), context.get("num_layers"))
    if num_layers is not None:
        result["num_layers"] = num_layers
    return MappingProxyType(result)


def collect_module_contexts(model: nn.Module, context: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Propagate configuration in one top-down pass before replacement and sharding.

    Args:
        model: Complete model tree, before any pipeline partitioning.
        context: Root job context, optionally initialized from the full model config.

    Returns:
        Read-only FQN-to-context mapping, including aliases and no module references.
        Each module inherits its parent's global depth until its own config overrides it.
    """
    contexts = {}
    root_context = {name: value for name, value in context.items() if name != "module_contexts"}
    for fqn, module in model.named_modules(remove_duplicate=False):
        parent_context = contexts[fqn.rpartition(".")[0]] if fqn else root_context
        contexts[fqn] = context_with_config(parent_context, getattr(module, "config", None))
    return MappingProxyType(contexts)

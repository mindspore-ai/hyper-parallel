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
"""Compile Transformer decoder layers as independent graph segments."""

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from hyper_parallel.auto_models.trainer.config import CompileConfig

logger = logging.getLogger(__name__)


_MAPPING_GET_POLYFILL_INSTALLED = False

_MODEL_LAYER_PATHS = {
    "gpt2": ("transformer.h",),
    "llama": ("model.layers",),
    "qwen2": ("model.layers",),
    "qwen3": ("model.layers",),
    "qwen3_5": ("model.layers",),
    "qwen3_5_moe": ("model.layers",),
    "glm5": ("model.layers",),
}


def _get_attribute(root: Any, path: str) -> Any:
    """Read a dotted module path, including numeric ModuleList indexes."""
    value = root
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def _normalize_declared_layers(declared: Any) -> list[tuple[str, nn.Module]]:
    """Normalize the model-owned decoder-layer contract."""
    if isinstance(declared, nn.ModuleList):
        return [(str(index), layer) for index, layer in enumerate(declared)]
    if not isinstance(declared, Iterable) or isinstance(declared, (str, bytes)):
        raise TypeError(
            "get_compile_layers() must return an iterable of modules or "
            "(fqn, module) pairs"
        )

    layers = []
    for index, item in enumerate(declared):
        if isinstance(item, nn.Module):
            layers.append((str(index), item))
            continue
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], nn.Module)
        ):
            layers.append(item)
            continue
        raise TypeError(
            "get_compile_layers() entries must be modules or (fqn, module) pairs, "
            f"but entry {index} is {type(item).__name__}"
        )
    return layers


def _qualify_declared_layers(
    model: nn.Module,
    layers: list[tuple[str, nn.Module]],
) -> list[tuple[str, nn.Module]]:
    """Replace generated numeric names with the modules' model-owned FQNs."""
    module_names = {id(module): name for name, module in model.named_modules()}
    return [
        (module_names.get(id(layer), name) if name.isdigit() else name, layer)
        for name, layer in layers
    ]


def _layers_from_path(model: nn.Module, path: str) -> list[tuple[str, nn.Module]]:
    """Return indexed layers from one declared container path."""
    try:
        container = _get_attribute(model, path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return []
    if not isinstance(container, (nn.ModuleList, nn.Sequential)):
        return []
    return [(f"{path}.{index}", layer) for index, layer in enumerate(container)]


def get_compile_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return stable decoder-layer segments declared by a supported model."""
    declared_getter = getattr(model, "get_compile_layers", None)
    if callable(declared_getter):
        layers = _qualify_declared_layers(
            model,
            _normalize_declared_layers(declared_getter()),
        )
        if layers:
            return layers

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    candidate_paths = list(_MODEL_LAYER_PATHS.get(model_type, ()))
    candidate_paths.extend(("layers", "model.layers"))

    seen_containers = set()
    for path in candidate_paths:
        layers = _layers_from_path(model, path)
        if not layers:
            continue
        container_id = id(_get_attribute(model, path))
        if container_id in seen_containers:
            continue
        seen_containers.add(container_id)
        return layers

    raise ValueError(
        "compile is enabled, but the model exposes no decoder-layer compile contract; "
        "define get_compile_layers() or use a supported model layer container"
    )


def _install_dynamo_mapping_get_polyfill() -> None:
    """Make ``Mapping.get`` traceable without changing Transformers source.

    Transformers' attention registry calls ``Mapping.get`` from every decoder
    layer. TorchDynamo treats the standard-library implementation as a
    skipfile, so the call creates a graph break. ``substitute_in_graph`` only
    replaces the implementation while Dynamo inlines it; eager execution
    continues to use the original method.
    """
    # Module-level idempotency flag: the polyfill must be installed at most
    # once per process, so a global statement is required here.
    global _MAPPING_GET_POLYFILL_INSTALLED  # pylint: disable=global-statement
    if _MAPPING_GET_POLYFILL_INSTALLED:
        return

    substitute_in_graph = getattr(torch.compiler, "substitute_in_graph", None)
    if substitute_in_graph is None:
        logger.warning(
            "Torch does not provide compiler.substitute_in_graph; "
            "Mapping.get graph breaks cannot be removed"
        )
        _MAPPING_GET_POLYFILL_INSTALLED = True
        return

    def _mapping_get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    try:
        substitute_in_graph(Mapping.get)(_mapping_get)
    except ValueError as exc:
        if "Duplicate dispatch rule" not in str(exc):
            raise
        logger.debug("Mapping.get already has a TorchDynamo substitution")
    _MAPPING_GET_POLYFILL_INSTALLED = True


def resolve_compile_kwargs(config: CompileConfig) -> dict[str, Any]:
    """Convert ``CompileConfig`` into keyword arguments for ``Module.compile``."""
    kwargs: dict[str, Any] = {
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic,
    }
    if config.backend is not None:
        kwargs["backend"] = config.backend
    if config.options:
        kwargs["options"] = dict(config.options)
    else:
        kwargs["mode"] = config.mode
    return kwargs


def apply_compile(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Compile each decoder layer in place while preserving module identity."""
    if not config.enabled:
        return model

    _install_dynamo_mapping_get_polyfill()
    torch._dynamo.config.cache_size_limit = config.dynamo_cache_size_limit  # pylint: disable=W0212
    compile_kwargs = resolve_compile_kwargs(config)
    layers = get_compile_layers(model)
    for layer_fqn, layer in layers:
        try:
            layer.compile(**compile_kwargs)
        except Exception as exc:
            raise RuntimeError(f"failed to compile segment {layer_fqn}") from exc

    logger.info(
        "Compiled %d decoder layers (backend=%s, mode=%s, dynamic=%s, fullgraph=%s)",
        len(layers),
        config.backend,
        config.mode,
        config.dynamic,
        config.fullgraph,
    )
    return model


__all__ = [
    "apply_compile",
    "get_compile_layers",
    "resolve_compile_kwargs",
]

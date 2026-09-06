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
from typing import Any, Optional, Union

import torch
from torch import nn

from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager

logger = logging.getLogger(__name__)


_MAPPING_GET_POLYFILL_INSTALLED = False

# No per-model layer-path table: the model-owned ``get_compile_layers()``
# contract comes first, and the generic HF-convention fallback paths below
# cover every supported family. A family whose container lives elsewhere
# declares the contract on its model class instead of registering here.
_GENERIC_LAYER_PATHS = ("layers", "model.layers")


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
    """Return stable decoder-layer segments declared by a supported model.

    Resolution order: the model-owned ``get_compile_layers()`` contract
    first, then the generic HF-convention container paths
    (``_GENERIC_LAYER_PATHS``). A family whose container follows neither
    declares ``get_compile_layers()`` on its model class.
    """
    declared_getter = getattr(model, "get_compile_layers", None)
    if callable(declared_getter):
        layers = _qualify_declared_layers(
            model,
            _normalize_declared_layers(declared_getter()),
        )
        if layers:
            return layers

    seen_containers = set()
    for path in _GENERIC_LAYER_PATHS:
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


def _resolve_compile_config(
    compile_config: Optional[Union[CompileConfig, dict]],
    validate_placement: bool,
    fsdp2_manager: Optional[FSDP2Manager],
) -> tuple[Optional[CompileConfig], bool]:
    """Normalize compile configuration and validate its FSDP interaction.

    Moved from ``_transformers/infrastructure.py`` (05 §15.2.1): compile
    option normalization and the fullgraph/FSDP conflict check follow the
    compile feature.
    """
    if isinstance(compile_config, dict):
        compile_config = CompileConfig(enabled=True, **compile_config)
    if compile_config is not None and not isinstance(compile_config, CompileConfig):
        raise TypeError("compile_config must be a CompileConfig, mapping, or None")
    compile_for_execution = bool(
        not validate_placement and compile_config is not None and compile_config.enabled
    )
    if validate_placement and compile_config is not None and compile_config.enabled:
        logger.info("Skipping decoder-layer compile during placement validation")
    if compile_for_execution and compile_config.fullgraph and isinstance(fsdp2_manager, FSDP2Manager):
        raise ValueError(
            "compile.fullgraph=True is incompatible with FSDP hooks kept eager "
            "by _dynamo_disable; set compile.fullgraph=False"
        )
    return compile_config, compile_for_execution

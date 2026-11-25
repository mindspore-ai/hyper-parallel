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
"""Model-agnostic activation checkpoint and swap installation."""

import logging
import weakref
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from torch import nn

from hyper_parallel.core.activation_memory import (
    ActivationWrapper,
    PinnedMemoryPool,
    SwapManager,
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
    swap_wrapper,
)
from hyper_parallel.core.activation_memory.swap import _teardown_wired_swap_layers

logger = logging.getLogger(__name__)

_ACTIVE_MODES = frozenset(("full", "selective", "hybrid"))
_MIB = 1024 * 1024


@dataclass(frozen=True)
class _TransformerBlockInfo:
    """One transformer block in stable model registration order."""

    fqn: str
    module: nn.Module
    parent: nn.Module
    child_name: str
    container_fqn: str


@dataclass(frozen=True)
class _LayerContainerInfo:
    """One repeated-block container and its registered children."""

    container: nn.Module
    path: str
    blocks: tuple[_TransformerBlockInfo, ...]


@dataclass(frozen=True)
class _ResolvedTarget:
    """One validated layer-relative module replacement target."""

    config_path: str
    layer_index: int
    layer_fqn: str
    module_path: str
    module_fqn: str
    module_order: int
    parent: nn.Module
    child_name: str
    module: nn.Module


@dataclass(frozen=True)
class _ResolvedModelTarget:
    """One validated model-root-relative module replacement target."""

    config_path: str
    module_path: str
    module_fqn: str
    parent: nn.Module
    child_name: str
    module: nn.Module


@dataclass(frozen=True)
class _ActivationPlan:
    """Fully resolved activation-wrapper installation plan."""

    mode: str
    full_layers: tuple[tuple[int, _TransformerBlockInfo], ...]
    full_save_targets: tuple[_ResolvedTarget, ...]
    full_swap_targets: tuple[_ResolvedTarget, ...]
    selective_recompute_targets: tuple[_ResolvedTarget, ...]
    selective_recompute_model_targets: tuple[_ResolvedModelTarget, ...]
    selective_swap_targets: tuple[_ResolvedTarget, ...]
    full_group_swap: bool
    full_cpu_pool_mb: Optional[int]
    selective_group_swap: bool
    selective_cpu_pool_mb: Optional[int]

    def for_stage(self, model: nn.Module) -> "_ActivationPlan":
        """Keep this stage's targets under their validated global layer numbers.

        Args:
            model: Pruned and sharded stage. Its layer container has changed
                but retained modules still have their original qualified names.
        """
        modules = dict(model.named_modules(remove_duplicate=False))

        def _local_targets(targets):
            return tuple(
                replace(target, parent=modules[target.module_fqn.rpartition(".")[0]], module=modules[target.module_fqn])
                for target in targets if target.module_fqn in modules
            )

        return replace(
            self,
            full_layers=tuple(
                (index, replace(block, parent=modules[block.container_fqn], module=modules[block.fqn]))
                for index, block in self.full_layers if block.fqn in modules
            ),
            full_save_targets=_local_targets(self.full_save_targets),
            full_swap_targets=_local_targets(self.full_swap_targets),
            selective_recompute_targets=_local_targets(self.selective_recompute_targets),
            selective_recompute_model_targets=_local_targets(self.selective_recompute_model_targets),
            selective_swap_targets=_local_targets(self.selective_swap_targets),
        )


@dataclass(frozen=True)
class _ReplacementRecord:
    """One reversible child-module replacement."""

    parent: nn.Module
    child_name: str
    original: nn.Module


def _is_repeated_block_container(container: nn.Module, children: list[tuple[str, nn.Module]]) -> bool:
    """Return whether one child module structurally represents repeated blocks."""
    if isinstance(container, nn.Sequential):
        # Sequential stems and heads may contain unrelated operations rather than repeated blocks.
        block_types = set()
        for _, child in children:
            wrapped_module = _get_checkpoint_wrapped_module(child)
            block_types.add(type(child if wrapped_module is None else wrapped_module))
        return len(block_types) == 1
    if isinstance(container, (nn.ModuleList, nn.ModuleDict)):
        return bool(children)
    # Ordinary modules may contain same-type stages with different roles, such as an aligner's projections.
    # Repeated blocks must use an explicit collection container so they do not enter layer numbering by accident.
    return False


def _get_checkpoint_wrapped_module(module: nn.Module) -> Optional[nn.Module]:
    """Return a directly wrapped module for discovery de-duplication."""
    for attr_name in ("_wrapped_module", "_checkpoint_wrapped_module"):
        wrapped_module = getattr(module, attr_name, None)
        if isinstance(wrapped_module, nn.Module):
            return wrapped_module
    return None


def _find_transformer_block_modules(
    model: nn.Module,
) -> tuple[list[_TransformerBlockInfo], set[int]]:
    """Find repeated transformer blocks below HF checkpointing owners.

    The owner marker and depth-first registered child order are structural
    contracts. No model family, class name, or conventional ``model.layers``
    path is used.
    """
    block_infos = []
    discovered_block_ids = set()
    visited_module_ids = set()

    def _visit(owner_fqn: str, owner: nn.Module) -> None:
        owner_id = id(owner)
        if owner_id in visited_module_ids or owner_id in discovered_block_ids:
            return
        visited_module_ids.add(owner_id)
        is_checkpoint_owner = hasattr(owner, "gradient_checkpointing")

        for container_name, container in owner.named_children():
            container_fqn = f"{owner_fqn}.{container_name}" if owner_fqn else container_name
            children = list(container.named_children())
            if not is_checkpoint_owner or not _is_repeated_block_container(container, children):
                _visit(container_fqn, container)
                continue

            visited_module_ids.add(id(container))
            for block_name, block in children:
                if id(block) in discovered_block_ids:
                    continue
                discovered_block_ids.add(id(block))
                wrapped_module = _get_checkpoint_wrapped_module(block)
                if wrapped_module is not None:
                    discovered_block_ids.add(id(wrapped_module))
                block_infos.append(
                    _TransformerBlockInfo(
                        fqn=f"{container_fqn}.{block_name}",
                        module=block,
                        parent=container,
                        child_name=block_name,
                        container_fqn=container_fqn,
                    )
                )

    _visit("", model)
    return block_infos, discovered_block_ids


def _find_transformer_layer_container_infos(model: nn.Module) -> list[_LayerContainerInfo]:
    """Group discovered blocks by repeated-block container."""
    block_infos, _ = _find_transformer_block_modules(model)
    blocks_by_container: dict[int, list[_TransformerBlockInfo]] = {}
    for block_info in block_infos:
        blocks_by_container.setdefault(id(block_info.parent), []).append(block_info)

    containers = []
    seen_container_ids = set()
    for block_info in block_infos:
        container_id = id(block_info.parent)
        if container_id in seen_container_ids:
            continue
        seen_container_ids.add(container_id)
        containers.append(
            _LayerContainerInfo(
                container=block_info.parent,
                path=block_info.container_fqn,
                blocks=tuple(blocks_by_container[container_id]),
            )
        )
    return containers


def _flatten_layer_container_infos(
    containers: list[_LayerContainerInfo],
) -> list[_TransformerBlockInfo]:
    """Flatten container metadata into stable global layer numbering."""
    return [block for container in containers for block in container.blocks]


def _resolve_activation_checkpoint_containers(
    model: nn.Module,
    *,
    required: bool = True,
) -> list[_LayerContainerInfo]:
    """Discover repeated transformer containers and optionally require one."""
    containers = _find_transformer_layer_container_infos(model)
    if required and not containers:
        raise ValueError(
            f"{type(model).__name__} has no module with a 'gradient_checkpointing' "
            "attribute and a non-empty repeated block container"
        )
    return containers


def _current_block(block_info: _TransformerBlockInfo) -> nn.Module:
    """Return the current registered block and reject structure drift."""
    block = block_info.parent._modules.get(block_info.child_name)  # pylint: disable=protected-access
    if not isinstance(block, nn.Module):
        raise ValueError(f"discovered layer {block_info.fqn} is no longer an nn.Module")
    return block


def _module_registration_order(layer: nn.Module) -> dict[str, int]:
    """Return stable registered-module order within one layer."""
    try:
        iterator = layer.named_modules(remove_duplicate=False)
    except TypeError:
        iterator = layer.named_modules()
    return {module_path: index for index, (module_path, _) in enumerate(iterator)}


def _raise_incompatible_wrapper(
    module: nn.Module,
    *,
    config_path: str,
    layer_index: Optional[int],
    module_path: str,
    module_fqn: str,
) -> None:
    """Reject existing or partially removed activation wrappers."""
    target_location = f"layer {layer_index}" if layer_index is not None else "model target"
    for descendant in module.modules():
        if isinstance(descendant, ActivationWrapper) or getattr(descendant, "_is_wrapped", False):
            raise ValueError(
                f"{config_path}: {target_location}, module {module_path!r} ({module_fqn}) "
                "is already covered by an incompatible activation wrapper"
            )


def _resolve_module_target(
    block_info: _TransformerBlockInfo,
    *,
    layer_index: int,
    module_path: str,
    config_path: str,
) -> _ResolvedTarget:
    """Resolve one exact relative module path inside a transformer layer."""
    layer = _current_block(block_info)
    current = layer
    parent = layer
    parts = module_path.split(".")
    for part_index, part in enumerate(parts):
        if isinstance(current, ActivationWrapper):
            raise ValueError(
                f"{config_path}: layer {layer_index}, module {module_path!r} crosses "
                f"an existing activation wrapper at {block_info.fqn}"
            )
        parent = current
        child = current._modules.get(part)  # pylint: disable=protected-access
        if not isinstance(child, nn.Module):
            prefix = ".".join(parts[:part_index + 1])
            full_fqn = f"{block_info.fqn}.{prefix}"
            if hasattr(current, part):
                raise ValueError(
                    f"{config_path}: layer {layer_index}, module {module_path!r} resolves "
                    f"{full_fqn} to non-nn.Module {type(getattr(current, part)).__name__}"
                )
            raise ValueError(
                f"{config_path}: layer {layer_index}, module {module_path!r} does not exist "
                f"at {full_fqn}"
            )
        current = child

    module_fqn = f"{block_info.fqn}.{module_path}"
    _raise_incompatible_wrapper(
        current,
        config_path=config_path,
        layer_index=layer_index,
        module_path=module_path,
        module_fqn=module_fqn,
    )
    module_order = _module_registration_order(layer).get(module_path)
    if module_order is None:
        raise ValueError(
            f"{config_path}: layer {layer_index}, module {module_path!r} ({module_fqn}) "
            "has no stable registered forward order"
        )
    return _ResolvedTarget(
        config_path=config_path,
        layer_index=layer_index,
        layer_fqn=block_info.fqn,
        module_path=module_path,
        module_fqn=module_fqn,
        module_order=module_order,
        parent=parent,
        child_name=parts[-1],
        module=current,
    )


def _resolve_model_target(
    model: nn.Module,
    blocks: list[_TransformerBlockInfo],
    *,
    module_path: str,
    config_path: str,
) -> _ResolvedModelTarget:
    """Resolve one exact model-root-relative module outside transformer layers."""
    current = model
    parent = model
    parts = module_path.split(".")
    for part_index, part in enumerate(parts):
        if isinstance(current, ActivationWrapper):
            prefix = ".".join(parts[:part_index]) or "<model>"
            raise ValueError(
                f"{config_path}: model module {module_path!r} crosses an existing "
                f"activation wrapper at {prefix}"
            )
        parent = current
        child = current._modules.get(part)  # pylint: disable=protected-access
        prefix = ".".join(parts[:part_index + 1])
        if not isinstance(child, nn.Module):
            if hasattr(current, part):
                raise ValueError(
                    f"{config_path}: model module {module_path!r} resolves {prefix} "
                    f"to non-nn.Module {type(getattr(current, part)).__name__}"
                )
            raise ValueError(
                f"{config_path}: model module {module_path!r} does not exist at {prefix}"
            )
        current = child

    target_module_ids = {id(module) for module in current.modules()}
    for layer_index, block_info in enumerate(blocks):
        layer = _current_block(block_info)
        layer_module_ids = {id(module) for module in layer.modules()}
        if id(current) in layer_module_ids or id(layer) in target_module_ids:
            raise ValueError(
                f"{config_path}: model module {module_path!r} must be outside transformer "
                f"layers, but overlaps layer {layer_index} ({block_info.fqn})"
            )

    _raise_incompatible_wrapper(
        current,
        config_path=config_path,
        layer_index=None,
        module_path=module_path,
        module_fqn=module_path,
    )
    return _ResolvedModelTarget(
        config_path=config_path,
        module_path=module_path,
        module_fqn=module_path,
        parent=parent,
        child_name=parts[-1],
        module=current,
    )


def _validate_layer_index(layer_index: int, layer_count: int, *, config_path: str) -> None:
    """Validate one normalized layer index against discovered model layers."""
    if layer_index < 0 or layer_index >= layer_count:
        maximum = layer_count - 1
        raise ValueError(
            f"{config_path}: layer {layer_index} is outside the discovered range 0-{maximum}"
        )


def _resolve_regions(
    blocks: list[_TransformerBlockInfo],
    regions: tuple[Any, ...],
    *,
    config_path: str,
) -> tuple[_ResolvedTarget, ...]:
    """Resolve every normalized region spec without modifying the model."""
    targets = []
    for region in regions:
        for layer_index in region.layers:
            _validate_layer_index(layer_index, len(blocks), config_path=config_path)
            targets.append(
                _resolve_module_target(
                    blocks[layer_index],
                    layer_index=layer_index,
                    module_path=region.module_path,
                    config_path=f"{config_path}.{region.module_path}",
                )
            )
    return tuple(targets)


def _resolve_model_targets(
    model: nn.Module,
    blocks: list[_TransformerBlockInfo],
    modules: tuple[Any, ...],
    *,
    config_path: str,
) -> tuple[_ResolvedModelTarget, ...]:
    """Resolve every normalized model-root-relative module target."""
    return tuple(
        _resolve_model_target(
            model,
            blocks,
            module_path=module.module_path,
            config_path=f"{config_path}.{module.module_path}",
        )
        for module in modules
    )


def _target_key(target: _ResolvedTarget) -> tuple[int, str]:
    """Return one layer/path conflict key."""
    return target.layer_index, target.module_path


def _validate_nested_targets(targets: tuple[_ResolvedTarget, ...], *, config_path: str) -> None:
    """Reject ancestor/descendant target pairs within one layer."""
    by_layer: dict[int, list[_ResolvedTarget]] = {}
    for target in targets:
        by_layer.setdefault(target.layer_index, []).append(target)
    for layer_index, layer_targets in by_layer.items():
        for index, target in enumerate(layer_targets):
            for other in layer_targets[index + 1:]:
                if target.module_path.startswith(f"{other.module_path}.") or other.module_path.startswith(
                    f"{target.module_path}."
                ):
                    raise ValueError(
                        f"{config_path}: layer {layer_index} has nested activation targets "
                        f"{target.module_path!r} ({target.module_fqn}) and "
                        f"{other.module_path!r} ({other.module_fqn})"
                    )


def _target_description(target: Any) -> str:
    """Format one resolved target without inventing a model-level layer index."""
    if isinstance(target, _ResolvedTarget):
        return (
            f"{target.config_path}: layer {target.layer_index}, module "
            f"{target.module_path!r} ({target.module_fqn})"
        )
    return f"{target.config_path}: model module {target.module_path!r} ({target.module_fqn})"


def _validate_target_aliases(targets: tuple[Any, ...]) -> None:
    """Reject one module object selected through multiple layer/path identities."""
    by_identity: dict[int, Any] = {}
    for target in targets:
        previous = by_identity.get(id(target.module))
        if previous is not None:
            raise ValueError(
                f"{_target_description(target)} aliases {_target_description(previous)}"
            )
        by_identity[id(target.module)] = target


def _validate_model_target_nesting(
    model_targets: tuple[_ResolvedModelTarget, ...],
    all_targets: tuple[Any, ...],
) -> None:
    """Reject model targets nested with any other activation target."""
    for target in model_targets:
        target_descendants = {id(module) for module in target.module.modules()}
        for other in all_targets:
            if other is target or other.module is target.module:
                continue
            other_descendants = {id(module) for module in other.module.modules()}
            if id(other.module) in target_descendants or id(target.module) in other_descendants:
                raise ValueError(
                    "activation_checkpoint.selective has nested activation targets: "
                    f"{_target_description(target)} and {_target_description(other)}"
                )


def _validate_resolved_conflicts(plan: _ActivationPlan) -> None:
    """Validate model-level conflicts across all resolved active targets."""
    full_layer_indices = {layer_index for layer_index, _ in plan.full_layers}
    full_targets = plan.full_save_targets + plan.full_swap_targets
    for target in full_targets:
        if target.layer_index not in full_layer_indices:
            raise ValueError(
                f"{target.config_path}: layer {target.layer_index}, module {target.module_path!r} "
                "is outside activation_checkpoint.full.recompute_layer"
            )

    full_conflicts = {_target_key(target) for target in plan.full_save_targets} & {
        _target_key(target) for target in plan.full_swap_targets
    }
    if full_conflicts:
        layer_index, module_path = sorted(full_conflicts)[0]
        raise ValueError(
            "activation_checkpoint.full exclude save/swap conflict on "
            f"layer {layer_index}, module {module_path!r}"
        )
    selective_conflicts = {
        _target_key(target) for target in plan.selective_recompute_targets
    } & {_target_key(target) for target in plan.selective_swap_targets}
    if selective_conflicts:
        layer_index, module_path = sorted(selective_conflicts)[0]
        raise ValueError(
            "activation_checkpoint.selective recompute/swap conflict on "
            f"layer {layer_index}, module {module_path!r}"
        )

    if plan.mode == "hybrid":
        selective_layers = {
            target.layer_index
            for target in plan.selective_recompute_targets + plan.selective_swap_targets
        }
        overlap = sorted(full_layer_indices & selective_layers)
        if overlap:
            raise ValueError(
                "activation_checkpoint.mode=hybrid requires disjoint full and selective layers; "
                f"overlap={overlap}"
            )

    _validate_nested_targets(full_targets, config_path="activation_checkpoint.full.exclude_region")
    _validate_nested_targets(
        plan.selective_recompute_targets + plan.selective_swap_targets,
        config_path="activation_checkpoint.selective",
    )
    layer_targets = full_targets + plan.selective_recompute_targets + plan.selective_swap_targets
    all_targets = layer_targets + plan.selective_recompute_model_targets
    _validate_target_aliases(all_targets)
    _validate_model_target_nesting(plan.selective_recompute_model_targets, all_targets)


def _resolve_full_layers(
    blocks: list[_TransformerBlockInfo],
    full_config: Any,
    *,
    default_all: bool,
) -> tuple[tuple[int, _TransformerBlockInfo], ...]:
    """Resolve selected full layers and reject existing wrappers."""
    if full_config is None:
        layer_indices = range(len(blocks)) if default_all else ()
    else:
        layer_indices = full_config.recompute_layers
    full_layers = []
    for layer_index in layer_indices:
        _validate_layer_index(
            layer_index,
            len(blocks),
            config_path="activation_checkpoint.full.recompute_layer",
        )
        block = blocks[layer_index]
        _raise_incompatible_wrapper(
            _current_block(block),
            config_path="activation_checkpoint.full.recompute_layer",
            layer_index=layer_index,
            module_path="<layer>",
            module_fqn=block.fqn,
        )
        full_layers.append((layer_index, block))
    return tuple(full_layers)


def _resolve_optional_regions(
    blocks: list[_TransformerBlockInfo],
    config: Any,
    attribute: str,
    *,
    config_path: str,
) -> tuple[_ResolvedTarget, ...]:
    """Resolve one region collection from an optional active config."""
    if config is None:
        return ()
    return _resolve_regions(
        blocks,
        getattr(config, attribute),
        config_path=config_path,
    )


def _resolve_activation_plan(model: nn.Module, config: Any) -> _ActivationPlan:
    """Resolve and validate all active targets before installing any wrapper."""
    mode = _require_active_mode(config)
    full_config, selective_config = _resolve_mode_configs(config, mode)

    containers = _resolve_activation_checkpoint_containers(
        model,
        required=_requires_layer_targets(mode, full_config, selective_config),
    )
    blocks = _flatten_layer_container_infos(containers)
    for layer_index, block in enumerate(blocks):
        logger.info("Activation checkpoint layer index=%d fqn=%s", layer_index, block.fqn)

    plan = _ActivationPlan(
        mode=mode,
        full_layers=_resolve_full_layers(
            blocks,
            full_config,
            default_all=mode == "full",
        ),
        full_save_targets=_resolve_optional_regions(
            blocks,
            full_config,
            "save_regions",
            config_path="activation_checkpoint.full.exclude_region.save",
        ),
        full_swap_targets=_resolve_optional_regions(
            blocks,
            full_config,
            "swap_regions",
            config_path="activation_checkpoint.full.exclude_region.swap",
        ),
        selective_recompute_targets=_resolve_optional_regions(
            blocks,
            selective_config,
            "recompute_regions",
            config_path="activation_checkpoint.selective.recompute",
        ),
        selective_recompute_model_targets=_resolve_optional_model_targets(
            model,
            blocks,
            selective_config,
        ),
        selective_swap_targets=_resolve_optional_regions(
            blocks,
            selective_config,
            "swap_regions",
            config_path="activation_checkpoint.selective.swap",
        ),
        full_group_swap=bool(getattr(full_config, "group_swap", False)),
        full_cpu_pool_mb=getattr(full_config, "cpu_pool_mb", None),
        selective_group_swap=bool(getattr(selective_config, "group_swap", False)),
        selective_cpu_pool_mb=getattr(selective_config, "cpu_pool_mb", None),
    )
    _validate_resolved_conflicts(plan)
    return plan


def _require_active_mode(config: Any) -> str:
    """Return the normalized mode, rejecting missing or inactive values."""
    mode = getattr(config, "mode", None)
    if mode not in _ACTIVE_MODES:
        raise ValueError(
            "activation_checkpoint runtime requires normalized mode full, selective, or hybrid; "
            f"got {mode!r}"
        )
    return mode


def _resolve_mode_configs(config: Any, mode: str) -> tuple[Any, Any]:
    """Select and validate the sub-configurations active for one runtime mode."""
    full_config = getattr(config, "full", None) if mode in ("full", "hybrid") else None
    selective_config = (
        getattr(config, "selective", None) if mode in ("selective", "hybrid") else None
    )
    if mode == "hybrid" and full_config is None:
        raise ValueError("activation_checkpoint.mode=hybrid requires activation_checkpoint.full")
    if mode in ("selective", "hybrid") and selective_config is None:
        raise ValueError(
            f"activation_checkpoint.mode={mode} requires activation_checkpoint.selective"
        )
    return full_config, selective_config


def _requires_layer_targets(mode: str, full_config: Any, selective_config: Any) -> bool:
    """Return whether the active configuration needs transformer-block discovery."""
    if mode == "full" or full_config is not None:
        return True
    if selective_config is None:
        return False
    return bool(selective_config.recompute_regions or selective_config.swap_regions)


def _resolve_optional_model_targets(
    model: nn.Module,
    blocks: list[_TransformerBlockInfo],
    selective_config: Any,
) -> tuple[_ResolvedTarget, ...]:
    """Resolve model-level selective targets when a selective config is active."""
    if selective_config is None:
        return ()
    return _resolve_model_targets(
        model,
        blocks,
        getattr(selective_config, "recompute_modules", ()),
        config_path="activation_checkpoint.selective.recompute",
    )


def _snapshot_overlap_marks(model: nn.Module) -> tuple[tuple[object, bool, Any], ...]:
    """Capture wrapper overlap marks so a failed apply can restore them."""
    snapshots = []
    seen = set()
    for module in model.modules():
        candidates = [module]
        candidates.extend(
            value
            for value in vars(module).values()
            if callable(value) and not isinstance(value, nn.Module) and hasattr(value, "__dict__")
        )
        for candidate in candidates:
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            exists = hasattr(candidate, "_is_wrapped")
            snapshots.append((candidate, exists, getattr(candidate, "_is_wrapped", None)))
    return tuple(snapshots)


def _restore_overlap_marks(snapshots: tuple[tuple[object, bool, Any], ...]) -> None:
    """Restore overlap marks captured before wrapper construction."""
    for candidate, existed, value in snapshots:
        if existed:
            setattr(candidate, "_is_wrapped", value)
        elif hasattr(candidate, "_is_wrapped"):
            delattr(candidate, "_is_wrapped")


def _replace_target(
    target: Any,
    wrapped: nn.Module,
    records: list[_ReplacementRecord],
) -> None:
    """Install one already-constructed wrapper and record its inverse."""
    current = target.parent._modules.get(target.child_name)  # pylint: disable=protected-access
    if current is not target.module:
        raise RuntimeError(
            f"{target.config_path}: target {target.module_fqn} changed during activation wrapper apply"
        )
    setattr(target.parent, target.child_name, wrapped)
    records.append(_ReplacementRecord(target.parent, target.child_name, target.module))


def _replace_layer(
    block_info: _TransformerBlockInfo,
    wrapped: nn.Module,
    records: list[_ReplacementRecord],
) -> None:
    """Install one full-layer wrapper and record its inverse."""
    current = _current_block(block_info)
    setattr(block_info.parent, block_info.child_name, wrapped)
    records.append(_ReplacementRecord(block_info.parent, block_info.child_name, current))


def _create_cpu_pool(cpu_pool_mb: Optional[int], *, has_targets: bool) -> Optional[PinnedMemoryPool]:
    """Create one runtime pool only for an active swap target domain."""
    if not has_targets or cpu_pool_mb is None:
        return None
    return PinnedMemoryPool(max_host_bytes=cpu_pool_mb * _MIB)


def _dedupe_modules(modules: list[nn.Module]) -> list[nn.Module]:
    """De-duplicate modules while preserving forward order."""
    return list(dict.fromkeys(modules))


def _wire_prefetch_chain(
    model: nn.Module,
    nodes: list[tuple[int, int, nn.Module, str]],
) -> list[nn.Module]:
    """Wire all active swap nodes through one ordered SwapManager flow."""
    nodes.sort(key=lambda item: (item[0], item[1]))
    swap_manager = SwapManager()
    wired_modules = []
    try:
        for current, next_node in zip(nodes, nodes[1:]):
            current_module, next_module = current[2], next_node[2]
            wired_modules.extend((current_module, next_module))
            swap_manager.set_forward_prefetch_layer(current_module, next_module)
            logger.info("Activation swap prefetch %s -> %s", current[3], next_node[3])
    except BaseException:
        _teardown_wired_swap_layers(_dedupe_modules(wired_modules))
        raise
    wired_modules = _dedupe_modules(wired_modules)
    if wired_modules:
        weakref.finalize(model, _teardown_wired_swap_layers, tuple(wired_modules))
    return wired_modules


def _pool_description(cpu_pool_mb: Optional[int], pool: Optional[PinnedMemoryPool]) -> str:
    """Format one pool for the activation summary log."""
    if pool is None or cpu_pool_mb is None:
        return "none"
    return f"{cpu_pool_mb} MiB ({pool.max_host_bytes} bytes)"


def _sorted_targets(targets: Sequence[_ResolvedTarget]) -> list[_ResolvedTarget]:
    """Order replacement targets by layer index, then model registration order."""
    return sorted(targets, key=lambda item: (item.layer_index, item.module_order))


def _install_full_save_wrappers(
    targets: Sequence[_ResolvedTarget],
    replacements: list[_ReplacementRecord],
) -> None:
    """Wrap full-mode save targets in memory-saving-only wrappers."""
    for target in _sorted_targets(targets):
        wrapped = checkpoint_exclude_wrapper(target.module, swap=False)
        _replace_target(target, wrapped, replacements)
        logger.info(
            "Activation wrapper=CheckpointExcludeWrapper(save) layer=%d layer_fqn=%s "
            "module=%s module_fqn=%s pool=none",
            target.layer_index,
            target.layer_fqn,
            target.module_path,
            target.module_fqn,
        )


def _install_full_swap_wrappers(
    targets: Sequence[_ResolvedTarget],
    group_swap: bool,
    cpu_pool: Optional[PinnedMemoryPool],
    replacements: list[_ReplacementRecord],
) -> None:
    """Wrap full-mode swap targets in exclusion wrappers backed by the full pool."""
    for target in _sorted_targets(targets):
        wrapped = checkpoint_exclude_wrapper(
            target.module,
            swap=True,
            group_swap=group_swap,
            cpu_pool=cpu_pool,
        )
        _replace_target(target, wrapped, replacements)
        logger.info(
            "Activation wrapper=CheckpointExcludeWrapper(swap) layer=%d layer_fqn=%s "
            "module=%s module_fqn=%s pool=full",
            target.layer_index,
            target.layer_fqn,
            target.module_path,
            target.module_fqn,
        )


def _install_full_layer_wrappers(
    full_layers: Sequence[tuple[int, _TransformerBlockInfo]],
    replacements: list[_ReplacementRecord],
) -> dict[int, nn.Module]:
    """Wrap full-mode transformer blocks, returning wrappers keyed by layer index."""
    full_wrappers: dict[int, nn.Module] = {}
    for layer_index, block_info in full_layers:
        current_layer = _current_block(block_info)
        wrapped = checkpoint_wrapper(current_layer)
        _replace_layer(block_info, wrapped, replacements)
        full_wrappers[layer_index] = wrapped
        logger.info(
            "Activation wrapper=CheckpointWrapper(full) layer=%d layer_fqn=%s",
            layer_index,
            block_info.fqn,
        )
    return full_wrappers


def _install_selective_recompute_wrappers(
    targets: Sequence[_ResolvedTarget],
    replacements: list[_ReplacementRecord],
) -> None:
    """Wrap selective-mode layer-relative recompute targets."""
    for target in _sorted_targets(targets):
        wrapped = checkpoint_wrapper(target.module)
        _replace_target(target, wrapped, replacements)
        logger.info(
            "Activation wrapper=CheckpointWrapper(selective) layer=%d layer_fqn=%s "
            "module=%s module_fqn=%s",
            target.layer_index,
            target.layer_fqn,
            target.module_path,
            target.module_fqn,
        )


def _install_selective_model_wrappers(
    targets: Sequence[_ResolvedModelTarget],
    replacements: list[_ReplacementRecord],
) -> None:
    """Wrap selective-mode model-root-relative recompute targets."""
    for target in targets:
        wrapped = checkpoint_wrapper(target.module)
        _replace_target(target, wrapped, replacements)
        logger.info(
            "Activation wrapper=CheckpointWrapper(selective) scope=model "
            "module=%s module_fqn=%s",
            target.module_path,
            target.module_fqn,
        )


def _install_selective_swap_wrappers(
    targets: Sequence[_ResolvedTarget],
    group_swap: bool,
    cpu_pool: Optional[PinnedMemoryPool],
    replacements: list[_ReplacementRecord],
) -> list[tuple[_ResolvedTarget, nn.Module]]:
    """Wrap selective-mode swap targets, returning the installed wrapper pairs."""
    swap_wrappers: list[tuple[_ResolvedTarget, nn.Module]] = []
    for target in _sorted_targets(targets):
        wrapped = swap_wrapper(
            target.module,
            group_swap=group_swap,
            cpu_pool=cpu_pool,
        )
        _replace_target(target, wrapped, replacements)
        swap_wrappers.append((target, wrapped))
        logger.info(
            "Activation wrapper=SwapWrapper layer=%d layer_fqn=%s module=%s "
            "module_fqn=%s pool=selective",
            target.layer_index,
            target.layer_fqn,
            target.module_path,
            target.module_fqn,
        )
    return swap_wrappers


def _collect_prefetch_nodes(
    plan: _ActivationPlan,
    full_wrappers: dict[int, nn.Module],
    selective_swap_wrappers: Sequence[tuple[_ResolvedTarget, nn.Module]],
) -> list[tuple[int, int, nn.Module, str]]:
    """Collect every swap node in layer order for one prefetch chain."""
    full_swap_layers = {target.layer_index for target in plan.full_swap_targets}
    prefetch_nodes = [
        (layer_index, -1, full_wrappers[layer_index], block_info.fqn)
        for layer_index, block_info in plan.full_layers
        if layer_index in full_swap_layers
    ]
    prefetch_nodes.extend(
        (target.layer_index, target.module_order, wrapper, target.module_fqn)
        for target, wrapper in selective_swap_wrappers
    )
    return prefetch_nodes


def _apply_activation_plan(model: nn.Module, plan: _ActivationPlan) -> nn.Module:
    """Create pools, install wrappers, and register one prefetch chain."""
    full_pool = _create_cpu_pool(
        plan.full_cpu_pool_mb,
        has_targets=bool(plan.full_swap_targets),
    )
    selective_pool = _create_cpu_pool(
        plan.selective_cpu_pool_mb,
        has_targets=bool(plan.selective_swap_targets),
    )
    overlap_marks = _snapshot_overlap_marks(model)
    replacements: list[_ReplacementRecord] = []
    wired_modules: list[nn.Module] = []

    try:
        _install_full_save_wrappers(plan.full_save_targets, replacements)
        _install_full_swap_wrappers(
            plan.full_swap_targets,
            plan.full_group_swap,
            full_pool,
            replacements,
        )
        full_wrappers = _install_full_layer_wrappers(plan.full_layers, replacements)
        _install_selective_recompute_wrappers(plan.selective_recompute_targets, replacements)
        _install_selective_model_wrappers(plan.selective_recompute_model_targets, replacements)
        selective_swap_wrappers = _install_selective_swap_wrappers(
            plan.selective_swap_targets,
            plan.selective_group_swap,
            selective_pool,
            replacements,
        )
        prefetch_nodes = _collect_prefetch_nodes(plan, full_wrappers, selective_swap_wrappers)
        wired_modules = _wire_prefetch_chain(model, prefetch_nodes)
    except BaseException:
        if wired_modules:
            _teardown_wired_swap_layers(wired_modules)
        for record in reversed(replacements):
            setattr(record.parent, record.child_name, record.original)
        _restore_overlap_marks(overlap_marks)
        raise

    logger.info(
        "Activation checkpoint mode=%s: full_layers=%s, full_save_regions=%d, "
        "full_swap_regions=%d, full_pool=%s; selective_recompute_regions=%d, "
        "selective_recompute_modules=%d, selective_swap_regions=%d, selective_pool=%s",
        plan.mode,
        [layer_index for layer_index, _ in plan.full_layers],
        len(plan.full_save_targets),
        len(plan.full_swap_targets),
        _pool_description(plan.full_cpu_pool_mb, full_pool),
        len(plan.selective_recompute_targets),
        len(plan.selective_recompute_model_targets),
        len(plan.selective_swap_targets),
        _pool_description(plan.selective_cpu_pool_mb, selective_pool),
    )
    return model


def _apply_activation_checkpointing(
    model: nn.Module,
    activation_checkpoint: Optional[Any],
    enable_compile: bool = False,
    *,
    plan: Optional[_ActivationPlan] = None,
) -> nn.Module:
    """Apply one normalized activation-checkpoint configuration atomically.

    ``None`` and ``mode=off`` are no-ops. Full mode checkpoints every
    discovered transformer layer when no ``full`` sub-configuration is given.
    Selective and hybrid modes retain their explicit module-target semantics.
    A supplied plan carries globally validated targets rebound to the local pipeline stage.
    Legacy string modes are intentionally unsupported.
    """
    if activation_checkpoint is None or getattr(activation_checkpoint, "mode", None) == "off":
        return model
    if isinstance(activation_checkpoint, str):
        raise ValueError(
            "legacy activation_checkpoint string modes are unsupported; pass the normalized "
            "activation_checkpoint configuration object"
        )
    _require_active_mode(activation_checkpoint)
    if enable_compile:
        raise ValueError(
            "activation_checkpoint modes full, selective, and hybrid are not supported with torch.compile"
        )
    if plan is None:
        plan = _resolve_activation_plan(model, activation_checkpoint)
    disable_native_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable_native_checkpointing):
        disable_native_checkpointing()
    return _apply_activation_plan(model, plan)


__all__ = [
    "_apply_activation_checkpointing",
    "_find_transformer_block_modules",
    "_find_transformer_layer_container_infos",
]

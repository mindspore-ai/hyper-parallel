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
"""Post-materialization lifecycle for model-owned derived state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel import DTensor, distribute_tensor


MaterializationReason = Literal["random_init", "checkpoint_load"]
MaterializedStateHook = Callable[[nn.Module, "MaterializationContext"], None]


@dataclass(frozen=True)
class MaterializationContext:
    """Describe why and where empty model storage was materialized."""

    reason: MaterializationReason
    device: torch.device
    strict: bool = True


@dataclass(frozen=True)
class RebuildableBufferSpec:
    """Recipe for restoring one registered buffer after ``to_empty()``."""

    name: str
    source: torch.Tensor | None
    factory: Callable[[MaterializationContext], torch.Tensor] | None
    preserve_layout: bool


@dataclass(frozen=True)
class _ProtectedStateSnapshot:
    """Identity, layout, and mutation state protected across model hooks."""

    tensor_id: int
    local_tensor_id: int
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    layout_id: int | None
    placements: tuple[object, ...] | None
    version: int | None


def _factory_initial_value(
    factory: Callable[[MaterializationContext], torch.Tensor],
) -> torch.Tensor:
    """Build the registration-time template for a deterministic factory."""
    context = MaterializationContext(
        reason="random_init",
        device=torch.device("cpu"),
        strict=True,
    )
    value = factory(context)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "A rebuildable buffer factory must return torch.Tensor, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, DTensor):
        raise TypeError("A rebuildable buffer factory must return a plain global tensor")
    if value.is_meta:
        raise ValueError("A rebuildable buffer factory cannot return a meta tensor")
    return value


def _cpu_source(value: torch.Tensor, name: str) -> torch.Tensor:
    """Retain a source outside ``Module._apply`` and materialized storage."""
    if isinstance(value, DTensor):
        raise TypeError(f"Rebuildable buffer {name!r} requires a plain global tensor source")
    if value.is_meta:
        raise ValueError(f"Rebuildable buffer {name!r} cannot use a meta tensor source")
    return value.detach().to(device="cpu").clone()


def register_rebuildable_buffer(
    module: nn.Module,
    name: str,
    *,
    value: torch.Tensor | None = None,
    factory: Callable[[MaterializationContext], torch.Tensor] | None = None,
    persistent: bool = False,
    preserve_layout: bool = True,
) -> None:
    """Attach a deterministic rebuild recipe to any ``nn.Module`` buffer.

    When ``name`` already identifies a direct non-persistent buffer, omitting
    both ``value`` and ``factory`` captures its current value. This lets a model
    adapter support an unmodified native HF module. Otherwise exactly one recipe
    must be supplied, and a missing buffer is registered without changing the
    module's class.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be torch.nn.Module, got {type(module).__name__}")
    if not isinstance(name, str) or not name or "." in name:
        raise ValueError(f"Rebuildable buffer name must be a direct non-empty name, got {name!r}")
    if persistent:
        raise ValueError(
            "Rebuildable buffers must be non-persistent; register checkpoint state "
            "with nn.Module.register_buffer instead"
        )

    specs = module.__dict__.setdefault("_hp_rebuildable_buffer_specs", {})
    if name in specs:
        raise ValueError(f"Rebuildable buffer {name!r} is already registered")

    existing = module._buffers.get(name)  # pylint: disable=protected-access
    has_existing = name in module._buffers  # pylint: disable=protected-access
    if has_existing:
        if existing is None:
            raise ValueError(f"Existing rebuildable buffer {name!r} cannot be None")
        non_persistent = module._non_persistent_buffers_set  # pylint: disable=protected-access
        if name not in non_persistent:
            raise ValueError(
                f"Existing buffer {name!r} is persistent and must be restored from checkpoint"
            )
        if value is None and factory is None:
            value = existing
    elif (value is None) == (factory is None):
        raise ValueError(
            "A new rebuildable buffer requires exactly one of value and factory"
        )

    if has_existing and value is not None and factory is not None:
        raise ValueError("Exactly one of value and factory may define a rebuildable buffer")
    initial_value = value if value is not None else _factory_initial_value(factory)
    if not isinstance(initial_value, torch.Tensor):
        raise TypeError(
            f"Rebuildable buffer {name!r} must be a torch.Tensor, "
            f"got {type(initial_value).__name__}"
        )
    source = _cpu_source(initial_value, name) if value is not None else None

    if has_existing:
        if initial_value.dtype != existing.dtype or tuple(initial_value.shape) != tuple(existing.shape):
            raise ValueError(
                f"Existing buffer {name!r} metadata does not match its rebuild recipe: "
                f"buffer=({tuple(existing.shape)}, {existing.dtype}), "
                f"recipe=({tuple(initial_value.shape)}, {initial_value.dtype})"
            )
    else:
        module.register_buffer(name, initial_value.detach().clone(), persistent=False)
    specs[name] = RebuildableBufferSpec(
        name=name,
        source=source,
        factory=factory,
        preserve_layout=preserve_layout,
    )


def register_materialized_state_hook(
    module: nn.Module,
    hook: MaterializedStateHook,
) -> None:
    """Register an adapter-owned recovery hook without modifying a model class."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be torch.nn.Module, got {type(module).__name__}")
    if not callable(hook):
        raise TypeError(f"materialized state hook must be callable, got {type(hook).__name__}")
    hooks = module.__dict__.setdefault("_hp_materialized_state_hooks", [])
    if hook in hooks:
        raise ValueError("Materialized state hook is already registered on the module")
    hooks.append(hook)


@torch.no_grad()
def _rebuild_registered_buffers(module: nn.Module, context: MaterializationContext) -> None:
    """Restore all declaratively registered buffers on one module in place."""
    specs = module.__dict__.get("_hp_rebuildable_buffer_specs", {})
    for spec in specs.values():
        if spec.source is not None:
            value = spec.source
        elif spec.factory is not None:
            value = spec.factory(context)
        else:
            raise RuntimeError(f"buffer {spec.name!r} has no rebuild recipe")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"buffer {spec.name!r} factory returned {type(value).__name__}, expected torch.Tensor"
            )
        if isinstance(value, DTensor):
            raise TypeError(f"buffer {spec.name!r} rebuild recipe must return a plain tensor")
        if value.is_meta:
            raise ValueError(f"buffer {spec.name!r} rebuild recipe returned a meta tensor")
        target = module.get_buffer(spec.name)
        _copy_rebuilt_buffer(spec, value, target)


def _copy_rebuilt_buffer(
    spec: RebuildableBufferSpec,
    value: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Copy one full or rank-local derived value into existing storage."""
    destination = target.to_local() if isinstance(target, DTensor) else target
    if destination.is_meta:
        raise ValueError(f"buffer {spec.name!r} still has meta storage")
    if value.dtype != target.dtype:
        raise ValueError(
            f"buffer {spec.name!r} dtype mismatch: rebuilt={value.dtype}, target={target.dtype}"
        )

    local_value = value
    if isinstance(target, DTensor) and spec.preserve_layout:
        layout = target.layout
        if layout is None:
            raise ValueError(f"buffer {spec.name!r} has a DTensor without layout metadata")
        if any(placement.is_partial() for placement in layout.placements):
            raise ValueError(f"buffer {spec.name!r} cannot rebuild a Partial DTensor")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"buffer {spec.name!r} global shape mismatch: "
                f"rebuilt={tuple(value.shape)}, target={tuple(target.shape)}"
            )
        local_value = distribute_tensor(
            value,
            layout.mesh,
            layout.alias_placements,
            src_data_rank=None,
        ).to_local()

    if tuple(local_value.shape) != tuple(destination.shape):
        raise ValueError(
            f"buffer {spec.name!r} local shape mismatch: "
            f"rebuilt={tuple(local_value.shape)}, target={tuple(destination.shape)}"
        )
    destination.copy_(local_value.to(device=destination.device))


def _snapshot_tensor(tensor: torch.Tensor) -> _ProtectedStateSnapshot:
    """Capture protected structure and the local tensor mutation counter."""
    local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    layout = tensor.layout if isinstance(tensor, DTensor) else getattr(tensor, "_sharding_spec", None)
    placements = tuple(layout.placements) if layout is not None else None
    return _ProtectedStateSnapshot(
        tensor_id=id(tensor),
        local_tensor_id=id(local_tensor),
        global_shape=tuple(tensor.shape),
        local_shape=tuple(local_tensor.shape),
        layout_id=id(layout) if layout is not None else None,
        placements=placements,
        version=getattr(local_tensor, "_version", None),
    )


def _protected_model_state(model: nn.Module) -> dict[str, _ProtectedStateSnapshot]:
    """Snapshot parameters and persistent buffers that hooks may not mutate."""
    snapshots = {}
    for module_fqn, module in model.named_modules():
        prefix = f"{module_fqn}." if module_fqn else ""
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            if parameter is not None:
                snapshots[f"parameter:{prefix}{name}"] = _snapshot_tensor(parameter)
        non_persistent = module._non_persistent_buffers_set  # pylint: disable=protected-access
        for name, buffer in module._buffers.items():  # pylint: disable=protected-access
            if buffer is not None and name not in non_persistent:
                snapshots[f"buffer:{prefix}{name}"] = _snapshot_tensor(buffer)
    return snapshots


def _validate_protected_model_state(
    model: nn.Module,
    expected: dict[str, _ProtectedStateSnapshot],
) -> None:
    """Reject a hook that replaces, reshapes, relayouts, or mutates owned state."""
    actual = _protected_model_state(model)
    if actual.keys() != expected.keys():
        raise RuntimeError(
            "Materialized state hook changed parameters or persistent buffers: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for name, expected_snapshot in expected.items():
        if actual[name] != expected_snapshot:
            raise RuntimeError(
                f"Materialized state hook changed protected {name}: "
                f"expected={expected_snapshot}, actual={actual[name]}"
            )


@torch.no_grad()
def rebuild_materialized_state(
    model: nn.Module,
    context: MaterializationContext,
) -> None:
    """Restore declared model state once after storage and weights are ready."""
    protected = _protected_model_state(model) if context.strict else {}
    for module_fqn, module in model.named_modules():
        specs = module.__dict__.get("_hp_rebuildable_buffer_specs", {})
        registered_hooks = module.__dict__.get("_hp_materialized_state_hooks", ())
        custom_hook = getattr(module, "rebuild_materialized_state_", None)
        if not specs and not registered_hooks and not callable(custom_hook):
            continue
        label = module_fqn or "<root>"
        try:
            if specs:
                _rebuild_registered_buffers(module, context)
            for hook in registered_hooks:
                hook(module, context)
            if callable(custom_hook):
                custom_hook(context)
        except Exception as exc:
            raise RuntimeError(
                "Materialized state rebuild failed for "
                f"{label}: reason={context.reason}, device={context.device}"
            ) from exc
    if context.strict:
        _validate_protected_model_state(model, protected)


__all__ = [
    "MaterializationContext",
    "MaterializationReason",
    "MaterializedStateHook",
    "RebuildableBufferSpec",
    "rebuild_materialized_state",
    "register_materialized_state_hook",
    "register_rebuildable_buffer",
]

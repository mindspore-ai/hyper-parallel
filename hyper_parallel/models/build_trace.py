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
"""Optional, context-local observation of resolved model build patches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Optional

from torch import nn  # pylint: disable=forbidden-backend-import


def _freeze_mapping(values: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return an immutable shallow snapshot without copying runtime objects."""
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class ModuleReplacementRecord:
    """One successfully installed module replacement at its final FQN."""

    module_fqn: str
    source_type: type[nn.Module]
    replacement_type: type[nn.Module]
    factory: Callable[..., nn.Module]
    factory_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class LocalComputeRecord:
    """One resolved and installed local-region compute patch."""

    module_fqn: str
    module_type: type[nn.Module]
    factory: Optional[Callable[..., Any]]
    factory_kwargs: Mapping[str, Any]
    resolved_compute: Callable[..., Any]
    region_dispatch: Optional[bool]
    original_forward: Optional[Callable[..., Any]] = None
    installed_forward: Optional[Callable[..., Any]] = None
    boundary: Any = None
    excluded_subtrees: tuple[str, ...] = ()


@dataclass(frozen=True)
class InnerWrapperRecord:
    """One resolved and atomically installed inner-forward wrapper."""

    module_fqn: str
    target_fqn: str
    wrapper: Any
    request: Any
    secondary_requests: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ModelPatchRecord:
    """Immutable snapshot of patches observed during one model build."""

    module_replacements: tuple[ModuleReplacementRecord, ...] = ()
    local_computes: tuple[LocalComputeRecord, ...] = ()
    inner_wrappers: tuple[InnerWrapperRecord, ...] = ()


@dataclass
class ModelPatchRecorder:
    """Mutable collector scoped by :func:`record_model_patches`."""

    _module_replacements: list[ModuleReplacementRecord] = field(default_factory=list)
    _local_computes: list[LocalComputeRecord] = field(default_factory=list)
    _inner_wrappers: list[InnerWrapperRecord] = field(default_factory=list)

    def freeze(self) -> ModelPatchRecord:
        """Freeze the events collected so far into an immutable snapshot."""
        return ModelPatchRecord(
            module_replacements=tuple(self._module_replacements),
            local_computes=tuple(self._local_computes),
            inner_wrappers=tuple(self._inner_wrappers),
        )


_ACTIVE_RECORDER: ContextVar[Optional[ModelPatchRecorder]] = ContextVar(
    "hyper_parallel_model_patch_recorder",
    default=None,
)


@contextmanager
def record_model_patches() -> Iterator[ModelPatchRecorder]:
    """Collect model patches in the current context until the scope exits."""
    recorder = ModelPatchRecorder()
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def record_module_replacement(
    *,
    module_fqn: str,
    source: nn.Module,
    replacement: nn.Module,
    factory: Callable[..., nn.Module],
    factory_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Record a replacement when observation is active; otherwise do nothing."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        return
    recorder._module_replacements.append(  # pylint: disable=protected-access
        ModuleReplacementRecord(
            module_fqn=module_fqn,
            source_type=type(source),
            replacement_type=type(replacement),
            factory=factory,
            factory_kwargs=_freeze_mapping(factory_kwargs),
        )
    )


def record_local_compute(
    *,
    module_fqn: str,
    module: nn.Module,
    factory: Optional[Callable[..., Any]],
    factory_kwargs: Optional[Mapping[str, Any]],
    resolved_compute: Callable[..., Any],
    region_dispatch: Optional[bool],
    original_forward: Optional[Callable[..., Any]] = None,
    installed_forward: Optional[Callable[..., Any]] = None,
    boundary: Any = None,
    excluded_subtrees: tuple[str, ...] = (),
) -> None:
    """Record an installed local compute when observation is active."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        return
    recorder._local_computes.append(  # pylint: disable=protected-access
        LocalComputeRecord(
            module_fqn=module_fqn,
            module_type=type(module),
            factory=factory,
            factory_kwargs=_freeze_mapping(factory_kwargs),
            resolved_compute=resolved_compute,
            region_dispatch=region_dispatch,
            original_forward=original_forward,
            installed_forward=installed_forward,
            boundary=boundary,
            excluded_subtrees=excluded_subtrees,
        )
    )


def record_inner_wrapper(
    *,
    module_fqn: str,
    target_fqn: str,
    wrapper: Any,
    request: Any,
    secondary_requests: tuple[Any, ...] = (),
) -> None:
    """Record an installed inner wrapper when observation is active."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        return
    recorder._inner_wrappers.append(  # pylint: disable=protected-access
        InnerWrapperRecord(
            module_fqn=module_fqn,
            target_fqn=target_fqn,
            wrapper=wrapper,
            request=request,
            secondary_requests=secondary_requests,
        )
    )


__all__ = [
    "InnerWrapperRecord",
    "LocalComputeRecord",
    "ModelPatchRecord",
    "ModelPatchRecorder",
    "ModuleReplacementRecord",
    "record_inner_wrapper",
    "record_local_compute",
    "record_model_patches",
    "record_module_replacement",
]

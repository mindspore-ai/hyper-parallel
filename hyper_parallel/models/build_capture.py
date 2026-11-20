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
"""Build-local capture of shared model preparation and sharding decisions."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
from typing import Any, Optional

from hyper_parallel.components.modules.module_family import _family_mesh
from hyper_parallel.distributed.apply import _prepare_sharding_meshes
from hyper_parallel.models.preparation_manifest import (
    PreparationManifest,
    capture_preparation_manifest,
    get_preparation_manifest,
    record_native_bindings,
)


@dataclass
class ModelBuildCapture:
    """One isolated result populated by the normal model build pipeline."""

    prepare_model_spec: Optional[Callable[[Any], None]] = field(default=None, repr=False)
    source: Optional[dict[str, Any]] = None
    manifest: Optional[PreparationManifest] = None
    model: Optional[Any] = field(default=None, repr=False)
    plan: Optional[Any] = field(default=None, repr=False)
    mesh: Optional[Any] = field(default=None, repr=False)
    reason: Optional[str] = None
    planning_finished: bool = False
    _spec_prepared: bool = field(default=False, init=False, repr=False)

    @property
    def complete(self) -> bool:
        """Return whether source, preparation, and planning have finished."""
        return self.source is not None and self.manifest is not None and self.planning_finished


_ACTIVE_CAPTURE: ContextVar[Optional[ModelBuildCapture]] = ContextVar(
    "hyper_parallel_model_build_capture",
    default=None,
)


@contextmanager
def capture_model_build_plan(
    *,
    prepare_model_spec: Optional[Callable[[Any], None]] = None,
) -> Iterator[ModelBuildCapture]:
    """Capture one normal model build without changing any public model API.

    Args:
        prepare_model_spec: Optional build-local source compatibility hook.

    Yields:
        An isolated capture populated by shared preparation and planning.
    """
    capture = ModelBuildCapture(prepare_model_spec=prepare_model_spec)
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        with capture_preparation_manifest():
            yield capture
    finally:
        _ACTIVE_CAPTURE.reset(token)


def is_model_build_capture_active() -> bool:
    """Return whether the current context requests an offline model build."""
    return _ACTIVE_CAPTURE.get() is not None


def prepare_model_spec_for_capture(model_spec: Any) -> None:
    """Run a capture-owned source compatibility hook at most once."""
    capture = _ACTIVE_CAPTURE.get()
    if capture is None or capture._spec_prepared:  # pylint: disable=protected-access
        return
    capture._spec_prepared = True  # pylint: disable=protected-access
    if capture.prepare_model_spec is not None:
        capture.prepare_model_spec(model_spec)


def record_model_source_for_capture(
    model: Any,
    *,
    pretrained_model_name_or_path: Any,
) -> None:
    """Record deterministic source identity before shared preparation mutates the tree."""
    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        return
    if capture.source is not None:
        raise RuntimeError("model build capture source has already been recorded")
    config = getattr(model, "config", None)
    model_type = type(model)
    architectures = list(getattr(config, "architectures", None) or ())
    architecture = architectures[0] if architectures else model_type.__name__
    capture.source = {
        "model_type": getattr(config, "model_type", None),
        "architecture": architecture,
        "architectures": architectures,
        "model_class": f"{model_type.__module__}:{model_type.__qualname__}",
        "module_name": model_type.__module__,
        "file_path": inspect.getsourcefile(model_type),
        "pretrained_model_name_or_path": (
            None if pretrained_model_name_or_path is None else str(pretrained_model_name_or_path)
        ),
    }


def record_model_plan_for_capture(
    model: Any,
    plan: Any,
    mesh: Any,
    *,
    reason: Optional[str] = None,
) -> bool:
    """Record a shared planner result and request that runtime application stop."""
    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        return False
    if capture.planning_finished:
        raise RuntimeError("model build capture planning has already finished")
    if capture.source is None:
        raise RuntimeError("model source must be recorded before planning")
    if plan is None:
        capture.manifest = get_preparation_manifest(model)
    else:
        # MeshContext carries the logical expert mesh, so native preparation
        # validates the same axes and injections without creating groups.
        _prepare_sharding_meshes(model, plan, _family_mesh(model, mesh))
        capture.manifest = record_native_bindings(model, plan)
    capture.model = model
    capture.plan = plan
    capture.mesh = mesh
    capture.reason = reason
    capture.planning_finished = True
    return True


def record_model_without_plan_for_capture(model: Any, mesh: Any, sharding_planner: Any) -> None:
    """Explain the shared planner's early exit only for capture consumers."""
    if not is_model_build_capture_active():
        return
    if sharding_planner is None:
        reason = "planner_unavailable"
    elif mesh is None:
        reason = "mesh_unavailable"
    else:
        reason = "pure_data_parallel"
    record_model_plan_for_capture(model, None, mesh, reason=reason)


__all__ = [
    "ModelBuildCapture",
    "PreparationManifest",
    "capture_model_build_plan",
    "is_model_build_capture_active",
]

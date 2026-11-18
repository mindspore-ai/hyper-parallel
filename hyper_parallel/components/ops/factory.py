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
"""Model-local operator selection and direct function dispatch."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cache, wraps
from threading import RLock, local
from types import MappingProxyType
from typing import Any, cast

import torch

from .catalog import builtin_registry
from .registry import OpBuildContext, OpRegistry
from .spec import Fn, OpSpec

# Training execution is synchronous per thread. Dynamo can guard the current
# thread-local runtime and its resolved callables when specializing a graph.
_ACTIVE = local()


class OpRuntime:
    """A model's immutable selections and cache of successfully loaded operators."""

    def __init__(
        self, registry: OpRegistry, selections: Mapping[str, str], context: OpBuildContext,
    ) -> None:
        """Initialize a runtime with its validated configuration and device."""
        self._registry = registry
        self._selections = selections
        self._context = context
        self._resolved: dict[tuple[OpSpec[Any], torch.dtype | None, str, bool, tuple[str, ...]], tuple[str, Any]] = {}
        self._report: dict[tuple[str, str], str] = {}
        self._used_backends: dict[tuple[str, str, str], None] = {}
        self._lock = RLock()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Select this model's operator configuration for the current thread."""
        previous = getattr(_ACTIVE, "runtime", None)
        _ACTIVE.runtime = self
        try:
            yield
        finally:
            _ACTIVE.runtime = previous

    def report(self) -> dict[tuple[str, str], str]:
        """Return the most recently resolved backend for each called contract."""
        with self._lock:
            return self._report.copy()

    def used_backends(self) -> tuple[tuple[str, str, str], ...]:
        """Return every resolved operator, variant and backend used by this runtime."""
        with self._lock:
            return tuple(self._used_backends)

    def _load(
        self, spec: OpSpec[Fn], input_dtype: torch.dtype | None, selection: str,
        native_only: str | None, blocked_backends: tuple[tuple[str, str], ...],
    ) -> tuple[str, Fn]:
        context = self._context
        if input_dtype is not None:
            context = OpBuildContext(context.target_device, context.dtype, context.parallel, input_dtype)
        selected = self._registry.resolve(
            spec, selection, context, native_only=native_only, blocked_backends=blocked_backends,
        )
        self._report[(spec.config_key, spec.variant)] = selected[0]
        self._used_backends[(spec.config_key, spec.variant, selected[0])] = None
        return selected

    def _get_or_resolve(
        self, spec: OpSpec[Fn], x: torch.Tensor, native_only: str | None,
        blocked_backends: tuple[tuple[str, str], ...], default_backend: str,
    ) -> Fn:
        # CUDA grouped kernels can have an input dtype constraint independent
        # of parameter initialization and FSDP mixed precision.
        input_dtype = x.dtype if self._context.target_device.type == "cuda" else None
        selection = self._selections.get(spec.config_key, default_backend)
        key = (spec, input_dtype, selection, native_only is not None, tuple(name for name, _ in blocked_backends))
        selected = self._resolved.get(key)
        if selected is None:
            if torch.compiler.is_compiling():
                # The graph already captures the selected callable. Avoid
                # publishing a Python callable as a side effect of tracing.
                return self._load(spec, input_dtype, selection, native_only, blocked_backends)[1]
            with self._lock:
                selected = self._resolved.get(key)
                if selected is None:
                    selected = self._resolved[key] = self._load(
                        spec, input_dtype, selection, native_only, blocked_backends,
                    )
        return cast(Fn, selected[1])


class _OpRuntimeFactory:
    """Create model-local operator runtimes."""

    def configure(
        self,
        *,
        registry: OpRegistry,
        selections: Mapping[str, str],
        target_device: torch.device | str,
        dtype: torch.dtype | None = None,
        parallel: Mapping[str, int | bool] | None = None,
    ) -> OpRuntime:
        """Validate a selection and build an isolated operator runtime."""
        registry.validate_selections(selections)
        target = torch.device(target_device)
        if target.type not in {"cpu", "cuda", "npu"}:
            raise ValueError(f"Unsupported operator execution device {target}")
        return OpRuntime(
            registry, MappingProxyType(dict(selections)),
            OpBuildContext(target, dtype, MappingProxyType(dict(parallel or {}))),
        )


ops = _OpRuntimeFactory()


def _with_runtime(function: Any, runtime: OpRuntime) -> Any:
    @wraps(function)
    def run(*args: Any, **kwargs: Any) -> Any:
        if getattr(_ACTIVE, "runtime", None) is runtime:
            return function(*args, **kwargs)
        with runtime.activate():
            return function(*args, **kwargs)

    return run


def capture_op_runtime(function: Fn) -> Fn:
    """Preserve the active runtime when activation checkpointing replays a call."""
    runtime = getattr(_ACTIVE, "runtime", None)
    return function if runtime is None else _with_runtime(function, runtime)


def _bind_hf_checkpoint(checkpoint: Any, runtime: OpRuntime) -> Any:
    @wraps(checkpoint)
    def run(function: Any, *args: Any, **kwargs: Any) -> Any:
        return checkpoint(_with_runtime(function, runtime), *args, **kwargs)

    return run


def bind_model_runtime(model: torch.nn.Module, runtime: OpRuntime) -> None:
    """Activate a runtime for direct model calls and HF-native recomputation."""
    model._hp_op_runtime = runtime  # pylint: disable=protected-access
    model.forward = _with_runtime(model.forward, runtime)
    for module in model.modules():
        checkpoint = getattr(module, "_gradient_checkpointing_func", None)
        if callable(checkpoint):
            module._gradient_checkpointing_func = _bind_hf_checkpoint(  # pylint: disable=protected-access
                checkpoint, runtime,
            )


@cache
def _standalone_runtime(device: torch.device) -> OpRuntime:
    return ops.configure(registry=builtin_registry, selections={}, target_device=device)


def _select_for_call(
    spec: OpSpec[Fn], x: torch.Tensor, *, native_only: str | None = None,
    blocked_backends: tuple[tuple[str, str], ...] = (), default_backend: str = "auto",
) -> Fn:
    runtime = getattr(_ACTIVE, "runtime", None)
    if runtime is None:
        if torch.compiler.is_compiling():
            return builtin_registry.resolve(
                spec, default_backend, OpBuildContext(x.device, None, {}, x.dtype), native_only=native_only,
                blocked_backends=blocked_backends,
            )[1]
        runtime = _standalone_runtime(x.device)
    return runtime._get_or_resolve(  # pylint: disable=protected-access
        spec, x, native_only, blocked_backends, default_backend,
    )


def _dispatch(spec: OpSpec[Fn]) -> Fn:
    def call(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return _select_for_call(spec, x)(x, *args, **kwargs)

    return cast(Fn, call)

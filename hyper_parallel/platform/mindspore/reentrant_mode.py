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
"""Backward-phase selection for MindSpore reentrant checkpointing."""
import contextlib
import contextvars
import enum
from functools import wraps
from typing import Any, Callable, Iterator


class ReentrantBackwardMode(enum.Enum):
    """Select which gradients one replay graph traversal returns."""

    NORMAL = "normal"
    DX = "dx"
    DW = "dw"


_CURRENT_BACKWARD_MODE: contextvars.ContextVar[ReentrantBackwardMode] = contextvars.ContextVar(
    "hyper_parallel_reentrant_backward_mode",
    default=ReentrantBackwardMode.NORMAL,
)
_EXPOSE_PARAMETERS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hyper_parallel_reentrant_expose_parameters",
    default=False,
)


def reentrant_backward_mode(mode: ReentrantBackwardMode) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate one GradFunction entry point with its replay backward mode."""
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = _CURRENT_BACKWARD_MODE.set(mode)
            try:
                return function(*args, **kwargs)
            finally:
                _CURRENT_BACKWARD_MODE.reset(token)

        return wrapper

    return decorator


@contextlib.contextmanager
def reentrant_parameter_exposure_ctx(expose: bool = True) -> Iterator[None]:
    """Control whether checkpoint parameters are explicit Function inputs."""
    token = _EXPOSE_PARAMETERS.set(expose)
    try:
        yield
    finally:
        _EXPOSE_PARAMETERS.reset(token)


def get_reentrant_backward_mode() -> ReentrantBackwardMode:
    """Return the active GradFunction replay mode."""
    return _CURRENT_BACKWARD_MODE.get()


def should_expose_reentrant_parameters() -> bool:
    """Return whether checkpoint parameters must be explicit Function inputs."""
    return _EXPOSE_PARAMETERS.get()


__all__ = [
    "ReentrantBackwardMode",
    "get_reentrant_backward_mode",
    "reentrant_backward_mode",
    "reentrant_parameter_exposure_ctx",
    "should_expose_reentrant_parameters",
]
